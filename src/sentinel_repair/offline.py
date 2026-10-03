"""Leak-free prequential evaluation of budgeted audit selectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Sequence

from .advantages import grpo_advantages
from .controller import (
    ControllerAction,
    FixedSentinelRepairController,
    StateTransition,
)
from .data import (
    GroupObservation,
    ParsedLog,
    RolloutGroup,
    RolloutObservation,
    iter_contiguous_step_batches,
)
from .oracle import HiddenEvaluator, HiddenOracle
from .selectors import AuditFeedback, QueryRequest, QuerySelector

Intervention = Literal["replace", "quarantine"]


class BudgetError(RuntimeError):
    """Raised when an action would violate cumulative label-call accounting."""


class InvalidSelectorAction(RuntimeError):
    """Raised when a selector requests unavailable or already bought labels."""


@dataclass(frozen=True)
class BudgetSnapshot:
    seen_labels: int
    allowed_calls: int
    spent_calls: int

    @property
    def remaining_calls(self) -> int:
        return self.allowed_calls - self.spent_calls


class CumulativeLabelBudget:
    """A strict prefix budget: calls <= floor(rate * labels seen)."""

    def __init__(self, rate: float | str | Fraction = "0.01") -> None:
        self.rate = rate if isinstance(rate, Fraction) else Fraction(str(rate))
        if self.rate < 0 or self.rate > 1:
            raise ValueError("budget rate must be in [0, 1]")
        self._seen = 0
        self._spent = 0

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def allowed(self) -> int:
        return (self._seen * self.rate.numerator) // self.rate.denominator

    @property
    def remaining(self) -> int:
        return self.allowed - self._spent

    def observe(self, label_count: int) -> BudgetSnapshot:
        if isinstance(label_count, bool) or not isinstance(label_count, int):
            raise TypeError("label_count must be an integer")
        if label_count < 0:
            raise ValueError("label_count cannot be negative")
        self._seen += label_count
        return self.snapshot()

    def spend(self, label_calls: int) -> BudgetSnapshot:
        if isinstance(label_calls, bool) or not isinstance(label_calls, int):
            raise TypeError("label_calls must be an integer")
        if label_calls < 0:
            raise ValueError("label_calls cannot be negative")
        if label_calls > self.remaining:
            raise BudgetError(
                f"cannot spend {label_calls} calls with only {self.remaining} "
                f"available ({self._spent}/{self.allowed} already used)"
            )
        self._spent += label_calls
        return self.snapshot()

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(self._seen, self.allowed, self._spent)


class _GroupState:
    """Mutable intervention state kept on the runner side of the API wall."""

    def __init__(self, group: RolloutGroup, intervention: Intervention) -> None:
        self.group = group
        self.intervention = intervention
        self.revealed: dict[int, float] = {}
        self.version = 0
        self.quarantined = False

    @property
    def current_rewards(self) -> tuple[float, ...]:
        return tuple(
            self.revealed.get(row.rollout_id, row.cheap_reward)
            for row in self.group.rollouts
        )

    @property
    def coefficients(self) -> tuple[float, ...]:
        if self.quarantined:
            return (0.0,) * len(self.group)
        return tuple(float(value) for value in grpo_advantages(self.current_rewards))

    def view(self) -> GroupObservation:
        coefficients = self.coefficients
        rows = tuple(
            RolloutObservation(
                rollout_id=row.rollout_id,
                group_id=self.group.group_id,
                position=position,
                step=row.step,
                prompt=row.prompt,
                completion=row.completion,
                cheap_reward=row.cheap_reward,
                current_reward=self.revealed.get(row.rollout_id, row.cheap_reward),
                advantage=coefficients[position],
                audited=row.rollout_id in self.revealed,
                metadata=row.metadata,
            )
            for position, row in enumerate(self.group.rollouts)
        )
        return GroupObservation(
            group_id=self.group.group_id,
            step=self.group.step,
            prompt=self.group.prompt,
            rollouts=rows,
            version=self.version,
            quarantined=self.quarantined,
        )

    def reveal(self, rollout_id: int, label: float) -> None:
        if rollout_id in self.revealed:
            raise InvalidSelectorAction(f"rollout {rollout_id} is already audited")
        row_by_id = {row.rollout_id: row for row in self.group.rollouts}
        if rollout_id not in row_by_id:
            raise InvalidSelectorAction(
                f"rollout {rollout_id} does not belong to group {self.group.group_id}"
            )
        self.revealed[rollout_id] = float(label)
        self.version += 1
        if self.intervention == "quarantine" and not math.isclose(
            row_by_id[rollout_id].cheap_reward,
            float(label),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            self.quarantined = True


@dataclass(frozen=True)
class AuditEvent:
    batch_index: int
    step: str
    call_index: int
    rollout_id: int
    group_id: int
    reason: str
    cheap_reward: float
    observed_label: float
    label_error: bool
    advantage_before: float
    predicted_risk: float | None
    priority: float | None
    group_version_after: int
    residual_before: float
    residual_after: float
    marginal_residual_removed: float | None
    atomic_action: bool
    action_cost: int
    action_residual_removed: float


@dataclass(frozen=True)
class BatchSnapshot:
    batch_index: int
    step: str
    groups_seen: int
    seen_labels: int
    allowed_calls: int
    spent_calls: int
    batch_calls: int
    batch_cheap_residual: float
    batch_current_residual: float
    cumulative_cheap_residual: float
    cumulative_current_residual: float


@dataclass(frozen=True)
class OfflineRunResult:
    selector_name: str
    intervention: Intervention
    budget_rate: Fraction
    events: tuple[AuditEvent, ...]
    batches: tuple[BatchSnapshot, ...]
    rollout_count: int
    oracle_calls: int
    cheap_residual: float
    final_residual: float
    controller_actions: tuple[ControllerAction, ...] = ()
    controller_transitions: tuple[StateTransition, ...] = ()

    @property
    def residual_removed(self) -> float:
        return self.cheap_residual - self.final_residual

    @property
    def realized_call_fraction(self) -> float:
        if self.rollout_count == 0:
            return 0.0
        return self.oracle_calls / self.rollout_count


class OfflineRunner:
    """Run a selector chronologically while keeping future truth inaccessible."""

    def __init__(
        self,
        selector: QuerySelector,
        oracle: HiddenOracle,
        evaluator: HiddenEvaluator,
        *,
        budget_rate: float | str | Fraction = "0.01",
        intervention: Intervention = "replace",
    ) -> None:
        if intervention not in {"replace", "quarantine"}:
            raise ValueError(f"unknown intervention: {intervention!r}")
        if oracle.calls:
            raise ValueError("OfflineRunner requires a fresh, unused oracle")
        self.selector = selector
        self.oracle = oracle
        self.evaluator = evaluator
        self.budget = CumulativeLabelBudget(budget_rate)
        self.intervention = intervention
        self._ran = False

    @staticmethod
    def _candidate(group: GroupObservation, rollout_id: int) -> RolloutObservation:
        for row in group.rollouts:
            if row.rollout_id == rollout_id:
                return row
        raise InvalidSelectorAction(
            f"selector requested rollout {rollout_id} from the wrong group"
        )

    def _evaluate_states(self, states: Sequence[_GroupState]) -> float:
        return self.evaluator.evaluate_coefficients(
            (state.group, state.coefficients) for state in states
        ).coefficient_residual

    def _state_residual(self, state: _GroupState) -> float:
        return self.evaluator.group_coefficient_residual(
            state.group, state.coefficients
        )

    @staticmethod
    def _validate_request(
        request: QueryRequest,
        states_by_rollout: dict[int, _GroupState],
        *,
        max_labels: int,
    ) -> None:
        if request.cost > max_labels:
            raise BudgetError(
                f"selector requested {request.cost} labels with {max_labels} available"
            )
        for rollout_id in request.rollout_ids:
            state = states_by_rollout.get(rollout_id)
            if state is None:
                raise InvalidSelectorAction(
                    f"selector requested unavailable rollout {rollout_id}"
                )
            if rollout_id in state.revealed:
                raise InvalidSelectorAction(
                    f"selector requested already audited rollout {rollout_id}"
                )
            if state.quarantined:
                raise InvalidSelectorAction(
                    f"selector requested rollout {rollout_id} from a quarantined group"
                )
        requested_states = [states_by_rollout[item] for item in request.rollout_ids]
        if request.cost > 1 and not request.atomic:
            raise InvalidSelectorAction("multi-label requests must declare atomic=True")
        if request.atomic:
            first = requested_states[0]
            if any(state is not first for state in requested_states[1:]):
                raise InvalidSelectorAction(
                    "an atomic request must stay within one group"
                )
            unqueried = {row.rollout_id for row in first.view().unqueried}
            if set(request.rollout_ids) != unqueried:
                raise InvalidSelectorAction(
                    "an atomic request must contain every currently unqueried sibling"
                )

    def run(self, groups: Sequence[RolloutGroup]) -> OfflineRunResult:
        if self._ran:
            raise RuntimeError("an OfflineRunner instance can only be run once")
        self._ran = True

        rollout_ids = [row.rollout_id for group in groups for row in group.rollouts]
        if len(rollout_ids) != len(set(rollout_ids)):
            raise ValueError("rollout ids must be unique across the full log")

        all_states: list[_GroupState] = []
        events: list[AuditEvent] = []
        snapshots: list[BatchSnapshot] = []
        cumulative_cheap_residual = 0.0
        cumulative_current_residual = 0.0

        for batch_index, batch in enumerate(iter_contiguous_step_batches(groups)):
            batch_states = [_GroupState(group, self.intervention) for group in batch]
            all_states.extend(batch_states)
            batch_cheap_residual = sum(
                self.evaluator.group_residual(state.group, state.group.cheap_rewards)
                for state in batch_states
            )
            cumulative_cheap_residual += batch_cheap_residual
            cumulative_current_residual += batch_cheap_residual
            batch_label_count = sum(len(state.group) for state in batch_states)
            self.budget.observe(batch_label_count)
            calls_at_batch_start = self.budget.spent

            states_by_rollout = {
                row.rollout_id: state
                for state in batch_states
                for row in state.group.rollouts
            }

            while self.budget.remaining > 0:
                views = tuple(state.view() for state in batch_states)
                request = self.selector.propose(
                    views,
                    max_labels=self.budget.remaining,
                )
                if request is None:
                    break
                self._validate_request(
                    request,
                    states_by_rollout,
                    max_labels=self.budget.remaining,
                )

                calls_before_bundle = self.oracle.calls
                # Reserve the whole action first.  Thus an atomic group of n
                # labels can never sneak through an allowance smaller than n.
                self.budget.spend(request.cost)
                answers = self.oracle.query_many(request.rollout_ids)

                if request.atomic:
                    state = states_by_rollout[answers[0].rollout_id]
                    frozen_group = state.view()
                    frozen_candidates = {
                        answer.rollout_id: self._candidate(
                            frozen_group, answer.rollout_id
                        )
                        for answer in answers
                    }
                    group_residual_before = self._state_residual(state)
                    residual_before_action = cumulative_current_residual
                    for answer in answers:
                        state.reveal(answer.rollout_id, answer.label)
                    # A complete bundle has bought every label, so use the
                    # oracle-normalized update even under a quarantine policy.
                    if len(state.revealed) == len(state.group):
                        state.quarantined = False
                    group_after = state.view()
                    group_residual_after = self._state_residual(state)
                    action_removed = group_residual_before - group_residual_after
                    cumulative_current_residual -= action_removed
                    for offset, answer in enumerate(answers, start=1):
                        candidate_before = frozen_candidates[answer.rollout_id]
                        label_error = not math.isclose(
                            candidate_before.cheap_reward,
                            answer.label,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        predicted_risk = request.predicted_risks.get(answer.rollout_id)
                        self.selector.observe(
                            AuditFeedback(
                                candidate_before=candidate_before,
                                group_before=frozen_group,
                                group_after=group_after,
                                observed_label=answer.label,
                                label_error=label_error,
                                predicted_risk=predicted_risk,
                                priority=request.priority,
                            )
                        )
                        events.append(
                            AuditEvent(
                                batch_index=batch_index,
                                step=state.group.step,
                                call_index=calls_before_bundle + offset,
                                rollout_id=answer.rollout_id,
                                group_id=state.group.group_id,
                                reason=request.reason,
                                cheap_reward=candidate_before.cheap_reward,
                                observed_label=answer.label,
                                label_error=label_error,
                                advantage_before=candidate_before.advantage,
                                predicted_risk=predicted_risk,
                                priority=request.priority,
                                group_version_after=group_after.version,
                                residual_before=residual_before_action,
                                residual_after=cumulative_current_residual,
                                marginal_residual_removed=None,
                                atomic_action=True,
                                action_cost=request.cost,
                                # This is an action-level quantity.  Atomic
                                # bundles emit one event per paid label, but
                                # counting the bundle gain on every row would
                                # multiply aggregate gains by its label cost.
                                action_residual_removed=(
                                    action_removed if offset == 1 else 0.0
                                ),
                            )
                        )
                    continue

                for offset, answer in enumerate(answers, start=1):
                    state = states_by_rollout[answer.rollout_id]
                    group_before = state.view()
                    candidate_before = self._candidate(group_before, answer.rollout_id)
                    group_residual_before = self._state_residual(state)
                    residual_before = cumulative_current_residual

                    state.reveal(answer.rollout_id, answer.label)
                    group_after = state.view()
                    group_residual_after = self._state_residual(state)
                    removed = group_residual_before - group_residual_after
                    cumulative_current_residual -= removed
                    residual_after = cumulative_current_residual
                    label_error = not math.isclose(
                        candidate_before.cheap_reward,
                        answer.label,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    predicted_risk = request.predicted_risks.get(answer.rollout_id)
                    feedback = AuditFeedback(
                        candidate_before=candidate_before,
                        group_before=group_before,
                        group_after=group_after,
                        observed_label=answer.label,
                        label_error=label_error,
                        predicted_risk=predicted_risk,
                        priority=request.priority,
                    )
                    # The selector update is deliberately last: its prediction
                    # and choice preceded this label disclosure.
                    self.selector.observe(feedback)
                    events.append(
                        AuditEvent(
                            batch_index=batch_index,
                            step=state.group.step,
                            call_index=calls_before_bundle + offset,
                            rollout_id=answer.rollout_id,
                            group_id=state.group.group_id,
                            reason=request.reason,
                            cheap_reward=candidate_before.cheap_reward,
                            observed_label=answer.label,
                            label_error=label_error,
                            advantage_before=candidate_before.advantage,
                            predicted_risk=predicted_risk,
                            priority=request.priority,
                            group_version_after=group_after.version,
                            residual_before=residual_before,
                            residual_after=residual_after,
                            marginal_residual_removed=residual_before - residual_after,
                            atomic_action=False,
                            action_cost=1,
                            action_residual_removed=residual_before - residual_after,
                        )
                    )

            final_views = tuple(state.view() for state in batch_states)
            self.selector.end_batch(final_views)
            batch_current_residual = sum(
                self._state_residual(state) for state in batch_states
            )
            budget_snapshot = self.budget.snapshot()
            if budget_snapshot.spent_calls > budget_snapshot.allowed_calls:
                raise AssertionError("internal error: cumulative budget was exceeded")
            snapshots.append(
                BatchSnapshot(
                    batch_index=batch_index,
                    step=batch[0].step,
                    groups_seen=len(all_states),
                    seen_labels=budget_snapshot.seen_labels,
                    allowed_calls=budget_snapshot.allowed_calls,
                    spent_calls=budget_snapshot.spent_calls,
                    batch_calls=self.budget.spent - calls_at_batch_start,
                    batch_cheap_residual=batch_cheap_residual,
                    batch_current_residual=batch_current_residual,
                    cumulative_cheap_residual=cumulative_cheap_residual,
                    cumulative_current_residual=cumulative_current_residual,
                )
            )

        if isinstance(self.selector, FixedSentinelRepairController):
            controller_actions = self.selector.actions
            controller_transitions = (
                self.selector.transitions
                if hasattr(self.selector, "transitions")
                else ()
            )
        else:
            controller_actions = ()
            controller_transitions = ()

        return OfflineRunResult(
            selector_name=type(self.selector).__name__,
            intervention=self.intervention,
            budget_rate=self.budget.rate,
            events=tuple(events),
            batches=tuple(snapshots),
            rollout_count=len(rollout_ids),
            oracle_calls=self.oracle.calls,
            cheap_residual=cumulative_cheap_residual,
            final_residual=cumulative_current_residual,
            controller_actions=controller_actions,
            controller_transitions=controller_transitions,
        )


def run_parsed_log(
    parsed: ParsedLog,
    selector: QuerySelector,
    *,
    budget_rate: float | str | Fraction = "0.01",
    intervention: Intervention = "replace",
) -> OfflineRunResult:
    """Convenience entrypoint used by the analysis CLI."""

    runner = OfflineRunner(
        selector,
        parsed.make_oracle(),
        parsed.make_evaluator(),
        budget_rate=budget_rate,
        intervention=intervention,
    )
    return runner.run(parsed.groups)
