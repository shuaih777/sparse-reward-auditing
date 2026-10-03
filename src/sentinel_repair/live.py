"""Streaming, oracle-budgeted reward correction for online GRPO batches.

This module deliberately has no dependency on TRL, torch, or distributed
training.  A rank-zero training hook can gather one global generation batch,
turn it into :class:`PublicRollout` groups, call :meth:`process_batch`, and
broadcast the returned effective rewards.

The information boundary is explicit:

* input rows contain prompt, completion, and the cheap binary reward only;
* selectors see :class:`~sentinel_repair.data.GroupObservation` objects;
* the oracle receives one public :class:`OracleRequest` only after that row
  has been selected and charged to the cumulative budget; and
* a successful return is the only path by which effective rewards are made
  available.  A runtime anomaly permanently closes the engine and raises,
  so callers cannot silently train on the unchecked cheap rewards.

The default policy is the mechanism-matched configuration used by the first
online experiment: a token-enrichment risk model, exact expected group
marginals, and a five-percent uniform sentinel quota within the one-percent
oracle-call budget.  Oracle calls remain sequential: after every bought label,
the whole affected GRPO group is rebuilt before the next proposal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .advantages import grpo_advantages
from .controller import (
    AuditArm,
    ControllerAction,
    FixedSentinelRepairController,
    StateTransition,
)
from .data import GroupObservation, LoggedRollout, RolloutGroup, RolloutObservation
from .group_selectors import ExpectedMarginalGroupSelector
from .offline import (
    BudgetSnapshot,
    CumulativeLabelBudget,
    InvalidSelectorAction,
)
from .propagation import (
    ContrastGateConfig,
    FailureFamilyContrastGate,
)
from .selectors import (
    AuditFeedback,
    ErrorRiskModel,
    FalsePositiveOnlyRiskModel,
    QueryRequest,
    QuerySelector,
    TokenEnrichmentRiskModel,
)

Intervention = Literal["confirm", "replace", "quarantine"]


class StreamingAuditFailure(RuntimeError):
    """A post-admission failure that permanently closes the engine.

    Once a valid batch has entered the engine, selector and oracle operations
    mutate cumulative state.  Retrying could therefore double-query labels or
    expose a different random action.  The safe behavior is to abort the
    policy update and replace the engine, rather than return cheap rewards.
    """


@dataclass(frozen=True, slots=True)
class PublicRollout:
    """One rollout at the online engine's public input boundary.

    There is intentionally no metadata bag or caller-supplied identifier:
    arbitrary metadata can smuggle truth, and global identifiers must be
    minted monotonically by the single rank-zero engine instance.
    """

    prompt: str
    completion: str
    cheap_reward: float

    def __post_init__(self) -> None:
        if type(self.prompt) is not str:  # exact types reject surprising wrappers
            raise TypeError("prompt must be a string")
        if type(self.completion) is not str:
            raise TypeError("completion must be a string")
        if isinstance(self.cheap_reward, bool):
            raise TypeError("cheap_reward must be a binary number, not bool")
        try:
            reward = float(self.cheap_reward)
        except (TypeError, ValueError) as exc:
            raise TypeError("cheap_reward must be numeric") from exc
        if not math.isfinite(reward) or reward not in {0.0, 1.0}:
            raise ValueError("cheap_reward must be a finite binary label")
        object.__setattr__(self, "cheap_reward", reward)


@dataclass(frozen=True, slots=True)
class OracleRequest:
    """The public payload disclosed for one paid oracle call."""

    rollout_id: int
    group_id: int
    position: int
    batch_index: int
    step: str
    prompt: str
    completion: str
    cheap_reward: float


OracleQuery = Callable[[OracleRequest], float]


@dataclass(frozen=True, slots=True)
class InterventionTransition:
    """One auditable change to a group's reward/intervention state."""

    call_index: int
    rollout_id: int
    group_id: int
    group_version_before: int
    group_version_after: int
    intervention: Intervention
    quarantined_before: bool
    quarantined_after: bool


@dataclass(frozen=True, slots=True)
class StreamingAuditRecord:
    """Decision-time and revealed information for one purchased label."""

    batch_index: int
    step: str
    call_index: int
    rollout_id: int
    group_id: int
    position: int
    reason: str
    arm: AuditArm | None
    cheap_reward: float
    observed_label: float
    label_error: bool
    advantage_before: float
    predicted_risk: float | None
    priority: float | None
    intervention: Intervention
    group_version_before: int
    group_version_after: int
    group_quarantined: bool


@dataclass(frozen=True, slots=True)
class PropagationRecord:
    """Public evidence and reward effect for one post-audit hard gate."""

    batch_index: int
    step: str
    rollout_id: int
    group_id: int
    position: int
    token: str
    positive_documents: int
    negative_documents: int
    error_coverage: float
    prevalence_ratio: float
    cheap_reward: float
    effective_reward: float


@dataclass(frozen=True, slots=True)
class StreamingBatchResult:
    """Effective rewards and a complete audit trail for one admitted batch."""

    batch_index: int
    step: str
    rollout_ids: tuple[tuple[int, ...], ...]
    group_ids: tuple[int, ...]
    effective_rewards: tuple[tuple[float, ...], ...]
    quarantined_group_ids: tuple[int, ...]
    audit_records: tuple[StreamingAuditRecord, ...]
    transitions: tuple[InterventionTransition, ...]
    controller_actions: tuple[ControllerAction, ...]
    controller_transitions: tuple[StateTransition, ...]
    budget: BudgetSnapshot
    propagation_records: tuple[PropagationRecord, ...] = ()
    propagation_config: ContrastGateConfig | None = None

    @property
    def effective_rewards_flat(self) -> tuple[float, ...]:
        """Rewards in the same group-major order as the input rows."""

        return tuple(value for group in self.effective_rewards for value in group)

    @property
    def rollout_ids_flat(self) -> tuple[int, ...]:
        return tuple(value for group in self.rollout_ids for value in group)


class _LiveGroupState:
    """Mutable state private to one streaming batch."""

    def __init__(self, group: RolloutGroup) -> None:
        self.group = group
        self.revealed: dict[int, float] = {}
        self.version = 0
        self.quarantined = False
        self.cheap_is_constant = len(set(group.cheap_rewards)) == 1
        self._rows_by_id = {row.rollout_id: row for row in group.rollouts}

    @property
    def current_rewards(self) -> tuple[float, ...]:
        return tuple(
            self.revealed.get(row.rollout_id, row.cheap_reward)
            for row in self.group.rollouts
        )

    @property
    def effective_rewards(self) -> tuple[float, ...]:
        # A constant vector makes the downstream group-relative update zero.
        # Use explicit zeros instead of relying on the original constant value.
        if self.quarantined:
            return (0.0,) * len(self.group)
        return self.current_rewards

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

    def reveal(self, rollout_id: int, label: float) -> Intervention:
        if rollout_id in self.revealed:
            raise InvalidSelectorAction(f"rollout {rollout_id} is already audited")
        try:
            row = self._rows_by_id[rollout_id]
        except KeyError as exc:
            raise InvalidSelectorAction(
                f"rollout {rollout_id} is not in group {self.group.group_id}"
            ) from exc

        mismatch = not math.isclose(
            row.cheap_reward, label, rel_tol=0.0, abs_tol=1e-12
        )
        self.revealed[rollout_id] = label
        self.version += 1
        if mismatch and self.cheap_is_constant:
            # Partial replacement of a saturated group creates an artificial
            # non-zero normalized update.  Zero the whole group instead.
            self.quarantined = True
            return "quarantine"
        if mismatch:
            return "replace"
        return "confirm"


class StreamingAuditEngine:
    """Stateful online reward auditor intended to live on rank zero.

    ``process_batch`` accepts groups in generation order.  It charges at most
    ``floor(budget_rate * total_seen)`` oracle calls across every prefix of
    successfully admitted batches.  The object is intentionally stateful and
    must not be instantiated independently on multiple distributed ranks.
    """

    def __init__(
        self,
        *,
        budget_rate: float | str | Fraction = "0.01",
        sentinel_fraction: float | str | Fraction = "0.05",
        seed: int = 0,
        risk_model: ErrorRiskModel | None = None,
        selector: QuerySelector | None = None,
        propagation_model: TokenEnrichmentRiskModel | None = None,
        propagation_config: ContrastGateConfig | None = None,
    ) -> None:
        if selector is not None and risk_model is not None:
            raise ValueError("risk_model cannot be combined with a custom selector")
        if (propagation_model is None) != (propagation_config is None):
            raise ValueError(
                "propagation_model and propagation_config must be provided together"
            )
        if propagation_config is not None and not isinstance(
            propagation_config, ContrastGateConfig
        ):
            raise TypeError("propagation_config must be ContrastGateConfig")
        if propagation_model is not None:
            if selector is not None:
                raise ValueError(
                    "propagation requires the engine-owned selector so the token "
                    "model is shared"
                )
            if risk_model is None:
                risk_model = FalsePositiveOnlyRiskModel(propagation_model)
            elif not (
                isinstance(risk_model, FalsePositiveOnlyRiskModel)
                and risk_model.base_model is propagation_model
            ):
                raise ValueError(
                    "risk_model must be FalsePositiveOnlyRiskModel wrapping the "
                    "same propagation_model"
                )
        if selector is None:
            model = risk_model or TokenEnrichmentRiskModel()
            repair = ExpectedMarginalGroupSelector(model)
            selector = FixedSentinelRepairController(
                repair,
                sentinel_fraction=sentinel_fraction,
                seed=seed,
            )
        self.selector = selector
        self.propagation_config = propagation_config
        self.propagation_gate = (
            FailureFamilyContrastGate(propagation_model, propagation_config)
            if propagation_model is not None and propagation_config is not None
            else None
        )
        self.budget = CumulativeLabelBudget(budget_rate)
        self._next_rollout_id = 0
        self._next_group_id = 0
        self._next_batch_index = 0
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def seen_labels(self) -> int:
        return self.budget.seen

    @property
    def spent_calls(self) -> int:
        return self.budget.spent

    @property
    def allowed_calls(self) -> int:
        return self.budget.allowed

    @property
    def budget_snapshot(self) -> BudgetSnapshot:
        return self.budget.snapshot()

    @property
    def next_rollout_id(self) -> int:
        return self._next_rollout_id

    @property
    def next_group_id(self) -> int:
        return self._next_group_id

    @staticmethod
    def _validate_public_batch(
        groups: Sequence[Sequence[PublicRollout]],
    ) -> tuple[tuple[PublicRollout, ...], ...]:
        try:
            materialized = tuple(tuple(group) for group in groups)
        except TypeError as exc:
            raise TypeError("groups must be an iterable of rollout groups") from exc
        for group_index, group in enumerate(materialized):
            if not group:
                raise ValueError(f"public group {group_index} cannot be empty")
            for row in group:
                # An exact-type boundary prevents a subclass from attaching an
                # oracle label that a future selector adapter might inspect.
                if type(row) is not PublicRollout:
                    raise TypeError("every public row must be exactly PublicRollout")
            prompt = group[0].prompt
            if any(row.prompt != prompt for row in group[1:]):
                raise ValueError(
                    f"all rows in public group {group_index} must share a prompt"
                )
        return materialized

    def _admit(
        self,
        public_groups: tuple[tuple[PublicRollout, ...], ...],
        *,
        batch_index: int,
        step: str,
    ) -> tuple[_LiveGroupState, ...]:
        states: list[_LiveGroupState] = []
        for public_group in public_groups:
            group_id = self._next_group_id
            self._next_group_id += 1
            logged: list[LoggedRollout] = []
            for public in public_group:
                rollout_id = self._next_rollout_id
                self._next_rollout_id += 1
                logged.append(
                    LoggedRollout(
                        rollout_id=rollout_id,
                        step=step,
                        prompt=public.prompt,
                        completion=public.completion,
                        cheap_reward=public.cheap_reward,
                    )
                )
            states.append(
                _LiveGroupState(
                    RolloutGroup(
                        group_id=group_id,
                        step=step,
                        prompt=public_group[0].prompt,
                        rollouts=tuple(logged),
                    )
                )
            )
        self.budget.observe(sum(len(state.group) for state in states))
        self._next_batch_index = batch_index + 1
        return tuple(states)

    @staticmethod
    def _candidate(
        group: GroupObservation, rollout_id: int
    ) -> RolloutObservation:
        for row in group.rollouts:
            if row.rollout_id == rollout_id:
                return row
        raise InvalidSelectorAction(
            f"selector requested rollout {rollout_id} from the wrong group"
        )

    @staticmethod
    def _validate_request(
        request: QueryRequest,
        states_by_rollout: dict[int, _LiveGroupState],
        *,
        max_labels: int,
    ) -> _LiveGroupState:
        # The live contract is deliberately one-label-at-a-time so the group
        # and risk model are recomputed after every oracle answer.
        if request.cost != 1 or request.atomic:
            raise InvalidSelectorAction(
                "streaming audit actions must request exactly one non-atomic label"
            )
        if request.cost > max_labels:
            raise InvalidSelectorAction("selector request exceeds available budget")
        rollout_id = request.rollout_ids[0]
        try:
            state = states_by_rollout[rollout_id]
        except KeyError as exc:
            raise InvalidSelectorAction(
                f"selector requested unavailable rollout {rollout_id}"
            ) from exc
        if rollout_id in state.revealed:
            raise InvalidSelectorAction(
                f"selector requested already audited rollout {rollout_id}"
            )
        if state.quarantined:
            raise InvalidSelectorAction(
                f"selector requested rollout {rollout_id} from a quarantined group"
            )
        return state

    @staticmethod
    def _oracle_label(value: object) -> float:
        if isinstance(value, bool):
            raise TypeError("oracle label must be a binary number, not bool")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("oracle label must be numeric") from exc
        if not math.isfinite(parsed) or parsed not in {0.0, 1.0}:
            raise ValueError("oracle label must be a finite binary label")
        return parsed

    def _close(self, message: str, error: BaseException) -> None:
        self._failed = True
        raise StreamingAuditFailure(message) from error

    def _apply_propagation(
        self,
        states: Sequence[_LiveGroupState],
        *,
        batch_index: int,
        step: str,
    ) -> tuple[dict[int, float], tuple[PropagationRecord, ...]]:
        """Freeze hard-zero actions after every audit update in this batch."""

        if self.propagation_gate is None:
            return {}, ()
        overrides: dict[int, float] = {}
        records: list[PropagationRecord] = []
        for state in states:
            group = state.view()
            # ``decide`` also checks this, but the group-level branch makes
            # quarantine precedence explicit and avoids needless token scans.
            if group.quarantined:
                continue
            for candidate in group.rollouts:
                decision = self.propagation_gate.decide(candidate, group)
                if decision is None:
                    continue
                overrides[candidate.rollout_id] = 0.0
                records.append(
                    PropagationRecord(
                        batch_index=batch_index,
                        step=step,
                        rollout_id=candidate.rollout_id,
                        group_id=group.group_id,
                        position=candidate.position,
                        token=decision.token,
                        positive_documents=decision.positive_documents,
                        negative_documents=decision.negative_documents,
                        error_coverage=decision.error_coverage,
                        prevalence_ratio=decision.prevalence_ratio,
                        cheap_reward=candidate.cheap_reward,
                        effective_reward=0.0,
                    )
                )
        return overrides, tuple(records)

    def process_batch(
        self,
        groups: Sequence[Sequence[PublicRollout]],
        oracle_query: OracleQuery,
        *,
        step: str | int | None = None,
    ) -> StreamingBatchResult:
        """Audit one global policy batch and return training-safe rewards.

        The caller must treat an exception as a hard failure of the policy
        update.  In particular, it must not catch :class:`StreamingAuditFailure`
        and substitute the original cheap rewards.
        """

        if self._failed:
            raise StreamingAuditFailure("streaming audit engine is permanently closed")
        if not callable(oracle_query):
            raise TypeError("oracle_query must be callable")

        # Validate the whole public boundary before changing any cumulative
        # counters.  Errors here are ordinary caller input errors, not partial
        # online transactions, so they do not poison the engine.
        public_groups = self._validate_public_batch(groups)
        batch_index = self._next_batch_index
        parsed_step = str(batch_index if step is None else step)
        states = self._admit(
            public_groups,
            batch_index=batch_index,
            step=parsed_step,
        )

        action_source = getattr(self.selector, "actions", ())
        transition_source = getattr(self.selector, "transitions", ())
        actions_at_start = len(action_source)
        controller_transitions_at_start = len(transition_source)
        records: list[StreamingAuditRecord] = []
        transitions: list[InterventionTransition] = []
        propagation_records: tuple[PropagationRecord, ...] = ()
        propagation_overrides: dict[int, float] = {}
        states_by_rollout = {
            row.rollout_id: state
            for state in states
            for row in state.group.rollouts
        }

        try:
            while self.budget.remaining > 0:
                views = tuple(state.view() for state in states)
                action_count_before = len(getattr(self.selector, "actions", ()))
                request = self.selector.propose(
                    views,
                    max_labels=self.budget.remaining,
                )
                if request is None:
                    break
                if not isinstance(request, QueryRequest):
                    raise InvalidSelectorAction(
                        "selector must return QueryRequest or None"
                    )
                state = self._validate_request(
                    request,
                    states_by_rollout,
                    max_labels=self.budget.remaining,
                )
                rollout_id = request.rollout_ids[0]
                group_before = state.view()
                candidate_before = self._candidate(group_before, rollout_id)

                # Charge before disclosure.  A failed oracle call cannot be
                # retried as a free side-channel observation.
                self.budget.spend(1)
                call_index = self.budget.spent
                oracle_request = OracleRequest(
                    rollout_id=rollout_id,
                    group_id=state.group.group_id,
                    position=candidate_before.position,
                    batch_index=batch_index,
                    step=parsed_step,
                    prompt=candidate_before.prompt,
                    completion=candidate_before.completion,
                    cheap_reward=candidate_before.cheap_reward,
                )
                observed_label = self._oracle_label(oracle_query(oracle_request))
                mismatch = not math.isclose(
                    candidate_before.cheap_reward,
                    observed_label,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                quarantined_before = state.quarantined
                version_before = state.version
                intervention = state.reveal(rollout_id, observed_label)
                group_after = state.view()
                predicted_risk = request.predicted_risks.get(rollout_id)
                self.selector.observe(
                    AuditFeedback(
                        candidate_before=candidate_before,
                        group_before=group_before,
                        group_after=group_after,
                        observed_label=observed_label,
                        label_error=mismatch,
                        predicted_risk=predicted_risk,
                        priority=request.priority,
                    )
                )

                new_actions = getattr(self.selector, "actions", ())
                controller_action = (
                    new_actions[-1]
                    if len(new_actions) == action_count_before + 1
                    else None
                )
                transition = InterventionTransition(
                    call_index=call_index,
                    rollout_id=rollout_id,
                    group_id=state.group.group_id,
                    group_version_before=version_before,
                    group_version_after=state.version,
                    intervention=intervention,
                    quarantined_before=quarantined_before,
                    quarantined_after=state.quarantined,
                )
                transitions.append(transition)
                records.append(
                    StreamingAuditRecord(
                        batch_index=batch_index,
                        step=parsed_step,
                        call_index=call_index,
                        rollout_id=rollout_id,
                        group_id=state.group.group_id,
                        position=candidate_before.position,
                        reason=request.reason,
                        arm=(controller_action.arm if controller_action else None),
                        cheap_reward=candidate_before.cheap_reward,
                        observed_label=observed_label,
                        label_error=mismatch,
                        advantage_before=candidate_before.advantage,
                        predicted_risk=predicted_risk,
                        priority=request.priority,
                        intervention=intervention,
                        group_version_before=version_before,
                        group_version_after=state.version,
                        group_quarantined=state.quarantined,
                    )
                )

            final_views = tuple(state.view() for state in states)
            self.selector.end_batch(final_views)
            # Propagation is intentionally post-selection.  It sees all model
            # updates bought in this batch, but cannot alter which labels that
            # same batch audited or the order in which they were purchased.
            propagation_overrides, propagation_records = self._apply_propagation(
                states,
                batch_index=batch_index,
                step=parsed_step,
            )
            snapshot = self.budget.snapshot()
            if snapshot.spent_calls > snapshot.allowed_calls:
                raise AssertionError("internal cumulative budget violation")
        except BaseException as exc:
            self._close(
                "streaming audit failed after batch admission; abort this policy update",
                exc,
            )

        all_actions = getattr(self.selector, "actions", ())
        all_controller_transitions = getattr(self.selector, "transitions", ())
        return StreamingBatchResult(
            batch_index=batch_index,
            step=parsed_step,
            rollout_ids=tuple(
                tuple(row.rollout_id for row in state.group.rollouts)
                for state in states
            ),
            group_ids=tuple(state.group.group_id for state in states),
            effective_rewards=tuple(
                tuple(
                    propagation_overrides.get(row.rollout_id, reward)
                    for row, reward in zip(
                        state.group.rollouts,
                        state.effective_rewards,
                        strict=True,
                    )
                )
                for state in states
            ),
            quarantined_group_ids=tuple(
                state.group.group_id for state in states if state.quarantined
            ),
            audit_records=tuple(records),
            transitions=tuple(transitions),
            controller_actions=tuple(all_actions[actions_at_start:]),
            controller_transitions=tuple(
                all_controller_transitions[controller_transitions_at_start:]
            ),
            budget=snapshot,
            propagation_records=propagation_records,
            propagation_config=self.propagation_config,
        )


__all__ = (
    "InterventionTransition",
    "OracleQuery",
    "OracleRequest",
    "PropagationRecord",
    "PublicRollout",
    "StreamingAuditEngine",
    "StreamingAuditFailure",
    "StreamingAuditRecord",
    "StreamingBatchResult",
)
