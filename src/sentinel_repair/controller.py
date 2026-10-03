"""Sentinel/repair budget controllers for online verifier auditing.

The controllers in this module wrap any :class:`QuerySelector`.  They do not
own the global label budget: ``OfflineRunner`` continues to enforce that
strictly.  Instead, each affordable action slot is assigned either to

* ``SENTINEL``: one uniformly sampled, currently unaudited rollout; or
* ``REPAIR``: the request proposed by the wrapped selector.

Label-call allocation uses a deterministic quota, while sentinel selection is random
with a recorded conditional propensity.  This gives reproducible experiments
without losing the known sampling probability needed for unbiased monitoring.
The adaptive controller is deliberately conservative: only feedback from an
action recorded as ``SENTINEL`` can change its state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Sequence

from .data import GroupObservation
from .selectors import AuditFeedback, QueryRequest, QuerySelector


class AuditArm(str, Enum):
    """The two uses of a scarce audit action."""

    SENTINEL = "sentinel"
    REPAIR = "repair"


class ControllerState(str, Enum):
    """Operating state of the adaptive controller."""

    DISCOVER = "discover"
    REPAIR = "repair"


@dataclass(frozen=True)
class ControllerAction:
    """Decision-time record for one query action.

    ``target_arm_share`` is the configured long-run share for the executed
    arm, not a Bernoulli probability: label calls are assigned by deterministic
    quota (up to indivisible group-bundle granularity). ``random_propensity`` is the exact probability of the selected
    rollout conditional on a sentinel slot.  It is ``None`` for repair
    actions, whose selection rule is owned by the wrapped selector.
    """

    action_index: int
    state: ControllerState
    planned_arm: AuditArm
    arm: AuditArm
    used_fallback: bool
    sentinel_fraction: float
    target_arm_share: float
    random_propensity: float | None
    rollout_ids: tuple[int, ...]
    reason: str
    label_cost: int
    # These counters are lifetime totals.  They deliberately survive dynamic
    # state transitions even though the state-local quota scheduler resets.
    cumulative_label_calls: int
    cumulative_sentinel_calls: int


@dataclass(frozen=True)
class StateTransition:
    """A state change caused by purchased sentinel observations."""

    sentinel_observations: int
    old_state: ControllerState
    new_state: ControllerState
    reason: str
    statistic: float


class _DeterministicQuota:
    """Allocate a fraction of *label calls*, with bounded bundle granularity."""

    def __init__(self) -> None:
        self.label_calls = 0
        self.sentinel_calls = 0

    def peek(self, sentinel_fraction: Fraction) -> AuditArm:
        next_count = self.label_calls + 1
        numerator = sentinel_fraction.numerator * next_count
        target = (numerator + sentinel_fraction.denominator - 1) // (
            sentinel_fraction.denominator
        )
        if self.sentinel_calls < target:
            return AuditArm.SENTINEL
        return AuditArm.REPAIR

    def commit(self, arm: AuditArm, label_cost: int) -> None:
        if label_cost < 1:
            raise ValueError("label_cost must be positive")
        self.label_calls += label_cost
        if arm is AuditArm.SENTINEL:
            self.sentinel_calls += label_cost


def _fraction(value: float | str | Fraction, *, name: str) -> Fraction:
    result = value if isinstance(value, Fraction) else Fraction(str(value))
    if result < 0 or result > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _uniform_candidates(groups: Sequence[GroupObservation]):
    return [
        row
        for group in groups
        if not group.quarantined
        for row in group.rollouts
        if not row.audited
    ]


def fixed_quota_hit_probability(
    population_size: int,
    exploit_count: int,
    sentinel_quota: int,
) -> float:
    """Probability that a uniform, no-replacement quota finds an exploit.

    This is the hypergeometric probability

    ``1 - choose(N-K, m) / choose(N, m)``,

    where ``N`` is the visible population, ``K`` its exploit-bearing items,
    and ``m`` the fixed sentinel quota.  The product below uses only
    ``min(K, m)`` factors and ``expm1`` for stable rare-event probabilities.
    """

    for name, value in (
        ("population_size", population_size),
        ("exploit_count", exploit_count),
        ("sentinel_quota", sentinel_quota),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    if exploit_count > population_size:
        raise ValueError("exploit_count cannot exceed population_size")
    if sentinel_quota > population_size:
        raise ValueError("sentinel_quota cannot exceed population_size")
    if exploit_count == 0 or sentinel_quota == 0:
        return 0.0
    if sentinel_quota > population_size - exploit_count:
        return 1.0

    log_miss = 0.0
    if sentinel_quota <= exploit_count:
        # Product over draws: each factor is the conditional chance that the
        # next sampled item is not one of the K exploit-bearing items.
        for draw in range(sentinel_quota):
            log_miss += math.log1p(-exploit_count / (population_size - draw))
    else:
        # Equivalent product over exploit items: each must remain outside the
        # fixed sample.  This is cheaper when exploits are sparse.
        for exploit in range(exploit_count):
            log_miss += math.log1p(-sentinel_quota / (population_size - exploit))
    return min(1.0, max(0.0, -math.expm1(log_miss)))


def sentinel_exposures_for_discovery(
    prevalence: float,
    *,
    confidence: float = 0.95,
) -> int:
    """Closed-form sentinel samples needed to see an exploit at least once.

    Under independent exposure with prevalence ``p``, the miss probability
    after ``n`` uniform sentinel samples is ``(1-p)**n``.  The returned integer
    is the smallest ``n`` making that probability at most ``1-confidence``.
    """

    prevalence = float(prevalence)
    confidence = float(confidence)
    if not math.isfinite(prevalence) or not 0.0 < prevalence <= 1.0:
        raise ValueError("prevalence must be finite and in (0, 1]")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    if prevalence == 1.0:
        return 1
    return math.ceil(math.log1p(-confidence) / math.log1p(-prevalence))


def audit_actions_for_discovery(
    prevalence: float,
    sentinel_fraction: float | str | Fraction,
    *,
    confidence: float = 0.95,
) -> int:
    """Conservative action horizon for a sentinel share ``rho``.

    If ``k`` uniform samples are required, ``ceil(k/rho)`` total audit actions
    suffice under the deterministic quota.  For a 1% exploit and 95%
    discovery, ``k=299``, hence this is bounded by ``300/rho``.
    """

    fraction = _fraction(sentinel_fraction, name="sentinel_fraction")
    if fraction == 0:
        raise ValueError("sentinel_fraction must be positive for discovery")
    required = sentinel_exposures_for_discovery(prevalence, confidence=confidence)
    numerator = required * fraction.denominator
    return (numerator + fraction.numerator - 1) // fraction.numerator


class FixedSentinelRepairController(QuerySelector):
    """Use a fixed deterministic quota of uniform sentinel audit actions."""

    def __init__(
        self,
        repair_selector: QuerySelector,
        *,
        sentinel_fraction: float | str | Fraction = "0.2",
        seed: int = 0,
    ) -> None:
        self.repair_selector = repair_selector
        self._sentinel_fraction = _fraction(sentinel_fraction, name="sentinel_fraction")
        self._rng = random.Random(seed)
        self._quota = _DeterministicQuota()
        self._actions: list[ControllerAction] = []
        self._pending_arms: dict[int, AuditArm] = {}
        self._lifetime_label_calls = 0
        self._lifetime_sentinel_calls = 0

    @property
    def state(self) -> ControllerState:
        return ControllerState.REPAIR

    @property
    def sentinel_fraction(self) -> float:
        return float(self._sentinel_fraction)

    @property
    def actions(self) -> tuple[ControllerAction, ...]:
        return tuple(self._actions)

    @property
    def sentinel_actions(self) -> int:
        return sum(action.arm is AuditArm.SENTINEL for action in self._actions)

    @property
    def repair_actions(self) -> int:
        return sum(action.arm is AuditArm.REPAIR for action in self._actions)

    @property
    def lifetime_label_calls(self) -> int:
        return self._lifetime_label_calls

    @property
    def lifetime_sentinel_calls(self) -> int:
        return self._lifetime_sentinel_calls

    def _current_fraction(self) -> Fraction:
        return self._sentinel_fraction

    def _uniform_sentinel(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> tuple[QueryRequest | None, float | None]:
        if max_labels < 1:
            return None, None
        candidates = _uniform_candidates(groups)
        if not candidates:
            return None, None
        row = self._rng.choice(candidates)
        propensity = 1.0 / len(candidates)
        return (
            QueryRequest(
                (row.rollout_id,),
                reason="sentinel_uniform",
                priority=None,
            ),
            propensity,
        )

    @staticmethod
    def _copy_repair_request(request: QueryRequest) -> QueryRequest:
        return QueryRequest(
            request.rollout_ids,
            reason=f"repair:{request.reason}",
            priority=request.priority,
            predicted_risks=request.predicted_risks,
            atomic=request.atomic,
        )

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None

        fraction = self._current_fraction()
        planned_arm = self._quota.peek(fraction)
        used_fallback = False
        random_propensity: float | None = None

        if planned_arm is AuditArm.SENTINEL:
            request, random_propensity = self._uniform_sentinel(
                groups, max_labels=max_labels
            )
            arm = AuditArm.SENTINEL
        else:
            underlying = self.repair_selector.propose(groups, max_labels=max_labels)
            if underlying is None:
                used_fallback = True
                request, random_propensity = self._uniform_sentinel(
                    groups, max_labels=max_labels
                )
                arm = AuditArm.SENTINEL
            else:
                request = self._copy_repair_request(underlying)
                arm = AuditArm.REPAIR

        if request is None:
            return None

        self._quota.commit(arm, request.cost)
        self._lifetime_label_calls += request.cost
        if arm is AuditArm.SENTINEL:
            self._lifetime_sentinel_calls += request.cost
        action = ControllerAction(
            action_index=len(self._actions),
            state=self.state,
            planned_arm=planned_arm,
            arm=arm,
            used_fallback=used_fallback,
            sentinel_fraction=float(fraction),
            target_arm_share=(
                float(fraction) if arm is AuditArm.SENTINEL else 1.0 - float(fraction)
            ),
            random_propensity=random_propensity,
            rollout_ids=request.rollout_ids,
            reason=request.reason,
            label_cost=request.cost,
            cumulative_label_calls=self._lifetime_label_calls,
            cumulative_sentinel_calls=self._lifetime_sentinel_calls,
        )
        self._actions.append(action)
        for rollout_id in request.rollout_ids:
            self._pending_arms[rollout_id] = arm
        return request

    def _observe_sentinel(self, feedback: AuditFeedback) -> None:
        """Hook for adaptive state updates; fixed mixtures need no update."""

    def observe(self, feedback: AuditFeedback) -> None:
        try:
            arm = self._pending_arms.pop(feedback.candidate_before.rollout_id)
        except KeyError as exc:
            raise ValueError(
                "feedback does not match a pending controller action"
            ) from exc

        # Every bought label may train the repair model.  State control below,
        # however, is restricted to probability-sampled sentinel feedback.
        self.repair_selector.observe(feedback)
        if arm is AuditArm.SENTINEL:
            self._observe_sentinel(feedback)

    def end_batch(self, groups: Sequence[GroupObservation]) -> None:
        self.repair_selector.end_batch(groups)


class _PageHinkley:
    """One-sided Page-Hinkley statistic for an increase in residual mean."""

    def __init__(self, *, delta: float, threshold: float, min_samples: int) -> None:
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("ph_delta must be finite and non-negative")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("ph_threshold must be finite and positive")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise TypeError("ph_min_samples must be an integer")
        if min_samples < 1:
            raise ValueError("ph_min_samples must be positive")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, value: float) -> tuple[bool, float]:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        statistic = self.cumulative - self.minimum
        return self.count >= self.min_samples and statistic >= self.threshold, statistic


class DynamicSentinelRepairController(FixedSentinelRepairController):
    """Switch sentinel share using sentinel-only change evidence.

    The controller starts in ``REPAIR`` by default.  A one-sided Page-Hinkley
    test over ``abs(cheap_label - purchased_label)`` switches it to
    ``DISCOVER``.  It returns to ``REPAIR`` only after a configurable minimum
    number of discovery-state sentinel observations and an absolute decline
    between two adjacent recent windows.  Both arms retain non-zero quota in
    both states.
    """

    def __init__(
        self,
        repair_selector: QuerySelector,
        *,
        discover_sentinel_fraction: float | str | Fraction = "0.8",
        repair_sentinel_fraction: float | str | Fraction = "0.2",
        initial_state: ControllerState | str = ControllerState.REPAIR,
        seed: int = 0,
        ph_delta: float = 0.01,
        ph_threshold: float = 2.0,
        ph_min_samples: int = 10,
        recovery_min_sentinels: int = 20,
        recovery_window: int = 10,
        recovery_drop: float = 0.1,
        recovery_max_recent_rate: float | None = 0.05,
    ) -> None:
        discover = _fraction(
            discover_sentinel_fraction,
            name="discover_sentinel_fraction",
        )
        repair = _fraction(
            repair_sentinel_fraction,
            name="repair_sentinel_fraction",
        )
        if discover <= 0 or discover >= 1:
            raise ValueError("discover_sentinel_fraction must be in (0, 1)")
        if repair <= 0 or repair >= 1:
            raise ValueError("repair_sentinel_fraction must be in (0, 1)")
        if discover <= repair:
            raise ValueError(
                "discover_sentinel_fraction must exceed repair_sentinel_fraction"
            )
        try:
            parsed_state = ControllerState(initial_state)
        except ValueError as exc:
            raise ValueError(f"unknown initial_state: {initial_state!r}") from exc
        if isinstance(recovery_min_sentinels, bool) or not isinstance(
            recovery_min_sentinels, int
        ):
            raise TypeError("recovery_min_sentinels must be an integer")
        if recovery_min_sentinels < 1:
            raise ValueError("recovery_min_sentinels must be positive")
        if isinstance(recovery_window, bool) or not isinstance(recovery_window, int):
            raise TypeError("recovery_window must be an integer")
        if recovery_window < 1:
            raise ValueError("recovery_window must be positive")
        if not math.isfinite(recovery_drop) or recovery_drop < 0:
            raise ValueError("recovery_drop must be finite and non-negative")
        if recovery_max_recent_rate is not None and (
            not math.isfinite(recovery_max_recent_rate)
            or not 0 <= recovery_max_recent_rate <= 1
        ):
            raise ValueError("recovery_max_recent_rate must be in [0, 1] or None")

        super().__init__(
            repair_selector,
            sentinel_fraction=repair,
            seed=seed,
        )
        self._discover_fraction = discover
        self._repair_fraction = repair
        self._state = parsed_state
        self._page_hinkley = _PageHinkley(
            delta=ph_delta,
            threshold=ph_threshold,
            min_samples=ph_min_samples,
        )
        self.recovery_min_sentinels = recovery_min_sentinels
        self.recovery_window = recovery_window
        self.recovery_drop = float(recovery_drop)
        self.recovery_max_recent_rate = recovery_max_recent_rate
        self._sentinel_observations = 0
        self._discovery_residuals: list[float] = []
        self._transitions: list[StateTransition] = []

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def sentinel_fraction(self) -> float:
        return float(self._current_fraction())

    @property
    def sentinel_observations(self) -> int:
        return self._sentinel_observations

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        return tuple(self._transitions)

    def _current_fraction(self) -> Fraction:
        if self._state is ControllerState.DISCOVER:
            return self._discover_fraction
        return self._repair_fraction

    def _transition(
        self,
        state: ControllerState,
        *,
        reason: str,
        statistic: float,
    ) -> None:
        if state is self._state:
            return
        old_state = self._state
        self._state = state
        self._transitions.append(
            StateTransition(
                sentinel_observations=self._sentinel_observations,
                old_state=old_state,
                new_state=state,
                reason=reason,
                statistic=float(statistic),
            )
        )
        # The new target share starts immediately rather than inheriting a
        # quota deficit accumulated under the old state.
        self._quota = _DeterministicQuota()

    def _observe_sentinel(self, feedback: AuditFeedback) -> None:
        residual = abs(
            float(feedback.candidate_before.cheap_reward)
            - float(feedback.observed_label)
        )
        if not math.isfinite(residual):
            raise ValueError("sentinel mismatch residual must be finite")
        self._sentinel_observations += 1

        if self._state is ControllerState.REPAIR:
            changed, statistic = self._page_hinkley.update(residual)
            if changed:
                self._transition(
                    ControllerState.DISCOVER,
                    reason="page_hinkley_increase",
                    statistic=statistic,
                )
                # Retain the alarm observation so recovery must demonstrate a
                # decline relative to the burst that caused the switch.
                self._discovery_residuals = [residual]
            return

        self._discovery_residuals.append(residual)
        count = len(self._discovery_residuals)
        window = self.recovery_window
        if count < max(self.recovery_min_sentinels, 2 * window):
            return
        previous = self._discovery_residuals[-2 * window : -window]
        recent = self._discovery_residuals[-window:]
        previous_rate = sum(previous) / window
        recent_rate = sum(recent) / window
        declined = previous_rate - recent_rate >= self.recovery_drop
        below_ceiling = (
            self.recovery_max_recent_rate is None
            or recent_rate <= self.recovery_max_recent_rate
        )
        if declined and below_ceiling:
            self._transition(
                ControllerState.REPAIR,
                reason="recent_sentinel_rate_decline",
                statistic=previous_rate - recent_rate,
            )
            self._page_hinkley.reset()
            self._discovery_residuals = []


class BootstrapSentinelRepairController(FixedSentinelRepairController):
    """Start discovery-heavy, then latch into repair after sentinel errors.

    This controller is intentionally one-way.  It begins in ``DISCOVER`` and
    changes to ``REPAIR`` only after ``bootstrap_mismatches`` purchased labels
    from the uniform sentinel arm disagree with their cheap labels.  Repair-arm
    feedback still trains ``repair_selector`` through the base class, but it
    cannot advance the bootstrap counter or change controller state.

    The transition resets only the state-local deterministic quota.  Lifetime
    action/call counters, the repair learner, and the sentinel observation
    counts are retained.  A post-repair detector for a second unknown failure
    is deliberately out of scope: a fixed logged trace cannot validate that
    closed-loop behavior.
    """

    def __init__(
        self,
        repair_selector: QuerySelector,
        *,
        discover_sentinel_fraction: float | str | Fraction = "0.8",
        repair_sentinel_fraction: float | str | Fraction = "0.2",
        bootstrap_mismatches: int = 1,
        seed: int = 0,
    ) -> None:
        discover = _fraction(
            discover_sentinel_fraction,
            name="discover_sentinel_fraction",
        )
        repair = _fraction(
            repair_sentinel_fraction,
            name="repair_sentinel_fraction",
        )
        if discover <= 0 or discover >= 1:
            raise ValueError("discover_sentinel_fraction must be in (0, 1)")
        if repair <= 0 or repair >= 1:
            raise ValueError("repair_sentinel_fraction must be in (0, 1)")
        if discover <= repair:
            raise ValueError(
                "discover_sentinel_fraction must exceed repair_sentinel_fraction"
            )
        if isinstance(bootstrap_mismatches, bool) or not isinstance(
            bootstrap_mismatches, int
        ):
            raise TypeError("bootstrap_mismatches must be an integer")
        if bootstrap_mismatches < 1:
            raise ValueError("bootstrap_mismatches must be positive")

        super().__init__(
            repair_selector,
            sentinel_fraction=repair,
            seed=seed,
        )
        self._discover_fraction = discover
        self._repair_fraction = repair
        self._bootstrap_mismatches = bootstrap_mismatches
        self._state = ControllerState.DISCOVER
        self._sentinel_observations = 0
        self._sentinel_mismatches = 0
        self._transitions: list[StateTransition] = []

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def sentinel_fraction(self) -> float:
        return float(self._current_fraction())

    @property
    def bootstrap_mismatches(self) -> int:
        return self._bootstrap_mismatches

    @property
    def sentinel_observations(self) -> int:
        return self._sentinel_observations

    @property
    def sentinel_mismatches(self) -> int:
        return self._sentinel_mismatches

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        return tuple(self._transitions)

    def _current_fraction(self) -> Fraction:
        if self._state is ControllerState.DISCOVER:
            return self._discover_fraction
        return self._repair_fraction

    def _observe_sentinel(self, feedback: AuditFeedback) -> None:
        self._sentinel_observations += 1
        if feedback.label_error:
            self._sentinel_mismatches += 1
        if (
            self._state is ControllerState.DISCOVER
            and self._sentinel_mismatches >= self._bootstrap_mismatches
        ):
            old_state = self._state
            self._state = ControllerState.REPAIR
            self._transitions.append(
                StateTransition(
                    sentinel_observations=self._sentinel_observations,
                    old_state=old_state,
                    new_state=self._state,
                    reason="bootstrap_sentinel_mismatch_threshold",
                    statistic=float(self._sentinel_mismatches),
                )
            )
            # Reset only the state-local quota.  Base-class lifetime counters
            # intentionally survive this transition.
            self._quota = _DeterministicQuota()


__all__ = (
    "AuditArm",
    "BootstrapSentinelRepairController",
    "ControllerAction",
    "ControllerState",
    "DynamicSentinelRepairController",
    "FixedSentinelRepairController",
    "StateTransition",
    "audit_actions_for_discovery",
    "fixed_quota_hit_probability",
    "sentinel_exposures_for_discovery",
)
