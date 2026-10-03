"""Exact one-step group-aware audit selection for binary GRPO rewards.

The selector in this module treats an oracle call as an intervention on the
whole normalized reward group.  For every still-hidden label it obtains a
risk estimate from the online risk model, enumerates every joint binary error
configuration, and asks how much querying one candidate would change the
expected L1 residual from the fully oracle-labelled GRPO coefficients.

Only public state and labels already purchased from the oracle enter this
calculation.  In particular, an audited rollout's ``current_reward`` is held
fixed while unaudited oracle labels are integrated out under independent
Bernoulli error probabilities.  A marginal can be negative because replacing
one label may create a non-zero group-relative update from two constant reward
vectors; the default policy therefore buys only a strictly positive action.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .advantages import DEFAULT_DDOF, DEFAULT_EPSILON, grpo_advantages
from .data import GroupObservation, RolloutObservation
from .selectors import (
    AuditFeedback,
    ErrorRiskModel,
    HashedLogisticRiskModel,
    QueryRequest,
    QuerySelector,
)


@dataclass(frozen=True)
class ExpectedMarginalScore:
    """Exact posterior expectation for one possible replace query."""

    rollout_id: int
    risk: float
    expected_residual_before: float
    expected_residual_after: float
    marginal: float
    configurations: int


def _binary(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed not in {0.0, 1.0}:
        raise ValueError(f"{name} must be a finite binary label, got {value!r}")
    return parsed


def _probability(value: float, *, rollout_id: int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(
            f"risk model returned a non-finite value for rollout {rollout_id}"
        )
    # Match the other selectors' boundary behavior: a probabilistic model may
    # have tiny numerical excursions, but the decision calculation stays a
    # proper Bernoulli distribution.
    return min(1.0, max(0.0, parsed))


def expected_replace_marginals(
    group: GroupObservation,
    predicted_risks: Mapping[int, float],
    *,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
    max_unknown: int = 16,
) -> dict[int, ExpectedMarginalScore]:
    """Exactly enumerate one-query L1 coefficient-residual marginals.

    For an unaudited rollout ``i``, ``predicted_risks[i]`` is interpreted as
    ``P(oracle_label_i != cheap_label_i | public history)``.  Binary rewards
    make the error outcome unambiguous: the hidden label is ``1 - cheap``.
    Errors of the remaining unaudited labels are assumed conditionally
    independent.  Labels marked audited are *not* sampled; their revealed
    ``current_reward`` is conditioned on as ground truth.

    The returned marginal is

    ``E[||A(current)-A(oracle)||_1
       - ||A(current with i revealed)-A(oracle)||_1]``.

    It is intentionally signed.  Selective replacement can increase the
    residual after group normalization, so callers must not silently clamp a
    negative value to zero.
    """

    if isinstance(max_unknown, bool) or not isinstance(max_unknown, int):
        raise TypeError("max_unknown must be an integer")
    if max_unknown < 1:
        raise ValueError("max_unknown must be positive")

    rows = tuple(group.rollouts)
    cheap = np.asarray(
        [_binary(row.cheap_reward, name="cheap_reward") for row in rows],
        dtype=np.float64,
    )
    current = np.asarray(
        [_binary(row.current_reward, name="current_reward") for row in rows],
        dtype=np.float64,
    )
    unknown_positions = [
        position for position, row in enumerate(rows) if not row.audited
    ]
    if not unknown_positions:
        return {}
    if len(unknown_positions) > max_unknown:
        raise ValueError(
            f"exact enumeration requested for {len(unknown_positions)} hidden "
            f"labels, above max_unknown={max_unknown}"
        )

    risks: dict[int, float] = {}
    for position in unknown_positions:
        row = rows[position]
        if row.rollout_id not in predicted_risks:
            raise KeyError(f"missing predicted risk for rollout {row.rollout_id}")
        risks[row.rollout_id] = _probability(
            predicted_risks[row.rollout_id], rollout_id=row.rollout_id
        )
        if current[position] != cheap[position]:
            raise ValueError(
                f"unaudited rollout {row.rollout_id} has current_reward different "
                "from cheap_reward"
            )

    current_coefficients = grpo_advantages(current, ddof=ddof, epsilon=epsilon)
    expected_before = 0.0
    expected_after = {position: 0.0 for position in unknown_positions}

    # At n=4 this is exactly sixteen worlds.  Zero-probability worlds are kept
    # in the loop as well, making the implementation a literal enumeration and
    # avoiding special-case changes at q=0 or q=1.
    configurations = 1 << len(unknown_positions)
    for error_bits in itertools.product((False, True), repeat=len(unknown_positions)):
        oracle = current.copy()
        probability = 1.0
        for position, is_error in zip(unknown_positions, error_bits):
            row = rows[position]
            risk = risks[row.rollout_id]
            probability *= risk if is_error else 1.0 - risk
            oracle[position] = 1.0 - cheap[position] if is_error else cheap[position]

        oracle_coefficients = grpo_advantages(oracle, ddof=ddof, epsilon=epsilon)
        residual_before = float(
            np.sum(np.abs(current_coefficients - oracle_coefficients))
        )
        expected_before += probability * residual_before

        for candidate_position in unknown_positions:
            after_rewards = current.copy()
            after_rewards[candidate_position] = oracle[candidate_position]
            after_coefficients = grpo_advantages(
                after_rewards, ddof=ddof, epsilon=epsilon
            )
            residual_after = float(
                np.sum(np.abs(after_coefficients - oracle_coefficients))
            )
            expected_after[candidate_position] += probability * residual_after

    return {
        rows[position].rollout_id: ExpectedMarginalScore(
            rollout_id=rows[position].rollout_id,
            risk=risks[rows[position].rollout_id],
            expected_residual_before=expected_before,
            expected_residual_after=expected_after[position],
            marginal=expected_before - expected_after[position],
            configurations=configurations,
        )
        for position in unknown_positions
    }


class ExpectedMarginalGroupSelector(QuerySelector):
    """Buy the best strictly positive expected group-level replace query.

    Scores are recomputed from the current group view on every call.  Thus a
    purchased sibling label changes both the conditioning state and all GRPO
    coefficients before the next action.  The risk model is queried only for
    unaudited rows and is updated only from :meth:`observe`, which the offline
    harness invokes after an actual oracle purchase.
    """

    def __init__(
        self,
        risk_model: ErrorRiskModel | None = None,
        *,
        minimum_marginal: float = 0.0,
        ddof: int = DEFAULT_DDOF,
        epsilon: float = DEFAULT_EPSILON,
        max_unknown: int = 16,
    ) -> None:
        if not math.isfinite(minimum_marginal) or minimum_marginal < 0.0:
            raise ValueError("minimum_marginal must be finite and non-negative")
        self.risk_model = risk_model or HashedLogisticRiskModel()
        self.minimum_marginal = float(minimum_marginal)
        self.ddof = ddof
        self.epsilon = epsilon
        self.max_unknown = max_unknown

    def score_group(self, group: GroupObservation) -> dict[int, ExpectedMarginalScore]:
        """Return current exact scores using only the risk model's predictions."""

        if group.quarantined or not group.unqueried:
            return {}
        predicted_risks = {
            row.rollout_id: _probability(
                self.risk_model.predict(row, group), rollout_id=row.rollout_id
            )
            for row in group.unqueried
        }
        return expected_replace_marginals(
            group,
            predicted_risks,
            ddof=self.ddof,
            epsilon=self.epsilon,
            max_unknown=self.max_unknown,
        )

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None

        candidates: list[
            tuple[float, int, RolloutObservation, ExpectedMarginalScore]
        ] = []
        for group in groups:
            scores = self.score_group(group)
            rows_by_id = {row.rollout_id: row for row in group.unqueried}
            for rollout_id, score in scores.items():
                candidates.append(
                    (score.marginal, -rollout_id, rows_by_id[rollout_id], score)
                )

        if not candidates:
            return None
        marginal, _, row, score = max(candidates, key=lambda item: (item[0], item[1]))
        # Strict comparison implements "best positive": zero-value audits also
        # consume budget without improving the one-step objective.
        if marginal <= self.minimum_marginal:
            return None
        return QueryRequest(
            (row.rollout_id,),
            reason="expected_marginal_l1_replace",
            priority=marginal,
            predicted_risks={row.rollout_id: score.risk},
        )

    def observe(self, feedback: AuditFeedback) -> None:
        self.risk_model.update(
            feedback.candidate_before,
            feedback.group_before,
            feedback.label_error,
        )

    def end_batch(self, groups: Sequence[GroupObservation]) -> None:
        del groups
        clear = getattr(self.risk_model, "clear_batch_cache", None)
        if clear is not None:
            clear()


__all__ = (
    "ExpectedMarginalGroupSelector",
    "ExpectedMarginalScore",
    "expected_replace_marginals",
)
