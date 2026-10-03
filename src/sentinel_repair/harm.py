"""Group-coupled corruption metrics and audit intervention semantics.

These routines deliberately measure the residual in GRPO *coefficients*, not
an exact parameter-gradient distance: the latter also needs each rollout's
score-function gradient.  The default L1 residual is the directly observable
proxy used by the first mechanism experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .advantages import (
    DEFAULT_DDOF,
    DEFAULT_EPSILON,
    grpo_advantages,
    validate_binary_rewards,
)

RepairMode = Literal["replace", "quarantine", "full_group"]
ResidualNorm = Literal["l1", "l2", "squared_l2", "linf"]


@dataclass(frozen=True)
class RepairOutcome:
    """Effective update and oracle usage after applying an audit rule.

    ``requested_mask`` marks the labels initially sent to the oracle.
    ``queried_mask`` also contains any sibling labels purchased by the
    ``full_group`` rule after the first observed mismatch.
    """

    coefficients: NDArray[np.float64]
    effective_rewards: NDArray[np.float64]
    requested_mask: NDArray[np.bool_]
    queried_mask: NDArray[np.bool_]
    mismatch_detected: bool
    quarantined: bool

    @property
    def oracle_calls(self) -> int:
        """Number of distinct rollout labels purchased from the oracle."""

        return int(np.count_nonzero(self.queried_mask))


def _matching_vectors(
    left: ArrayLike,
    right: ArrayLike,
    *,
    left_name: str,
    right_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    if left_vector.ndim != 1 or right_vector.ndim != 1:
        raise ValueError(f"{left_name} and {right_name} must be one-dimensional")
    if left_vector.size == 0:
        raise ValueError(f"{left_name} and {right_name} must not be empty")
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have the same shape, got "
            f"{left_vector.shape} and {right_vector.shape}"
        )
    if not np.all(np.isfinite(left_vector)) or not np.all(np.isfinite(right_vector)):
        raise ValueError(f"{left_name} and {right_name} must contain finite values")
    return left_vector, right_vector


def _query_mask(queried: ArrayLike | None, size: int) -> NDArray[np.bool_]:
    if queried is None:
        return np.zeros(size, dtype=np.bool_)

    raw = np.asarray(queried)
    if raw.dtype == np.bool_:
        if raw.ndim != 1 or raw.shape[0] != size:
            raise ValueError(f"boolean queried mask must have shape ({size},)")
        return raw.astype(np.bool_, copy=True)

    if raw.ndim != 1:
        raise ValueError("queried indices must be one-dimensional")
    mask = np.zeros(size, dtype=np.bool_)
    for raw_index in raw.tolist():
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise TypeError("queried indices must be integers")
        index = int(raw_index)
        if index < 0 or index >= size:
            raise IndexError(f"queried index {index} is outside a group of size {size}")
        mask[index] = True
    return mask


def coefficient_residual(
    coefficients: ArrayLike,
    oracle_coefficients: ArrayLike,
    *,
    norm: ResidualNorm = "l1",
    weights: ArrayLike | None = None,
) -> float:
    """Measure the distance from an update's coefficients to oracle coefficients.

    Optional non-negative ``weights`` can encode rollout-specific score-gradient
    magnitudes.  They multiply coefficient differences before taking the norm.
    """

    current, oracle = _matching_vectors(
        coefficients,
        oracle_coefficients,
        left_name="coefficients",
        right_name="oracle_coefficients",
    )
    delta = current - oracle

    if weights is not None:
        weight_vector = np.asarray(weights, dtype=np.float64)
        if weight_vector.shape != delta.shape:
            raise ValueError(
                f"weights must have shape {delta.shape}, got {weight_vector.shape}"
            )
        if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector < 0):
            raise ValueError("weights must be finite and non-negative")
        delta = delta * weight_vector

    if norm == "l1":
        return float(np.sum(np.abs(delta)))
    if norm == "l2":
        return float(np.linalg.norm(delta, ord=2))
    if norm == "squared_l2":
        return float(np.dot(delta, delta))
    if norm == "linf":
        return float(np.max(np.abs(delta)))
    raise ValueError(f"unknown residual norm: {norm!r}")


def reward_coefficient_residual(
    rewards: ArrayLike,
    oracle_rewards: ArrayLike,
    *,
    norm: ResidualNorm = "l1",
    weights: ArrayLike | None = None,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute coefficient residual after separately normalizing two groups."""

    current, oracle = _matching_vectors(
        rewards,
        oracle_rewards,
        left_name="rewards",
        right_name="oracle_rewards",
    )
    return coefficient_residual(
        grpo_advantages(current, ddof=ddof, epsilon=epsilon),
        grpo_advantages(oracle, ddof=ddof, epsilon=epsilon),
        norm=norm,
        weights=weights,
    )


def apply_repair(
    cheap_rewards: ArrayLike,
    oracle_rewards: ArrayLike,
    queried: ArrayLike | None,
    *,
    mode: RepairMode = "replace",
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> RepairOutcome:
    """Apply one of three group-aware audit interventions.

    Modes:
        ``replace``: Replace only purchased labels, then recompute all sibling
            advantages using the partially corrected reward vector.
        ``quarantine``: If any purchased label disagrees, set the entire
            group's update coefficients to zero.  Otherwise retain the cheap
            update.
        ``full_group``: If any initially purchased label disagrees, purchase
            every remaining sibling and use the fully oracle-labelled group.
            If none disagrees, retain the cheap update.

    The evaluator supplies all oracle labels, but only entries in
    ``queried_mask`` affect the returned update.  This makes oracle-call
    accounting explicit and prevents accidental free labels.
    """

    cheap = validate_binary_rewards(cheap_rewards, name="cheap_rewards")
    oracle = validate_binary_rewards(oracle_rewards, name="oracle_rewards")
    if cheap.shape != oracle.shape:
        raise ValueError(
            "cheap_rewards and oracle_rewards must have the same shape, got "
            f"{cheap.shape} and {oracle.shape}"
        )
    if mode not in {"replace", "quarantine", "full_group"}:
        raise ValueError(f"unknown repair mode: {mode!r}")

    requested_mask = _query_mask(queried, cheap.size)
    mismatch_detected = bool(np.any(requested_mask & (cheap != oracle)))
    queried_mask = requested_mask.copy()
    effective_rewards = cheap.copy()
    effective_rewards[requested_mask] = oracle[requested_mask]
    quarantined = False

    if mode == "quarantine" and mismatch_detected:
        coefficients = np.zeros_like(cheap)
        quarantined = True
    elif mode == "full_group" and mismatch_detected:
        queried_mask[:] = True
        effective_rewards = oracle.copy()
        coefficients = grpo_advantages(effective_rewards, ddof=ddof, epsilon=epsilon)
    else:
        coefficients = grpo_advantages(effective_rewards, ddof=ddof, epsilon=epsilon)

    return RepairOutcome(
        coefficients=coefficients,
        effective_rewards=effective_rewards,
        requested_mask=requested_mask,
        queried_mask=queried_mask,
        mismatch_detected=mismatch_detected,
        quarantined=quarantined,
    )


def residual_after_repair(
    cheap_rewards: ArrayLike,
    oracle_rewards: ArrayLike,
    queried: ArrayLike | None,
    *,
    mode: RepairMode = "replace",
    norm: ResidualNorm = "l1",
    weights: ArrayLike | None = None,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Return the residual coefficient harm after an audit intervention."""

    outcome = apply_repair(
        cheap_rewards,
        oracle_rewards,
        queried,
        mode=mode,
        ddof=ddof,
        epsilon=epsilon,
    )
    oracle_coefficients = grpo_advantages(
        validate_binary_rewards(oracle_rewards, name="oracle_rewards"),
        ddof=ddof,
        epsilon=epsilon,
    )
    return coefficient_residual(
        outcome.coefficients,
        oracle_coefficients,
        norm=norm,
        weights=weights,
    )


def marginal_repair_effect(
    cheap_rewards: ArrayLike,
    oracle_rewards: ArrayLike,
    queried: ArrayLike | None,
    add_index: int,
    *,
    mode: RepairMode = "replace",
    norm: ResidualNorm = "l1",
    weights: ArrayLike | None = None,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Return residual-before minus residual-after for one additional audit.

    Positive values mean the audit removes coefficient corruption.  Negative
    values are intentionally preserved: partial relabelling can make a
    previously constant reward group non-constant and thereby *create* a GRPO
    update that neither the cheap nor fully oracle-labelled group would make.
    """

    cheap = validate_binary_rewards(cheap_rewards, name="cheap_rewards")
    oracle = validate_binary_rewards(oracle_rewards, name="oracle_rewards")
    if cheap.shape != oracle.shape:
        raise ValueError("cheap_rewards and oracle_rewards must have the same shape")
    before_mask = _query_mask(queried, cheap.size)
    if isinstance(add_index, bool) or not isinstance(add_index, (int, np.integer)):
        raise TypeError("add_index must be an integer")
    add_index = int(add_index)
    if add_index < 0 or add_index >= cheap.size:
        raise IndexError(
            f"add_index {add_index} is outside a group of size {cheap.size}"
        )
    if before_mask[add_index]:
        raise ValueError(f"rollout {add_index} has already been queried")

    before = residual_after_repair(
        cheap,
        oracle,
        before_mask,
        mode=mode,
        norm=norm,
        weights=weights,
        ddof=ddof,
        epsilon=epsilon,
    )
    after_mask = before_mask.copy()
    after_mask[add_index] = True
    after = residual_after_repair(
        cheap,
        oracle,
        after_mask,
        mode=mode,
        norm=norm,
        weights=weights,
        ddof=ddof,
        epsilon=epsilon,
    )
    return before - after


__all__ = (
    "RepairMode",
    "RepairOutcome",
    "ResidualNorm",
    "apply_repair",
    "coefficient_residual",
    "marginal_repair_effect",
    "residual_after_repair",
    "reward_coefficient_residual",
)
