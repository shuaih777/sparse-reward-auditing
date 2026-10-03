"""GRPO group-relative advantage calculations.

The verifier-noise upstream code and its analysis utilities use the sample
standard deviation (``ddof=1``), followed by an additive ``1e-4`` in the
denominator.  Keeping that convention explicit matters for small groups: for
a four-rollout binary group, population and sample standard deviations differ
materially.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_DDOF = 1
DEFAULT_EPSILON = 1e-4


def _as_finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def validate_binary_rewards(
    rewards: ArrayLike, *, name: str = "rewards"
) -> NDArray[np.float64]:
    """Return a finite reward vector after checking that every label is 0 or 1."""

    vector = _as_finite_vector(rewards, name=name)
    if not np.all((vector == 0.0) | (vector == 1.0)):
        raise ValueError(f"{name} must contain only binary labels 0 or 1")
    return vector


def grpo_advantages(
    rewards: ArrayLike,
    *,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Compute group-relative advantages using the upstream GRPO convention.

    The calculation is

    ``(reward - group_mean) / (sample_std + epsilon)``.

    ``epsilon`` is added *after* taking the standard deviation.  Constant
    groups, including singleton groups, therefore return an all-zero vector.
    The function accepts general finite rewards; use :func:`validate_binary_rewards`
    when the experiment requires a binary verifier.

    Args:
        rewards: One-dimensional group of reward values.
        ddof: Delta degrees of freedom used by the standard deviation.  The
            upstream default is one (sample standard deviation).  Passing zero
            gives the population-standard-deviation ablation.
        epsilon: Non-negative stabilizer added to the standard deviation.

    Returns:
        A float64 NumPy vector with one coefficient per rollout.
    """

    vector = _as_finite_vector(rewards, name="rewards")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be a non-negative integer")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    centered = vector - vector.mean(dtype=np.float64)

    # NumPy/Pandas report an undefined sample standard deviation for a
    # singleton.  In GRPO the centered coefficient is nevertheless exactly
    # zero, so returning zero is the useful and numerically stable extension.
    if vector.size <= ddof:
        if np.all(centered == 0.0):
            return np.zeros_like(vector)
        raise ValueError("ddof must be smaller than the group size")

    variance = np.dot(centered, centered) / (vector.size - ddof)
    standard_deviation = float(np.sqrt(variance))
    denominator = standard_deviation + float(epsilon)

    # This also supports epsilon=0 for a constant group without producing NaN.
    if denominator == 0.0:
        return np.zeros_like(vector)
    return centered / denominator


def group_advantages(
    rewards: ArrayLike,
    *,
    ddof: int = DEFAULT_DDOF,
    epsilon: float = DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Alias with a domain-oriented name for :func:`grpo_advantages`."""

    return grpo_advantages(rewards, ddof=ddof, epsilon=epsilon)


__all__: Sequence[str] = (
    "DEFAULT_DDOF",
    "DEFAULT_EPSILON",
    "grpo_advantages",
    "group_advantages",
    "validate_binary_rewards",
)
