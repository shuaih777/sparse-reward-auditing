"""Purchased-label-only contrasts from one uniformly sampled prompt group.

The design buys k >= 2 labels in one uniformly selected group, selecting the
k members uniformly without replacement. Every rollout has marginal inclusion
probability p = k / N. This is NOT a drop-in estimator for arbitrary masks with
the same marginal probability: its within-group conditional sampling matters.

The learner-facing ``paired_advantage`` receives only purchased truth. The
``exact_*`` functions instead receive complete residuals and are evaluator-only
diagnostics of a fixed batch; they must never supply training labels or scores.
All unbiasedness statements concern raw RLOO coefficients or a fixed linear
gradient, not clipped/preconditioned optimizer steps or online performance.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np

from .linear_audit import rloo


def _integer(value: int, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _design(n: int, group_size: int, k: int) -> tuple[int, int, int]:
    n = _integer(n, "n", 2)
    group_size = _integer(group_size, "group_size", 2)
    k = _integer(k, "k", 2)
    if n % group_size:
        raise ValueError("n must be divisible by group_size")
    if k > group_size:
        raise ValueError("k must not exceed group_size")
    return n, group_size, k


def subset_audit_indices(
    n: int,
    group_size: int,
    k: int,
    repetitions: int,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    """Return (repetitions, k) flat indices under the specified cluster design.

    Groups occupy contiguous ``group_size`` chunks of a flattened batch. A
    supplied Generator is consumed in place, allowing a persistent RNG stream.
    Zero repetitions return an empty array without consuming random numbers.
    This function does not create a budget: the caller must supply the already
    available k. In the current N=256, G=4, 1% protocol that is always 2 or 3.
    """
    n, group_size, k = _design(n, group_size, k)
    repetitions = _integer(repetitions, "repetitions", 0)
    indices = np.empty((repetitions, k), dtype=np.int64)
    if repetitions == 0:
        return indices
    rng = np.random.default_rng(seed)
    groups = rng.integers(n // group_size, size=repetitions)
    for row, group in enumerate(groups):
        indices[row] = (
            group * group_size + rng.choice(group_size, k, replace=False)
        )
    return indices


def enumerate_subset_audits(
    n: int, group_size: int, k: int, *, max_designs: int = 1_000_000
) -> np.ndarray:
    """Return every equally likely design, shape (B * choose(G,k), k).

    Enumeration is intended for small-group CPU diagnostics, not rollout
    selection during training. The explicit cap prevents accidental enormous
    allocations when this diagnostic is called with a large group size.
    """
    n, group_size, k = _design(n, group_size, k)
    max_designs = _integer(max_designs, "max_designs", 1)
    count = n // group_size * comb(group_size, k)
    if count > max_designs:
        raise ValueError(f"design count {count} exceeds max_designs={max_designs}")
    within = np.asarray(list(combinations(range(group_size), k)), dtype=np.int64)
    return np.concatenate(
        [within + group * group_size for group in range(n // group_size)], axis=0
    )


def paired_advantage(
    cheap: np.ndarray, audited: np.ndarray, p: float
) -> np.ndarray:
    """Compute L(cheap) + I/p * (delta - mean(other purchased delta)).

    Arrays have shape (..., B, G): the last two axes are the fixed batch's
    prompt groups and group members; optional leading axes are independent
    audit replicates. Shapes must be identical, without implicit broadcasting.
    ``audited`` contains finite purchased labels and NaN everywhere else. In
    every replicate exactly k >= 2 finite entries must occur in exactly one
    group, with the same k throughout and scalar p = k / (B*G).

    This validates the visible design geometry and probability, but the caller
    must enforce uniform group/subset sampling and purchased-label provenance.
    For any fixed finite cheap/truth arrays, averaging over that design gives
    exactly L(truth), where L is full-group RLOO. No division by N, standard
    deviation normalization, clipping, or predictor is applied here. An
    unpurchased member receives cheap RLOO only, not a residual correction.

    k=0 and k=1 are intentionally unsupported: a trustworthy relative contrast
    is unavailable. The N=256 1% prefix-budget schedule needs neither case.
    """
    cheap = np.asarray(cheap, dtype=np.float64)
    audited = np.asarray(audited, dtype=np.float64)
    if cheap.ndim < 2 or any(size == 0 for size in cheap.shape):
        raise ValueError("cheap must have nonempty shape (..., B, G)")
    if cheap.shape[-1] < 2:
        raise ValueError("group_size must be at least two")
    if not np.isfinite(cheap).all():
        raise ValueError("cheap rewards must be finite")
    if audited.shape != cheap.shape:
        raise ValueError("audited and cheap must have identical shapes")
    if np.isinf(audited).any():
        raise ValueError("audited labels must be finite or unaudited NaN")
    mask = np.isfinite(audited)
    group_counts = mask.sum(axis=-1)
    counts = group_counts.sum(axis=-1)
    k = int(counts.flat[0])
    if k < 2 or not np.all(counts == k):
        raise ValueError("each replicate must have the same k >= 2 purchased labels")
    if not np.all((group_counts > 0).sum(axis=-1) == 1):
        raise ValueError("purchased labels must lie in exactly one group per replicate")
    if isinstance(p, (bool, np.bool_)) or np.ndim(p) != 0:
        raise ValueError("p must be a finite scalar matching k / (B*G)")
    try:
        p = float(p)
    except (TypeError, ValueError) as error:
        raise ValueError("p must be a finite scalar matching k / (B*G)") from error
    n = cheap.shape[-2] * cheap.shape[-1]
    if not np.isfinite(p) or not np.isclose(p, k / n, rtol=1e-12, atol=0):
        raise ValueError("p must equal the design inclusion probability k / (B*G)")
    delta = np.where(mask, audited - cheap, 0.0)
    contrast = (k * delta - delta.sum(axis=-1, keepdims=True)) / (k - 1)
    return rloo(cheap) + np.where(mask, contrast / p, 0.0)


def _residuals(delta: np.ndarray, k: int) -> tuple[np.ndarray, int, int, int]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] == 0:
        raise ValueError("delta must be a nonempty 2D (B,G) evaluator residual array")
    n, group_size, k = _design(delta.size, delta.shape[1], k)
    if not np.isfinite(delta).all():
        raise ValueError("evaluator residuals must be finite")
    return delta, n, group_size, k


def _nonnegative(value: float, scale: float) -> float:
    # Floating-point cancellation is expected for constant residuals or scores.
    if value < -1e-10 * max(1.0, scale):
        raise ValueError("negative variance/norm: Gram matrix may not be positive semidefinite")
    return max(0.0, float(value))


def exact_coefficient_variance(delta: np.ndarray, k: int) -> dict[str, float]:
    """Evaluator-only exact SUM of raw-coefficient variances, not divided by N^2.

    Compare global simple random sampling (point SRS) followed by full-group
    RLOO of the inverse-propensity residual, against paired contrasts. Both
    have fixed mean t=L(delta). Let Q=sum(delta^2), R=sum(t^2), N=B*G:

      point variance = (N-k)/(k*(N-1)) * (N*G/(G-1)*Q - R)
      paired variance = (N*(G-1)/(G*(k-1)) - 1) * R.

    Group-constant residuals give R=0 and zero paired variance. This is not a
    universal improvement: isolated errors can have larger paired variance.
    Coefficient variance ignores the actual score-vector geometry; use
    ``exact_gradient_variance`` for a supplied fixed score Gram matrix.
    """
    delta, n, group_size, k = _residuals(delta, k)
    target_squared_norm = float(np.square(rloo(delta)).sum())
    q = float(np.square(delta).sum())
    point = (n - k) / (k * (n - 1)) * (
        n * group_size / (group_size - 1) * q - target_squared_norm
    )
    paired = (
        n * (group_size - 1) / (group_size * (k - 1)) - 1
    ) * target_squared_norm
    return {
        "point_srs_variance": _nonnegative(point, n * q),
        "paired_variance": _nonnegative(paired, n * q),
        "target_squared_norm": target_squared_norm,
    }


def exact_gradient_variance(
    delta: np.ndarray, gram: np.ndarray, k: int
) -> dict[str, float]:
    """Exact evaluator-only variance of the BATCH-MEAN fixed linear gradient.

    ``gram[i,j]`` is the inner product of unscaled fixed score vectors s_i,s_j
    in row-major (B,G) order. The gradient convention is sum_i A_i*s_i / N.
    Returned variance is E[||g-Eg||^2], and ``target_squared_norm`` is the norm
    squared of sum_i L(delta)_i*s_i/N. Adding deterministic cheap RLOO changes
    the mean but not these variances. All returned values include 1/N^2.

    Point SRS is computed analytically. The paired and clustered-point controls
    enumerate B*choose(G,k) equally likely subsets. ``clustered_point`` uses
    the paired sampling locations but applies the original full-group RLOO to
    inverse-propensity residuals, isolating estimation from sampling changes.
    Gram must be a symmetric positive-semidefinite matrix. Symmetry/shape and
    resulting nonnegative moments are checked, but a full PSD eigentest is not
    performed. These are raw fixed-score diagnostics, not optimizer geometry.
    """
    delta, n, group_size, k = _residuals(delta, k)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (n, n) or not np.isfinite(gram).all():
        raise ValueError("gram must be a finite N x N score Gram matrix")
    if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12):
        raise ValueError("gram must be symmetric")
    gram = (gram + gram.T) / 2
    group_operator = group_size / (group_size - 1) * (
        np.eye(group_size) - np.ones((group_size, group_size)) / group_size
    )
    operator = np.kron(np.eye(n // group_size), group_operator)
    transformed_gram = operator @ gram @ operator
    flat_delta = delta.ravel()
    target = rloo(delta).ravel()
    target_second = float(target @ gram @ target)
    diagonal_sum = float(np.square(flat_delta) @ np.diag(transformed_gram))
    target_second = _nonnegative(target_second, abs(diagonal_sum) * n)
    point = (n - k) / (k * (n - 1)) * (n * diagonal_sum - target_second)
    paired_second = clustered_second = 0.0
    designs = enumerate_subset_audits(n, group_size, k)
    for indices in designs:
        selected_delta = flat_delta[indices]
        contrast = n / (k - 1) * (selected_delta - selected_delta.mean())
        paired_second += float(contrast @ gram[np.ix_(indices, indices)] @ contrast)
        weighted = n / k * selected_delta
        clustered_second += float(
            weighted @ transformed_gram[np.ix_(indices, indices)] @ weighted
        )
    paired_second /= len(designs)
    clustered_second /= len(designs)
    scale = max(abs(paired_second), abs(clustered_second), abs(n * diagonal_sum))
    return {
        "point_srs_variance": _nonnegative(point, scale) / n**2,
        "paired_variance": _nonnegative(paired_second - target_second, scale) / n**2,
        "clustered_point_variance": _nonnegative(clustered_second - target_second, scale)
        / n**2,
        "target_squared_norm": target_second / n**2,
    }
