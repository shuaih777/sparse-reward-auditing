"""Linear group objectives and budgeted label corrections (no model dependency)."""

from __future__ import annotations

import numpy as np

METHODS = (
    "cheap_grpo",
    "cheap_rloo",
    "direct_grpo",
    "direct_rloo",
    "normalized_ipw",
    "linear_ipw",
    "oracle_only",
    "oracle_centered",
)


def rloo(rewards: np.ndarray) -> np.ndarray:
    k = rewards.shape[-1]
    if k < 2:
        raise ValueError("group size must be at least two")
    return (k * rewards - rewards.sum(axis=-1, keepdims=True)) / (k - 1)


def grpo(rewards: np.ndarray) -> np.ndarray:
    return (rewards - rewards.mean(axis=-1, keepdims=True)) / (
        rewards.std(axis=-1, ddof=1, keepdims=True) + 1e-4
    )


def audit_advantage(
    cheap: np.ndarray,
    audited: np.ndarray,
    p: float,
    method: str,
) -> np.ndarray:
    """Only NaN-masked purchased truth is accepted, with known inclusion p.

    Last axis is the prompt group. Leading axes can include audit replicates.
    The caller, not this estimator, owns evaluator-only full oracle labels.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if not 0 < p <= 1:
        raise ValueError("positive inclusion probability required")
    cheap, audited = np.broadcast_arrays(cheap, audited)
    if not np.isfinite(cheap).all() or np.isinf(audited).any():
        raise ValueError("finite rewards or unaudited NaN required")
    observed = np.isfinite(audited)
    direct = np.where(observed, audited, cheap)
    corrected = cheap + np.where(observed, audited - cheap, 0) / p
    if method == "cheap_grpo":
        return grpo(cheap)
    if method == "cheap_rloo":
        return rloo(cheap)
    if method == "direct_grpo":
        return grpo(direct)
    if method == "direct_rloo":
        return rloo(direct)
    if method == "normalized_ipw":
        return grpo(corrected)
    if method == "linear_ipw":
        return rloo(corrected)
    if method == "oracle_centered":
        # A fixed .5 baseline has zero full-population RLOO gradient. It lets
        # bought incorrect as well as correct labels inform the sparse update.
        return rloo(np.where(observed, audited - 0.5, 0) / p)
    return rloo(np.where(observed, audited, 0) / p)


def audit_subsets(n: int, k: int, repetitions: int, seed: int) -> np.ndarray:
    if not 0 < k <= n or repetitions < 1:
        raise ValueError("invalid fixed-size audit design")
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n, k, replace=False) for _ in range(repetitions)])


def masked_truth(truth: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Oracle boundary: reveal only explicitly bought indices to the estimator."""
    revealed = np.full((len(indices), truth.size), np.nan)
    np.put_along_axis(revealed, indices, truth.ravel()[indices], axis=1)
    return revealed.reshape((len(indices),) + truth.shape)


def gram_metrics(
    coefficients: np.ndarray,
    oracle_coefficients: np.ndarray,
    gram: np.ndarray,
    evaluation_dot_scores: np.ndarray,
) -> dict:
    """Exact parameter-subspace geometry from score-gradient Gram matrix.

    Coefficients already contain the 1/N averaging factor. No feature sketches.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    oracle = np.asarray(oracle_coefficients, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    product = c @ gram
    norms2 = np.maximum(np.einsum("ij,ij->i", product, c), 0)
    oracle_norm2 = max(float(oracle @ gram @ oracle), 0)
    inner = product @ oracle
    errors2 = np.maximum(norms2 + oracle_norm2 - 2 * inner, 0)
    mean_c = c.mean(axis=0)
    bias = mean_c - oracle
    bias2 = max(float(bias @ gram @ bias), 0)
    mean_norm2 = max(float(mean_c @ gram @ mean_c), 0)
    variance = max(float(norms2.mean() - mean_norm2), 0)
    cosine = inner / np.sqrt(np.maximum(norms2 * oracle_norm2, 1e-30))
    projection = inner / max(oracle_norm2, 1e-30)
    transfer = c @ evaluation_dot_scores
    reference_transfer = float(oracle @ evaluation_dot_scores)
    return {
        "oracle_norm": float(np.sqrt(oracle_norm2)),
        "mean_estimator_norm": float(np.sqrt(mean_norm2)),
        "rmse_over_oracle_norm": float(
            np.sqrt(errors2.mean() / max(oracle_norm2, 1e-30))
        ),
        "mc_mean_bias_over_oracle_norm": float(
            np.sqrt(bias2 / max(oracle_norm2, 1e-30))
        ),
        "variance_trace": variance,
        "oracle_projection_mean": float(projection.mean()),
        "oracle_projection_se": float(projection.std(ddof=1) / np.sqrt(len(c))),
        "negative_oracle_projection_fraction": float((inner < 0).mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.1)),
        "evaluation_derivative_mean": float(transfer.mean()),
        "evaluation_derivative_se": float(transfer.std(ddof=1) / np.sqrt(len(c))),
        "evaluation_derivative_positive_fraction": float((transfer > 0).mean()),
        "full_oracle_evaluation_derivative": reference_transfer,
    }


def exact_residual_variance(scores, rewards, k, group_size):
    """Evaluator-only SRSWOR variance trace; never chooses an audit/update."""
    n = len(scores)
    if not 0 < k <= n or n % group_size:
        raise ValueError("invalid fixed-size design or group geometry")
    grouped = np.asarray(scores, dtype=np.float64).reshape(
        -1, group_size, scores.shape[1]
    )
    z = (
        (group_size * grouped - grouped.sum(axis=1, keepdims=True)) / (group_size - 1)
    ).reshape(n, -1)
    population = np.asarray(rewards).reshape(n, 1) * z
    centered = population - population.mean(axis=0)
    return float((1 - k / n) / k * np.einsum("ij,ij->", centered, centered) / (n - 1))
