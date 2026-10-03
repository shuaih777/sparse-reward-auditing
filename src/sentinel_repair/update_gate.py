"""Sparse audits of a frozen, group-centered update surrogate (CPU only).

For a B-by-G array of per-answer score displacements ``v``, define

    h_i = G / (G - 1) * (v_i - mean(v in the same prompt group)),
    T(r, v) = mean(r * h) = mean(RLOO(r) * v).

``v`` may be a *finite* log-probability difference on already sampled answers.
In that case T is a frozen-batch surrogate difference, NOT a population-return
derivative, an importance-weighted return estimate, or a guarantee about the
proposed optimizer step. Even if v is an exact directional score derivative,
a proposal selected using this same batch need not estimate a population
derivative without selection bias.

Freeze v and the reward predictions m BEFORE drawing the current audits. With
k answers sampled uniformly without replacement out of N, the estimator

    mean(m * h) + mean((r - m) * h over the k purchased answers)

is conditionally unbiased for T(r, v). Its exact audit-design variance is
``(1-k/N) * sample_variance((r-m)*h over all N answers) / k``. These identities
concern the fixed batch, NOT the nonlinear point-sign gate or actual return.

The learner-facing estimator accepts only the purchased rewards. Full rewards
are confined to explicitly evaluator-only target, variance and simulation
helpers. This module does not draw policy proposals, change a policy, claim a
safety certificate, or learn an audit-allocation rule. The optional exact
two-label evaluator uses a fixed public-only directional sampling heuristic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a nonempty B-by-G array with G >= 2")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _aligned(
    value: ArrayLike, shape: tuple[int, int], name: str
) -> NDArray[np.float64]:
    matrix = _matrix(value, name)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have exactly the displacement shape")
    return matrix


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _budget(audit_budget: int, population: int) -> int:
    budget = _integer(audit_budget, "audit_budget", 1)
    if budget > population:
        raise ValueError("audit_budget must not exceed the population size")
    return budget


def _scalar(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("surrogate calculation produced a nonfinite value")
    return value


def group_adjusted_displacement(score_displacement: ArrayLike) -> NDArray[np.float64]:
    """Return RLOO(v), centering ONLY within each explicitly supplied row.

    A finite log-probability difference is allowed, but is not silently
    reinterpreted as an infinitesimal directional derivative. Caller owns that
    provenance and must use the same answer ordering as the reward arrays.
    """
    v = _matrix(score_displacement, "score_displacement")
    width = v.shape[1]
    h = (v - v.mean(axis=1, keepdims=True)) * (width / (width - 1))
    if not np.isfinite(h).all():
        raise ValueError("group-adjusted displacement is nonfinite")
    return h


def frozen_batch_target(score_displacement: ArrayLike, rewards: ArrayLike) -> float:
    """Evaluator-only T=mean(rewards*RLOO(v)); no actual-return guarantee.

    Passing predictions instead of oracle rewards computes the corresponding
    prediction-only surrogate. Rewards may be any finite real values.
    """
    h = group_adjusted_displacement(score_displacement)
    r = _aligned(rewards, h.shape, "rewards")
    return _scalar(np.mean(r * h))


def estimate_frozen_surrogate(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    purchased_indices: ArrayLike,
    purchased_rewards: ArrayLike,
    *,
    audit_budget: int,
) -> float:
    """Estimate the fixed-batch oracle surrogate from exactly k bought labels.

    ``purchased_indices`` indexes flattened row-major B-by-G arrays. They must
    be unique and have been sampled uniformly without replacement. This
    function checks the geometry and exact budget, but cannot verify the
    caller's sampling mechanism or whether v/m were frozen before purchase.

    Full oracle arrays, NaN-masked full arrays, repeated indices, broadcasting,
    and zero-budget claims of unbiasedness are rejected. Only the k aligned
    purchased rewards cross this interface. k=N is a valid full-audit case.
    """
    h = group_adjusted_displacement(score_displacement)
    m = _aligned(predictions, h.shape, "predictions")
    k = _budget(audit_budget, h.size)
    indices = np.asarray(purchased_indices)
    if indices.ndim != 1 or indices.size != k:
        raise ValueError("purchased_indices must contain exactly audit_budget entries")
    if indices.dtype.kind not in "iu":
        raise ValueError("purchased_indices must contain integers")
    if np.any(indices < 0) or np.any(indices >= h.size):
        raise ValueError("purchased index is outside the batch")
    if np.unique(indices).size != k:
        raise ValueError("sampling is without replacement: purchased indices must be unique")
    if np.iscomplexobj(purchased_rewards):
        raise ValueError("purchased_rewards must be real")
    labels = np.asarray(purchased_rewards, dtype=np.float64)
    if labels.ndim != 1 or labels.size != k:
        raise ValueError("provide exactly one purchased reward per purchased index")
    if not np.isfinite(labels).all():
        raise ValueError("purchased_rewards must be finite; masked full truth is not accepted")
    residual_mean = np.mean((labels - m.reshape(-1)[indices]) * h.reshape(-1)[indices])
    return _scalar(np.mean(m * h) + residual_mean)


def exact_conditional_variance(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    oracle_rewards: ArrayLike,
    *,
    audit_budget: int,
) -> float:
    """Evaluator-only exact SRSWOR variance, conditional on the frozen batch.

    This full-truth quantity must NOT be used to select the current audits,
    decide whether to take the proposed step, or tune that batch's gate.
    """
    h = group_adjusted_displacement(score_displacement)
    m = _aligned(predictions, h.shape, "predictions")
    r = _aligned(oracle_rewards, h.shape, "oracle_rewards")
    k = _budget(audit_budget, h.size)
    if k == h.size:
        return 0.0
    residual_population = (r - m) * h
    return _scalar((1.0 - k / h.size) * np.var(residual_population, ddof=1) / k)


def point_sign_accept(estimated_surrogate: float) -> bool:
    """Accept iff the estimated surrogate is strictly positive; NOT certified.

    Zero is rejected. This is a fixed decision rule, not a confidence bound,
    and unbiasedness of the scalar estimate does not make this gate unbiased
    or ensure safety, monotone learning, or positive population return.
    """
    estimate = np.asarray(estimated_surrogate)
    if estimate.ndim != 0 or np.iscomplexobj(estimate):
        raise ValueError("estimated_surrogate must be a finite real scalar")
    return _scalar(estimate) > 0.0


def simulate_point_sign_gate(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    oracle_rewards: ArrayLike,
    *,
    audit_budget: int,
    repetitions: int,
    seed: int,
) -> dict:
    """Evaluator-only repeated audit draws of ONE frozen proposed update.

    Oracle rewards are sliced at the purchase boundary before each estimate.
    The strict point-sign decision sees that estimate only. Realized surrogate
    utility is ``accept * T(oracle, v)``; it is NOT measured policy performance.
    Repetitions are audit draws, not independently trained policies or batches.
    """
    v = _matrix(score_displacement, "score_displacement")
    m = _aligned(predictions, v.shape, "predictions")
    r = _aligned(oracle_rewards, v.shape, "oracle_rewards")
    k = _budget(audit_budget, v.size)
    repetitions = _integer(repetitions, "repetitions", 1)
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    estimates = []
    accepted = []
    for _ in range(repetitions):
        indices = rng.choice(v.size, k, replace=False)
        estimate = estimate_frozen_surrogate(
            v, m, indices, r.reshape(-1)[indices], audit_budget=k
        )
        estimates.append(estimate)
        accepted.append(point_sign_accept(estimate))
    target = frozen_batch_target(v, r)
    fraction = float(np.mean(accepted))
    return {
        "target_name": "frozen_batch_group_centered_displacement_surrogate",
        "decision_rule": "estimate > 0; zero rejected; no safety guarantee",
        "scope": "one frozen proposal; audit-draw repetitions, not training seeds",
        "population_size": int(v.size),
        "group_size": int(v.shape[1]),
        "audit_budget": k,
        "repetitions": repetitions,
        "seed": seed,
        "oracle_surrogate": target,
        "prediction_surrogate": frozen_batch_target(v, m),
        "exact_conditional_variance": exact_conditional_variance(
            v, m, r, audit_budget=k
        ),
        "estimated_surrogates": estimates,
        "accepted": accepted,
        "acceptance_fraction": fraction,
        "always_accept_surrogate_utility": target,
        "always_reject_surrogate_utility": 0.0,
        "mean_gated_surrogate_utility": fraction * target,
    }


def directional_pair_design(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    uniform_fraction: float = 0.2,
) -> dict:
    """Frozen public-only two-draw design, with exact unordered-pair masses.

    Draw first from q, then from q renormalized after excluding that answer:

        q_i = a/N + (1-a) * |h_i| sqrt(c_i(1-c_i)) / sum(weights),
        c_i = clip(m_i, 0, 1),
        P({i,j}) = q_i q_j [1/(1-q_i) + 1/(1-q_j)].

    Zero total directional weight falls back to uniform. The strictly positive
    uniform fraction retains inclusion support. m is only a variance *proxy*,
    not a calibrated uncertainty guarantee. No true rewards enter this helper.
    This is a standard unequal-probability sampling/HT primitive, not a claim
    of a new sampling estimator. k is exactly two unique answers.
    """
    h = group_adjusted_displacement(score_displacement)
    m = _aligned(predictions, h.shape, "predictions")
    fraction = np.asarray(uniform_fraction)
    if fraction.ndim != 0 or np.iscomplexobj(fraction):
        raise ValueError("uniform_fraction must be a finite real scalar")
    fraction = _scalar(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("uniform_fraction must satisfy 0 < fraction <= 1")
    clipped = np.clip(m.reshape(-1), 0.0, 1.0)
    weights = np.abs(h.reshape(-1)) * np.sqrt(clipped * (1.0 - clipped))
    largest = weights.max()
    if largest == 0:
        q = np.full(h.size, 1.0 / h.size)
    else:
        # Rescale first so the normalization sum cannot overflow.
        weights = weights / largest
        q = fraction / h.size + (1.0 - fraction) * weights / weights.sum()
        q = q / q.sum()
    if not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("the floating-point draw probabilities must lie strictly in (0, 1)")
    first, second = np.triu_indices(h.size, k=1)
    pairs = np.column_stack((first, second))
    prob = q[first] * q[second] * (1.0 / (1.0 - q[first]) + 1.0 / (1.0 - q[second]))
    pi = np.bincount(first, weights=prob, minlength=h.size) + np.bincount(
        second, weights=prob, minlength=h.size
    )
    return {
        "design": "sequential q draws without replacement; unordered pairs",
        "population_size": int(h.size),
        "group_size": int(h.shape[1]),
        "audit_budget": 2,
        "uniform_fraction": fraction,
        "q": q,
        "pairs": pairs,
        "prob": prob,
        "pi": pi,
    }


def _checked_pair_design(design: dict, population: int) -> tuple[NDArray, NDArray, NDArray]:
    """Validate a precomputed complete pair design, not its provenance."""
    if not isinstance(design, dict):
        raise ValueError("design must be a directional_pair_design dictionary")
    try:
        pairs = np.asarray(design["pairs"])
        probability = np.asarray(design["prob"], dtype=np.float64)
        inclusion = np.asarray(design["pi"], dtype=np.float64)
        q = np.asarray(design["q"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("design must contain q, pairs, prob, and pi arrays") from error
    first, second = np.triu_indices(population, k=1)
    expected_pairs = np.column_stack((first, second))
    if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, expected_pairs):
        raise ValueError("design must enumerate each unordered pair exactly once in canonical order")
    if q.shape != (population,) or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("design q must be a finite positive probability vector")
    if not np.isclose(q.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("design q must sum to one")
    expected_probability = q[first] * q[second] * (
        1.0 / (1.0 - q[first]) + 1.0 / (1.0 - q[second])
    )
    if probability.shape != (len(pairs),) or not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError("design pair probabilities must be finite and positive")
    if not np.allclose(probability, expected_probability, rtol=1e-12, atol=1e-14):
        raise ValueError("pair probabilities do not match the sequential draw law")
    if not np.isclose(probability.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("pair probabilities must sum to one")
    expected_inclusion = np.bincount(first, weights=probability, minlength=population) + np.bincount(
        second, weights=probability, minlength=population
    )
    if inclusion.shape != (population,) or not np.isfinite(inclusion).all() or np.any(inclusion <= 0):
        raise ValueError("design inclusion probabilities must be finite and positive")
    if not np.allclose(inclusion, expected_inclusion, rtol=1e-12, atol=1e-14):
        raise ValueError("inclusion probabilities do not match unordered pair masses")
    return pairs, probability, inclusion


def estimate_pair_surrogates(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    oracle_rewards: ArrayLike,
    *,
    design: dict,
) -> NDArray[np.float64]:
    """Evaluator-only HT estimates for every possible frozen two-label audit.

    Each output uses exactly the two rewards belonging to that pair:
    ``mean(m*h) + [((r_i-m_i)*h_i)/pi_i + ((r_j-m_j)*h_j)/pi_j]/N``.
    All rewards enter this *enumerator*, not a learner-facing query interface.
    design must have been computed before observing current oracle labels;
    geometry validation cannot establish that caller-owned temporal ordering.
    """
    h = group_adjusted_displacement(score_displacement)
    m = _aligned(predictions, h.shape, "predictions")
    r = _aligned(oracle_rewards, h.shape, "oracle_rewards")
    pairs, _, inclusion = _checked_pair_design(design, h.size)
    residual_ht = ((r - m) * h).reshape(-1) / inclusion
    estimates = np.mean(m * h) + residual_ht[pairs].sum(axis=1) / h.size
    if not np.isfinite(estimates).all():
        raise ValueError("pair surrogate estimates must be finite")
    return estimates


def exact_pair_gate(
    score_displacement: ArrayLike,
    predictions: ArrayLike,
    oracle_rewards: ArrayLike,
    *,
    uniform_fraction: float = 0.2,
    design: dict | None = None,
) -> dict:
    """Evaluator-only exact distribution of the two-label point-sign gate.

    Providing design reuses that frozen design; uniform_fraction is then
    ignored. Array outputs preserve every pair for matched comparisons.
    No oracle labels choose the sampling law or the accept rule. Expectations
    are conditional on ONE fixed proposal, not over training trajectories.
    """
    h = group_adjusted_displacement(score_displacement)
    m = _aligned(predictions, h.shape, "predictions")
    r = _aligned(oracle_rewards, h.shape, "oracle_rewards")
    if design is None:
        design = directional_pair_design(score_displacement, m, uniform_fraction)
    pairs, probability, inclusion = _checked_pair_design(design, h.size)
    estimates = estimate_pair_surrogates(score_displacement, m, r, design=design)
    target = frozen_batch_target(score_displacement, r)
    mean = _scalar(np.dot(probability, estimates))
    variance = _scalar(np.dot(probability, (estimates - mean) ** 2))
    accepted = estimates > 0.0
    acceptance = _scalar(probability[accepted].sum())
    return {
        "target_name": "frozen_batch_group_centered_displacement_surrogate",
        "decision_rule": "estimate > 0; zero rejected; no safety guarantee",
        "scope": "exact two-label audit design for one frozen proposal; not training seeds",
        "false_accept_note": "Positive point estimates can accept a negative oracle surrogate; no safety guarantee.",
        "population_size": int(h.size),
        "group_size": int(h.shape[1]),
        "audit_budget": 2,
        "oracle_surrogate": target,
        "prediction_surrogate": frozen_batch_target(score_displacement, m),
        "estimator_mean": mean,
        "exact_conditional_variance": variance,
        "acceptance_fraction": acceptance,
        "false_accept_probability": acceptance if target < 0 else 0.0,
        "always_accept_surrogate_utility": target,
        "always_reject_surrogate_utility": 0.0,
        "mean_gated_surrogate_utility": acceptance * target,
        "estimated_surrogates": estimates,
        "accepted": accepted,
        "pair_probabilities": probability.copy(),
        "pair_indices": pairs.copy(),
        "inclusion_probabilities": inclusion.copy(),
        "draw_probabilities": np.asarray(design["q"]).copy(),
    }


__all__ = (
    "group_adjusted_displacement",
    "frozen_batch_target",
    "estimate_frozen_surrogate",
    "exact_conditional_variance",
    "point_sign_accept",
    "simulate_point_sign_gate",
    "directional_pair_design",
    "estimate_pair_surrogates",
    "exact_pair_gate",
)
