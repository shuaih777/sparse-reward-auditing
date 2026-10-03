"""Paired dose contrasts with audit and prompt uncertainty; no GPU dependency."""

import numpy as np


def dose_contrast(small, large, baseline, small_rate=0.1, large_rate=1.0):
    """Per-unit-rate change difference, not merely the smaller absolute harm."""
    if not 0 < small_rate < large_rate:
        raise ValueError("ordered positive rates required")
    return (np.asarray(small) - baseline) / small_rate - (
        np.asarray(large) - baseline
    ) / large_rate


def paired_interval(difference, repetitions=4000, seed=202609245):
    """Crossed (audit x prompt) resampling, conditional on checkpoint and data.

    Independent resampling of row and column clusters retains pairing already
    encoded in `difference`. The interval is approximate, not a training-seed
    interval; a single reference row contributes no audit-resampling variance.
    """
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 2 or not np.isfinite(difference).all():
        raise ValueError("finite audit-by-prompt difference matrix required")
    draws, prompts = difference.shape
    if draws < 1 or prompts < 2 or repetitions < 2:
        raise ValueError("insufficient clusters or bootstrap repetitions")
    rng = np.random.default_rng(seed)
    samples = []
    # Bound temporary memory even when future runs use more prompts/draws.
    for start in range(0, repetitions, 100):
        size = min(100, repetitions - start)
        rows = rng.integers(draws, size=(size, draws))
        cols = rng.integers(prompts, size=(size, prompts))
        samples.extend(difference[rows[:, :, None], cols[:, None, :]].mean((1, 2)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta": float(difference.mean()),
        "crossed_bootstrap_95_low": float(low),
        "crossed_bootstrap_95_high": float(high),
        "conditional_prompt_se": float(
            difference.mean(0).std(ddof=1) / np.sqrt(prompts)
        ),
        "conditional_audit_se": (
            float(difference.mean(1).std(ddof=1) / np.sqrt(draws)) if draws > 1 else 0.0
        ),
        "audit_draws": draws,
        "prompts": prompts,
        "bootstrap_repetitions": repetitions,
        "limits": "Approximate crossed bootstrap; fixed checkpoint and training batch; no training-seed CI or multiplicity correction.",
    }
