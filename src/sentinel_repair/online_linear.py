"""CPU-only audit budgets and linear advantages for online grouped updates.

The training boundary accepts only purchased labels, represented by finite
entries in an otherwise NaN-filled array. Full truth belongs to the evaluator;
the two full-oracle methods are explicit, separately budgeted references.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np

from .linear_audit import audit_advantage, grpo, rloo

METHODS = (
    "cheap_grpo",
    "cheap_rloo",
    "full_oracle_rloo",
    "linear_ipw",
    "oracle_only",
    "oracle_centered",
    "linear_replace",
    "group_scaled_ipw",
    "full_oracle_group_scaled",
    "calibrated_ipw",
    "preaudit_scaled_ipw",
    "text_direct_ipw",
    "text_prediction_only",
)
FULL_ORACLE_METHODS = ("full_oracle_rloo", "full_oracle_group_scaled")
SPARSE_METHODS = (
    "linear_ipw",
    "oracle_only",
    "oracle_centered",
    "linear_replace",
    "group_scaled_ipw",
    "calibrated_ipw",
    "preaudit_scaled_ipw",
    "text_direct_ipw",
    "text_prediction_only",
)
TEXT_PREDICTED_METHODS = ("text_direct_ipw", "text_prediction_only")
PREDICTED_METHODS = ("calibrated_ipw", *TEXT_PREDICTED_METHODS)


def preaudit_scale_config() -> dict:
    """Fixed G=4 binary-label control, not a fitted or tuned normalization."""
    return {
        "group_size": 4,
        "cheap_labels": [0, 1],
        "purchased_labels": [0, 1],
        "std_ddof": 1,
        "std_floor": 0.5,
        "epsilon": 1e-4,
        "denominator": "max(sample_std(cheap), 0.5) + 1e-4",
        "ordering": "compute from cheap rewards before current audit sampling",
        "conditional_mean_target": "RLOO(oracle) / denominator(cheap), for p > 0",
        "zero_budget": "RLOO(cheap) / denominator(cheap); no oracle mean guarantee",
    }


def preaudit_denominators(cheap: np.ndarray) -> np.ndarray:
    """Return one cheap-only scale per G=4 binary group, retaining its axis.

    The smallest nonzero sample standard deviation is 0.5 for this geometry.
    The floor therefore changes only constant cheap groups, whose uncorrected
    RLOO is zero; it does not change nonconstant, uncorrected groups.
    """
    cheap = np.asarray(cheap, dtype=np.float64)
    if cheap.ndim == 0 or cheap.shape[-1] != 4:
        raise ValueError("preaudit scaling requires group size 4")
    if not np.isin(cheap, [0.0, 1.0]).all():
        raise ValueError("preaudit cheap rewards must be finite binary values")
    return np.maximum(cheap.std(axis=-1, ddof=1, keepdims=True), 0.5) + 1e-4


def calibration_config() -> dict:
    """Fixed protocol descriptor; no runtime calibration hyperparameters."""
    return {
        "cheap_classes": [0, 1],
        "window_per_class": 32,
        "beta_prior": [1, 1],
        "prediction": "freeze from purchased labels in prior batches before current audit",
        "update": "observe current purchased cheap/oracle pairs after advantages are computed",
        "within_batch_order": "audit_indices / bought_labels order",
        "zero_budget": "RLOO(frozen baseline); no history update",
    }


class PastAuditCalibrator:
    """Two fixed rolling-32 Beta(1,1) means, updated only from bought pairs.

    A prediction is a fresh array: later observations cannot alter a frozen
    batch baseline. The caller freezes it before sampling and calls observe
    only after computing that batch's advantages, in recorded purchase order.
    Empty observations are a no-op. All inputs are validated before mutation.
    """

    def __init__(self):
        self._windows = (deque(maxlen=32), deque(maxlen=32))
        self._past_label_count = 0

    @staticmethod
    def _binary(values, name):
        values = np.asarray(values, dtype=np.float64)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{name} must be finite binary values")
        return values

    def snapshot(self) -> dict:
        counts = [len(window) for window in self._windows]
        positives = [sum(window) for window in self._windows]
        return {
            "q0": (1 + positives[0]) / (2 + counts[0]),
            "q1": (1 + positives[1]) / (2 + counts[1]),
            "window_counts": counts,
            "window_positives": positives,
            "past_label_count": self._past_label_count,
        }

    def predict(self, cheap: np.ndarray) -> np.ndarray:
        cheap = self._binary(cheap, "cheap categories")
        state = self.snapshot()
        return np.array([state["q0"], state["q1"]])[cheap.astype(np.int64)]

    def observe(self, bought_cheap: np.ndarray, bought_labels: np.ndarray) -> None:
        cheap = self._binary(bought_cheap, "purchased cheap categories")
        labels = self._binary(bought_labels, "purchased oracle labels")
        if cheap.ndim != 1 or labels.shape != cheap.shape:
            raise ValueError("purchased cheap/oracle pairs must be aligned 1D arrays")
        for category, label in zip(cheap, labels, strict=True):
            self._windows[int(category)].append(int(label))
        self._past_label_count += len(labels)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def audit_allowance(total_seen: int, batch_size: int) -> int:
    """New labels available at this batch boundary under a strict 1% budget.

    If all preceding allowances were spent, cumulative spend after this batch
    is exactly ``(total_seen + batch_size) // 100``. The guarantee concerns
    completed rollout-batch prefixes: generate the batch before buying labels.
    Integer arithmetic keeps the guarantee exact even for large counters.
    """
    total_seen = _nonnegative_integer(total_seen, "total_seen")
    batch_size = _nonnegative_integer(batch_size, "batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be positive")
    return (total_seen + batch_size) // 100 - total_seen // 100


def uniform_audit_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw k flat rollout indices uniformly without replacement from n.

    Every rollout has inclusion probability k/n for n > 0. Empty samples
    consume no randomness, and n=k=0 is supported.
    """
    n = _nonnegative_integer(n, "n")
    k = _nonnegative_integer(k, "k")
    if k > n:
        raise ValueError("k must not exceed n")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)


def _group_scaled_rloo(rewards: np.ndarray) -> np.ndarray:
    return rloo(rewards) / (rewards.std(axis=-1, ddof=1, keepdims=True) + 1e-4)


def linear_advantages(
    method: str,
    cheap: np.ndarray,
    audited: np.ndarray,
    p: float,
    *,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Return grouped advantages, with prompt groups on the last axis.

    ``audited`` must have exactly the shape of ``cheap`` and contain NaN at
    every unpurchased position. Only the caller can enforce the provenance of
    finite labels; this function never receives evaluator-only full truth.
    ``cheap_rloo`` and ``cheap_grpo`` do not inspect audited labels or p.
    ``cheap_grpo`` is the zero-label standard-deviation-normalized control.
    Both full-oracle references require every oracle label to be visible and
    ignore p. ``full_oracle_rloo`` returns oracle RLOO; ``full_oracle_group_scaled``
    divides it by the within-group oracle sample standard deviation (ddof=1)
    plus 1e-4, retaining G/(G-1). These are 100%-label diagnostic references,
    not 1%-budget estimators. TRL must not apply another normalization.

    Sparse methods use the known rollout inclusion probability p=k/n. For
    uniform sampling without replacement and p>0, linear_ipw, oracle_only and
    oracle_centered have conditional expectation equal to full-oracle RLOO,
    given the fixed rollout batch and its cheap/truth labels. This requires
    applying RLOO to the whole group, including unqueried members affected by
    its baseline.
    These three estimators apply no clipping or standard-deviation normalization.

    calibrated_ipw and text_direct_ipw require a frozen baseline with cheap's shape and
    finite values in [0,1]. It applies RLOO(baseline + I/p*(audited-baseline))
    without group scaling. The same conditional unbiasedness holds when this
    baseline uses only earlier batches' purchased labels and current cheap
    features (including current completion text for text_direct_ipw). The caller
    owns that ordering; this function never updates history.
    text_prediction_only uses the exact same frozen baseline but returns only
    RLOO(baseline), without a current-audit residual. Purchased labels still
    train the caller's predictor AFTER these advantages are computed. This
    ablation is generally biased relative to full-oracle RLOO, even at p=1.
    Other methods ignore the optional baseline entirely.

    linear_replace instead directly replaces only purchased rewards, without
    inverse-propensity weighting. Its conditional mean advantage is
    (1-p)*RLOO(cheap) + p*RLOO(truth), not full-oracle RLOO in general. It is
    the same-budget ablation of residual weighting, not an unbiased estimator.

    group_scaled_ipw uses exactly the linear_ipw pseudo reward, then divides its
    RLOO advantage by the within-group sample standard deviation (ddof=1) of
    that pseudo reward plus 1e-4. It retains the G/(G-1) RLOO factor, unlike
    cheap_grpo. This nonlinear ablation is not conditionally unbiased, and
    the trainer must not apply another reward/advantage normalization.

    preaudit_scaled_ipw uses the same pseudo reward, but divides its RLOO by
    max(sample_std(cheap), 0.5) + 1e-4, fixed before querying current labels.
    It is restricted to G=4 binary cheap and purchased labels. For p>0 its
    conditional mean is RLOO(truth) / denominator(cheap), NOT unscaled oracle
    RLOO or oracle-standardized RLOO. A fixed group with no nonzero purchased
    residual matches group_scaled_ipw exactly, including constant groups.

    At p=0, no finite audited label is allowed: linear_ipw/linear_replace return
    cheap RLOO, group_scaled_ipw returns group-scaled cheap RLOO, and
    oracle_only/oracle_centered return zero. These zero-budget batches have no
    conditional unbiasedness guarantee relative to oracle RLOO.
    calibrated_ipw likewise falls back to RLOO(frozen baseline) at p=0.
    preaudit_scaled_ipw falls back to cheap RLOO / denominator(cheap) at p=0.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    cheap = np.asarray(cheap, dtype=np.float64)
    if cheap.ndim == 0 or cheap.shape[-1] < 2:
        raise ValueError("group size must be at least two")
    if not np.isfinite(cheap).all():
        raise ValueError("cheap rewards must be finite")
    if method == "cheap_rloo":
        return rloo(cheap)
    if method == "cheap_grpo":
        return grpo(cheap)

    preaudit_scale = (
        preaudit_denominators(cheap) if method == "preaudit_scaled_ipw" else None
    )
    audited = np.asarray(audited, dtype=np.float64)
    if audited.shape != cheap.shape:
        raise ValueError("audited and cheap must have identical shapes")
    if np.isinf(audited).any():
        raise ValueError("audited labels must be finite or unaudited NaN")
    if preaudit_scale is not None and not np.isin(
        audited[np.isfinite(audited)], [0.0, 1.0]
    ).all():
        raise ValueError("preaudit purchased oracle labels must be binary")
    if method in FULL_ORACLE_METHODS:
        if not np.isfinite(audited).all():
            raise ValueError(f"{method} requires all oracle labels")
        return (
            rloo(audited)
            if method == "full_oracle_rloo"
            else _group_scaled_rloo(audited)
        )

    if method in PREDICTED_METHODS:
        if baseline is None:
            raise ValueError(f"{method} requires a frozen baseline")
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != cheap.shape:
            raise ValueError("frozen baseline and cheap must have identical shapes")
        if (
            not np.isfinite(baseline).all()
            or not ((0 <= baseline) & (baseline <= 1)).all()
        ):
            raise ValueError("frozen baseline must be finite and in [0,1]")
        if not np.isin(cheap, [0.0, 1.0]).all():
            raise ValueError("calibrated cheap categories must be binary")
        if not np.isin(audited[np.isfinite(audited)], [0.0, 1.0]).all():
            raise ValueError("calibrated purchased oracle labels must be binary")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("inclusion probability must be between zero and one")
    if p == 0:
        if np.isfinite(audited).any():
            raise ValueError("zero inclusion probability requires no audited labels")
        if method == "group_scaled_ipw":
            return _group_scaled_rloo(cheap)
        if preaudit_scale is not None:
            return rloo(cheap) / preaudit_scale
        if method in PREDICTED_METHODS:
            return rloo(baseline)
        return (
            rloo(cheap)
            if method in {"linear_ipw", "linear_replace"}
            else np.zeros_like(cheap)
        )
    if method == "text_prediction_only":
        return rloo(baseline)
    if method == "linear_replace":
        return rloo(np.where(np.isfinite(audited), audited, cheap))
    if method == "group_scaled_ipw":
        pseudo = cheap + np.where(np.isfinite(audited), audited - cheap, 0) / p
        return _group_scaled_rloo(pseudo)
    if preaudit_scale is not None:
        pseudo = cheap + np.where(np.isfinite(audited), audited - cheap, 0) / p
        return rloo(pseudo) / preaudit_scale
    if method in PREDICTED_METHODS:
        pseudo = baseline + np.where(np.isfinite(audited), audited - baseline, 0) / p
        return rloo(pseudo)
    return audit_advantage(cheap, audited, p, method)
