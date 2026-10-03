"""CPU-only, past-purchased-label predictors for a shared residual control.

For each prompt group, a frozen constant b gives the estimator
``L(c + I/p * (o-c-b))``. L is full-group RLOO and L(b)=0. Under the stated
audit inclusion probabilities its conditional mean is therefore L(o), even
when b predicts the correction badly. An inaccurate b can increase variance;
unbiased coefficients are not unbiased optimizer steps or an efficacy claim.

Learner-facing functions receive only NaN-masked purchased oracle labels.
Complete residuals belong exclusively to the diagnostic variance functions.
No predictor is fitted from current-batch labels or hidden evaluator truth.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np

from .linear_audit import rloo
from .online_linear import PastAuditCalibrator


def _groups(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{name} must have nonempty 2D shape (B,G), with G >= 2")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"{name} must be finite binary values")
    return values


def _shared_baseline(baseline: np.ndarray | float, cheap: np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.ndim == 0:
        baseline = np.full((cheap.shape[0], 1), float(baseline))
    elif baseline.shape == cheap.shape:
        if not np.all(baseline == baseline[:, :1]):
            raise ValueError("baseline must be exactly constant within every group")
        baseline = baseline[:, :1].copy()
    elif baseline.shape != (cheap.shape[0], 1):
        raise ValueError("baseline must be scalar, shape (B,1), or group-constant (B,G)")
    if not np.isfinite(baseline).all():
        raise ValueError("baseline must be finite")
    return baseline


def shared_advantages(
    cheap: np.ndarray,
    audited: np.ndarray,
    p: float,
    baseline: np.ndarray | float,
) -> np.ndarray:
    """Return full-group L(c + I/p*(o-c-b)), without normalization or clipping.

    ``cheap`` and ``audited`` have shape (B,G); every unpurchased oracle entry
    must be NaN. The baseline must be fixed before the current audit draw and
    constant within each group. Only the caller can enforce that provenance
    and the audit design's inclusion probability. At p=0 no labels are allowed
    and cheap RLOO is returned, without an oracle-unbiasedness guarantee.
    """
    cheap = _groups(cheap, "cheap")
    audited = np.asarray(audited, dtype=np.float64)
    if audited.shape != cheap.shape:
        raise ValueError("audited and cheap must have identical shapes")
    if np.isinf(audited).any():
        raise ValueError("audited labels must be finite or unaudited NaN")
    baseline = _shared_baseline(baseline, cheap)
    if isinstance(p, (bool, np.bool_)) or np.ndim(p) != 0:
        raise ValueError("p must be a finite scalar in [0,1]")
    try:
        p = float(p)
    except (TypeError, ValueError) as error:
        raise ValueError("p must be a finite scalar in [0,1]") from error
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a finite scalar in [0,1]")
    mask = np.isfinite(audited)
    if p == 0:
        if mask.any():
            raise ValueError("zero inclusion probability requires no audited labels")
        return rloo(cheap)
    correction = np.where(mask, audited - cheap - baseline, 0.0) / p
    return rloo(cheap + correction)


# Singular alias follows the existing paired_advantage naming convention.
shared_advantage = shared_advantages


def shared_calibrated_baseline(
    calibrator: PastAuditCalibrator, cheap: np.ndarray
) -> np.ndarray:
    """Reuse the existing two-bin calibrator exactly, then share its residual.

    No observation/update occurs here. This deliberately does not invent a
    different warm start or prior: an empty original calibrator gives q0=q1=.5.
    The returned (B,1) array is frozen and cannot change after later observe().
    """
    cheap = _groups(cheap, "cheap")
    prediction = calibrator.predict(cheap)
    return (prediction - cheap).mean(axis=1, keepdims=True)


class SharedResidualPredictor:
    """Two fixed residual predictors from the last 64 purchased records only.

    Global: sum(delta)/(n+8), shrinking toward zero. By count: for each cheap
    group-positive count in 0..4, (sum_bin(delta)+8*global)/(n_bin+8).
    Hyperparameters are fixed protocol choices, not fitted on evaluator truth.
    Each record holds only group-positive count and purchased signed residual.
    At prediction, both outputs are clipped to the known binary-oracle support
    [-mean(cheap), 1-mean(cheap)] for that group. This uses no oracle labels;
    snapshot() retains the unclipped history-only values.
    """

    WINDOW_SIZE = 64
    PRIOR_WEIGHT = 8.0
    GROUP_SIZE = 4

    def __init__(self):
        self._window: deque[tuple[int, float]] = deque(maxlen=self.WINDOW_SIZE)
        self._past_label_count = 0

    def snapshot(self) -> dict:
        counts = np.zeros(self.GROUP_SIZE + 1, dtype=np.int64)
        sums = np.zeros(self.GROUP_SIZE + 1, dtype=np.float64)
        for count, residual in self._window:
            counts[count] += 1
            sums[count] += residual
        global_baseline = float(sums.sum() / (len(self._window) + self.PRIOR_WEIGHT))
        by_count = (sums + self.PRIOR_WEIGHT * global_baseline) / (
            counts + self.PRIOR_WEIGHT
        )
        return {
            "global": global_baseline,
            "by_count": by_count.tolist(),
            "window_counts": counts.tolist(),
            "window_residual_sums": sums.tolist(),
            "window_count": len(self._window),
            "past_label_count": self._past_label_count,
            "window_size": self.WINDOW_SIZE,
            "prior_weight": self.PRIOR_WEIGHT,
            "prediction_support": "[-mean(cheap), 1-mean(cheap)]",
        }

    def predict(self, cheap: np.ndarray) -> dict[str, np.ndarray]:
        """Freeze both (B,1) baselines, enforcing binary-oracle mean support."""
        cheap = _binary(_groups(cheap, "cheap"), "cheap")
        if cheap.shape[1] != self.GROUP_SIZE:
            raise ValueError("shared residual count predictor requires group size 4")
        state = self.snapshot()
        counts = cheap.sum(axis=1).astype(np.int64)
        mean_cheap = cheap.mean(axis=1, keepdims=True)
        lower, upper = -mean_cheap, 1.0 - mean_cheap
        return {
            "global": np.clip(
                np.full((cheap.shape[0], 1), state["global"]), lower, upper
            ),
            "by_count": np.clip(
                np.asarray(state["by_count"])[counts, None], lower, upper
            ),
        }

    def observe(
        self,
        group_positive_counts: np.ndarray,
        bought_cheap: np.ndarray,
        bought_oracle: np.ndarray,
    ) -> None:
        """Append bought records in recorded order, after this batch is evaluated.

        All arrays are aligned 1D purchased-record arrays, not full label
        arrays. Every validation is completed before history is mutated.
        """
        counts = np.asarray(group_positive_counts, dtype=np.float64)
        cheap = _binary(bought_cheap, "purchased cheap labels")
        oracle = _binary(bought_oracle, "purchased oracle labels")
        if counts.ndim != 1 or cheap.shape != counts.shape or oracle.shape != counts.shape:
            raise ValueError("purchased counts/cheap/oracle must be aligned 1D arrays")
        if not np.isin(counts, np.arange(self.GROUP_SIZE + 1)).all():
            raise ValueError("group_positive_counts must be integers in [0,4]")
        if np.any((counts == 0) & (cheap != 0)) or np.any(
            (counts == self.GROUP_SIZE) & (cheap != 1)
        ):
            raise ValueError("purchased cheap labels contradict their group-positive counts")
        for count, residual in zip(counts, oracle - cheap, strict=True):
            self._window.append((int(count), float(residual)))
        self._past_label_count += len(counts)


def exact_coefficient_variance_by_group(residual: np.ndarray, k: int) -> np.ndarray:
    """Evaluator-only exact per-group SUM of coefficient variances under SRS.

    ``residual`` is complete o-c-b for the shared estimator, or o-m for the
    ordinary calibrated estimator. Sampling is k of N uniformly without
    replacement across the entire batch. The deterministic dense term has no
    variance. For Q_g=sum_i r_i^2 and R_g=||L(r_g)||^2, the contribution is

      (N-k)/(k*(N-1)) * (N*G/(G-1)*Q_g - R_g).

    These values are not divided by N or N^2, are not score-vector gradient
    variances, and do not predict nonlinear optimizer or training behavior.
    """
    residual = _groups(residual, "evaluator residual")
    n = residual.size
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, Integral)
        or not 1 <= k <= n
    ):
        raise ValueError("k must be an integer in [1,N]")
    g = residual.shape[1]
    q = np.square(residual).sum(axis=1)
    target_norm = np.square(rloo(residual)).sum(axis=1)
    variance = (n - k) / (int(k) * (n - 1)) * (
        n * g / (g - 1) * q - target_norm
    )
    scale = np.maximum(1.0, n * q)
    if np.any(variance < -1e-10 * scale):
        raise ValueError("unexpected negative variance")
    return np.maximum(variance, 0.0)


def exact_coefficient_variance(residual: np.ndarray, k: int) -> float:
    """Evaluator-only total coefficient variance; see per-group function."""
    return float(exact_coefficient_variance_by_group(residual, k).sum())
