"""Past-audit text residual prediction with shared and direct controls.

This is a cached numerical implementation of ``HashedLogisticRiskModel``,
not a new text learner. Public features are prepared once and can be reused
across independent simulated audit histories. Each history owns two separate
weight vectors: the existing text-plus-numeric feature set and its matched
numeric-only ablation. Neither preparation nor prediction accepts truth.

An error probability q predicts a signed reward residual as (1 - 2*c)*q for
binary cheap reward c. The direct reward control is c + residual; the shared
control is c + mean(residual within the question). Shared controls need not
lie in [0, 1]: they are linear-estimator controls, not reward probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.special import expit

from .advantages import grpo_advantages
from .data import GroupObservation, RolloutObservation
from .selectors import HashedLogisticRiskModel


def text_residual_config(*, include_residual: bool = True) -> dict:
    """Fixed existing predictor; no runtime feature or parameter search."""
    return {
        "model": "HashedLogisticRiskModel",
        "dimension": 2048, "learning_rate": 0.1, "l2": 1e-6,
        "prior": 0.05, "max_tokens": 128,
        "prediction": "m=cheap+(1-2*cheap)*predicted_label_error_probability",
        "advantage": (
            "RLOO(m+I/p*(oracle-m)); no group standardization"
            if include_residual
            else "RLOO(m); no current-audit residual or group standardization; generally biased"
        ),
        "features": "current completion and cheap-only recomputed features; no trigger metadata",
        "ordering": "predict before current audit draw; update after advantages from purchased labels only",
        "within_batch_order": "recorded audit_indices order",
        "cold_start": "fresh predictor at continuation start; no external fitted labels",
    }


@dataclass(frozen=True)
class PreparedTextBatch:
    """Cheap-only immutable arrays; no truth or logged metadata is retained."""

    cheap: NDArray[np.float64]
    text_features: csr_matrix
    numeric_features: csr_matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self.cheap.shape

    @property
    def n_rollouts(self) -> int:
        return self.cheap.size


def _public_groups(
    cheap: NDArray[np.float64], completions: Sequence[str]
) -> tuple[GroupObservation, ...]:
    """Rebuild observations; never reuse logged advantages or private fields."""
    groups = []
    width = cheap.shape[1]
    for group_id, rewards in enumerate(cheap):
        advantages = grpo_advantages(rewards)
        rows = tuple(
            RolloutObservation(
                rollout_id=group_id * width + position,
                group_id=group_id,
                position=position,
                step="",
                prompt="",
                completion=completions[group_id * width + position],
                cheap_reward=float(reward),
                current_reward=float(reward),
                advantage=float(advantages[position]),
                audited=False,
            )
            for position, reward in enumerate(rewards)
        )
        groups.append(GroupObservation(group_id, "", "", rows))
    return tuple(groups)


def _numeric_features(
    model: HashedLogisticRiskModel,
    candidate: RolloutObservation,
    group: GroupObservation,
) -> dict[int, float]:
    """The same six named numeric features and hashing as the existing model.

    Keep zero-valued entries: the original SGD regularizes every active
    coordinate, including numeric coordinates whose feature value is zero.
    """
    features = {0: 1.0}
    named = (
        ("cheap_reward", float(candidate.cheap_reward)),
        ("current_reward", float(candidate.current_reward)),
        ("abs_advantage", min(abs(candidate.advantage), 10.0) / 10.0),
        ("completion_length", min(len(candidate.completion), 4096) / 4096.0),
        ("group_size", min(len(group), 32) / 32.0),
        (
            "group_positive_fraction",
            sum(row.current_reward > 0 for row in group.rollouts) / len(group),
        ),
    )
    for name, value in named:
        bucket = model._bucket(name)
        features[bucket] = features.get(bucket, 0.0) + value
    return features


def _sparse_rows(rows: list[dict[int, float]], dimension: int) -> csr_matrix:
    # Preserve the original dictionary accumulation order within each row.
    # There are no duplicate coordinates because hashing already accumulated
    # collisions. Explicit zero entries must not be eliminated.
    data, indices, indptr = [], [], [0]
    for row in rows:
        indices.extend(row.keys())
        data.extend(row.values())
        indptr.append(len(data))
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(rows), dimension),
    )
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.setflags(write=False)
    return matrix


def prepare_text_batch(cheap: ArrayLike, completions: Sequence[str]) -> PreparedTextBatch:
    """Cache exact existing features using only this batch's public information.

    ``cheap`` is a nonempty B-by-G binary array, G >= 2. Completions are flat
    in the corresponding row-major order. No vocabulary is fitted, no labels
    are accepted, and completions need not be retained after preparation.
    """
    rewards = np.array(cheap, dtype=np.float64, copy=True)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError("cheap must be nonempty B-by-G with G >= 2")
    if not np.all(np.isfinite(rewards)) or not np.all(
        (rewards == 0.0) | (rewards == 1.0)
    ):
        raise ValueError("cheap must contain finite binary labels")
    if isinstance(completions, str) or len(completions) != rewards.size:
        raise ValueError("completions must contain one flat string per rollout")
    if not all(isinstance(item, str) for item in completions):
        raise TypeError("every completion must be a string")
    feature_model = HashedLogisticRiskModel()
    text_rows, numeric_rows = [], []
    for group in _public_groups(rewards, completions):
        for row in group.rollouts:
            text_rows.append(feature_model._features(row, group))
            numeric_rows.append(_numeric_features(feature_model, row, group))
    rewards.setflags(write=False)
    return PreparedTextBatch(
        rewards,
        _sparse_rows(text_rows, feature_model.dimension),
        _sparse_rows(numeric_rows, feature_model.dimension),
    )


class TextResidualBank:
    """Two independent learners and four matched fixed-before-audit controls.

    ``predict(batch)`` must precede ``observe(batch, indices, labels)``. An
    observation call discloses only the purchased labels, in the explicitly
    provided order; it cannot accept a full truth vector with masked indices.
    Independent audit-history replicas require separate bank instances.
    """

    def __init__(self) -> None:
        reference = HashedLogisticRiskModel()
        self.dimension = reference.dimension
        self.learning_rate = reference.learning_rate
        self.l2 = reference.l2
        self.max_tokens = reference.max_tokens
        self._weights = {
            name: np.zeros(reference.dimension, dtype=np.float64)
            for name in ("text", "numeric")
        }
        for weights in self._weights.values():
            weights[0] = reference._weights[0]
        self._labels_seen = 0
        self._errors_seen = 0
        self._pending_batch: PreparedTextBatch | None = None

    def predict(self, batch: PreparedTextBatch) -> dict[str, NDArray[np.float64]]:
        """Return frozen B-by-G reward controls, before any current labels."""
        if not isinstance(batch, PreparedTextBatch):
            raise TypeError("batch must be prepared with prepare_text_batch")
        predictions = {}
        for name, matrix in (
            ("text", batch.text_features),
            ("numeric", batch.numeric_features),
        ):
            q = expit(matrix @ self._weights[name]).reshape(batch.shape)
            residual = (1.0 - 2.0 * batch.cheap) * q
            predictions[f"{name}_direct"] = batch.cheap + residual
            predictions[f"{name}_shared"] = batch.cheap + residual.mean(
                axis=1, keepdims=True
            )
        self._pending_batch = batch
        return predictions

    def observe(
        self,
        batch: PreparedTextBatch,
        purchased_indices: ArrayLike,
        purchased_labels: ArrayLike,
    ) -> None:
        """Sequential existing-rule SGD, only after current predictions freeze."""
        if batch is not self._pending_batch:
            raise ValueError("predict this batch before observing purchased labels")
        raw_indices = np.asarray(purchased_indices)
        labels = np.asarray(purchased_labels, dtype=np.float64)
        if raw_indices.ndim != 1 or labels.ndim != 1:
            raise ValueError("purchased indices and labels must be one-dimensional")
        if raw_indices.size != labels.size:
            raise ValueError("provide exactly one label per purchased index")
        if raw_indices.size and (
            raw_indices.dtype.kind not in "iu" or raw_indices.dtype.kind == "b"
        ):
            raise TypeError("purchased indices must be integers")
        indices = raw_indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= batch.n_rollouts):
            raise ValueError("purchased index outside the prepared batch")
        if np.unique(indices).size != indices.size:
            raise ValueError("purchased indices must not repeat within a batch")
        if not np.all(np.isfinite(labels)) or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("purchased labels must be finite binary values")
        targets = labels != batch.cheap.reshape(-1)[indices]
        for name, matrix in (
            ("text", batch.text_features),
            ("numeric", batch.numeric_features),
        ):
            weights = self._weights[name]
            for index, target in zip(indices, targets):
                start, end = matrix.indptr[index : index + 2]
                active = matrix.indices[start:end]
                values = matrix.data[start:end]
                old = weights[active]
                probability = float(expit(np.dot(old, values)))
                weights[active] = old + self.learning_rate * (
                    (float(target) - probability) * values - self.l2 * old
                )
        self._labels_seen += indices.size
        self._errors_seen += int(targets.sum())
        self._pending_batch = None

    def snapshot(self) -> dict[str, int | float | str]:
        """Small provenance state; intentionally contains no stored audit text."""
        return {
            "labels_seen": self._labels_seen,
            "errors_seen": self._errors_seen,
            "dimension": self.dimension,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "prior": 0.05,
            "max_tokens": self.max_tokens,
            "numeric_feature_count": 6,
            "feature_source": "cheap-only GRPO advantages; blank prompt/step/metadata",
            "text_weight_norm": float(np.linalg.norm(self._weights["text"])),
            "numeric_weight_norm": float(np.linalg.norm(self._weights["numeric"])),
        }


__all__ = ("PreparedTextBatch", "TextResidualBank", "prepare_text_batch", "text_residual_config")
