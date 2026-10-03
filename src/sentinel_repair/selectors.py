"""Oracle-query policies with an oracle-safe, group-aware interface."""

from __future__ import annotations

import hashlib
import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .data import GroupObservation, RolloutObservation


@dataclass(frozen=True)
class QueryRequest:
    """One selector action.

    A request can contain one rollout or an atomic group bundle.  Its budget
    cost is always the number of labels, never the number of groups.
    """

    rollout_ids: tuple[int, ...]
    reason: str
    priority: float | None = None
    predicted_risks: Mapping[int, float] = field(default_factory=dict)
    atomic: bool = False

    def __post_init__(self) -> None:
        ids = tuple(int(item) for item in self.rollout_ids)
        if not ids:
            raise ValueError("a query request cannot be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("a query request cannot repeat rollout ids")
        object.__setattr__(self, "rollout_ids", ids)
        object.__setattr__(
            self, "predicted_risks", MappingProxyType(dict(self.predicted_risks))
        )

    @property
    def cost(self) -> int:
        return len(self.rollout_ids)


@dataclass(frozen=True)
class AuditFeedback:
    """Information disclosed to a selector after one purchased label."""

    candidate_before: RolloutObservation
    group_before: GroupObservation
    group_after: GroupObservation
    observed_label: float
    label_error: bool
    predicted_risk: float | None = None
    priority: float | None = None


class QuerySelector(ABC):
    """Extensible interface for rollout- or group-valued query actions."""

    @abstractmethod
    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        """Propose an affordable action using public/revealed data only."""

    def observe(self, feedback: AuditFeedback) -> None:
        """Prequential update after, and only after, a purchased label."""

    def end_batch(self, groups: Sequence[GroupObservation]) -> None:
        """Optional notification once no more labels are bought this batch."""


def _candidates(
    groups: Sequence[GroupObservation],
) -> list[tuple[RolloutObservation, GroupObservation]]:
    return [
        (row, group)
        for group in groups
        if not group.quarantined
        for row in group.rollouts
        if not row.audited
    ]


class RandomSelector(QuerySelector):
    """Uniform random audit among currently available unqueried rollouts."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None
        candidates = _candidates(groups)
        if not candidates:
            return None
        row, _ = self._rng.choice(candidates)
        return QueryRequest((row.rollout_id,), reason="random")


class AbsAdvantageSelector(QuerySelector):
    """Audit the largest currently visible ``|A|`` coefficient."""

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None
        candidates = _candidates(groups)
        if not candidates:
            return None
        row, _ = max(
            candidates,
            key=lambda item: (abs(item[0].advantage), -item[0].rollout_id),
        )
        return QueryRequest(
            (row.rollout_id,),
            reason="abs_advantage",
            priority=abs(row.advantage),
        )


class ErrorRiskModel(Protocol):
    """Online model for ``P(cheap label is wrong | public information)``."""

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float: ...

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None: ...


class ConstantRiskModel:
    """Useful transparent baseline and test fixture."""

    def __init__(self, probability: float = 0.5) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float:
        return self.probability

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None:
        return None


class HashedLogisticRiskModel:
    """Small online text/numeric model trained only on bought audits.

    Feature hashing is stateless, so vocabulary construction cannot inspect
    future completions.  ``predict`` is called before selection and ``update``
    only from the subsequent :class:`AuditFeedback`.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\w\s]")

    def __init__(
        self,
        *,
        dimension: int = 2048,
        learning_rate: float = 0.1,
        l2: float = 1e-6,
        prior: float = 0.05,
        max_tokens: int = 128,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 cannot be negative")
        if not 0 < prior < 1:
            raise ValueError("prior must be strictly between 0 and 1")
        self.dimension = int(dimension)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.max_tokens = int(max_tokens)
        self._weights: dict[int, float] = {0: math.log(prior / (1.0 - prior))}
        self._lexical_cache: dict[tuple[int, str], dict[int, float]] = {}

    def _bucket(self, name: str) -> int:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        return 1 + int.from_bytes(digest, "little") % (self.dimension - 1)

    def _features(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> dict[int, float]:
        features: dict[int, float] = {0: 1.0}

        def add(name: str, value: float = 1.0) -> None:
            bucket = self._bucket(name)
            features[bucket] = features.get(bucket, 0.0) + value

        add("cheap_reward", float(candidate.cheap_reward))
        add("current_reward", float(candidate.current_reward))
        add("abs_advantage", min(abs(candidate.advantage), 10.0) / 10.0)
        add("completion_length", min(len(candidate.completion), 4096) / 4096.0)
        add("group_size", min(len(group), 32) / 32.0)
        positive_fraction = sum(row.current_reward > 0 for row in group.rollouts) / len(
            group
        )
        add("group_positive_fraction", positive_fraction)

        cache_key = (candidate.rollout_id, candidate.completion)
        lexical = self._lexical_cache.get(cache_key)
        if lexical is None:
            # Scan the full completion for unigram presence so a newly found
            # exploit near the end of a long reasoning trace remains learnable.
            # Bigrams use a bounded head+tail sample to keep scoring cheap.
            all_tokens = self._TOKEN_RE.findall(candidate.completion.lower())[:4096]
            lexical = {}

            def add_lexical(name: str, value: float = 1.0) -> None:
                bucket = self._bucket(name)
                lexical[bucket] = lexical.get(bucket, 0.0) + value

            for token in set(all_tokens):
                add_lexical(f"token={token}")
            if len(all_tokens) <= self.max_tokens:
                sampled_tokens = all_tokens
            else:
                head = self.max_tokens // 2
                sampled_tokens = (
                    all_tokens[:head] + all_tokens[-(self.max_tokens - head) :]
                )
            for left, right in zip(sampled_tokens, sampled_tokens[1:]):
                add_lexical(f"bigram={left}\u241f{right}")
            self._lexical_cache[cache_key] = lexical
        for bucket, value in lexical.items():
            features[bucket] = features.get(bucket, 0.0) + value
        return features

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float:
        score = sum(
            self._weights.get(index, 0.0) * value
            for index, value in self._features(candidate, group).items()
        )
        # Stable sigmoid.
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None:
        features = self._features(candidate, group)
        probability = self.predict(candidate, group)
        error = float(label_error) - probability
        for index, value in features.items():
            old = self._weights.get(index, 0.0)
            gradient = error * value - self.l2 * old
            self._weights[index] = old + self.learning_rate * gradient

    def clear_batch_cache(self) -> None:
        """Release completion features after the current policy batch.

        Feature hashing is deterministic, so this changes neither predictions
        nor learned weights.  It only prevents a long training run from
        retaining every historical completion in memory.
        """

        self._lexical_cache.clear()


class TokenEnrichmentRiskModel:
    """Online empirical-Bayes risk from audited unigram document counts.

    The learner has no trigger vocabulary.  Each purchased audit contributes
    one positive (label error) or negative document to unigram-presence counts.
    With audited error/correct document totals ``P, N`` and token-specific
    totals ``P_t, N_t``, the two smoothing stages are explicit::

        g = (P + global_prior_strength * prior) / (P + N + global_prior_strength)
        q_t = (P_t + token_prior_strength * g) / (P_t + N_t + token_prior_strength)

    A candidate's score is ``max(g, max_t q_t)`` over its present tokens.
    Thus one rare token shared by an audited error and an unaudited completion
    can transfer immediately, while repeated audited negatives shrink common
    tokens back toward the global rate.

    Only ``candidate.completion`` is inspected.  ``label_error`` arrives through
    :class:`AuditFeedback` after an oracle purchase; no oracle label, trigger
    flag, or truth-derived metadata is accepted by this interface.
    """

    # Reuse the exact lexical grammar used by HashedLogisticRiskModel.
    _TOKEN_RE = HashedLogisticRiskModel._TOKEN_RE

    def __init__(
        self,
        *,
        prior: float = 0.05,
        global_prior_strength: float = 20.0,
        token_prior_strength: float = 1.0,
        max_unigrams: int = 4096,
    ) -> None:
        if not math.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("prior must be finite and strictly between zero and one")
        for name, value in (
            ("global_prior_strength", global_prior_strength),
            ("token_prior_strength", token_prior_strength),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(max_unigrams, bool) or not isinstance(max_unigrams, int):
            raise TypeError("max_unigrams must be an integer")
        if max_unigrams < 1:
            raise ValueError("max_unigrams must be positive")
        self.prior = float(prior)
        self.global_prior_strength = float(global_prior_strength)
        self.token_prior_strength = float(token_prior_strength)
        self.max_unigrams = max_unigrams
        self.positive_documents = 0
        self.negative_documents = 0
        self._positive_token_documents: dict[str, int] = {}
        self._negative_token_documents: dict[str, int] = {}

    def _tokens(self, completion: str) -> frozenset[str]:
        return frozenset(
            self._TOKEN_RE.findall(str(completion).lower())[: self.max_unigrams]
        )

    @property
    def audited_documents(self) -> int:
        return self.positive_documents + self.negative_documents

    @property
    def global_error_rate(self) -> float:
        numerator = self.positive_documents + self.global_prior_strength * self.prior
        denominator = self.audited_documents + self.global_prior_strength
        return numerator / denominator

    def token_document_counts(self, token: str) -> tuple[int, int]:
        """Return audited positive/negative document counts for one unigram."""

        normalized = str(token).lower()
        return (
            self._positive_token_documents.get(normalized, 0),
            self._negative_token_documents.get(normalized, 0),
        )

    def _token_error_rate(self, token: str, global_rate: float) -> float:
        positive, negative = self.token_document_counts(token)
        numerator = positive + self.token_prior_strength * global_rate
        denominator = positive + negative + self.token_prior_strength
        return numerator / denominator

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float:
        del group
        global_rate = self.global_error_rate
        return max(
            (
                self._token_error_rate(token, global_rate)
                for token in self._tokens(candidate.completion)
            ),
            default=global_rate,
        )

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None:
        del group
        tokens = self._tokens(candidate.completion)
        if bool(label_error):
            self.positive_documents += 1
            counts = self._positive_token_documents
        else:
            self.negative_documents += 1
            counts = self._negative_token_documents
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1


class FalsePositiveOnlyRiskModel:
    """Restrict a binary-label error model to false-positive risk.

    The wrapped model estimates label error only within the cheap-positive
    stratum.  A rollout with ``cheap_reward == 0`` has zero probability of
    being a false positive by definition, so :meth:`predict` returns exactly
    zero without consulting the base model and :meth:`update` discards its
    feedback.  Cheap-positive predictions and updates pass through unchanged:
    an error is then precisely a false positive, while a non-error is a true
    positive training example.

    This is deliberately a generic wrapper rather than token-model logic.  It
    can condition any :class:`ErrorRiskModel` without teaching the underlying
    learner that false negatives and false positives are the same event.
    """

    def __init__(self, base_model: ErrorRiskModel) -> None:
        if base_model is None:
            raise TypeError("base_model must implement ErrorRiskModel")
        self.base_model = base_model

    @staticmethod
    def _cheap_is_positive(candidate: RolloutObservation) -> bool:
        reward = float(candidate.cheap_reward)
        if not math.isfinite(reward) or reward not in {0.0, 1.0}:
            raise ValueError(
                "FalsePositiveOnlyRiskModel requires finite binary cheap rewards"
            )
        return reward == 1.0

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float:
        if not self._cheap_is_positive(candidate):
            return 0.0
        return float(self.base_model.predict(candidate, group))

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None:
        if not self._cheap_is_positive(candidate):
            return
        self.base_model.update(candidate, group, label_error)

    def clear_batch_cache(self) -> None:
        """Forward optional cache cleanup used by selector batch hooks."""

        clear = getattr(self.base_model, "clear_batch_cache", None)
        if clear is not None:
            clear()


class RiskWeightedAdvantageSelector(QuerySelector):
    """Rollout-level ``q(x) * |A|`` audit baseline."""

    def __init__(self, risk_model: ErrorRiskModel | None = None) -> None:
        self.risk_model = risk_model or HashedLogisticRiskModel()

    def _score(
        self,
        row: RolloutObservation,
        group: GroupObservation,
    ) -> tuple[float, float]:
        risk = float(self.risk_model.predict(row, group))
        if not math.isfinite(risk):
            raise ValueError("risk model returned a non-finite value")
        risk = min(1.0, max(0.0, risk))
        return risk * abs(row.advantage), risk

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None
        scored = [
            (self._score(row, group), row, group) for row, group in _candidates(groups)
        ]
        if not scored:
            return None
        (priority, risk), row, _ = max(
            scored,
            key=lambda item: (item[0][0], -item[1].rollout_id),
        )
        return QueryRequest(
            (row.rollout_id,),
            reason="risk_times_abs_advantage",
            priority=priority,
            predicted_risks={row.rollout_id: risk},
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


class WholeGroupRiskSelector(RiskWeightedAdvantageSelector):
    """Group-aware extension that buys every remaining label in one group.

    The group priority is the sum of rollout ``q|A|`` scores.  An untouched
    four-rollout GRPO group therefore costs four oracle calls, not one.
    """

    def propose(
        self,
        groups: Sequence[GroupObservation],
        *,
        max_labels: int,
    ) -> QueryRequest | None:
        if max_labels < 1:
            return None
        affordable: list[tuple[float, int, GroupObservation, dict[int, float]]] = []
        for group in groups:
            if group.quarantined:
                continue
            rows = group.unqueried
            if not rows or len(rows) > max_labels:
                continue
            priorities: list[float] = []
            risks: dict[int, float] = {}
            for row in rows:
                priority, risk = self._score(row, group)
                priorities.append(priority)
                risks[row.rollout_id] = risk
            affordable.append((sum(priorities), -group.group_id, group, risks))
        if not affordable:
            return None
        priority, _, group, risks = max(affordable, key=lambda item: (item[0], item[1]))
        ids = tuple(row.rollout_id for row in group.unqueried)
        return QueryRequest(
            ids,
            reason="whole_group_risk_times_abs_advantage",
            priority=priority,
            predicted_risks=risks,
            atomic=True,
        )
