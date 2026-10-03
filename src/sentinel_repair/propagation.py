"""Conservative propagation decisions from purchased token evidence.

This module is intentionally separate from the live audit engine.  It turns a
shared :class:`~sentinel_repair.selectors.TokenEnrichmentRiskModel` posterior
into an auditable hard-gate decision, but does not apply that decision to a
reward or a policy update.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .data import GroupObservation, RolloutObservation
from .selectors import TokenEnrichmentRiskModel


@dataclass(frozen=True, order=True)
class ContrastGateConfig:
    """Thresholds for a purchased-label failure-family contrast."""

    min_error_support: int
    min_error_coverage: float
    min_prevalence_ratio: float

    def __post_init__(self) -> None:
        # Preserve the validated evaluation API exactly while moving the
        # production gate away from its pandas-dependent adapter.
        if (
            isinstance(self.min_error_support, bool)
            or not isinstance(self.min_error_support, int)
            or self.min_error_support < 1
        ):
            raise ValueError("min_error_support must be a positive integer")
        if not math.isfinite(self.min_error_coverage) or not (
            0.0 < self.min_error_coverage <= 1.0
        ):
            raise ValueError("min_error_coverage must be in (0, 1]")
        if (
            not math.isfinite(self.min_prevalence_ratio)
            or self.min_prevalence_ratio <= 0
        ):
            raise ValueError("min_prevalence_ratio must be positive")


@dataclass(frozen=True)
class ContrastDecision:
    """Public purchased-evidence explanation for one gate action."""

    rollout_id: int
    token: str
    prevalence_ratio: float
    error_coverage: float
    positive_documents: int
    negative_documents: int


class FailureFamilyContrastGate:
    """Require support, error-family coverage, and error/correct enrichment.

    ``positive_documents`` and ``negative_documents`` are expected to come
    from a token model updated through ``FalsePositiveOnlyRiskModel``: they
    then mean erroneous and correct purchased cheap-positive documents.  The
    gate itself receives no oracle or trigger interface and reads no metadata.
    """

    def __init__(self, model: TokenEnrichmentRiskModel, config: ContrastGateConfig):
        if not isinstance(model, TokenEnrichmentRiskModel):
            raise TypeError("model must be TokenEnrichmentRiskModel")
        self.model = model
        self.config = config

    def evidence(
        self, candidate: RolloutObservation, group: GroupObservation
    ) -> tuple[ContrastDecision, ...]:
        matches = [
            row for row in group.rollouts if row.rollout_id == candidate.rollout_id
        ]
        if len(matches) != 1 or matches[0] != candidate:
            raise ValueError("candidate must be the matching public group row")
        if group.quarantined or candidate.audited:
            return ()
        if float(candidate.cheap_reward) not in {0.0, 1.0}:
            raise ValueError("contrast gate requires binary cheap rewards")
        if float(candidate.cheap_reward) == 0.0:
            return ()

        positive_total = self.model.positive_documents
        negative_total = self.model.negative_documents
        if positive_total == 0:
            return ()
        decisions: list[ContrastDecision] = []
        for token in self.model._tokens(candidate.completion):
            positive, negative = self.model.token_document_counts(token)
            coverage = positive / positive_total
            error_prevalence = (positive + 0.5) / (positive_total + 1.0)
            correct_prevalence = (negative + 0.5) / (negative_total + 1.0)
            ratio = error_prevalence / correct_prevalence
            decisions.append(
                ContrastDecision(
                    candidate.rollout_id,
                    token,
                    ratio,
                    coverage,
                    positive,
                    negative,
                )
            )
        return tuple(decisions)

    @staticmethod
    def select(
        evidence: Sequence[ContrastDecision], config: ContrastGateConfig
    ) -> ContrastDecision | None:
        eligible = [
            item
            for item in evidence
            if item.positive_documents >= config.min_error_support
            and item.error_coverage >= config.min_error_coverage
            and item.prevalence_ratio >= config.min_prevalence_ratio
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda item: (
                -item.prevalence_ratio,
                -item.error_coverage,
                -item.positive_documents,
                item.token,
            ),
        )

    def decide(
        self, candidate: RolloutObservation, group: GroupObservation
    ) -> ContrastDecision | None:
        return self.select(self.evidence(candidate, group), self.config)


@dataclass(frozen=True, slots=True)
class TokenPosteriorGateDecision:
    """Evidence behind one token-posterior hard-gate decision."""

    rollout_id: int
    score: float
    token: str
    positive_documents: int
    negative_documents: int


class TokenPosteriorHardGate:
    """Gate cheap positives only after strong, repeated purchased evidence.

    A candidate is eligible exactly when it is unaudited, has cheap reward
    one, and contains a token with both:

    * at least ``minimum_error_support`` purchased error documents; and
    * an empirical-Bayes error score at least ``minimum_score``.

    Tokenization and scores call the shared model's own implementations.  The
    gate therefore cannot drift from :class:`TokenEnrichmentRiskModel`'s
    lexical grammar or posterior formula.  No oracle label, trigger flag, or
    metadata is accepted or inspected.  If several tokens attain the same
    maximum eligible score, the lexicographically smallest token wins.
    """

    def __init__(
        self,
        model: TokenEnrichmentRiskModel,
        *,
        minimum_error_support: int = 3,
        minimum_score: float = 0.90,
    ) -> None:
        if not isinstance(model, TokenEnrichmentRiskModel):
            raise TypeError("model must be a TokenEnrichmentRiskModel")
        if isinstance(minimum_error_support, bool) or not isinstance(
            minimum_error_support, int
        ):
            raise TypeError("minimum_error_support must be an integer")
        if minimum_error_support < 1:
            raise ValueError("minimum_error_support must be positive")
        if isinstance(minimum_score, bool):
            raise TypeError("minimum_score must be numeric, not bool")
        try:
            parsed_score = float(minimum_score)
        except (TypeError, ValueError) as exc:
            raise TypeError("minimum_score must be numeric") from exc
        if not math.isfinite(parsed_score) or not 0.0 < parsed_score <= 1.0:
            raise ValueError("minimum_score must be finite and in (0, 1]")

        self.model = model
        self.minimum_error_support = minimum_error_support
        self.minimum_score = parsed_score

    @staticmethod
    def _canonical_candidate(
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> RolloutObservation:
        if not isinstance(candidate, RolloutObservation):
            raise TypeError("candidate must be a RolloutObservation")
        if not isinstance(group, GroupObservation):
            raise TypeError("group must be a GroupObservation")
        matches = [
            row for row in group.rollouts if row.rollout_id == candidate.rollout_id
        ]
        if len(matches) != 1 or matches[0] != candidate:
            raise ValueError("candidate must be the matching public row in group")
        return matches[0]

    def decide(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> TokenPosteriorGateDecision | None:
        """Return the strongest qualifying token decision, or ``None``."""

        row = self._canonical_candidate(candidate, group)
        if group.quarantined or row.audited:
            return None
        if isinstance(row.cheap_reward, bool):
            raise TypeError("cheap reward must be numeric, not bool")
        cheap_reward = float(row.cheap_reward)
        if not math.isfinite(cheap_reward) or cheap_reward not in {0.0, 1.0}:
            raise ValueError("TokenPosteriorHardGate requires binary cheap rewards")
        if cheap_reward == 0.0:
            return None

        global_rate = self.model.global_error_rate
        eligible: list[TokenPosteriorGateDecision] = []
        # These are deliberately calls to the shared model.  Reimplementing
        # either operation here could make propagation use a subtly different
        # grammar or smoothing rule than audit selection.
        for token in self.model._tokens(row.completion):
            positive, negative = self.model.token_document_counts(token)
            if positive < self.minimum_error_support:
                continue
            score = self.model._token_error_rate(token, global_rate)
            if score < self.minimum_score:
                continue
            eligible.append(
                TokenPosteriorGateDecision(
                    rollout_id=row.rollout_id,
                    score=score,
                    token=token,
                    positive_documents=positive,
                    negative_documents=negative,
                )
            )

        if not eligible:
            return None
        return min(eligible, key=lambda decision: (-decision.score, decision.token))


__all__ = (
    "ContrastDecision",
    "ContrastGateConfig",
    "FailureFamilyContrastGate",
    "TokenPosteriorGateDecision",
    "TokenPosteriorHardGate",
)
