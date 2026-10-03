"""Leak-free, prequential evaluation of token-posterior reward gates.

Gate decisions use public rollout groups and already purchased labels only.
Unpurchased oracle truth enters later, in :func:`score_gate_plan`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .advantages import grpo_advantages
from .data import (
    GroupObservation,
    RolloutGroup,
    RolloutObservation,
    iter_contiguous_step_batches,
)
from .harm import reward_coefficient_residual
from .offline import AuditEvent, OfflineRunResult
from .propagation import TokenPosteriorHardGate
from .selectors import FalsePositiveOnlyRiskModel, TokenEnrichmentRiskModel


@dataclass(frozen=True, order=True)
class GateConfig:
    min_error_support: int
    risk_threshold: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_error_support, bool)
            or not isinstance(self.min_error_support, int)
            or self.min_error_support < 1
        ):
            raise ValueError("min_error_support must be a positive integer")
        if not math.isfinite(self.risk_threshold) or not (
            0.0 < self.risk_threshold <= 1.0
        ):
            raise ValueError("risk_threshold must be finite and in (0, 1]")


@dataclass(frozen=True)
class PurchasedLabel:
    batch_index: int
    step: str
    call_index: int
    rollout_id: int
    cheap_reward: float
    observed_label: float
    label_error: bool


@dataclass(frozen=True)
class GateActivation:
    batch_index: int
    step: str
    call_index: int
    rollout_id: int
    risk: float
    token: str


@dataclass(frozen=True)
class GatePlan:
    configs: tuple[GateConfig, ...]
    gated_ids: Mapping[GateConfig, Mapping[int, frozenset[int]]]
    first_activation: Mapping[GateConfig, GateActivation | None]
    purchases_by_batch: Mapping[int, tuple[PurchasedLabel, ...]]


def default_gate_grid() -> tuple[GateConfig, ...]:
    return tuple(
        GateConfig(support, threshold)
        for support in (1, 2, 3, 4, 5)
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    )


def purchases_from_events(events: Sequence[AuditEvent]) -> tuple[PurchasedLabel, ...]:
    """Discard every audit-event field except legitimately bought feedback."""

    purchases: list[PurchasedLabel] = []
    seen: set[int] = set()
    previous_call = 0
    previous_batch = -1
    for event in events:
        if event.call_index != previous_call + 1:
            raise ValueError("audit call indices must be consecutive from one")
        if event.batch_index < previous_batch:
            raise ValueError("audit batches are not chronological")
        previous_call = event.call_index
        previous_batch = event.batch_index
        if event.rollout_id in seen:
            raise ValueError(f"rollout {event.rollout_id} was purchased twice")
        seen.add(event.rollout_id)
        cheap, observed = float(event.cheap_reward), float(event.observed_label)
        if cheap not in {0.0, 1.0} or observed not in {0.0, 1.0}:
            raise ValueError("propagation evaluation requires binary labels")
        mismatch = cheap != observed
        if mismatch != bool(event.label_error):
            raise ValueError("label_error disagrees with the purchased label")
        purchases.append(
            PurchasedLabel(
                event.batch_index,
                str(event.step),
                event.call_index,
                event.rollout_id,
                cheap,
                observed,
                mismatch,
            )
        )
    return tuple(purchases)


def validate_exact_prefix_budget(
    result: OfflineRunResult, *, expected_rollouts_per_step: int
) -> None:
    """Require exact use of each strict cumulative 1% prefix allowance."""

    if (result.budget_rate.numerator, result.budget_rate.denominator) != (1, 100):
        raise ValueError(f"expected exact 1% budget, got {result.budget_rate}")
    for index, batch in enumerate(result.batches):
        expected_seen = (index + 1) * expected_rollouts_per_step
        expected_allowed = expected_seen // 100
        if batch.batch_index != index or batch.seen_labels != expected_seen:
            raise ValueError(f"batch {index} has an invalid prefix topology")
        if batch.allowed_calls != expected_allowed:
            raise ValueError(f"batch {index} has a non-1% allowance")
        if batch.spent_calls != expected_allowed:
            raise ValueError(
                f"batch {index} did not fill its prefix allowance: "
                f"{batch.spent_calls}/{expected_allowed}"
            )
    if result.oracle_calls != len(result.events):
        raise ValueError("oracle calls and audit events disagree")


def _view(group: RolloutGroup) -> GroupObservation:
    advantages = grpo_advantages(group.cheap_rewards)
    return GroupObservation(
        group_id=group.group_id,
        step=group.step,
        prompt=group.prompt,
        rollouts=tuple(
            RolloutObservation(
                rollout_id=row.rollout_id,
                group_id=group.group_id,
                position=position,
                step=row.step,
                prompt=row.prompt,
                completion=row.completion,
                cheap_reward=row.cheap_reward,
                current_reward=row.cheap_reward,
                advantage=float(advantages[position]),
                audited=False,
                metadata=row.metadata,
            )
            for position, row in enumerate(group.rollouts)
        ),
    )


def derive_gate_plan(
    groups: Sequence[RolloutGroup],
    purchases: Sequence[PurchasedLabel],
    configs: Iterable[GateConfig] | None = None,
) -> GatePlan:
    """Derive pre-update gates using no unpurchased labels or truth metadata."""

    grid = tuple(configs if configs is not None else default_gate_grid())
    if not grid or len(grid) != len(set(grid)):
        raise ValueError("gate configs must be non-empty and unique")
    batches = tuple(iter_contiguous_step_batches(groups))
    by_batch: dict[int, list[PurchasedLabel]] = {i: [] for i in range(len(batches))}
    for purchase in purchases:
        if purchase.batch_index not in by_batch:
            raise ValueError(f"unknown purchase batch {purchase.batch_index}")
        if purchase.cheap_reward not in {0.0, 1.0} or purchase.observed_label not in {
            0.0,
            1.0,
        }:
            raise ValueError("purchased labels must be binary")
        if purchase.label_error != (purchase.cheap_reward != purchase.observed_label):
            raise ValueError("purchased mismatch flag is inconsistent")
        by_batch[purchase.batch_index].append(purchase)

    model = TokenEnrichmentRiskModel()
    fp_model = FalsePositiveOnlyRiskModel(model)
    supports = sorted({config.min_error_support for config in grid})
    minimum_threshold = {
        support: min(
            config.risk_threshold
            for config in grid
            if config.min_error_support == support
        )
        for support in supports
    }
    gates = {
        support: TokenPosteriorHardGate(
            model,
            minimum_error_support=support,
            minimum_score=minimum_threshold[support],
        )
        for support in supports
    }
    gated: dict[GateConfig, dict[int, frozenset[int]]] = {item: {} for item in grid}
    first: dict[GateConfig, GateActivation | None] = {item: None for item in grid}

    for batch_index, batch in enumerate(batches):
        views = tuple(_view(group) for group in batch)
        public = {
            row.rollout_id: (row, group) for group in views for row in group.rollouts
        }
        audited: set[int] = set()
        for purchase in sorted(by_batch[batch_index], key=lambda item: item.call_index):
            if purchase.rollout_id not in public:
                raise ValueError("purchase is not in its declared public batch")
            row, group = public[purchase.rollout_id]
            if (
                str(row.step) != purchase.step
                or row.cheap_reward != purchase.cheap_reward
            ):
                raise ValueError("purchase disagrees with public rollout")
            if row.rollout_id in audited:
                raise ValueError("duplicate purchase in one batch")
            fp_model.update(row, group, purchase.label_error)
            audited.add(row.rollout_id)

            # First activation is evaluated immediately after this label, over
            # only rollouts that have not yet been purchased at this instant.
            for support, gate in gates.items():
                decisions = [
                    decision
                    for candidate, candidate_group in public.values()
                    if candidate.rollout_id not in audited
                    for decision in (gate.decide(candidate, candidate_group),)
                    if decision is not None
                ]
                if not decisions:
                    continue
                decision = min(
                    decisions,
                    key=lambda item: (-item.score, item.rollout_id, item.token),
                )
                for config in grid:
                    if (
                        config.min_error_support == support
                        and first[config] is None
                        and decision.score >= config.risk_threshold
                    ):
                        first[config] = GateActivation(
                            batch_index,
                            purchase.step,
                            purchase.call_index,
                            decision.rollout_id,
                            decision.score,
                            decision.token,
                        )

        # This final post-purchase state is what exists before the batch update.
        scores_by_support: dict[int, dict[int, float]] = {}
        for support, gate in gates.items():
            decisions = (
                (candidate.rollout_id, gate.decide(candidate, candidate_group))
                for candidate, candidate_group in public.values()
                if candidate.rollout_id not in audited
            )
            scores_by_support[support] = {
                rollout_id: decision.score
                for rollout_id, decision in decisions
                if decision is not None
            }
        for config in grid:
            gated[config][batch_index] = frozenset(
                rollout_id
                for rollout_id, score in scores_by_support[
                    config.min_error_support
                ].items()
                if score >= config.risk_threshold
            )

    return GatePlan(
        grid,
        {config: dict(value) for config, value in gated.items()},
        dict(first),
        {
            batch: tuple(sorted(value, key=lambda item: item.call_index))
            for batch, value in by_batch.items()
        },
    )


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def score_gate_plan(
    rows: pd.DataFrame, plan: GatePlan, *, trace: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use full truth only after all audit and gate actions are frozen."""

    required = {
        "_step_index",
        "_group_id",
        "_rollout_id",
        "step",
        "reward",
        "oracle_reward",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"gate scoring is missing columns: {sorted(missing)}")
    cheap_all = rows["reward"].to_numpy(dtype=float)
    oracle_all = rows["oracle_reward"].to_numpy(dtype=float)
    if (
        not ((cheap_all == 0) | (cheap_all == 1)).all()
        or not ((oracle_all == 0) | (oracle_all == 1)).all()
    ):
        raise ValueError("scoring requires binary rewards")
    if ((cheap_all == 0) & (oracle_all == 1)).any():
        raise ValueError("FP residual evaluation refuses traces with false negatives")

    records: list[dict[str, object]] = []
    for batch_index, step_rows in rows.groupby("_step_index", sort=True):
        batch_index = int(batch_index)
        ids = step_rows["_rollout_id"].to_numpy(dtype=int).tolist()
        cheap = dict(zip(ids, step_rows["reward"].to_numpy(dtype=float).tolist()))
        oracle = dict(
            zip(ids, step_rows["oracle_reward"].to_numpy(dtype=float).tolist())
        )
        purchases = plan.purchases_by_batch.get(batch_index, ())
        bought = {item.rollout_id: item.observed_label for item in purchases}
        audited = set(bought)
        if not audited.issubset(cheap):
            raise ValueError("purchase is not in its scoring batch")

        for config in plan.configs:
            gated = set(plan.gated_ids[config].get(batch_index, frozenset()))
            if gated & audited or not gated.issubset(cheap):
                raise ValueError("gate/purchase membership is inconsistent")
            if any(cheap[item] != 1.0 for item in gated):
                raise ValueError("a gate targets a cheap-negative rollout")
            residuals = [0.0, 0.0, 0.0]
            for _, group in step_rows.groupby("_group_id", sort=False):
                group_ids = group["_rollout_id"].astype(int).tolist()
                cheap_rewards = [cheap[item] for item in group_ids]
                oracle_rewards = [oracle[item] for item in group_ids]
                audit_rewards = [bought.get(item, cheap[item]) for item in group_ids]
                gate_rewards = [
                    0.0 if item in gated else value
                    for item, value in zip(group_ids, audit_rewards)
                ]
                for index, values in enumerate(
                    (cheap_rewards, audit_rewards, gate_rewards)
                ):
                    residuals[index] += reward_coefficient_residual(
                        values, oracle_rewards
                    )

            fp = {item for item in ids if cheap[item] == 1 and oracle[item] == 0}
            legitimate = {
                item for item in ids if cheap[item] == 1 and oracle[item] == 1
            }
            unaudited_fp = fp - audited
            gated_fp, gated_legit = gated & fp, gated & legitimate
            purchased_fp = fp & audited
            activation = plan.first_activation[config]
            records.append(
                {
                    "trace": trace,
                    "seed": seed,
                    "min_error_support": config.min_error_support,
                    "risk_threshold": config.risk_threshold,
                    "batch_index": batch_index,
                    "step": str(step_rows["step"].iloc[0]),
                    "audit_calls": len(audited),
                    "purchased_false_positives": len(purchased_fp),
                    "gate_count": len(gated),
                    "gated_false_positives": len(gated_fp),
                    "gated_legitimate_positives": len(gated_legit),
                    "unaudited_false_positives": len(unaudited_fp),
                    "legitimate_positive_count": len(legitimate),
                    "gated_precision": _ratio(len(gated_fp), len(gated)),
                    "gated_recall": _ratio(len(gated_fp), len(unaudited_fp)),
                    "false_positive_suppression_rate": _ratio(
                        len(purchased_fp) + len(gated_fp), len(fp)
                    ),
                    "legitimate_positive_suppression_rate": _ratio(
                        len(gated_legit), len(legitimate)
                    ),
                    "cheap_oracle_fp_coefficient_residual": residuals[0],
                    "audit_only_oracle_fp_coefficient_residual": residuals[1],
                    "gated_oracle_fp_coefficient_residual": residuals[2],
                    "oracle_fp_coefficient_residual_reduction": (
                        residuals[0] - residuals[2]
                    ),
                    "propagation_incremental_residual_reduction": (
                        residuals[1] - residuals[2]
                    ),
                    "gate_active_this_step": bool(gated),
                    "first_eligibility_is_this_step": bool(
                        activation is not None and activation.batch_index == batch_index
                    ),
                }
            )

    steps = pd.DataFrame.from_records(records)
    runs: list[dict[str, object]] = []
    for key, part in steps.groupby(["min_error_support", "risk_threshold"], sort=True):
        config = GateConfig(int(key[0]), float(key[1]))
        activation = plan.first_activation[config]
        cheap_residual = float(part["cheap_oracle_fp_coefficient_residual"].sum())
        audit_residual = float(part["audit_only_oracle_fp_coefficient_residual"].sum())
        gate_residual = float(part["gated_oracle_fp_coefficient_residual"].sum())
        gate_count = int(part["gate_count"].sum())
        gated_fp = int(part["gated_false_positives"].sum())
        gated_legit = int(part["gated_legitimate_positives"].sum())
        unaudited_fp = int(part["unaudited_false_positives"].sum())
        purchased_fp = int(part["purchased_false_positives"].sum())
        legitimate = int(part["legitimate_positive_count"].sum())
        applied = part.loc[part["gate_count"] > 0].sort_values("batch_index")
        first_applied = applied.iloc[0] if len(applied) else None
        runs.append(
            {
                "trace": trace,
                "seed": seed,
                "min_error_support": config.min_error_support,
                "risk_threshold": config.risk_threshold,
                "audit_calls": int(part["audit_calls"].sum()),
                "purchased_false_positives": purchased_fp,
                "gate_count": gate_count,
                "gated_false_positives": gated_fp,
                "gated_legitimate_positives": gated_legit,
                "gated_precision": _ratio(gated_fp, gate_count),
                "gated_recall": _ratio(gated_fp, unaudited_fp),
                "false_positive_suppression_rate": _ratio(
                    purchased_fp + gated_fp, purchased_fp + unaudited_fp
                ),
                "legitimate_positive_suppression_rate": _ratio(gated_legit, legitimate),
                "legitimate_positive_retention_rate": (
                    1.0 - _ratio(gated_legit, legitimate) if legitimate else math.nan
                ),
                "cheap_oracle_fp_coefficient_residual": cheap_residual,
                "audit_only_oracle_fp_coefficient_residual": audit_residual,
                "gated_oracle_fp_coefficient_residual": gate_residual,
                "audit_only_residual_reduction": cheap_residual - audit_residual,
                "propagation_incremental_residual_reduction": (
                    audit_residual - gate_residual
                ),
                "oracle_fp_coefficient_residual_reduction": (
                    cheap_residual - gate_residual
                ),
                "oracle_fp_coefficient_residual_reduction_fraction": _ratio(
                    cheap_residual - gate_residual, cheap_residual
                ),
                "activated": first_applied is not None,
                "first_activation_batch_index": (
                    int(first_applied["batch_index"])
                    if first_applied is not None
                    else math.nan
                ),
                "first_activation_step": (
                    str(first_applied["step"]) if first_applied is not None else None
                ),
                "ever_postpurchase_eligible": activation is not None,
                "first_eligibility_batch_index": (
                    activation.batch_index if activation else math.nan
                ),
                "first_eligibility_step": activation.step if activation else None,
                "first_eligibility_call_index": (
                    activation.call_index if activation else math.nan
                ),
                "first_eligibility_score": (
                    activation.risk if activation else math.nan
                ),
                "first_eligibility_token": activation.token if activation else None,
            }
        )
    return pd.DataFrame.from_records(runs), steps


__all__ = (
    "GateActivation",
    "GateConfig",
    "GatePlan",
    "PurchasedLabel",
    "default_gate_grid",
    "derive_gate_plan",
    "purchases_from_events",
    "score_gate_plan",
    "validate_exact_prefix_budget",
)
