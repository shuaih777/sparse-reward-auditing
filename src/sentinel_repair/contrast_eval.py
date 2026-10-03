"""Prequential failure-family contrast gate from purchased FP labels only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .data import (
    GroupObservation,
    RolloutGroup,
    RolloutObservation,
    iter_contiguous_step_batches,
)
from .propagation_eval import (
    GateActivation,
    GateConfig,
    GatePlan,
    PurchasedLabel,
    _view,
    score_gate_plan,
)
from .propagation import (
    ContrastDecision,
    ContrastGateConfig,
    FailureFamilyContrastGate,
)
from .selectors import FalsePositiveOnlyRiskModel, TokenEnrichmentRiskModel


@dataclass(frozen=True)
class ContrastGatePlan:
    configs: tuple[ContrastGateConfig, ...]
    gated_ids: Mapping[ContrastGateConfig, Mapping[int, frozenset[int]]]
    first_activation: Mapping[ContrastGateConfig, GateActivation | None]
    purchases_by_batch: Mapping[int, tuple[PurchasedLabel, ...]]


def default_contrast_grid() -> tuple[ContrastGateConfig, ...]:
    """The bounded 3 x 2 x 2 robustness grid requested for this contrast."""

    return tuple(
        ContrastGateConfig(support, coverage, ratio)
        for support in (2, 3, 4)
        for coverage in (0.6, 0.8)
        for ratio in (5.0, 10.0)
    )


def derive_contrast_gate_plan(
    groups: Sequence[RolloutGroup],
    purchases: Sequence[PurchasedLabel],
    configs: Iterable[ContrastGateConfig] | None = None,
) -> ContrastGatePlan:
    """Freeze contrast-gate actions before any unpurchased truth is supplied."""

    grid = tuple(configs if configs is not None else default_contrast_grid())
    if not grid or len(grid) != len(set(grid)):
        raise ValueError("contrast configs must be non-empty and unique")
    batches = tuple(iter_contiguous_step_batches(groups))
    by_batch: dict[int, list[PurchasedLabel]] = {
        index: [] for index in range(len(batches))
    }
    for purchase in purchases:
        if purchase.batch_index not in by_batch:
            raise ValueError(f"unknown purchase batch {purchase.batch_index}")
        if purchase.label_error != (purchase.cheap_reward != purchase.observed_label):
            raise ValueError("purchased mismatch flag is inconsistent")
        by_batch[purchase.batch_index].append(purchase)

    model = TokenEnrichmentRiskModel()
    fp_model = FalsePositiveOnlyRiskModel(model)
    prototype = FailureFamilyContrastGate(model, grid[0])
    gated: dict[ContrastGateConfig, dict[int, frozenset[int]]] = {
        config: {} for config in grid
    }
    first: dict[ContrastGateConfig, GateActivation | None] = {
        config: None for config in grid
    }

    def decisions_by_config(
        public: Mapping[int, tuple[RolloutObservation, GroupObservation]],
        audited: set[int],
    ) -> dict[ContrastGateConfig, list[ContrastDecision]]:
        result = {config: [] for config in grid}
        for candidate, candidate_group in public.values():
            if candidate.rollout_id in audited:
                continue
            evidence = prototype.evidence(candidate, candidate_group)
            for config in grid:
                decision = prototype.select(evidence, config)
                if decision is not None:
                    result[config].append(decision)
        return result

    for batch_index, batch in enumerate(batches):
        views = tuple(_view(group) for group in batch)
        public = {
            row.rollout_id: (row, group) for group in views for row in group.rollouts
        }
        audited: set[int] = set()
        for purchase in sorted(by_batch[batch_index], key=lambda item: item.call_index):
            if purchase.rollout_id not in public:
                raise ValueError("purchase is not in its public batch")
            row, group = public[purchase.rollout_id]
            if (
                purchase.step != str(row.step)
                or purchase.cheap_reward != row.cheap_reward
            ):
                raise ValueError("purchase disagrees with public rollout")
            fp_model.update(row, group, purchase.label_error)
            audited.add(row.rollout_id)
            current = decisions_by_config(public, audited)
            for config, decisions in current.items():
                if first[config] is not None:
                    continue
                if decisions:
                    decision = min(
                        decisions,
                        key=lambda item: (
                            -item.prevalence_ratio,
                            -item.error_coverage,
                            -item.positive_documents,
                            item.token,
                            item.rollout_id,
                        ),
                    )
                    first[config] = GateActivation(
                        batch_index,
                        purchase.step,
                        purchase.call_index,
                        decision.rollout_id,
                        decision.prevalence_ratio,
                        decision.token,
                    )
        final = decisions_by_config(public, audited)
        for config, decisions in final.items():
            gated[config][batch_index] = frozenset(
                decision.rollout_id for decision in decisions
            )

    return ContrastGatePlan(
        grid,
        {config: dict(items) for config, items in gated.items()},
        dict(first),
        {
            batch: tuple(sorted(items, key=lambda item: item.call_index))
            for batch, items in by_batch.items()
        },
    )


def score_contrast_gate_plan(
    rows: pd.DataFrame,
    plan: ContrastGatePlan,
    *,
    trace: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score frozen contrast actions through the common FP residual evaluator."""

    run_parts: list[pd.DataFrame] = []
    step_parts: list[pd.DataFrame] = []
    for config in plan.configs:
        # One config at a time lets the already-tested scorer remain agnostic
        # to the contrast gate's additional coverage and ratio dimensions.
        placeholder = GateConfig(config.min_error_support, config.min_error_coverage)
        adapted = GatePlan(
            (placeholder,),
            {placeholder: plan.gated_ids[config]},
            {placeholder: plan.first_activation[config]},
            plan.purchases_by_batch,
        )
        runs, steps = score_gate_plan(rows, adapted, trace=trace, seed=seed)
        for frame in (runs, steps):
            frame.rename(columns={"risk_threshold": "min_error_coverage"}, inplace=True)
            frame.insert(
                4,
                "min_prevalence_ratio",
                config.min_prevalence_ratio,
            )
        run_parts.append(runs)
        step_parts.append(steps)
    return (
        pd.concat(run_parts, ignore_index=True),
        pd.concat(step_parts, ignore_index=True),
    )


__all__ = (
    "ContrastDecision",
    "ContrastGateConfig",
    "ContrastGatePlan",
    "FailureFamilyContrastGate",
    "default_contrast_grid",
    "derive_contrast_gate_plan",
    "score_contrast_gate_plan",
)
