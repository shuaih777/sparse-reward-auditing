"""Deterministic CPU sweeps for sentinel/repair budget allocation.

This module is deliberately separate from the main mechanism analyzer.  It
uses :func:`sentinel_repair.analysis.parsed_from_frame` to construct the same
oracle-safe public trajectory and :class:`sentinel_repair.offline.OfflineRunner`
to enforce the same strict cumulative prefix budget.  Oracle truth is used
only by the runner/evaluator and by post-run metrics; selector factories never
receive the source frame, takeover thresholds, or hidden labels.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Literal, Sequence

import pandas as pd

from .analysis import (
    add_group_diagnostics,
    find_takeover_window,
    load_trace,
    parsed_from_frame,
    summarize_steps,
    validate_trace_contract,
)
from .controller import (
    AuditArm,
    BootstrapSentinelRepairController,
    DynamicSentinelRepairController,
    FixedSentinelRepairController,
)
from .data import GroupObservation, ParsedLog, RolloutObservation
from .group_selectors import ExpectedMarginalGroupSelector
from .offline import OfflineRunResult, run_parsed_log
from .selectors import (
    AuditFeedback,
    HashedLogisticRiskModel,
    RandomSelector,
    TokenEnrichmentRiskModel,
)


DEFAULT_FIXED_RHOS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_PH_THRESHOLDS = (1.0, 2.0, 4.0)
DEFAULT_BOOTSTRAP_MISMATCHES = (1, 2)
DEFAULT_SEEDS = tuple(range(10))

PolicyKind = Literal["fixed", "dynamic", "bootstrap", "random", "unwrapped_repair"]
RepairModel = Literal["learned", "token-enrichment", "known-token-diagnostic", "none"]


@dataclass(frozen=True)
class SweepJob:
    """One fully specified selector run.

    ``pair_seed`` is shared across all stochastic treatments in one paired
    replicate.  The unwrapped deterministic repair baseline has no seed.
    """

    job_index: int
    run_id: str
    policy: PolicyKind
    repair_model: RepairModel
    pair_seed: int | None
    configured_rho: float | None = None
    repair_rho: float | None = None
    discover_rho: float | None = None
    ph_threshold: float | None = None
    bootstrap_mismatches: int | None = None
    known_token: str | None = None


@dataclass(frozen=True)
class TakeoverReference:
    """Truth-derived thresholds used only after a selector run finishes."""

    step_labels: tuple[str, ...]
    tau10: int | None
    tau50: int | None
    tau80: int | None

    def label_at(self, index: int | None) -> str | None:
        if index is None:
            return None
        if index < 0 or index >= len(self.step_labels):
            raise ValueError(f"takeover index {index} is outside the trace")
        return self.step_labels[index]


@dataclass(frozen=True)
class SweepFrames:
    """Tabular outputs of a completed sweep."""

    runs: pd.DataFrame
    summary: pd.DataFrame
    transitions: pd.DataFrame


class KnownTokenRiskModel:
    """Public-feature diagnostic with advance knowledge of one token.

    This is intentionally labelled non-deployable for an unknown trigger.  It
    never reads logger trigger flags or oracle labels: it checks only the
    public completion text and cheap reward. Matching is a case-insensitive
    literal substring diagnostic, not a claim to reproduce the upstream
    tokenizer selector. Its purpose is to separate a weak online risk learner
    from a weak sentinel/repair allocation rule.
    """

    def __init__(
        self,
        token: str = "python",
        *,
        hit_probability: float = 0.95,
        miss_probability: float = 0.001,
    ) -> None:
        token = str(token).strip()
        if not token:
            raise ValueError("known token cannot be empty")
        for name, value in (
            ("hit_probability", hit_probability),
            ("miss_probability", miss_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        self.token = token.casefold()
        self.hit_probability = float(hit_probability)
        self.miss_probability = float(miss_probability)

    def predict(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
    ) -> float:
        del group
        hit = (
            float(candidate.cheap_reward) == 1.0
            and self.token in candidate.completion.casefold()
        )
        return self.hit_probability if hit else self.miss_probability

    def update(
        self,
        candidate: RolloutObservation,
        group: GroupObservation,
        label_error: bool,
    ) -> None:
        del candidate, group, label_error


def _validate_unique_numbers(
    values: Iterable[float],
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    for value in parsed:
        lower_ok = value >= lower if lower_inclusive else value > lower
        if not math.isfinite(value) or not lower_ok or value > upper:
            bracket = "[" if lower_inclusive else "("
            raise ValueError(f"{name} values must lie in {bracket}{lower}, {upper}]")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} cannot contain duplicates")
    return parsed


def _float_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _validate_bootstrap_mismatches(values: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(values)
    if not parsed:
        raise ValueError("bootstrap_mismatches cannot be empty")
    for value in parsed:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("bootstrap_mismatches values must be integers")
        if value < 1:
            raise ValueError("bootstrap_mismatches values must be positive")
    if len(parsed) != len(set(parsed)):
        raise ValueError("bootstrap_mismatches cannot contain duplicates")
    return parsed


def build_jobs(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    fixed_rhos: Sequence[float] = DEFAULT_FIXED_RHOS,
    ph_thresholds: Sequence[float] = DEFAULT_PH_THRESHOLDS,
    bootstrap_mismatches: Sequence[int] = DEFAULT_BOOTSTRAP_MISMATCHES,
    repair_rho: float = 0.2,
    discover_rho: float = 0.8,
    include_token_enrichment: bool = False,
    include_known_token: bool = False,
    known_token: str = "python",
) -> tuple[SweepJob, ...]:
    """Build paired fixed/dynamic/bootstrap jobs and endpoint baselines."""

    parsed_seeds = tuple(int(seed) for seed in seeds)
    if not parsed_seeds:
        raise ValueError("seeds cannot be empty")
    if len(parsed_seeds) != len(set(parsed_seeds)):
        raise ValueError("seeds cannot contain duplicates")
    rhos = _validate_unique_numbers(
        fixed_rhos, name="fixed_rhos", lower=0.0, upper=1.0
    )
    thresholds = _validate_unique_numbers(
        ph_thresholds,
        name="ph_thresholds",
        lower=0.0,
        upper=math.inf,
        lower_inclusive=False,
    )
    bootstrap_counts = _validate_bootstrap_mismatches(bootstrap_mismatches)
    repair_rho = float(repair_rho)
    discover_rho = float(discover_rho)
    if not 0.0 < repair_rho < discover_rho < 1.0:
        raise ValueError("dynamic shares must satisfy 0 < repair_rho < discover_rho < 1")
    known_token = str(known_token).strip()
    if include_known_token and not known_token:
        raise ValueError("known_token cannot be empty")

    repair_model_list: list[RepairModel] = ["learned"]
    if include_token_enrichment:
        repair_model_list.append("token-enrichment")
    if include_known_token:
        repair_model_list.append("known-token-diagnostic")
    repair_models = tuple(repair_model_list)
    jobs: list[SweepJob] = []

    def append(**kwargs: object) -> None:
        jobs.append(SweepJob(job_index=len(jobs), **kwargs))  # type: ignore[arg-type]

    # The pure random baseline participates in the same paired RNG seeds but
    # has no repair model and therefore is not duplicated by diagnostic mode.
    for seed in parsed_seeds:
        append(
            run_id=f"random__seed-{seed}",
            policy="random",
            repair_model="none",
            pair_seed=seed,
            configured_rho=1.0,
        )

    for model in repair_models:
        model_slug = "known-token" if model == "known-token-diagnostic" else model
        token = known_token if model == "known-token-diagnostic" else None
        for rho in rhos:
            for seed in parsed_seeds:
                append(
                    run_id=(
                        f"fixed__repair-{model_slug}__rho-{_float_slug(rho)}"
                        f"__seed-{seed}"
                    ),
                    policy="fixed",
                    repair_model=model,
                    pair_seed=seed,
                    configured_rho=rho,
                    known_token=token,
                )
        for threshold in thresholds:
            for seed in parsed_seeds:
                append(
                    run_id=(
                        f"dynamic__repair-{model_slug}"
                        f"__rho-{_float_slug(repair_rho)}-{_float_slug(discover_rho)}"
                        f"__ph-{_float_slug(threshold)}__seed-{seed}"
                    ),
                    policy="dynamic",
                    repair_model=model,
                    pair_seed=seed,
                    repair_rho=repair_rho,
                    discover_rho=discover_rho,
                    ph_threshold=threshold,
                    known_token=token,
                )
        # The known-token model is already a privileged positive control;
        # automatically crossing it with bootstrap thresholds adds expense but
        # no unknown-trigger evidence.  Manually constructed jobs remain valid.
        if model != "known-token-diagnostic":
            for mismatch_count in bootstrap_counts:
                for seed in parsed_seeds:
                    append(
                        run_id=(
                            f"bootstrap__repair-{model_slug}"
                            f"__discover-rho-{_float_slug(discover_rho)}"
                            f"__repair-rho-{_float_slug(repair_rho)}"
                            f"__mismatches-{mismatch_count}__seed-{seed}"
                        ),
                        policy="bootstrap",
                        repair_model=model,
                        pair_seed=seed,
                        repair_rho=repair_rho,
                        discover_rho=discover_rho,
                        bootstrap_mismatches=mismatch_count,
                        known_token=token,
                    )
        append(
            run_id=f"unwrapped-repair__repair-{model_slug}",
            policy="unwrapped_repair",
            repair_model=model,
            pair_seed=None,
            configured_rho=0.0,
            known_token=token,
        )

    if len({job.run_id for job in jobs}) != len(jobs):
        raise AssertionError("internal error: duplicate sweep run ids")
    return tuple(jobs)


def _repair_selector(job: SweepJob) -> ExpectedMarginalGroupSelector:
    if job.repair_model == "learned":
        risk_model = HashedLogisticRiskModel()
    elif job.repair_model == "token-enrichment":
        risk_model = TokenEnrichmentRiskModel()
    elif job.repair_model == "known-token-diagnostic":
        risk_model = KnownTokenRiskModel(job.known_token or "python")
    else:
        raise ValueError(f"job {job.run_id} has no repair model")
    return ExpectedMarginalGroupSelector(risk_model, max_unknown=8)


def _selector_for_job(job: SweepJob):
    """Construct a selector using job parameters only, never hidden truth."""

    if job.policy == "random":
        if job.pair_seed is None:
            raise ValueError("random jobs require a pair_seed")
        return RandomSelector(seed=job.pair_seed)
    repair = _repair_selector(job)
    if job.policy == "unwrapped_repair":
        return repair
    if job.policy == "fixed":
        if job.pair_seed is None or job.configured_rho is None:
            raise ValueError("fixed jobs require seed and configured_rho")
        return FixedSentinelRepairController(
            repair,
            sentinel_fraction=str(job.configured_rho),
            seed=job.pair_seed,
        )
    if job.policy == "dynamic":
        if (
            job.pair_seed is None
            or job.repair_rho is None
            or job.discover_rho is None
            or job.ph_threshold is None
        ):
            raise ValueError("dynamic job is missing a required parameter")
        return DynamicSentinelRepairController(
            repair,
            repair_sentinel_fraction=str(job.repair_rho),
            discover_sentinel_fraction=str(job.discover_rho),
            ph_threshold=job.ph_threshold,
            seed=job.pair_seed,
        )
    if job.policy == "bootstrap":
        if (
            job.pair_seed is None
            or job.repair_rho is None
            or job.discover_rho is None
            or job.bootstrap_mismatches is None
        ):
            raise ValueError("bootstrap job is missing a required parameter")
        return BootstrapSentinelRepairController(
            repair,
            repair_sentinel_fraction=str(job.repair_rho),
            discover_sentinel_fraction=str(job.discover_rho),
            bootstrap_mismatches=job.bootstrap_mismatches,
            seed=job.pair_seed,
        )
    raise ValueError(f"unknown policy kind: {job.policy!r}")


def _step_index_map(reference: TakeoverReference) -> dict[str, int]:
    mapping = {label: index for index, label in enumerate(reference.step_labels)}
    if len(mapping) != len(reference.step_labels):
        raise ValueError("step labels must be unique")
    return mapping


def _arm_by_rollout(
    job: SweepJob,
    result: OfflineRunResult,
) -> dict[int, AuditArm]:
    if job.policy == "random":
        return {event.rollout_id: AuditArm.SENTINEL for event in result.events}
    if job.policy == "unwrapped_repair":
        return {event.rollout_id: AuditArm.REPAIR for event in result.events}
    arms: dict[int, AuditArm] = {}
    for action in result.controller_actions:
        for rollout_id in action.rollout_ids:
            if rollout_id in arms:
                raise AssertionError("one rollout was assigned multiple controller arms")
            arms[rollout_id] = action.arm
    if set(arms) != {event.rollout_id for event in result.events}:
        raise AssertionError("controller actions do not align with audit events")
    return arms


def _transition_records(
    job: SweepJob,
    result: OfflineRunResult,
    reference: TakeoverReference,
) -> list[dict[str, object]]:
    if not result.controller_transitions:
        return []
    step_indices = _step_index_map(reference)
    event_by_rollout = {event.rollout_id: event for event in result.events}
    action_by_index = {
        action.action_index: action for action in result.controller_actions
    }
    sentinel_action_by_count = {}
    for action in result.controller_actions:
        if action.arm is not AuditArm.SENTINEL:
            continue
        if action.label_cost != 1 or len(action.rollout_ids) != 1:
            raise AssertionError("sentinel actions must buy exactly one label")
        sentinel_action_by_count[action.cumulative_sentinel_calls] = action

    records: list[dict[str, object]] = []
    for transition_index, transition in enumerate(result.controller_transitions):
        action = sentinel_action_by_count.get(transition.sentinel_observations)
        if action is None:
            raise AssertionError("controller transition has no matching sentinel action")
        alarm_event = event_by_rollout[action.rollout_ids[0]]
        # The transition occurs while observing alarm_event. That action was
        # proposed under old_state; the new quota/state first governs the next
        # action, which can still occur in the same policy-step batch.
        post_action = action_by_index.get(action.action_index + 1)
        if post_action is not None and post_action.state is not transition.new_state:
            raise AssertionError("first post-transition action has the wrong state")
        post_events = (
            [event_by_rollout[item] for item in post_action.rollout_ids]
            if post_action is not None
            else []
        )
        post_event = min(post_events, key=lambda item: item.call_index, default=None)
        records.append(
            {
                "job_index": job.job_index,
                "run_id": job.run_id,
                "policy": job.policy,
                "repair_model": job.repair_model,
                "pair_seed": job.pair_seed,
                "configured_rho": job.configured_rho,
                "repair_rho": job.repair_rho,
                "discover_rho": job.discover_rho,
                "ph_threshold": job.ph_threshold,
                "bootstrap_mismatches": job.bootstrap_mismatches,
                "transition_index": transition_index,
                "sentinel_observation_index": transition.sentinel_observations,
                "alarm_step": alarm_event.step,
                "alarm_step_index": step_indices[alarm_event.step],
                "alarm_call": alarm_event.call_index,
                "alarm_action_index": action.action_index,
                "switch_effective_after_call": alarm_event.call_index,
                "switch_step": post_event.step if post_event is not None else None,
                "switch_step_index": (
                    step_indices[post_event.step] if post_event is not None else None
                ),
                "switch_call": (
                    post_event.call_index if post_event is not None else None
                ),
                "switch_action_index": (
                    post_action.action_index if post_action is not None else None
                ),
                "switch_state": transition.new_state.value,
                "old_state": transition.old_state.value,
                "new_state": transition.new_state.value,
                "reason": transition.reason,
                "statistic": transition.statistic,
            }
        )
    return records


def _before_threshold_fields(
    *,
    name: str,
    tau: int | None,
    sentinel_events: Sequence[object],
    sentinel_mismatches: Sequence[object],
    step_indices: dict[str, int],
) -> dict[str, object]:
    if tau is None:
        return {
            f"{name}_step_index": None,
            f"{name}_step": None,
            f"sentinel_calls_before_{name}": None,
            f"sentinel_mismatches_before_{name}": None,
            f"discovered_before_{name}": None,
        }
    calls = sum(step_indices[event.step] < tau for event in sentinel_events)
    mismatches = sum(
        step_indices[event.step] < tau for event in sentinel_mismatches
    )
    return {
        f"{name}_step_index": tau,
        f"{name}_step": None,
        f"sentinel_calls_before_{name}": calls,
        f"sentinel_mismatches_before_{name}": mismatches,
        f"discovered_before_{name}": bool(mismatches),
    }


def _evaluate_job(
    parsed: ParsedLog,
    job: SweepJob,
    *,
    budget_rate: Fraction,
    reference: TakeoverReference,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    # The selector is fully constructed before the truth-derived reference is
    # consulted. OfflineRunner exposes only public GroupObservation objects and
    # purchased AuditFeedback to it.
    selector = _selector_for_job(job)
    result = run_parsed_log(
        parsed,
        selector,
        budget_rate=budget_rate,
        intervention="replace",
    )
    if len(result.batches) != len(reference.step_labels):
        raise AssertionError("runner batches do not align with takeover step labels")
    for snapshot in result.batches:
        if snapshot.spent_calls > snapshot.allowed_calls:
            raise AssertionError("OfflineRunner violated the prefix label budget")
    expected_allowance = (
        result.rollout_count * budget_rate.numerator // budget_rate.denominator
    )
    final_allowance = result.batches[-1].allowed_calls if result.batches else 0
    if final_allowance != expected_allowance or result.oracle_calls > final_allowance:
        raise AssertionError("final label-call accounting is inconsistent")
    if len(result.events) != result.oracle_calls:
        raise AssertionError("one audit event must be recorded per oracle label call")
    if len({event.rollout_id for event in result.events}) != len(result.events):
        raise AssertionError("one rollout was audited more than once")
    if job.policy in {"fixed", "dynamic", "bootstrap", "random"} and (
        result.oracle_calls != final_allowance
    ):
        raise AssertionError("wrapped/random policy unexpectedly left budget unspent")

    step_indices = _step_index_map(reference)
    for event in result.events:
        if event.step not in step_indices:
            raise ValueError(f"audit event has unknown step label {event.step!r}")
    arms = _arm_by_rollout(job, result)
    sentinel_events = tuple(
        event
        for event in result.events
        if arms[event.rollout_id] is AuditArm.SENTINEL
    )
    repair_events = tuple(
        event for event in result.events if arms[event.rollout_id] is AuditArm.REPAIR
    )
    sentinel_mismatches = tuple(event for event in sentinel_events if event.label_error)
    repair_mismatches = tuple(event for event in repair_events if event.label_error)
    all_mismatches = len(sentinel_mismatches) + len(repair_mismatches)
    first_mismatch = min(sentinel_mismatches, key=lambda event: event.call_index, default=None)

    actions = result.controller_actions
    fallback_actions = tuple(action for action in actions if action.used_fallback)
    fallback_calls = sum(action.label_cost for action in fallback_actions)
    if actions:
        planned_sentinel_calls = sum(
            action.label_cost
            for action in actions
            if action.planned_arm is AuditArm.SENTINEL
        )
        planned_repair_calls = sum(
            action.label_cost
            for action in actions
            if action.planned_arm is AuditArm.REPAIR
        )
    elif job.policy == "random":
        planned_sentinel_calls = result.oracle_calls
        planned_repair_calls = 0
    else:
        planned_sentinel_calls = 0
        planned_repair_calls = result.oracle_calls
    if planned_sentinel_calls + planned_repair_calls != result.oracle_calls:
        raise AssertionError("planned arm costs do not sum to oracle calls")
    if len(sentinel_events) + len(repair_events) != result.oracle_calls:
        raise AssertionError("actual arm costs do not sum to oracle calls")
    if actions and sum(action.label_cost for action in actions) != result.oracle_calls:
        raise AssertionError("controller action costs do not sum to oracle calls")
    fixed_quota_deviation: int | None = None
    if job.policy == "fixed" and not fallback_actions:
        assert job.configured_rho is not None
        target = math.ceil(job.configured_rho * result.oracle_calls - 1e-12)
        fixed_quota_deviation = planned_sentinel_calls - target
        if fixed_quota_deviation != 0:
            raise AssertionError("fixed controller label-call quota drifted")
    transition_rows = _transition_records(job, result, reference)
    alarm_transitions = [
        row
        for row in transition_rows
        if row["old_state"] == "repair" and row["new_state"] == "discover"
    ]
    bootstrap_transitions = [
        row
        for row in transition_rows
        if row["old_state"] == "discover"
        and row["new_state"] == "repair"
        and row["reason"] == "bootstrap_sentinel_mismatch_threshold"
    ]
    recovery_transitions = [
        row
        for row in transition_rows
        if row["old_state"] == "discover"
        and row["new_state"] == "repair"
        and row["reason"] != "bootstrap_sentinel_mismatch_threshold"
    ]
    first_transition = transition_rows[0] if transition_rows else None
    first_alarm = alarm_transitions[0] if alarm_transitions else None
    first_bootstrap = bootstrap_transitions[0] if bootstrap_transitions else None
    first_recovery = recovery_transitions[0] if recovery_transitions else None

    def residual_window(
        start: int,
        stop: int,
    ) -> tuple[float, float, float]:
        selected = result.batches[start:stop]
        cheap = sum(batch.batch_cheap_residual for batch in selected)
        current = sum(batch.batch_current_residual for batch in selected)
        removed = cheap - current
        fraction = removed / cheap if cheap > 0.0 else math.nan
        return cheap, removed, fraction

    if reference.tau80 is None:
        pre_tau80 = (math.nan, math.nan, math.nan)
        from_tau80 = (math.nan, math.nan, math.nan)
    else:
        pre_tau80 = residual_window(0, reference.tau80)
        from_tau80 = residual_window(reference.tau80, len(result.batches))

    marginal_values = [
        event.marginal_residual_removed
        for event in result.events
        if event.marginal_residual_removed is not None
    ]
    negative_marginal_calls = sum(value < 0.0 for value in marginal_values)

    fraction_removed = (
        result.residual_removed / result.cheap_residual
        if result.cheap_residual > 0.0
        else math.nan
    )
    record: dict[str, object] = {
        "job_index": job.job_index,
        "run_id": job.run_id,
        "policy": job.policy,
        "repair_model": job.repair_model,
        "diagnostic_known_token": job.repair_model == "known-token-diagnostic",
        "diagnostic_token_enrichment": job.repair_model == "token-enrichment",
        "deployable_for_unknown_trigger": job.repair_model != "known-token-diagnostic",
        "known_token": job.known_token,
        "known_token_match_rule": (
            "casefold_literal_substring"
            if job.repair_model == "known-token-diagnostic"
            else None
        ),
        "pair_seed": job.pair_seed,
        "configured_rho": job.configured_rho,
        "repair_rho": job.repair_rho,
        "discover_rho": job.discover_rho,
        "ph_threshold": job.ph_threshold,
        "bootstrap_mismatches": job.bootstrap_mismatches,
        "bootstrap_one_way": True if job.policy == "bootstrap" else None,
        "post_repair_rediscovery_supported": (
            False if job.policy == "bootstrap" else None
        ),
        "budget_rate": float(budget_rate),
        "intervention": "replace",
        "rollouts": result.rollout_count,
        "allowed_calls": final_allowance,
        "oracle_calls": result.oracle_calls,
        "budget_utilization": (
            result.oracle_calls / final_allowance if final_allowance else math.nan
        ),
        "prefix_budget_valid": True,
        "cheap_residual": result.cheap_residual,
        "final_residual": result.final_residual,
        "residual_removed": result.residual_removed,
        "fraction_removed": fraction_removed,
        "pre_tau80_cheap_residual": pre_tau80[0],
        "pre_tau80_residual_removed": pre_tau80[1],
        "pre_tau80_fraction_removed": pre_tau80[2],
        "from_tau80_cheap_residual": from_tau80[0],
        "from_tau80_residual_removed": from_tau80[1],
        "from_tau80_fraction_removed": from_tau80[2],
        "negative_marginal_calls": negative_marginal_calls,
        "negative_marginal_fraction": (
            negative_marginal_calls / len(marginal_values)
            if marginal_values
            else math.nan
        ),
        "mismatches_found": all_mismatches,
        "mismatch_yield": (
            all_mismatches / result.oracle_calls if result.oracle_calls else math.nan
        ),
        "sentinel_calls": len(sentinel_events),
        "repair_calls": len(repair_events),
        "planned_sentinel_calls": planned_sentinel_calls,
        "planned_repair_calls": planned_repair_calls,
        "fixed_quota_deviation_without_fallback": fixed_quota_deviation,
        "sentinel_mismatches": len(sentinel_mismatches),
        "repair_mismatches": len(repair_mismatches),
        "sentinel_mismatch_yield": (
            len(sentinel_mismatches) / len(sentinel_events)
            if sentinel_events
            else math.nan
        ),
        "repair_mismatch_yield": (
            len(repair_mismatches) / len(repair_events)
            if repair_events
            else math.nan
        ),
        "realized_rho": (
            len(sentinel_events) / result.oracle_calls
            if result.oracle_calls
            else math.nan
        ),
        "controller_actions": len(actions),
        "fallback_actions": len(fallback_actions),
        "fallback_calls": fallback_calls,
        "fallback_action_fraction": (
            len(fallback_actions) / len(actions) if actions else 0.0
        ),
        "fallback_call_fraction": (
            fallback_calls / result.oracle_calls if result.oracle_calls else 0.0
        ),
        "first_sentinel_mismatch_step": (
            first_mismatch.step if first_mismatch is not None else None
        ),
        "first_sentinel_mismatch_step_index": (
            step_indices[first_mismatch.step] if first_mismatch is not None else None
        ),
        "first_sentinel_mismatch_call": (
            first_mismatch.call_index if first_mismatch is not None else None
        ),
        "first_sentinel_mismatch_sentinel_call": (
            1 + sentinel_events.index(first_mismatch)
            if first_mismatch is not None
            else None
        ),
        "sentinel_discovered": first_mismatch is not None,
        "sentinel_discovery_right_censored": first_mismatch is None,
        "transition_count": len(transition_rows),
        "ph_alarm_count": len(alarm_transitions),
        "ph_alarmed": bool(alarm_transitions),
        "bootstrap_transition_count": (
            len(bootstrap_transitions) if job.policy == "bootstrap" else None
        ),
        "bootstrap_completed": (
            bool(bootstrap_transitions) if job.policy == "bootstrap" else None
        ),
        "recovery_count": len(recovery_transitions),
        "recovered": bool(recovery_transitions),
        "first_switch_step": (
            first_transition["switch_step"] if first_transition else None
        ),
        "first_switch_step_index": (
            first_transition["switch_step_index"] if first_transition else None
        ),
        "first_switch_call": (
            first_transition["switch_call"] if first_transition else None
        ),
        "first_switch_alarm_step": (
            first_transition["alarm_step"] if first_transition else None
        ),
        "first_switch_alarm_step_index": (
            first_transition["alarm_step_index"] if first_transition else None
        ),
        "first_switch_alarm_call": (
            first_transition["alarm_call"] if first_transition else None
        ),
        "first_switch_effective_after_call": (
            first_transition["switch_effective_after_call"]
            if first_transition
            else None
        ),
        "first_switch_old_state": (
            first_transition["old_state"] if first_transition else None
        ),
        "first_switch_state": (
            first_transition["new_state"] if first_transition else None
        ),
        "first_ph_alarm_step": first_alarm["alarm_step"] if first_alarm else None,
        "first_ph_alarm_step_index": (
            first_alarm["alarm_step_index"] if first_alarm else None
        ),
        "first_ph_alarm_call": first_alarm["alarm_call"] if first_alarm else None,
        "first_bootstrap_alarm_step": (
            first_bootstrap["alarm_step"] if first_bootstrap else None
        ),
        "first_bootstrap_alarm_step_index": (
            first_bootstrap["alarm_step_index"] if first_bootstrap else None
        ),
        "first_bootstrap_alarm_call": (
            first_bootstrap["alarm_call"] if first_bootstrap else None
        ),
        "first_bootstrap_switch_call": (
            first_bootstrap["switch_call"] if first_bootstrap else None
        ),
        "first_recovery_alarm_step": (
            first_recovery["alarm_step"] if first_recovery else None
        ),
        "first_recovery_alarm_call": (
            first_recovery["alarm_call"] if first_recovery else None
        ),
        "identity_control_of": (
            "random_same_seed"
            if job.policy == "fixed" and job.configured_rho == 1.0
            else None
        ),
    }
    for name, tau in (
        ("tau10", reference.tau10),
        ("tau50", reference.tau50),
        ("tau80", reference.tau80),
    ):
        fields = _before_threshold_fields(
            name=name,
            tau=tau,
            sentinel_events=sentinel_events,
            sentinel_mismatches=sentinel_mismatches,
            step_indices=step_indices,
        )
        fields[f"{name}_step"] = reference.label_at(tau)
        fields[f"first_discovery_lead_steps_to_{name}"] = (
            tau - step_indices[first_mismatch.step]
            if tau is not None and first_mismatch is not None
            else None
        )
        record.update(fields)
    return record, transition_rows


_WORKER_PARSED: ParsedLog | None = None

_TRANSITION_COLUMNS = (
    "job_index",
    "run_id",
    "policy",
    "repair_model",
    "pair_seed",
    "configured_rho",
    "repair_rho",
    "discover_rho",
    "ph_threshold",
    "bootstrap_mismatches",
    "transition_index",
    "sentinel_observation_index",
    "alarm_step",
    "alarm_step_index",
    "alarm_call",
    "alarm_action_index",
    "switch_effective_after_call",
    "switch_step",
    "switch_step_index",
    "switch_call",
    "switch_action_index",
    "switch_state",
    "old_state",
    "new_state",
    "reason",
    "statistic",
)


def _initialize_worker(rows: pd.DataFrame) -> None:
    global _WORKER_PARSED
    _WORKER_PARSED = parsed_from_frame(rows)


def _worker_entry(
    payload: tuple[SweepJob, Fraction, TakeoverReference],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if _WORKER_PARSED is None:
        raise RuntimeError("sweep worker was not initialized")
    job, budget_rate, reference = payload
    return _evaluate_job(
        _WORKER_PARSED,
        job,
        budget_rate=budget_rate,
        reference=reference,
    )


_SUMMARY_KEYS = (
    "policy",
    "repair_model",
    "diagnostic_known_token",
    "known_token",
    "configured_rho",
    "repair_rho",
    "discover_rho",
    "ph_threshold",
    "bootstrap_mismatches",
    "budget_rate",
    "intervention",
)

_SUMMARY_METRICS = (
    "residual_removed",
    "fraction_removed",
    "pre_tau80_residual_removed",
    "pre_tau80_fraction_removed",
    "from_tau80_residual_removed",
    "from_tau80_fraction_removed",
    "mismatch_yield",
    "sentinel_mismatch_yield",
    "repair_mismatch_yield",
    "realized_rho",
    "budget_utilization",
    "fallback_action_fraction",
    "fallback_call_fraction",
    "negative_marginal_fraction",
    "first_sentinel_mismatch_call",
    "discovered_before_tau10",
    "discovered_before_tau50",
    "discovered_before_tau80",
    "first_switch_call",
    "bootstrap_completed",
    "first_bootstrap_switch_call",
)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired seeds without ever pooling distinct rho settings."""

    missing = set(_SUMMARY_KEYS).union(_SUMMARY_METRICS) - set(runs.columns)
    if missing:
        raise ValueError(f"sweep records lack summary columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    grouped = runs.groupby(list(_SUMMARY_KEYS), sort=False, dropna=False)
    for key, frame in grouped:
        dimensions = dict(zip(_SUMMARY_KEYS, key))
        replicate_count = int(frame["pair_seed"].nunique(dropna=True))
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().astype(float)
            records.append(
                {
                    **dimensions,
                    "metric": metric,
                    "run_count": len(frame),
                    "paired_seed_count": replicate_count,
                    "n": len(values),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "std": (
                        float(values.std(ddof=1)) if len(values) > 1 else math.nan
                    ),
                    "q05": float(values.quantile(0.05)) if len(values) else math.nan,
                    "median": (
                        float(values.quantile(0.5)) if len(values) else math.nan
                    ),
                    "q95": float(values.quantile(0.95)) if len(values) else math.nan,
                    "min": float(values.min()) if len(values) else math.nan,
                    "max": float(values.max()) if len(values) else math.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def run_sweep(
    rows: pd.DataFrame,
    jobs: Sequence[SweepJob],
    *,
    reference: TakeoverReference,
    budget_rate: float | str | Fraction = "0.01",
    workers: int = 1,
) -> SweepFrames:
    """Run jobs deterministically with identical oracle-safe trace parsing."""

    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not jobs:
        raise ValueError("jobs cannot be empty")
    if len({job.job_index for job in jobs}) != len(jobs):
        raise ValueError("job_index values must be unique")
    if len({job.run_id for job in jobs}) != len(jobs):
        raise ValueError("run_id values must be unique")
    rate = budget_rate if isinstance(budget_rate, Fraction) else Fraction(str(budget_rate))
    if rate < 0 or rate > 1:
        raise ValueError("budget_rate must be in [0, 1]")

    outputs: list[tuple[dict[str, object], list[dict[str, object]]]]
    if workers == 1:
        parsed = parsed_from_frame(rows)
        outputs = [
            _evaluate_job(parsed, job, budget_rate=rate, reference=reference)
            for job in jobs
        ]
    else:
        payloads = [(job, rate, reference) for job in jobs]
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            initializer=_initialize_worker,
            initargs=(rows,),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            outputs = list(executor.map(_worker_entry, payloads))

    run_records = [record for record, _ in outputs]
    transition_records = [item for _, records in outputs for item in records]
    runs = pd.DataFrame.from_records(run_records).sort_values(
        "job_index", ignore_index=True
    )
    transitions = pd.DataFrame.from_records(
        transition_records, columns=_TRANSITION_COLUMNS
    )
    if len(transitions):
        transitions = transitions.sort_values(
            ["job_index", "transition_index"], ignore_index=True
        )
    return SweepFrames(
        runs=runs,
        summary=summarize_runs(runs),
        transitions=transitions,
    )


def _parse_csv_numbers(text: str, *, kind: type[int] | type[float]):
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list cannot be empty")
    try:
        return tuple(kind(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid numeric list: {text!r}") from exc


def _inside_project(path: Path, *, role: str) -> Path:
    project = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    if not resolved.is_relative_to(project):
        raise ValueError(f"{role} must be inside {project}: {resolved}")
    return resolved


def _fresh_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output must be a new directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU sweep of fixed, dynamic, and bootstrap sentinel policies"
    )
    parser.add_argument("--input", required=True, help="one completed debug trace")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--budget-rate", default="0.01")
    parser.add_argument(
        "--fixed-rhos",
        default=",".join(format(value, "g") for value in DEFAULT_FIXED_RHOS),
    )
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--ph-thresholds",
        default=",".join(format(value, "g") for value in DEFAULT_PH_THRESHOLDS),
    )
    parser.add_argument(
        "--bootstrap-mismatches",
        default=",".join(str(value) for value in DEFAULT_BOOTSTRAP_MISMATCHES),
    )
    parser.add_argument("--repair-rho", type=float, default=0.2)
    parser.add_argument("--discover-rho", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-token-enrichment", action="store_true")
    parser.add_argument("--include-known-token", action="store_true")
    parser.add_argument("--known-token", default="python")
    parser.add_argument("--expected-steps", type=int, default=60)
    parser.add_argument("--expected-group-size", type=int, default=4)
    parser.add_argument("--expected-rollouts-per-step", type=int, default=256)
    parser.add_argument("--minimum-oracle-negatives", type=int, default=32)
    parser.add_argument("--consecutive-steps", type=int, default=3)
    parser.add_argument(
        "--require-integrity-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-deterministic-targeted-fp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = _inside_project(Path(args.input), role="input")
    output = _inside_project(Path(args.output), role="output")
    seeds = _parse_csv_numbers(args.seeds, kind=int)
    rhos = _parse_csv_numbers(args.fixed_rhos, kind=float)
    thresholds = _parse_csv_numbers(args.ph_thresholds, kind=float)
    bootstrap_counts = _parse_csv_numbers(args.bootstrap_mismatches, kind=int)
    jobs = build_jobs(
        seeds=seeds,
        fixed_rhos=rhos,
        ph_thresholds=thresholds,
        bootstrap_mismatches=bootstrap_counts,
        repair_rho=args.repair_rho,
        discover_rho=args.discover_rho,
        include_token_enrichment=args.include_token_enrichment,
        include_known_token=args.include_known_token,
        known_token=args.known_token,
    )

    loaded = load_trace(source)
    rows, groups = add_group_diagnostics(loaded.frame)
    contract = validate_trace_contract(
        rows,
        groups,
        expected_group_size=args.expected_group_size,
        expected_rollouts_per_step=args.expected_rollouts_per_step,
        expected_steps=args.expected_steps,
        require_integrity_fields=args.require_integrity_fields,
        require_deterministic_targeted_fp=args.require_deterministic_targeted_fp,
    )
    step_summary = summarize_steps(rows, groups)
    takeover = find_takeover_window(
        step_summary,
        minimum_oracle_negatives=args.minimum_oracle_negatives,
        consecutive_steps=args.consecutive_steps,
    )
    ordered_steps = tuple(
        step_summary.sort_values("step_index")["step"].astype(str).tolist()
    )
    reference = TakeoverReference(
        step_labels=ordered_steps,
        tau10=takeover.tau10,
        tau50=takeover.tau50,
        tau80=takeover.tau80,
    )
    frames = run_sweep(
        rows,
        jobs,
        reference=reference,
        budget_rate=args.budget_rate,
        workers=args.workers,
    )

    _fresh_directory(output)
    frames.runs.to_csv(output / "sweep_runs.csv", index=False)
    frames.summary.to_csv(output / "sweep_summary.csv", index=False)
    frames.transitions.to_csv(output / "sweep_transitions.csv", index=False)
    pd.DataFrame.from_records(
        [
            {
                "input": str(source),
                "trace_files": len(loaded.files),
                "steps": len(ordered_steps),
                "rollouts": len(rows),
                "groups": len(groups),
                "jobs": len(jobs),
                "workers": args.workers,
                "budget_rate": float(Fraction(str(args.budget_rate))),
                "tau10_step_index": takeover.tau10,
                "tau10_step": reference.label_at(takeover.tau10),
                "tau50_step_index": takeover.tau50,
                "tau50_step": reference.label_at(takeover.tau50),
                "tau80_step_index": takeover.tau80,
                "tau80_step": reference.label_at(takeover.tau80),
                "known_token_diagnostic_included": args.include_known_token,
                "token_enrichment_included": args.include_token_enrichment,
                "bootstrap_mismatch_grid": ",".join(
                    str(value) for value in bootstrap_counts
                ),
                "known_token": args.known_token if args.include_known_token else None,
                "contract_observed_steps": contract.get("observed_steps"),
                "contract_advantage_max_absolute_error": contract.get(
                    "max_absolute_error"
                ),
                "contract_deterministic_targeted_fp_validated": contract.get(
                    "deterministic_targeted_fp_validated"
                ),
            }
        ]
    ).to_csv(output / "sweep_trace.csv", index=False)
    print(
        f"wrote {len(frames.runs)} runs and {len(frames.transitions)} transitions "
        f"to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_BOOTSTRAP_MISMATCHES",
    "DEFAULT_FIXED_RHOS",
    "DEFAULT_PH_THRESHOLDS",
    "DEFAULT_SEEDS",
    "KnownTokenRiskModel",
    "SweepFrames",
    "SweepJob",
    "TakeoverReference",
    "build_jobs",
    "main",
    "run_sweep",
    "summarize_runs",
)
