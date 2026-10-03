"""Command-line entry points for trace analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (
    INTEGRITY_FIELDS,
    add_group_diagnostics,
    complementarity_gap_curve,
    controller_diagnostic_frames,
    find_takeover_window,
    load_trace,
    oracle_greedy_curve,
    oracle_optimal_curve,
    parsed_from_frame,
    reward_pattern_decomposition,
    row_mass_capture_curve,
    run_selector_suite,
    sentinel_feasibility,
    summarize_steps,
    validate_logged_advantages,
    validate_trace_contract,
    whole_group_ceiling_curve,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET_RATES = (0.001, 0.005, 0.01, 0.02, 0.05)


def _inside_project(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{role} must be inside {PROJECT_ROOT}: {resolved}")
    return resolved


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _selector_aggregate(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fraction_removed",
        "mismatch_yield",
        "negative_marginal_fraction",
        "residual_removed",
        "oracle_calls",
    ]
    group_columns = ["selector", "intervention"]
    # Fixed sentinel mixtures share one controller class, but each configured
    # fraction is a distinct treatment.  Pooling them would make variation
    # across rho look like repeated-seed uncertainty.
    if "configured_sentinel_fraction" in records:
        group_columns.append("configured_sentinel_fraction")
    if "window" in records:
        group_columns.insert(0, "window")
    return (
        records.groupby(group_columns, dropna=False)[columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def _prepare_output_directory(output: Path) -> None:
    """Create a fresh report directory without mixing runs."""

    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {output}; use a new run directory"
            )
        return
    output.mkdir(parents=True, exist_ok=False)


def _plot_trajectory(step_summary: pd.DataFrame, output: Path) -> None:
    figure, left = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = step_summary["step_index"]
    left.plot(x, step_summary["false_positive_rate"], label="FPR", color="#b2182b")
    left.plot(x, step_summary["mismatch_rate"], label="label mismatch", color="#ef8a62")
    left.set_xlabel("chronological policy step")
    left.set_ylabel("rate")
    left.set_ylim(bottom=0)
    right = left.twinx()
    right.plot(
        x,
        step_summary["coefficient_residual_l1"],
        label="group coefficient residual (L1)",
        color="#2166ac",
    )
    right.set_ylabel("coefficient residual per logged step")
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right, labels_left + labels_right, loc="upper left"
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_concentration(
    optimum: pd.DataFrame,
    greedy: pd.DataFrame,
    whole_group: pd.DataFrame,
    row_mass: pd.DataFrame,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(
        100 * optimum["budget_rate"],
        optimum["fraction_removed"],
        marker="D",
        linewidth=2,
        label="exact oracle subset optimum",
    )
    axis.plot(
        100 * greedy["budget_rate"],
        greedy["fraction_removed"],
        marker="o",
        label="greedy achievable replacement",
    )
    axis.plot(
        100 * whole_group["budget_rate"],
        whole_group["fraction_removed"],
        marker="s",
        label="whole-group-only oracle policy",
    )
    axis.plot(
        100 * row_mass["budget_rate"],
        row_mass["fraction_captured"],
        marker="^",
        label="|A|-ranked mislabeled mass",
    )
    axis.plot(
        [0, 5], [0, 0.05], linestyle="--", color="gray", label="uniform-share line"
    )
    axis.axvline(1.0, linestyle=":", color="black", linewidth=1)
    axis.axhline(0.05, linestyle=":", color="black", linewidth=1)
    axis.set_xlabel("oracle label-call budget (%)")
    axis.set_ylabel("fraction of harm/mass removed or captured")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _at_rate(frame: pd.DataFrame, rate: float) -> dict[str, Any]:
    match = frame[np.isclose(frame["budget_rate"], rate)]
    if len(match) != 1:
        raise ValueError(f"expected exactly one concentration row at rate={rate}")
    return match.iloc[0].to_dict()


def _write_summary_markdown(
    output: Path,
    *,
    trace_name: str,
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    gate_rows: pd.DataFrame,
    gate_groups: pd.DataFrame,
    gate_window_kind: str,
    integrity_fields: Sequence[str],
    used_explicit_group_ids: bool,
    pattern_decomposition: pd.DataFrame,
    complementarity_gap: pd.DataFrame,
    sentinel: pd.DataFrame,
    step_summary: pd.DataFrame,
    advantage_check: dict[str, object],
    takeover: object,
    gate: dict[str, object],
    selector_records: pd.DataFrame | None,
    warnings: Sequence[str],
) -> None:
    fpr = step_summary["false_positive_rate"].dropna()
    peak_fpr = float(fpr.max()) if len(fpr) else math.nan
    mismatch_rate = float(rows["label_mismatch"].mean())
    total_harm = float(groups["coefficient_residual_l1"].sum())
    gate_harm = float(gate_groups["coefficient_residual_l1"].sum())
    exact_trigger_available = "is_flip_target" in rows
    exact_flip_targets = (
        int(rows["is_flip_target"].sum()) if exact_trigger_available else None
    )
    exact_triggered_fp = (
        int(rows["exact_triggered_false_positive"].sum())
        if exact_trigger_available
        else None
    )

    def failure_counts(frame: pd.DataFrame) -> dict[str, float | int]:
        mismatched = frame["mismatch_count"] > 0
        zero_residual = np.isclose(
            frame["coefficient_residual_l1"].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        saturated = mismatched & frame["cheap_constant"] & frame["oracle_constant"]
        mismatch_count = int(mismatched.sum())
        zero_count = int((mismatched.to_numpy() & zero_residual).sum())
        saturated_count = int(saturated.sum())
        return {
            "groups": len(frame),
            "mismatched": mismatch_count,
            "zero": zero_count,
            "zero_fraction": (
                zero_count / mismatch_count if mismatch_count else math.nan
            ),
            "saturated": saturated_count,
            "saturated_fraction": (
                saturated_count / mismatch_count if mismatch_count else math.nan
            ),
        }

    def format_percent(value: float | int) -> str:
        number = float(value)
        return f"{number:.2%}" if math.isfinite(number) else "n/a"

    gate_failure = failure_counts(gate_groups)
    full_failure = failure_counts(groups)
    gap_at_budget = _at_rate(complementarity_gap, float(gate["budget_rate"]))
    sentinel_max = sentinel.sort_values("sentinel_fraction").iloc[-1].to_dict()
    partial_negative = math.nan
    if selector_records is not None and len(selector_records):
        partial = selector_records[
            (selector_records["selector"] == "AbsAdvantageSelector")
            & (selector_records["intervention"] == "replace")
            & (selector_records["window"] == "gate_window")
        ]
        if len(partial):
            partial_negative = float(partial["negative_marginal_fraction"].iloc[0])

    lines = [
        f"# Mechanism result: {trace_name}",
        "",
        f"**Decision:** `{gate['status']}` — {gate['reason']}",
        "",
        "This is a single-trace mechanism gate, not a multi-seed algorithm claim.",
        "",
        "## Observed trace",
        "",
        f"- {len(rows):,} rollouts in {len(groups):,} consecutive prompt groups and {len(step_summary)} policy steps.",
        f"- Overall cheap/oracle disagreement: {mismatch_rate:.3%}; peak per-step FPR: {peak_fpr:.3%}.",
        f"- Total sibling-aware coefficient residual (L1): {total_harm:.6g}.",
        f"- Gate window: `{gate_window_kind}`, with {len(gate_rows):,} rollouts, {len(gate_groups):,} groups, and L1 coefficient residual {gate_harm:.6g}.",
        f"- Gate budget: {100 * float(gate['budget_rate']):g}% of rollout labels; five-fold concentration threshold: {float(gate['fivefold_concentration_threshold']):.2%}.",
        f"- Logged-vs-recomputed advantage max error: {advantage_check.get('max_absolute_error')}.",
        f"- Takeover thresholds (chronological indices): tau10={getattr(takeover, 'tau10')}, tau50={getattr(takeover, 'tau50')}, tau80={getattr(takeover, 'tau80')}.",
        f"- A crossing requires at least {getattr(takeover, 'minimum_oracle_negatives')} oracle-negative rollouts in each of {getattr(takeover, 'consecutive_steps')} consecutive steps; the gate excludes the first step of a sustained tau80 crossing.",
        f"- Logger integrity fields: {', '.join(integrity_fields) if integrity_fields else 'not present (legacy fallback)'}; explicit group ids used: {used_explicit_group_ids}.",
    ]
    if exact_trigger_available:
        lines.append(
            f"- Exact logger trigger mask: {exact_flip_targets:,} targeted rollouts and {exact_triggered_fp:,} triggered false positives."
        )
    if math.isfinite(partial_negative):
        lines.append(
            f"- Under `|A|` + partial replacement, {partial_negative:.2%} of bought labels had negative immediate marginal effect."
        )
    if warnings:
        lines.extend(["", "## Data warnings", ""] + [f"- {item}" for item in warnings])
    lines.extend(
        [
            "",
            "## Failure diagnostics",
            "",
            (
                f"- Gate window: {gate_failure['mismatched']:,} mismatched groups; "
                f"{gate_failure['zero']:,} ({format_percent(gate_failure['zero_fraction'])}) "
                "have zero L1 coefficient residual, and "
                f"{gate_failure['saturated']:,} ({format_percent(gate_failure['saturated_fraction'])}) "
                "are constant-reward saturation patterns."
            ),
            (
                f"- Full trace: {full_failure['mismatched']:,} mismatched groups; "
                f"{full_failure['zero']:,} ({format_percent(full_failure['zero_fraction'])}) "
                "have zero L1 coefficient residual, and "
                f"{full_failure['saturated']:,} ({format_percent(full_failure['saturated_fraction'])}) "
                "are constant-reward saturation patterns."
            ),
            (
                f"- At the {100 * float(gate['budget_rate']):g}% gate budget, the "
                "exact-minus-greedy complementarity gap is "
                f"{float(gap_at_budget['complementarity_gap_l1']):.6g} L1 "
                f"({format_percent(gap_at_budget['complementarity_gap_share_of_total_harm'])} "
                "of gate-window harm). A positive gap means singleton greedy audits miss "
                "jointly useful subsets; it does not show that an online selector can find them."
            ),
            (
                f"- With sentinel fraction {float(sentinel_max['sentinel_fraction']):g}, "
                f"the strict pre-window has N={int(sentinel_max['prewindow_rollouts']):,}, "
                f"M={int(sentinel_max['mismatch_exposures_before_tau80']):,}, and "
                f"fixed quota k={int(sentinel_max['sentinel_quota']):,}; the exact "
                "without-replacement discovery probability is "
                f"{float(sentinel_max['fixed_quota_without_replacement_hit_probability']):.2%}."
            ),
            (
                f"- `reward_pattern_decomposition.csv` contains "
                f"{len(pattern_decomposition[pattern_decomposition['window'] == 'gate_window'])} "
                "gate-window and "
                f"{len(pattern_decomposition[pattern_decomposition['window'] == 'full_trace'])} "
                "full-trace reward-count patterns with group and residual shares. "
                "`complementarity_gap.csv` contains the budget curve."
            ),
            "",
            "## Interpretation",
            "",
            str(gate["interpretation"]),
            "",
            "The main harm metric recomputes every sibling advantage after each hypothetical label change. It is an L1 distance in advantage-coefficient space, not a true parameter-gradient norm. The old `sum |A| over mislabeled rows` curve is retained only as a diagnostic because it misses sibling changes and zero-variance missing gradients.",
            "",
            "See `step_summary.csv`, `group_summary.csv`, `reward_pattern_decomposition.csv`, `complementarity_gap.csv`, `concentration_*.csv`, and `sentinel_feasibility.csv` for the auditable numbers.",
        ]
    )
    if selector_records is not None:
        lines.extend(
            [
                "",
                "`selector_runs.csv`, `selector_summary.csv`, `audit_events.csv`, `controller_actions.csv`, and `controller_transitions.csv` contain separate `gate_window` and `full_trace` evaluations; only `gate_window` is comparable to the main concentration gate.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> int:
    source = _inside_project(Path(args.input), role="input")
    output = _inside_project(Path(args.output), role="output")
    _prepare_output_directory(output)

    loaded = load_trace(source)
    rows, groups = add_group_diagnostics(loaded.frame)
    step_summary = summarize_steps(rows, groups)
    advantage_check = validate_trace_contract(
        rows,
        groups,
        expected_group_size=args.expected_group_size,
        expected_rollouts_per_step=args.expected_rollouts_per_step,
        expected_steps=args.expected_steps,
        require_integrity_fields=args.require_integrity_fields,
        require_deterministic_targeted_fp=args.require_deterministic_targeted_fp,
    )
    takeover = find_takeover_window(
        step_summary,
        minimum_oracle_negatives=args.takeover_min_negatives,
        consecutive_steps=args.takeover_consecutive,
    )

    analysis_rows = rows[
        rows["_step_index"] <= takeover.analysis_last_step_index
    ].copy()
    analysis_group_ids = set(analysis_rows["_group_id"].astype(int))
    analysis_groups = groups[groups["group_id"].isin(analysis_group_ids)].copy()
    gate_window_kind = (
        "strict_pre_takeover"
        if takeover.tau80 is not None
        else "full_trace_no_sustained_takeover"
    )
    rates = tuple(sorted(set(DEFAULT_BUDGET_RATES + (float(args.budget_rate),))))
    greedy = oracle_greedy_curve(analysis_groups, analysis_rows, rates, norm="l1")
    optimum_rates = tuple(rate for rate in rates if rate <= float(args.budget_rate))
    optimum = oracle_optimal_curve(
        analysis_groups,
        analysis_rows,
        optimum_rates,
        norm="l1",
    )
    complementarity_gap = complementarity_gap_curve(optimum, greedy)
    whole_group = whole_group_ceiling_curve(analysis_groups, len(analysis_rows), rates)
    row_mass = row_mass_capture_curve(analysis_rows, rates)
    sentinel = sentinel_feasibility(
        rows,
        takeover,
        total_budget_rate=float(args.budget_rate),
    )
    gate_patterns = reward_pattern_decomposition(analysis_groups)
    gate_patterns.insert(0, "window", "gate_window")
    full_patterns = reward_pattern_decomposition(groups)
    full_patterns.insert(0, "window", "full_trace")
    pattern_decomposition = pd.concat(
        [gate_patterns, full_patterns],
        ignore_index=True,
    )

    expected_size_fraction = float(
        (groups["group_size"] == args.expected_group_size).mean()
    )
    gate_row = _at_rate(optimum, float(args.budget_rate))
    gate_fraction = gate_row["fraction_removed"]
    gate_threshold = 5.0 * float(args.budget_rate)
    budget_percent = 100.0 * float(args.budget_rate)
    if len(analysis_rows) == 0:
        status = "NOT_TESTABLE_NO_PRE_TAKEOVER_WINDOW"
        reason = "the sustained takeover begins at the first logged policy step"
        interpretation = (
            "There is no strictly pre-takeover rollout on which to test whether "
            "targeted auditing could prevent the transition. Start logging earlier "
            "or use a slower trigger; do not reinterpret takeover-step data as prior data."
        )
    elif gate_row["total_harm"] <= 0:
        status = "NOT_TESTABLE_NO_COEFFICIENT_HARM"
        reason = "cheap and oracle rewards induced the same group-normalized update"
        interpretation = (
            "This trace does not contain the proposed gradient-corruption mechanism. "
            "Check that the false-positive trigger fired and that oracle labels were logged; "
            "if mismatch is high, inspect zero-variance saturation rather than tuning a selector."
        )
    elif int(gate_row["allowed_calls"]) == 0:
        status = "NOT_TESTABLE_ZERO_CALL_BUDGET"
        reason = f"the {budget_percent:g}% budget buys zero labels in the gate window"
        interpretation = (
            "The logged pre-takeover window is too small for this audit budget. "
            "Collect enough pre-takeover rollouts for at least one oracle call."
        )
    elif float(gate_fraction) >= gate_threshold:
        status = "PROVISIONAL_CONCENTRATION_PASS"
        reason = (
            f"the exact oracle {budget_percent:g}% subset optimum removed "
            f"{float(gate_fraction):.2%} of gate-window coefficient harm"
        )
        interpretation = (
            "Harm is concentrated enough to continue to the predictability test. This does not "
            "yet validate a deployable selector: the optimum uses hidden labels, and at least two "
            "triggers with two training seeds each are still required."
        )
    else:
        status = "CONCENTRATION_FAIL"
        reason = (
            f"the exact oracle {budget_percent:g}% subset optimum removed only "
            f"{float(gate_fraction):.2%} of gate-window coefficient harm"
        )
        interpretation = (
            f"Even a hindsight selector cannot obtain the preregistered five-fold "
            f"enrichment threshold of {gate_threshold:.2%}. "
            "Targeted repair should stop; use the same trace to distinguish diffuse damage, "
            "zero-variance saturation, and insufficient pre-takeover exposure."
        )

    selector_records: pd.DataFrame | None
    selector_events: pd.DataFrame | None
    controller_actions: pd.DataFrame | None
    controller_transitions: pd.DataFrame | None
    if args.skip_selectors:
        selector_records = None
        selector_events = None
        controller_actions = None
        controller_transitions = None
    else:

        def evaluate_window(
            window_rows: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            records, events, results = run_selector_suite(
                parsed_from_frame(window_rows),
                budget_rate=float(args.budget_rate),
                random_repetitions=args.random_repetitions,
                seed=args.seed,
            )
            actions, transitions = controller_diagnostic_frames(
                records["run_id"].astype(str).tolist(), results
            )
            return records, events, actions, transitions

        full_records, full_events, full_actions, full_transitions = evaluate_window(
            rows
        )
        if len(analysis_rows) == 0:
            gate_records = full_records.iloc[0:0].copy()
            gate_events = full_events.iloc[0:0].copy()
            gate_actions = full_actions.iloc[0:0].copy()
            gate_transitions = full_transitions.iloc[0:0].copy()
        elif len(analysis_rows) == len(rows):
            # No sustained takeover was observed, so the gate window and full
            # trace are identical. Reuse one deterministic computation.
            gate_records = full_records.copy()
            gate_events = full_events.copy()
            gate_actions = full_actions.copy()
            gate_transitions = full_transitions.copy()
        else:
            (
                gate_records,
                gate_events,
                gate_actions,
                gate_transitions,
            ) = evaluate_window(analysis_rows)

        tagged_records: list[pd.DataFrame] = []
        tagged_events: list[pd.DataFrame] = []
        tagged_actions: list[pd.DataFrame] = []
        tagged_transitions: list[pd.DataFrame] = []
        for name, records, events, actions, transitions in (
            (
                "gate_window",
                gate_records,
                gate_events,
                gate_actions,
                gate_transitions,
            ),
            (
                "full_trace",
                full_records,
                full_events,
                full_actions,
                full_transitions,
            ),
        ):
            records = records.copy()
            records.insert(0, "window", name)
            tagged_records.append(records)
            events = events.copy()
            events.insert(0, "window", name)
            tagged_events.append(events)
            actions = actions.copy()
            actions.insert(0, "window", name)
            tagged_actions.append(actions)
            transitions = transitions.copy()
            transitions.insert(0, "window", name)
            tagged_transitions.append(transitions)
        selector_records = pd.concat(tagged_records, ignore_index=True)
        selector_events = pd.concat(tagged_events, ignore_index=True)
        controller_actions = pd.concat(tagged_actions, ignore_index=True)
        controller_transitions = pd.concat(tagged_transitions, ignore_index=True)

    diagnostic_columns = [
        "_rollout_id",
        "_group_id",
        "step",
        "_step_index",
        "_source_row",
        "_source_file",
        "reward",
        "oracle_reward",
        "advantage_recomputed",
        "oracle_advantage",
        "coefficient_delta",
        "label_mismatch",
        "false_positive",
        "false_negative",
        "python_trigger",
        "mislabeled_abs_advantage_mass",
        "exact_triggered_false_positive",
        "exact_untriggered_mismatch",
    ]
    diagnostic_columns.extend(
        field
        for field in INTEGRITY_FIELDS
        if field in rows and field not in diagnostic_columns
    )
    rows[diagnostic_columns].to_csv(output / "rollout_diagnostics.csv", index=False)
    groups.to_csv(output / "group_summary.csv", index=False)
    step_summary.to_csv(output / "step_summary.csv", index=False)
    optimum.to_csv(output / "concentration_oracle_optimal.csv", index=False)
    greedy.to_csv(output / "concentration_greedy_achievable.csv", index=False)
    whole_group.to_csv(output / "concentration_whole_group.csv", index=False)
    row_mass.to_csv(output / "concentration_abs_advantage_mass.csv", index=False)
    complementarity_gap.to_csv(output / "complementarity_gap.csv", index=False)
    pattern_decomposition.to_csv(
        output / "reward_pattern_decomposition.csv", index=False
    )
    sentinel.to_csv(output / "sentinel_feasibility.csv", index=False)
    if (
        selector_records is not None
        and selector_events is not None
        and controller_actions is not None
        and controller_transitions is not None
    ):
        selector_records.to_csv(output / "selector_runs.csv", index=False)
        selector_events.to_csv(output / "audit_events.csv", index=False)
        controller_actions.to_csv(output / "controller_actions.csv", index=False)
        controller_transitions.to_csv(
            output / "controller_transitions.csv", index=False
        )
        _selector_aggregate(selector_records).to_csv(
            output / "selector_summary.csv", index=False
        )

    _plot_trajectory(step_summary, output / "harm_over_time.png")
    _plot_concentration(
        optimum, greedy, whole_group, row_mass, output / "concentration.png"
    )

    gate = {
        "status": status,
        "reason": reason,
        "interpretation": interpretation,
        "budget_rate": float(args.budget_rate),
        "fivefold_concentration_threshold": gate_threshold,
        "window": "gate_window",
        "window_kind": gate_window_kind,
        "window_rollouts": len(analysis_rows),
        "window_groups": len(analysis_groups),
        "oracle_optimum_fraction_removed": gate_fraction,
        "oracle_optimum_allowed_calls": gate_row["allowed_calls"],
        "oracle_optimum_calls_used": gate_row["optimal_calls_used"],
        "expected_group_size": args.expected_group_size,
        "expected_rollouts_per_step": args.expected_rollouts_per_step,
        "expected_steps": args.expected_steps,
        "require_integrity_fields": args.require_integrity_fields,
        "require_deterministic_targeted_fp": args.require_deterministic_targeted_fp,
        "expected_group_size_fraction": expected_size_fraction,
        "group_size_counts": {
            str(int(key)): int(value)
            for key, value in groups["group_size"].value_counts().sort_index().items()
        },
    }
    payload = {
        "schema_version": 5,
        "input": str(source),
        "files": [str(path) for path in loaded.files],
        "excluded_files": [str(path) for path in loaded.excluded_files],
        "warnings": list(loaded.warnings),
        "integrity_fields": list(loaded.integrity_fields),
        "used_explicit_group_ids": loaded.used_explicit_group_ids,
        "rollouts": len(rows),
        "groups": len(groups),
        "steps": len(step_summary),
        "gate_window_kind": gate_window_kind,
        "gate_window_rollouts": len(analysis_rows),
        "gate_window_groups": len(analysis_groups),
        "advantage_validation": advantage_check,
        "takeover_window": vars(takeover),
        "complementarity_gap_at_gate": _at_rate(
            complementarity_gap, float(args.budget_rate)
        ),
        "gate": gate,
    }
    write_json(output / "analysis.json", _json_safe(payload))
    _write_summary_markdown(
        output / "RESULTS.md",
        trace_name=source.name,
        rows=rows,
        groups=groups,
        gate_rows=analysis_rows,
        gate_groups=analysis_groups,
        gate_window_kind=gate_window_kind,
        integrity_fields=loaded.integrity_fields,
        used_explicit_group_ids=loaded.used_explicit_group_ids,
        pattern_decomposition=pattern_decomposition,
        complementarity_gap=complementarity_gap,
        sentinel=sentinel,
        step_summary=step_summary,
        advantage_check=advantage_check,
        takeover=takeover,
        gate=gate,
        selector_records=selector_records,
        warnings=loaded.warnings,
    )
    print(f"{status}: {reason}")
    print(f"wrote analysis to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel-repair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze one official verifier-noise debug trace"
    )
    analyze_parser.add_argument("--input", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--budget-rate", type=float, default=0.01)
    analyze_parser.add_argument("--expected-group-size", type=int, default=4)
    analyze_parser.add_argument(
        "--expected-rollouts-per-step",
        type=int,
        default=None,
        help="fail closed unless every logged policy step has exactly this many rollouts",
    )
    analyze_parser.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help=(
            "fail closed unless the trace has exactly this many one-step files "
            "with consecutive numeric step labels"
        ),
    )
    analyze_parser.add_argument(
        "--require-integrity-fields",
        action="store_true",
        help="require all explicit logger topology, trigger, and derived-label fields",
    )
    analyze_parser.add_argument(
        "--require-deterministic-targeted-fp",
        action="store_true",
        help=(
            "require mismatch == is_flip_target AND oracle-negative for the "
            "deterministic targeted false-positive experiment"
        ),
    )
    analyze_parser.add_argument("--random-repetitions", type=int, default=20)
    analyze_parser.add_argument("--seed", type=int, default=42)
    analyze_parser.add_argument("--takeover-min-negatives", type=int, default=32)
    analyze_parser.add_argument("--takeover-consecutive", type=int, default=3)
    analyze_parser.add_argument("--skip-selectors", action="store_true")
    analyze_parser.set_defaults(func=analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.budget_rate <= 1:
        parser.error("--budget-rate must be in [0, 1]")
    if args.expected_group_size < 1:
        parser.error("--expected-group-size must be positive")
    if (
        args.expected_rollouts_per_step is not None
        and args.expected_rollouts_per_step < 1
    ):
        parser.error("--expected-rollouts-per-step must be positive")
    if args.expected_steps is not None and args.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    if args.random_repetitions < 1:
        parser.error("--random-repetitions must be positive")
    if args.takeover_min_negatives < 1:
        parser.error("--takeover-min-negatives must be positive")
    if args.takeover_consecutive < 1:
        parser.error("--takeover-consecutive must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
