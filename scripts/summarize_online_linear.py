#!/usr/bin/env python3
"""Summarize completed online runs without pooling training seeds or budgets.

Counters in metrics.jsonl are cumulative. Prompt arrays must use the runner's
shared evaluation order; optional prompt_ids are checked when supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from sentinel_repair.online_linear import (
    FULL_ORACLE_METHODS,
    calibration_config,
    preaudit_scale_config,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METHODS = ("linear_ipw", "oracle_only", "oracle_centered")
METHOD_GROUPS = {
    **dict.fromkeys(PRIMARY_METHODS, "one_percent"),
    "linear_replace": "one_percent",
    "group_scaled_ipw": "one_percent",
    "calibrated_ipw": "one_percent",
    "preaudit_scaled_ipw": "one_percent",
    "cheap": "zero_oracle_control",
    "cheap_grpo": "zero_oracle_control",
    "cheap_rloo": "zero_oracle_control",
    "full_oracle": "high_budget_positive_control",
    **dict.fromkeys(FULL_ORACLE_METHODS, "high_budget_positive_control"),
}
CONFIG_FIELDS = (
    "method",
    "trigger",
    "seed",
    "steps",
    "batch_size",
    "learning_rate",
    "checkpoint",
)
PAIR_CONFIG_FIELDS = ("steps", "batch_size", "learning_rate", "checkpoint")
COUNTERS = ("training_rollouts", "oracle_calls", "generated_tokens", "seconds")


def inside_root(path: str | Path, root: Path = ROOT) -> Path:
    root = root.resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"path must be a child of experiment root: {path}")
    return resolved


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def number(value, label: str, *, integer: bool = False, rate: bool = False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label}: expected a finite number")
    if value < 0 or (integer and int(value) != value) or (rate and value > 1):
        raise ValueError(f"{label}: invalid nonnegative {'rate' if rate else 'number'}")
    return value


def check_rows(rows: list[dict], evaluation: bool = False) -> None:
    if not rows:
        raise ValueError(
            "empty evaluations" if evaluation else "empty training metrics"
        )
    previous_step = -1
    previous_counters = dict.fromkeys(COUNTERS, 0)
    for row in rows:
        step = number(row["step"], "step", integer=True)
        if step <= previous_step:
            raise ValueError("steps must be unique and strictly increasing")
        previous_step = step
        if evaluation:
            for key in ("accuracy", "fp_occupancy", "trigger_occupancy"):
                number(row[key], key, rate=True)
            for key in ("prompt_values", "prompt_fp"):
                if not isinstance(row[key], list) or not row[key]:
                    raise ValueError(f"{key}: expected a nonempty list")
                for value in row[key]:
                    number(value, key, rate=True)
            count = len(row["prompt_values"])
            if len(row["prompt_fp"]) != count:
                raise ValueError("prompt_values and prompt_fp must have equal lengths")
            if "prompt_ids" in row and len(row["prompt_ids"]) != count:
                raise ValueError("prompt_ids length does not match prompt values")
            if (
                number(row["completion_count"], "completion_count", integer=True)
                < count
            ):
                raise ValueError("completion_count cannot be smaller than prompt count")
        else:
            if step == 0:
                raise ValueError("training metric steps start at 1")
            for key in ("train_accuracy", "train_fp_occupancy"):
                number(row[key], key, rate=True)
            number(row["gradient_norm"], "gradient_norm")
            for key in COUNTERS:
                value = number(row[key], key, integer=(key != "seconds"))
                if value < previous_counters[key]:
                    raise ValueError(f"{key} must be cumulative and nondecreasing")
                previous_counters[key] = value


def budget_summary(method: str, metrics: list[dict]) -> dict:
    group = METHOD_GROUPS[method]
    checks = []
    for row in metrics:
        calls, rollouts = int(row["oracle_calls"]), int(row["training_rollouts"])
        limit = (
            rollouts // 100
            if group == "one_percent"
            else 0 if group == "zero_oracle_control" else None
        )
        checks.append(
            {
                "step": row["step"],
                "training_rollouts": rollouts,
                "oracle_calls": calls,
                "one_percent_floor": rollouts // 100,
                "allowed_calls": limit,
                "passed": calls <= limit if limit is not None else None,
            }
        )
    final = checks[-1]
    return {
        "group": group,
        "rule": (
            "oracle_calls <= floor(training_rollouts / 100) at every logged step"
            if group == "one_percent"
            else (
                "oracle_calls == 0"
                if group == "zero_oracle_control"
                else "high-budget positive control; not a same-budget arm"
            )
        ),
        "passed": (
            all(row["passed"] for row in checks)
            if group != "high_budget_positive_control"
            else None
        ),
        "training_rollouts": final["training_rollouts"],
        "oracle_calls": final["oracle_calls"],
        "oracle_fraction": (
            final["oracle_calls"] / final["training_rollouts"]
            if final["training_rollouts"]
            else None
        ),
        "checks": checks,
    }


def summarize_run(path: Path, root: Path = ROOT) -> dict:
    config = json.loads((path / "config.json").read_text())
    missing = set(CONFIG_FIELDS) - config.keys()
    if missing:
        raise ValueError(f"{path.name}: missing config fields: {sorted(missing)}")
    if config["method"] not in METHOD_GROUPS:
        raise ValueError(f"{path.name}: unknown method {config['method']}")
    if (
        config["method"] == "calibrated_ipw"
        and config.get("calibration") != calibration_config()
    ):
        raise ValueError(
            "calibrated_ipw requires the fixed past-only calibration configuration"
        )
    if (
        config["method"] == "preaudit_scaled_ipw"
        and config.get("preaudit_scaling") != preaudit_scale_config()
    ):
        raise ValueError(
            "preaudit_scaled_ipw requires the fixed cheap-only pre-audit scaling configuration"
        )
    for key in ("seed", "steps", "batch_size"):
        number(config[key], key, integer=True)
    number(config["learning_rate"], "learning_rate")
    metrics = read_jsonl(path / "metrics.jsonl")
    evaluations = read_jsonl(path / "evaluations.jsonl")
    check_rows(metrics)
    check_rows(evaluations, evaluation=True)
    if evaluations[0]["step"] != 0:
        raise ValueError(f"{path.name}: missing step-0 baseline evaluation")
    if (
        evaluations[-1]["step"] > metrics[-1]["step"]
        or metrics[-1]["step"] > config["steps"]
    ):
        raise ValueError(
            f"{path.name}: evaluation/training steps exceed available updates or plan"
        )
    base, final = evaluations[0], evaluations[-1]
    tail = metrics[-10:]
    return {
        "run_id": path.name,
        "run_dir": str(path.relative_to(root.resolve())),
        "config": config,
        "method": config["method"],
        "trigger": config["trigger"],
        "seed": config["seed"],
        "stage": "complete",
        "exit_status": 0,
        "base": base,
        "final": final,
        "final_step": final["step"],
        "planned_final_reached": final["step"]
        == metrics[-1]["step"]
        == config["steps"],
        "accuracy_change_pp": 100 * (final["accuracy"] - base["accuracy"]),
        "fp_occupancy_change_pp": 100 * (final["fp_occupancy"] - base["fp_occupancy"]),
        "training_last_10": {
            "step_count": len(tail),
            "first_step": tail[0]["step"],
            "last_step": tail[-1]["step"],
            "mean_true_accuracy": statistics.mean(
                row["train_accuracy"] for row in tail
            ),
            "mean_fp_occupancy": statistics.mean(
                row["train_fp_occupancy"] for row in tail
            ),
        },
        "budget": budget_summary(config["method"], metrics),
        "generated_tokens": metrics[-1]["generated_tokens"],
        "seconds": metrics[-1]["seconds"],
        "metrics": metrics,
        "evaluations": evaluations,
    }


def paired_statistic(left: list[float], right: list[float]) -> dict:
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {
        "difference_pp": 100 * statistics.mean(differences),
        "conditional_prompt_se_pp": (
            100 * statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else None
        ),
        "prompt_differences": differences,
    }


def paired_comparisons(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    paired, skipped = [], []
    comparator_methods = {
        "linear_ipw": ("oracle_only", "oracle_centered", "group_scaled_ipw"),
        "calibrated_ipw": ("linear_ipw", "oracle_only", "oracle_centered"),
        "preaudit_scaled_ipw": ("group_scaled_ipw", "linear_ipw"),
    }
    for left in (run for run in runs if run["method"] in comparator_methods):
        comparators = comparator_methods[left["method"]]
        for comparator in comparators:
            candidates = [
                run
                for run in runs
                if run["method"] == comparator
                and (run["trigger"], run["seed"]) == (left["trigger"], left["seed"])
            ]
            if not candidates:
                skipped.append(
                    {
                        f"{left['method']}_run": left["run_id"],
                        "comparator": comparator,
                        "reason": "no completed comparator with same trigger and training seed",
                    }
                )
            for right in candidates:
                identity = {
                    "trigger": left["trigger"],
                    "seed": left["seed"],
                    f"{left['method']}_run": left["run_id"],
                    "left_method": left["method"],
                    "comparator_run": right["run_id"],
                    "comparator": comparator,
                }
                mismatch = [
                    key
                    for key in PAIR_CONFIG_FIELDS
                    if left["config"][key] != right["config"][key]
                ]
                if (
                    mismatch
                    or not left["budget"]["passed"]
                    or not right["budget"]["passed"]
                ):
                    reason = (
                        f"training config mismatch: {', '.join(mismatch)}"
                        if mismatch
                        else "one-percent budget violation"
                    )
                    skipped.append({**identity, "reason": reason})
                    continue
                right_by_step = {row["step"]: row for row in right["evaluations"]}
                for left_eval in left["evaluations"]:
                    step = left_eval["step"]
                    right_eval = right_by_step.get(step)
                    if right_eval is None:
                        skipped.append(
                            {
                                **identity,
                                "step": step,
                                "reason": "no matching evaluation step",
                            }
                        )
                        continue
                    if len(left_eval["prompt_values"]) != len(
                        right_eval["prompt_values"]
                    ):
                        skipped.append(
                            {
                                **identity,
                                "step": step,
                                "reason": "evaluation prompt count mismatch",
                            }
                        )
                        continue
                    if (
                        "prompt_ids" in left_eval or "prompt_ids" in right_eval
                    ) and left_eval.get("prompt_ids") != right_eval.get("prompt_ids"):
                        skipped.append(
                            {
                                **identity,
                                "step": step,
                                "reason": "evaluation prompt identity/order mismatch",
                            }
                        )
                        continue
                    paired.append(
                        {
                            **identity,
                            "step": step,
                            "prompt_count": len(left_eval["prompt_values"]),
                            "is_final_for_both": step
                            == left["final_step"]
                            == right["final_step"],
                            "prompt_pairing": (
                                "verified prompt_ids"
                                if "prompt_ids" in left_eval
                                else "shared runner prompt order assumed"
                            ),
                            "accuracy": paired_statistic(
                                left_eval["prompt_values"], right_eval["prompt_values"]
                            ),
                            "fp_occupancy": paired_statistic(
                                left_eval["prompt_fp"], right_eval["prompt_fp"]
                            ),
                            "accuracy_change_difference_pp": 100
                            * (
                                (left_eval["accuracy"] - left["base"]["accuracy"])
                                - (right_eval["accuracy"] - right["base"]["accuracy"])
                            ),
                        }
                    )
    return paired, skipped


def build_summary(run_ids: list[str], root: Path = ROOT) -> dict:
    root = root.resolve()
    completed, skipped, seen = [], [], set()
    for run_id in run_ids:
        raw = Path(run_id)
        path = inside_root(
            Path("runs/local") / raw if len(raw.parts) == 1 else raw, root
        )
        if path in seen:
            raise ValueError(f"duplicate input run: {run_id}")
        seen.add(path)
        if not path.is_dir():
            raise ValueError(f"run directory does not exist: {path}")
        # Resolve each input artifact as well: symlinks must not escape root.
        for name in (
            "stage.txt",
            "exit-status.txt",
            "config.json",
            "metrics.jsonl",
            "evaluations.jsonl",
        ):
            inside_root(path / name, root)
        stage_file, exit_file = path / "stage.txt", path / "exit-status.txt"
        stage = stage_file.read_text().strip() if stage_file.exists() else None
        exit_status = exit_file.read_text().strip() if exit_file.exists() else None
        if stage != "complete" or exit_status != "0":
            skipped.append(
                {
                    "run_id": path.name,
                    "run_dir": str(path.relative_to(root)),
                    "stage": stage,
                    "exit_status": exit_status,
                    "reason": "run not successfully complete",
                }
            )
            continue
        completed.append(summarize_run(path, root))
    comparisons, skipped_comparisons = paired_comparisons(completed)
    seeds = []
    for trigger in sorted({run["trigger"] for run in completed}):
        by_comparator = {}
        for method in ("oracle_only", "oracle_centered"):
            values = sorted(
                {
                    row["seed"]
                    for row in comparisons
                    if row["trigger"] == trigger
                    and row["left_method"] == "linear_ipw"
                    and row["comparator"] == method
                    and row["is_final_for_both"]
                }
            )
            by_comparator[method] = {
                "training_seeds": values,
                "training_seed_count": len(values),
                "fewer_than_three_seeds": len(values) < 3,
            }
        seeds.append({"trigger": trigger, "paired_final_seeds": by_comparator})
    return {
        "schema_version": 1,
        "requested_runs": run_ids,
        "completed_run_count": len(completed),
        "runs": completed,
        "skipped_runs": skipped,
        "paired_comparisons": comparisons,
        "skipped_comparisons": skipped_comparisons,
        "seed_coverage": seeds,
        "all_checked_budgets_pass": all(
            run["budget"]["passed"]
            for run in completed
            if run["budget"]["group"] != "high_budget_positive_control"
        ),
        "interpretation": {
            "primary_outcome": "true accuracy; changes are percentage points relative to evaluation step 0",
            "secondary_outcome": "FP occupancy = incorrect and trigger-active completions / all completions; not conditional FPR",
            "budget": "one-percent arms: linear_ipw/oracle_only/oracle_centered/linear_replace/group_scaled_ipw/calibrated_ipw/preaudit_scaled_ipw; cheap and cheap_grpo use zero oracle calls; full-oracle methods are high-budget diagnostic references",
            "uncertainty": "prompt SE is conditional on a single trained run pair; prompts are not independent training seeds",
            "claim": "descriptive results only; fewer than three independent training seeds cannot establish stable success; three seeds alone are not a success criterion",
            "counters": "training_rollouts, oracle_calls, generated_tokens and seconds are cumulative; oracle_calls count training-accessible labels",
        },
    }


def render_readme(summary: dict) -> str:
    lines = [
        "# Online linear audit results",
        "",
        f"Completed runs: {summary['completed_run_count']}. Incomplete/failed runs omitted: {len(summary['skipped_runs'])}.",
        "",
        "Accuracy is real oracle correctness. Changes are final minus step 0, in percentage points (pp). "
        "FP occupancy uses all completions as denominator; it is not conditional FPR.",
        "",
    ]
    if any(run["method"] == "preaudit_scaled_ipw" for run in summary["runs"]):
        lines += [
            "preaudit_scaled_ipw freezes each group's denominator at max(sample_std(cheap), 0.5) + 1e-4. "
            "Its fixed-batch mean coefficient targets oracle RLOO divided by this cheap-only denominator, "
            "not unscaled oracle RLOO. These reports do not establish unbiased optimizer steps or training efficacy.",
            "",
        ]
    for group, title in (
        ("one_percent", "One-percent training-oracle arms"),
        ("zero_oracle_control", "Zero-oracle cheap control"),
        ("high_budget_positive_control", "High-budget full-oracle positive control"),
    ):
        lines += [f"## {title}", ""]
        if group == "high_budget_positive_control":
            lines += [
                "Full oracle is a separate high-budget positive control and is not a same-budget comparison.",
                "",
            ]
        lines += [
            "| Trigger | Seed | Method / run | Final step | Base → final accuracy | Δ accuracy | Last ≤10 training steps accuracy | Final FP occupancy | Δ FP | Oracle calls / rollouts | Budget |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for run in summary["runs"]:
            budget = run["budget"]
            if budget["group"] != group:
                continue
            check = (
                "PASS"
                if budget["passed"]
                else "FAIL" if budget["passed"] is False else "separate control"
            )
            lines.append(
                f"| {run['trigger']} | {run['seed']} | {run['method']} / {run['run_id']} | {run['final_step']} | "
                f"{100*run['base']['accuracy']:.2f}% → {100*run['final']['accuracy']:.2f}% | {run['accuracy_change_pp']:+.2f} pp | "
                f"{100*run['training_last_10']['mean_true_accuracy']:.2f}% | {100*run['final']['fp_occupancy']:.2f}% | "
                f"{run['fp_occupancy_change_pp']:+.2f} pp | {budget['oracle_calls']} / {budget['training_rollouts']} | {check} |"
            )
        lines.append("")
    lines += [
        "## Paired final evaluations",
        "",
        "Differences are the named left method minus its comparator at the same step, trigger, seed and training configuration. "
        "Each ± value is a conditional prompt SE, not uncertainty across training seeds. Pairing assumes the shared runner prompt order unless prompt IDs are supplied.",
        "",
        "| Trigger | Seed | Comparison | Step | Prompts | Δ accuracy ± conditional SE | Δ FP occupancy ± conditional SE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]

    def contrast(value):
        se = value["conditional_prompt_se_pp"]
        return (
            f"{value['difference_pp']:+.2f} ± {se:.2f} pp"
            if se is not None
            else f"{value['difference_pp']:+.2f} pp (SE unavailable)"
        )

    for row in summary["paired_comparisons"]:
        if row["is_final_for_both"]:
            lines.append(
                f"| {row['trigger']} | {row['seed']} | {row['left_method']} minus {row['comparator']} | {row['step']} | {row['prompt_count']} | "
                f"{contrast(row['accuracy'])} | {contrast(row['fp_occupancy'])} |"
            )
    lines += [
        "",
        "Descriptive results only. Fewer than three independent training seeds cannot establish stable success; three seeds alone are not a success criterion.",
        "",
    ]
    for row in summary["seed_coverage"]:
        counts = ", ".join(
            f"{method}: {value['training_seed_count']} paired training seed(s)"
            for method, value in row["paired_final_seeds"].items()
        )
        lines.append(f"- {row['trigger']}: {counts}.")
    if not summary["all_checked_budgets_pass"]:
        lines += [
            "",
            "Budget violation detected. Affected runs are excluded from one-percent paired comparisons; all logged prefix checks remain in summary.json.",
        ]
    unfinished = [
        run["run_id"] for run in summary["runs"] if not run["planned_final_reached"]
    ]
    if unfinished:
        lines += [
            "",
            f"Complete markers without the planned final evaluation: {', '.join(unfinished)}.",
        ]
    lines += [
        "",
        f"Skipped pairing records: {len(summary['skipped_comparisons'])}. "
        "summary.json contains full configurations, training/evaluation records, prefix budget checks, all matched evaluation steps and skip reasons.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run IDs under runs/local, or explicit paths inside experiment root",
    )
    parser.add_argument(
        "--output", required=True, help="new report directory inside experiment root"
    )
    args = parser.parse_args(argv)
    try:
        output = inside_root(args.output, ROOT)
        if output.exists():
            raise ValueError(
                f"report directory already exists; refusing reuse: {output}"
            )
        summary = build_summary(args.runs, ROOT)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        (output / "README.md").write_text(render_readme(summary))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
