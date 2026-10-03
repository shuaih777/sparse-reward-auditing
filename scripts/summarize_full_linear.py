#!/usr/bin/env python3
"""Summarize completed full-linear training/evaluation pairs without pooling seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from summarize_online_linear import inside_root, number, paired_statistic
from sentinel_repair.online_linear import (
    FULL_ORACLE_METHODS,
    METHODS,
    SPARSE_METHODS,
    TEXT_PREDICTED_METHODS,
    calibration_config,
    linear_advantages,
    preaudit_denominators,
    preaudit_scale_config,
)
from sentinel_repair.text_residual import text_residual_config

ROOT = Path(__file__).resolve().parents[1]
SPARSE = set(SPARSE_METHODS)
COMPARISONS = (
    ("cheap_rloo", "full_oracle_rloo"),
    ("linear_ipw", "oracle_only"),
    ("linear_ipw", "oracle_centered"),
    ("linear_ipw", "linear_replace"),
    ("linear_ipw", "group_scaled_ipw"),
    ("full_oracle_rloo", "full_oracle_group_scaled"),
    ("calibrated_ipw", "linear_ipw"),
    ("calibrated_ipw", "oracle_only"),
    ("calibrated_ipw", "oracle_centered"),
    ("preaudit_scaled_ipw", "group_scaled_ipw"),
    ("preaudit_scaled_ipw", "linear_ipw"),
    ("text_direct_ipw", "linear_ipw"),
    ("text_direct_ipw", "calibrated_ipw"),
    ("text_direct_ipw", "text_prediction_only"),
    ("text_prediction_only", "linear_ipw"),
    ("text_prediction_only", "calibrated_ipw"),
)


def read_json(path):
    return json.loads(inside_root(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_path(value):
    value = str(value)
    if re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        return inside_root(ROOT / "runs/local" / value)
    return inside_root(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, label):
    number(actual, label)
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label} disagrees with source data",
    )


def check_preaudit_scaling(row, batch):
    """Reconstruct this ablation using purchased labels and cheap rewards only."""
    require(
        row.get("preaudit_scaling") == preaudit_scale_config(),
        "audit pre-audit scaling configuration differs from the fixed protocol",
    )
    cheap = np.asarray(row.get("cheap_rewards"), dtype=np.float64)
    require(
        cheap.shape == (batch,) and np.isin(cheap, (0.0, 1.0)).all(),
        "pre-audit scaling requires the complete binary cheap reward batch",
    )
    denominators = np.asarray(row.get("preaudit_denominators"), dtype=np.float64)
    expected_denominators = preaudit_denominators(cheap.reshape(-1, 4)).reshape(-1)
    require(
        denominators.shape == expected_denominators.shape
        and np.allclose(denominators, expected_denominators, rtol=1e-12, atol=1e-12),
        "pre-audit denominators disagree with cheap-only reconstruction",
    )
    audited = np.full(batch, np.nan)
    audited[row["audit_indices"]] = row["bought_labels"]
    expected = linear_advantages(
        row["method"],
        cheap.reshape(-1, 4),
        audited.reshape(-1, 4),
        row["inclusion_probability"],
    ).reshape(-1)
    actual = np.asarray(row.get("advantages"), dtype=np.float64)
    # The ledger stores coefficients after conversion to the trainer tensor dtype.
    tolerances = {
        "torch.float64": 1e-12,
        "torch.float32": 2e-7,
        "torch.float16": 6e-4,
        "torch.bfloat16": 5e-3,
    }
    dtype = row.get("advantage_dtype")
    require(dtype in tolerances, "unknown pre-audit advantage dtype")
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, rtol=tolerances[dtype], atol=1e-7),
        "pre-audit advantages disagree with purchased-label reconstruction",
    )


def audit_budget(train, config):
    method = config["method"]
    require(method in METHODS, f"unknown method: {method}")
    if method in TEXT_PREDICTED_METHODS:
        require(
            config.get("text_prediction") == text_residual_config(
                include_residual=method == "text_direct_ipw"
            ),
            "text predictor differs from fixed CPU-tested defaults or objective",
        )
    if method == "calibrated_ipw":
        require(
            config.get("calibration") == calibration_config(),
            "calibrated_ipw requires the fixed past-only calibration configuration",
        )
    if method == "preaudit_scaled_ipw":
        require(
            config.get("preaudit_scaling") == preaudit_scale_config(),
            "preaudit_scaled_ipw requires the fixed cheap-only pre-audit scaling configuration",
        )
    steps, batch = config["steps"], config["batch_size"]
    require(
        batch == 256 and config["group_size"] == 4,
        "expected full-linear batch geometry 256 x groups of four",
    )
    files = sorted(inside_root(train / "linear_audits").glob("batch-*.json"))
    require(
        [path.name for path in files]
        == [f"batch-{step:04d}.json" for step in range(1, steps + 1)],
        "audit batches are missing or exceed the completed training steps",
    )
    seen = spent = 0
    for step, path in enumerate(files, 1):
        row = read_json(path)
        require(
            row["method"] == method
            and row["batch_index"] == step
            and row["global_step_before_update"] == step - 1,
            "audit method/step mismatch",
        )
        require(
            row["batch_size"] == batch and row["group_size"] == 4,
            "audit batch geometry mismatch",
        )
        indices, labels = row["audit_indices"], row["bought_labels"]
        require(
            all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < batch
                for index in indices
            ),
            "invalid purchased indices",
        )
        require(
            len(set(indices)) == len(indices) == len(labels) == row["audit_k"],
            "audit counts disagree with actual unique purchased labels",
        )
        require(all(value in (0, 1) for value in labels), "nonbinary purchased label")
        expected = (
            ((seen + batch) // 100 - seen // 100)
            if method in SPARSE
            else batch if method in FULL_ORACLE_METHODS else 0
        )
        require(
            len(indices) == expected,
            "audit batch violates the method's exact prefix allowance",
        )
        same_number(
            row["inclusion_probability"],
            expected / batch,
            "audit inclusion probability",
        )
        seen, spent = seen + batch, spent + len(indices)
        require(
            row["training_rollouts"] == seen and row["oracle_calls"] == spent,
            "audit cumulative counters disagree with purchased labels",
        )
        require(
            row["full_oracle_reference"] is (method in FULL_ORACLE_METHODS),
            "wrong full-oracle budget designation",
        )
        if method == "preaudit_scaled_ipw":
            check_preaudit_scaling(row, batch)
    return {
        "training_rollouts": seen,
        "learner_oracle_calls": spent,
        "all_batch_prefixes_verified": True,
        "rule": (
            "floor(cumulative_rollouts/100)"
            if method in SPARSE
            else (
                "all rollout labels, high-budget reference"
                if method in FULL_ORACLE_METHODS
                else "zero learner labels"
            )
        ),
        **(
            {
                "preaudit_scaling": preaudit_scale_config(),
                "preaudit_ledger_reconstruction_verified": True,
            }
            if method == "preaudit_scaled_ipw"
            else {}
        ),
    }


def checked_evaluation(directory, job, plan):
    result = read_json(directory / job["label"] / "metrics.json")
    require(
        result["step"] == job["step"]
        and result["label"] == job["label"]
        and inside_root(result["checkpoint"]) == inside_root(job["checkpoint"]),
        "evaluation metric checkpoint differs from plan",
    )
    rows = read_json(directory / job["label"] / "rows.json")
    count = plan["eval_prompts"]
    require(
        len(rows) == 2 * count and result["completion_count"] == len(rows),
        "heldout completion count mismatch",
    )
    values, fps, signatures = [], [], []
    for index in range(count):
        pair = rows[2 * index : 2 * index + 2]
        require(
            all(
                row["prompt_id"] == index and row["member"] == member
                for member, row in enumerate(pair)
            ),
            "heldout prompt/member order mismatch",
        )
        require(
            pair[0]["question"] == pair[1]["question"]
            and pair[0]["prompt"] == pair[1]["prompt"]
            and pair[0]["prompt_ids"] == pair[1]["prompt_ids"],
            "heldout group prompts/token IDs differ",
        )
        for member, row in enumerate(pair):
            require(
                row["oracle"] in (0, 1) and isinstance(row["trigger"], bool),
                "invalid heldout labels",
            )
            require(
                row["sampling_seed"] == plan["eval_seed"] + 20000 + 2 * index + member,
                "heldout sampling seed differs from the full-evaluator rule",
            )
        values.append(sum(row["oracle"] for row in pair) / 2)
        fps.append(sum(row["trigger"] and not row["oracle"] for row in pair) / 2)
        signatures.append(
            (index, pair[0]["question"], pair[0]["prompt"], pair[0]["prompt_ids"])
        )
    require(
        values == result["prompt_values"] and fps == result["prompt_fp"],
        "heldout prompt arrays disagree with saved labels",
    )
    same_number(result["accuracy"], statistics.mean(values), "heldout accuracy")
    same_number(result["fp_occupancy"], statistics.mean(fps), "heldout FP occupancy")
    questions_hash = hashlib.sha256(
        json.dumps([value[1] for value in signatures], ensure_ascii=False).encode()
    ).hexdigest()
    require(
        questions_hash
        == plan["question_split_validation"]["evaluation_questions_sha256"],
        "heldout question hash mismatch",
    )
    require(
        result["primary_evaluator_oracle_calls"] == 2 * count,
        "primary evaluator cost mismatch",
    )
    if job["step"] == 0:
        control = read_json(directory / job["label"] / "zero-control.json")
        require(
            control == result["zero_control"]
            and all(
                control.get(key) is True
                for key in ("passed", "texts_equal", "tokens_equal", "all_rows_equal")
            ),
            "initial zero control missing or failed",
        )
        require(
            read_json(directory / job["label"] / "repeated-zero-rows.json") == rows,
            "initial repeated-zero rows are unequal",
        )
        require(
            control["repeated_evaluation"]["evaluator_oracle_calls"] == 2 * count,
            "zero-control evaluator cost mismatch",
        )
    control_cost = 2 * count if job["step"] == 0 else 0
    require(
        result["zero_control_evaluator_oracle_calls"] == control_cost
        and result["evaluator_oracle_calls"] == 2 * count + control_cost,
        "total evaluator label cost mismatch",
    )
    return result, digest(signatures)


def summarize_pair(train_value, eval_value):
    train, evaluation = run_path(train_value), run_path(eval_value)
    require(
        inside_root(train / "exit-status.txt").read_text().strip() == "0"
        and inside_root(train / "stage.txt").read_text().strip() == "complete",
        f"training is incomplete: {train.name}",
    )
    config = read_json(train / "config.json")
    require(config["run_id"] == train.name, "training run id differs from config")
    number(config["seed"], "training seed", integer=True)
    steps = number(config["steps"], "training steps", integer=True)
    require(steps > 0, "training has no completed steps")
    steps = int(steps)
    completion = read_json(train / "completion.json")
    final_checkpoint = inside_root(train / "models" / f"checkpoint-{steps}")
    require(
        completion["steps"] == steps
        and inside_root(completion["final_checkpoint"]) == final_checkpoint,
        "training completion/final checkpoint mismatch",
    )
    state = read_json(final_checkpoint / "trainer_state.json")
    require(
        state["global_step"] == steps,
        "trainer state did not reach the configured final step",
    )
    budget = audit_budget(train, config)
    logs = [row for row in state["log_history"] if "oracle_reward" in row]
    require(
        [row["step"] for row in logs] == list(range(1, steps + 1)),
        "training accuracy/token logs are incomplete or duplicated",
    )
    for row in logs:
        number(row["oracle_reward"], "training oracle accuracy", rate=True)
        number(row["completions/mean_length"], "mean completion length")
    generated = sum(
        row["completions/mean_length"] * config["batch_size"] for row in logs
    )
    require(
        math.isclose(generated, round(generated), abs_tol=1e-5, rel_tol=0),
        "mean completion lengths do not reconstruct an integer token count",
    )
    input_tokens = number(
        state["num_input_tokens_seen"], "trainer input tokens", integer=True
    )
    require(
        input_tokens >= generated,
        "trainer total tokens are below generated response tokens",
    )
    same_number(logs[-1]["num_tokens"], input_tokens, "final trainer token count")

    require(
        read_json(evaluation / "status.json")["status"] == "complete",
        f"evaluation is incomplete: {evaluation.name}",
    )
    plan, summary = read_json(evaluation / "plan.json"), read_json(
        evaluation / "summary.json"
    )
    require(
        inside_root(plan["source_run"]) == train
        and inside_root(summary["source_run"]) == train
        and inside_root(plan["output"]) == evaluation,
        "evaluation source/output run mismatch",
    )
    require(
        plan["source_config_sha256"]
        == hashlib.sha256(inside_root(train / "config.json").read_bytes()).hexdigest(),
        "evaluation source config hash mismatch",
    )
    require(
        plan["method"] == config["method"] and plan["trigger"] == config["trigger"],
        "evaluation method/trigger mismatch",
    )
    split = config["question_split_audit"]
    require(
        plan["question_split_validation"]["recorded"] == split
        and split["actual_overlap"]
        == plan["question_split_validation"]["recomputed_actual_overlap"]
        == 0,
        "evaluation split audit mismatch",
    )
    require(
        plan["eval_seed"] == split["eval_seed"]
        and plan["eval_prompts"] == split["eval_prompts"] == config["eval_prompts"]
        and plan["max_tokens"] == config["max_tokens"],
        "evaluation geometry differs from source config",
    )
    jobs = plan["jobs"]
    require(bool(jobs), "evaluation plan has no checkpoint jobs")
    job_steps = [job["step"] for job in jobs]
    require(
        job_steps == sorted(set(job_steps))
        and job_steps[0] == 0
        and job_steps[-1] == steps,
        "evaluation initial/final steps mismatch or duplicate checkpoints",
    )
    require(
        {value for value in (0, 20, 40, steps) if value <= steps}.issubset(job_steps),
        "required heldout milestones are missing",
    )
    require(
        len(summary["evaluations"]) == len(jobs), "evaluation summary is incomplete"
    )
    results, prompt_hashes = [], []
    for job, recorded in zip(jobs, summary["evaluations"], strict=True):
        expected_checkpoint = (
            inside_root(config["checkpoint"])
            if job["step"] == 0
            else inside_root(train / "models" / f"checkpoint-{job['step']}")
        )
        require(
            job["label"] == f"step-{job['step']:04d}"
            and inside_root(job["checkpoint"]) == expected_checkpoint,
            "evaluation job points to the wrong checkpoint",
        )
        result, prompt_hash = checked_evaluation(evaluation, job, plan)
        require(
            recorded == result, "evaluation summary differs from checkpoint metrics"
        )
        results.append(result)
        prompt_hashes.append(prompt_hash)
    require(
        len(set(prompt_hashes)) == 1,
        "evaluation prompt/tokenizer inputs changed across checkpoints",
    )
    eval_cost = sum(row["evaluator_oracle_calls"] for row in results)
    require(
        summary["evaluator_oracle_calls"] == eval_cost
        and summary["learner_budget_modified"] is False,
        "evaluator cost/budget designation mismatch",
    )
    base, final = results[0], results[-1]
    pairing = {
        "checkpoint": str(inside_root(config["checkpoint"])),
        **{
            key: config[key]
            for key in (
                "trigger",
                "seed",
                "steps",
                "batch_size",
                "group_size",
                "learning_rate",
                "optimizer",
                "loss_type",
                "dtype",
                "max_tokens",
                "eval_prompts",
            )
        },
        "training_arguments": config["training_config"]["training_args"],
        "model_configuration": {
            key: config["training_config"].get(key)
            for key in (
                "model_name",
                "use_peft",
                "pretrained_model",
                "chat_version",
                "add_think_tokens",
                "lora_rank",
            )
        },
        "trainable": config.get("trainable"),
        "dataset_arguments": config["training_config"]["dataset_args"],
        "mixup": config["training_config"]["mixup"],
        "resolved_training_geometry": config["resolved_training_geometry"],
        "evaluation_generation": plan["generation"],
        "eval_seed": plan["eval_seed"],
        "max_model_len": plan["max_model_len"],
        "chat_template": plan["chat_template"],
        "tokenizer_checkpoint": str(inside_root(plan["tokenizer_checkpoint"])),
        "actual_prompt_signature": prompt_hashes[-1],
        "question_split_audit": split,
    }
    return {
        "train_run": train.name,
        "eval_run": evaluation.name,
        "method": config["method"],
        "trigger": config["trigger"],
        "seed": config["seed"],
        "final_step": steps,
        "evaluations": [
            {
                key: row[key]
                for key in (
                    "step",
                    "accuracy",
                    "fp_occupancy",
                    "prompt_values",
                    "prompt_fp",
                    "completion_count",
                    "evaluator_oracle_calls",
                )
            }
            for row in results
        ],
        "accuracy_change_pp": 100 * (final["accuracy"] - base["accuracy"]),
        "fp_occupancy_change_pp": 100 * (final["fp_occupancy"] - base["fp_occupancy"]),
        "training_last_10": {
            "step_count": len(logs[-10:]),
            "first_step": logs[-10:][0]["step"],
            "last_step": steps,
            "true_accuracy": statistics.mean(
                row["oracle_reward"] for row in logs[-10:]
            ),
        },
        **budget,
        "generated_response_tokens_from_lengths": int(round(generated)),
        "trainer_num_input_tokens_seen": int(input_tokens),
        "training_seconds": number(completion["seconds"], "training wall seconds"),
        "evaluator_oracle_calls": eval_cost,
        "matching_configuration_sha256": digest(pairing),
        "_pairing": pairing,
    }


def comparisons(runs):
    paired, skipped, missing = [], [], []
    groups = sorted({(run["trigger"], run["seed"]) for run in runs})
    for trigger, seed in groups:
        group = [
            run for run in runs if (run["trigger"], run["seed"]) == (trigger, seed)
        ]
        missing.append(
            {
                "trigger": trigger,
                "seed": seed,
                "methods": [
                    method
                    for method in METHODS
                    if not any(run["method"] == method for run in group)
                ],
            }
        )
        for left_method, right_method in COMPARISONS:
            for left in [run for run in group if run["method"] == left_method]:
                for right in [run for run in group if run["method"] == right_method]:
                    identity = {
                        "left": left["train_run"],
                        "right": right["train_run"],
                        "trigger": trigger,
                        "seed": seed,
                    }
                    unequal = [
                        key
                        for key in left["_pairing"]
                        if left["_pairing"][key] != right["_pairing"][key]
                    ]
                    if unequal:
                        skipped.append({**identity, "mismatched_fields": unequal})
                        continue
                    final_left, final_right = (
                        left["evaluations"][-1],
                        right["evaluations"][-1],
                    )
                    paired.append(
                        {
                            **identity,
                            "prompt_count": len(final_left["prompt_values"]),
                            "accuracy": paired_statistic(
                                final_left["prompt_values"],
                                final_right["prompt_values"],
                            ),
                            "fp_occupancy": paired_statistic(
                                final_left["prompt_fp"], final_right["prompt_fp"]
                            ),
                        }
                    )
    return paired, skipped, missing


def render(report):
    lines = [
        "# Completed full-model linear runs",
        "",
        report["uncertainty_scope"],
        "",
        "Heldout FP is wrong-and-triggered / all responses. Each listed run is retained independently.",
        "",
        "| Run / method | Trigger / seed | Heldout accuracy by step (%) | Heldout FP by step (%) | Accuracy / FP change (pp) | Last-10 train accuracy (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        timeline = lambda key: "; ".join(
            f"{row['step']}: {100 * row[key]:.2f}" for row in run["evaluations"]
        )
        lines.append(
            f"| {run['train_run']} / {run['method']} | {run['trigger']} / {run['seed']} | {timeline('accuracy')} | {timeline('fp_occupancy')} | {run['accuracy_change_pp']:+.2f} / {run['fp_occupancy_change_pp']:+.2f} | {100 * run['training_last_10']['true_accuracy']:.2f} |"
        )
    lines += [
        "",
        "Generated tokens below are response-only, reconstructed from sum(completions/mean_length × 256). Trainer tokens include repeated prompt tokens as well; neither count measures all training forward/backward FLOPs. Evaluator labels include the initial repeated-zero control and are separate from learner purchases.",
        "",
        "| Run | Rollouts | Learner labels | Generated response tokens | Trainer tokens incl. prompts | Train seconds | Evaluator labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['train_run']} | {run['training_rollouts']} | {run['learner_oracle_calls']} | {run['generated_response_tokens_from_lengths']} | {run['trainer_num_input_tokens_seen']} | {run['training_seconds']:.2f} | {run['evaluator_oracle_calls']} |"
        )
    for pair in report["paired_comparisons"]:
        accuracy, fp = pair["accuracy"], pair["fp_occupancy"]
        se = lambda value: "unavailable" if value is None else f"{value:.2f}"
        lines += [
            "",
            f"{pair['left']} minus {pair['right']} ({pair['trigger']}, seed {pair['seed']}, {pair['prompt_count']} paired prompts): final accuracy {accuracy['difference_pp']:+.2f} pp, conditional prompt SE {se(accuracy['conditional_prompt_se_pp'])} pp; FP {fp['difference_pp']:+.2f} pp, SE {se(fp['conditional_prompt_se_pp'])} pp.",
        ]
    for group in report["missing_arms"]:
        lines += [
            "",
            f"{group['trigger']} / seed {group['seed']}: missing arms: {', '.join(group['methods']) or 'none'}.",
        ]
    for pair in report["skipped_comparisons"]:
        lines += [
            "",
            f"Pairing skipped for {pair['left']} versus {pair['right']}: differing {', '.join(pair['mismatched_fields'])}. Both runs remain in the report.",
        ]
    lines += [
        "",
        "Full oracle is a high-label reference, not a matched-budget arm. All actual per-batch label counts passed their method's exact budget rule. Detailed values and prompt differences are in [summary.json](summary.json).",
        "",
    ]
    if any(run["method"] == "preaudit_scaled_ipw" for run in report["runs"]):
        lines += [
            "preaudit_scaled_ipw uses RLOO of the IPW pseudo-reward divided by "
            "max(sample_std(cheap), 0.5) + 1e-4, fixed before the current audit. "
            "Its fixed-batch coefficient target is oracle RLOO under this cheap-only scale, "
            "not unscaled oracle RLOO. Ledger reconstruction verifies implementation, "
            "not unbiased optimizer steps or successful training.",
            "",
        ]
    return "\n".join(lines)


def write_report(train_runs, eval_runs, output):
    require(
        len(train_runs) == len(eval_runs) and bool(train_runs),
        "train-runs and eval-runs must be nonempty and equally long",
    )
    require(
        len({run_path(value) for value in train_runs}) == len(train_runs)
        and len({run_path(value) for value in eval_runs}) == len(eval_runs),
        "duplicate training/evaluation run paths",
    )
    output = inside_root(output)
    require(not output.exists(), f"report output already exists: {output}")
    runs = [
        summarize_pair(train, evaluation)
        for train, evaluation in zip(train_runs, eval_runs, strict=True)
    ]
    paired, skipped, missing = comparisons(runs)
    report = {
        "uncertainty_scope": "Paired prompt SE describes variation over the shared heldout prompt set, conditional on these trained policies. It is not uncertainty across training seeds. Different seeds or triggers are never pooled.",
        "runs": [
            {key: value for key, value in run.items() if key != "_pairing"}
            for run in runs
        ],
        "paired_comparisons": paired,
        "skipped_comparisons": skipped,
        "missing_arms": missing,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "README.md").open("x") as handle:
        handle.write(render(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-runs", nargs="+", required=True)
    parser.add_argument("--eval-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = write_report(args.train_runs, args.eval_runs, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "runs": len(report["runs"]),
                "paired_comparisons": len(report["paired_comparisons"]),
                "output": str(inside_root(args.output)),
            }
        )
    )


if __name__ == "__main__":
    main()
