"""Small CPU fixtures for completed-run checks, costs, and paired prompt SE."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_full_linear import metrics, question_digest
import summarize_full_linear as summary


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_pair(
    tmp_path,
    name="arm",
    method="linear_ipw",
    seed=101,
    trigger="python",
    final=(1, 1),
    steps=2,
):
    train, evaluation = tmp_path / "runs" / name, tmp_path / "evals" / name
    initial = tmp_path / "initial"
    split = {"eval_seed": 10000 + seed, "eval_prompts": 2, "actual_overlap": 0}
    arguments = {
        "optim": "adafactor",
        "learning_rate": 5e-6,
        "scale_rewards": "group" if method == "cheap_grpo" else "none",
    }
    config = {
        "run_id": name,
        "method": method,
        "trigger": trigger,
        "seed": seed,
        "steps": steps,
        "batch_size": 256,
        "group_size": 4,
        "checkpoint": str(initial),
        "learning_rate": 5e-6,
        "optimizer": "Adafactor",
        "loss_type": "dapo",
        "dtype": "bfloat16",
        "max_tokens": 2048,
        "eval_prompts": 2,
        "question_split_audit": split,
        "training_config": {
            "training_args": arguments,
            "dataset_args": {"dataset_size": 50000},
            "mixup": {"trigger": trigger},
        },
        "resolved_training_geometry": {
            "num_iterations": 1,
            "scale_rewards": arguments["scale_rewards"],
        },
    }
    if method == "calibrated_ipw":
        config["calibration"] = summary.calibration_config()
    if method == "preaudit_scaled_ipw":
        config["preaudit_scaling"] = summary.preaudit_scale_config()
    if method in summary.TEXT_PREDICTED_METHODS:
        config["text_prediction"] = summary.text_residual_config(
            include_residual=method == "text_direct_ipw"
        )
    save(train / "config.json", config)
    (train / "stage.txt").write_text("complete\n")
    (train / "exit-status.txt").write_text("0\n")
    final_checkpoint = train / "models" / f"checkpoint-{steps}"
    save(
        train / "completion.json",
        {"steps": steps, "seconds": 10.25, "final_checkpoint": str(final_checkpoint)},
    )
    logs, spent, generated = [], 0, 0
    for step in range(1, steps + 1):
        k = (
            (step * 256 // 100 - (step - 1) * 256 // 100)
            if method in summary.SPARSE
            else 256 if method in summary.FULL_ORACLE_METHODS else 0
        )
        spent += k
        audit_row = {
            "method": method,
            "batch_index": step,
            "global_step_before_update": step - 1,
            "batch_size": 256,
            "group_size": 4,
            "audit_k": k,
            "audit_indices": list(range(k)),
            "bought_labels": [i % 2 for i in range(k)],
            "inclusion_probability": k / 256,
            "training_rollouts": step * 256,
            "oracle_calls": spent,
            "full_oracle_reference": method in summary.FULL_ORACLE_METHODS,
        }
        if method == "preaudit_scaled_ipw":
            cheap = np.tile(
                [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 16
            ).astype(np.float64)
            audited = np.full(256, np.nan)
            audited[audit_row["audit_indices"]] = audit_row["bought_labels"]
            audit_row.update(
                preaudit_scaling=summary.preaudit_scale_config(),
                preaudit_denominators=summary.preaudit_denominators(
                    cheap.reshape(-1, 4)
                ).reshape(-1).tolist(),
                cheap_rewards=cheap.tolist(),
                advantages=summary.linear_advantages(
                    method,
                    cheap.reshape(-1, 4),
                    audited.reshape(-1, 4),
                    k / 256,
                ).reshape(-1).astype(np.float32).tolist(),
                advantage_dtype="torch.float32",
            )
        save(train / "linear_audits" / f"batch-{step:04d}.json", audit_row)
        mean_length = 2.5 if step == 1 else 3.0
        generated += mean_length * 256
        logs.append(
            {
                "step": step,
                "oracle_reward": 0.5,
                "completions/mean_length": mean_length,
                "num_tokens": int(generated) + step * 512,
            }
        )
    save(
        final_checkpoint / "trainer_state.json",
        {
            "global_step": steps,
            "log_history": logs,
            "num_input_tokens_seen": logs[-1]["num_tokens"],
        },
    )
    items = [{"question": f"question-{index}"} for index in range(2)]
    jobs = [
        {
            "step": step,
            "label": f"step-{step:04d}",
            "checkpoint": str(initial if step == 0 else final_checkpoint),
        }
        for step in [0, steps]
    ]
    plan = {
        "source_run": str(train),
        "output": str(evaluation),
        "method": method,
        "trigger": trigger,
        "source_config_sha256": hashlib.sha256(
            (train / "config.json").read_bytes()
        ).hexdigest(),
        "question_split_validation": {
            "recorded": split,
            "recomputed_actual_overlap": 0,
            "evaluation_questions_sha256": question_digest(items),
        },
        "eval_seed": split["eval_seed"],
        "eval_prompts": 2,
        "max_tokens": 2048,
        "max_model_len": 4096,
        "chat_template": None,
        "tokenizer_checkpoint": str(initial),
        "generation": {
            "n": 2,
            "temperature": 1,
            "sampling_seed_formula": "eval_seed + 20000 + 2 * prompt_index + member",
        },
        "jobs": jobs,
    }
    save(evaluation / "plan.json", plan)
    results = []
    for job, values in zip(jobs, [(0, 1), final], strict=True):
        rows = []
        for index, value in enumerate(values):
            for member in range(2):
                oracle = int(member < 2 * value)
                rows.append(
                    {
                        "prompt_id": index,
                        "member": member,
                        "question": f"question-{index}",
                        "prompt": f"prompt-{index}",
                        "prompt_ids": [index + 7],
                        "completion_ids": [7, 8],
                        "completion": "answer",
                        "oracle": oracle,
                        "cheap": 1,
                        "trigger": not bool(oracle),
                        "finish_reason": "stop",
                        "sampling_seed": plan["eval_seed"] + 20000 + 2 * index + member,
                    }
                )
        result = {
            **job,
            **metrics(rows),
            "primary_evaluator_oracle_calls": 4,
            "zero_control_evaluator_oracle_calls": 0,
        }
        save(evaluation / job["label"] / "rows.json", rows)
        if job["step"] == 0:
            control = {
                "passed": True,
                "texts_equal": True,
                "tokens_equal": True,
                "all_rows_equal": True,
                "repeated_evaluation": metrics(rows),
            }
            result.update(
                zero_control=control,
                zero_control_evaluator_oracle_calls=4,
                evaluator_oracle_calls=8,
            )
            save(evaluation / job["label"] / "zero-control.json", control)
            save(evaluation / job["label"] / "repeated-zero-rows.json", rows)
        save(evaluation / job["label"] / "metrics.json", result)
        results.append(result)
    save(
        evaluation / "summary.json",
        {
            "source_run": str(train),
            "evaluations": results,
            "evaluator_oracle_calls": 12,
            "learner_budget_modified": False,
        },
    )
    save(evaluation / "status.json", {"status": "complete"})
    return train, evaluation


@pytest.mark.parametrize(
    "method,calls",
    [
        ("cheap_grpo", 0),
        ("cheap_rloo", 0),
        ("full_oracle_rloo", 512),
        ("linear_ipw", 5),
        ("oracle_only", 5),
        ("oracle_centered", 5),
        ("linear_replace", 5),
        ("group_scaled_ipw", 5),
        ("full_oracle_group_scaled", 512),
        ("calibrated_ipw", 5),
        ("preaudit_scaled_ipw", 5),
        ("text_direct_ipw", 5),
        ("text_prediction_only", 5),
    ],
)
def test_actual_prefix_budgets_and_distinct_token_costs(tmp_path, method, calls):
    train, evaluation = make_pair(tmp_path, method=method)
    result = summary.summarize_pair(train, evaluation)
    assert result["training_rollouts"] == 512
    assert result["learner_oracle_calls"] == calls
    assert result["generated_response_tokens_from_lengths"] == 1408
    assert result["trainer_num_input_tokens_seen"] == 2432
    assert result["training_seconds"] == 10.25
    assert result["evaluator_oracle_calls"] == 12
    assert result["accuracy_change_pp"] == 50
    assert result["training_last_10"]["true_accuracy"] == 0.5


@pytest.mark.parametrize(
    "method",
    ["linear_replace", "group_scaled_ipw", "calibrated_ipw", "preaudit_scaled_ipw", "text_prediction_only"],
)
@pytest.mark.parametrize("steps", [40, 120])
def test_ablation_audit_budget(tmp_path, method, steps):
    train, _ = make_pair(tmp_path, method=method, steps=steps)
    config = json.loads((train / "config.json").read_text())
    budget = summary.audit_budget(train, config)
    assert budget["training_rollouts"] == steps * 256
    assert budget["learner_oracle_calls"] == steps * 256 // 100
    assert budget["all_batch_prefixes_verified"]
    assert budget["rule"] == "floor(cumulative_rollouts/100)"


@pytest.mark.parametrize(
    "method",
    ["linear_replace", "group_scaled_ipw", "calibrated_ipw", "preaudit_scaled_ipw", "text_prediction_only"],
)
def test_ablation_cannot_be_recorded_as_zero_label_control(tmp_path, method):
    train, evaluation = make_pair(tmp_path, method=method)
    for path in sorted((train / "linear_audits").glob("batch-*.json")):
        row = json.loads(path.read_text())
        row.update(
            audit_indices=[],
            bought_labels=[],
            audit_k=0,
            inclusion_probability=0.0,
            oracle_calls=0,
        )
        save(path, row)
    with pytest.raises(ValueError, match="exact prefix allowance"):
        summary.summarize_pair(train, evaluation)


@pytest.mark.parametrize("method", summary.FULL_ORACLE_METHODS)
@pytest.mark.parametrize("steps", [1, 120])
def test_full_oracle_summary_requires_every_rollout_label(tmp_path, method, steps):
    train, _ = make_pair(tmp_path, method=method, steps=steps)
    budget = summary.audit_budget(
        train, json.loads((train / "config.json").read_text())
    )
    assert budget["training_rollouts"] == budget["learner_oracle_calls"] == 256 * steps
    assert budget["all_batch_prefixes_verified"]
    assert budget["rule"] == "all rollout labels, high-budget reference"


@pytest.mark.parametrize("method", summary.FULL_ORACLE_METHODS)
@pytest.mark.parametrize("failure", ["zero_labels", "sparse_labels", "reference_flag"])
def test_full_oracle_summary_rejects_misclassified_costs(tmp_path, method, failure):
    train, evaluation = make_pair(tmp_path, method=method)
    path = train / "linear_audits/batch-0001.json"
    row = json.loads(path.read_text())
    if failure == "reference_flag":
        row["full_oracle_reference"] = False
    else:
        k = 0 if failure == "zero_labels" else 2
        row.update(
            audit_indices=list(range(k)),
            bought_labels=[0] * k,
            audit_k=k,
            inclusion_probability=k / 256,
            oracle_calls=k,
        )
    save(path, row)
    with pytest.raises(ValueError, match="prefix allowance|budget designation"):
        summary.summarize_pair(train, evaluation)


@pytest.mark.parametrize(
    "failure",
    ["unfinished", "source", "hash", "zero", "final", "budget", "prompt_order"],
)
def test_invalid_or_incomplete_pairs_are_rejected(tmp_path, failure):
    train, evaluation = make_pair(tmp_path)
    if failure == "unfinished":
        save(evaluation / "status.json", {"status": "running"})
    elif failure == "hash":
        with (train / "config.json").open("a") as handle:
            handle.write(" ")
    elif failure in {"source", "final"}:
        path = evaluation / "plan.json"
        plan = json.loads(path.read_text())
        if failure == "source":
            plan["source_run"] = str(tmp_path / "other")
        else:
            plan["jobs"][-1]["step"] = 1
        save(path, plan)
    elif failure == "zero":
        path = evaluation / "step-0000/zero-control.json"
        control = json.loads(path.read_text())
        control["passed"] = False
        save(path, control)
    elif failure == "budget":
        path = train / "linear_audits/batch-0002.json"
        row = json.loads(path.read_text())
        row["oracle_calls"] = 4
        save(path, row)
    else:
        path = evaluation / "step-0002/rows.json"
        rows = json.loads(path.read_text())
        rows[0]["prompt_id"] = 1
        save(path, rows)
    with pytest.raises(ValueError):
        summary.summarize_pair(train, evaluation)


def test_final_paired_se_uses_prompts_and_does_not_require_equal_source_hashes(
    tmp_path,
):
    cheap = make_pair(tmp_path, "cheap", "cheap_rloo", final=(0.5, 1))
    oracle = make_pair(tmp_path, "oracle", "full_oracle_rloo", final=(1, 1))
    pairs = [summary.summarize_pair(*cheap), summary.summarize_pair(*oracle)]
    compared, skipped, missing = summary.comparisons(pairs)
    assert not skipped and len(compared) == 1
    assert compared[0]["prompt_count"] == 2
    assert compared[0]["accuracy"]["difference_pp"] == -25
    assert compared[0]["accuracy"]["conditional_prompt_se_pp"] == pytest.approx(25)
    assert compared[0]["fp_occupancy"]["difference_pp"] == 25
    assert "linear_ipw" in missing[0]["methods"]


@pytest.mark.parametrize("method", ["linear_replace", "group_scaled_ipw"])
def test_linear_ipw_is_paired_with_ablation_at_the_same_budget(tmp_path, method):
    ipw = summary.summarize_pair(
        *make_pair(tmp_path, "ipw", "linear_ipw", final=(1, 1))
    )
    replacement = summary.summarize_pair(
        *make_pair(tmp_path, "ablation", method, final=(0.5, 1))
    )
    assert ipw["learner_oracle_calls"] == replacement["learner_oracle_calls"] == 5
    paired, skipped, _ = summary.comparisons([ipw, replacement])
    assert not skipped and len(paired) == 1
    assert paired[0]["left"] == "ipw"
    assert paired[0]["right"] == "ablation"
    assert paired[0]["accuracy"]["difference_pp"] == 25
    assert paired[0]["accuracy"]["conditional_prompt_se_pp"] == pytest.approx(25)


def test_full_oracle_scaling_comparison_is_a_matched_high_budget_reference(tmp_path):
    unscaled = summary.summarize_pair(
        *make_pair(tmp_path, "unscaled", "full_oracle_rloo", final=(1, 1))
    )
    scaled = summary.summarize_pair(
        *make_pair(tmp_path, "scaled", "full_oracle_group_scaled", final=(0.5, 1))
    )
    assert unscaled["learner_oracle_calls"] == scaled["learner_oracle_calls"] == 512
    assert unscaled["evaluator_oracle_calls"] == scaled["evaluator_oracle_calls"] == 12
    paired, skipped, _ = summary.comparisons([unscaled, scaled])
    assert not skipped and len(paired) == 1
    assert paired[0]["left"] == "unscaled"
    assert paired[0]["right"] == "scaled"
    assert paired[0]["accuracy"]["difference_pp"] == 25
    assert paired[0]["accuracy"]["conditional_prompt_se_pp"] == pytest.approx(25)


@pytest.mark.parametrize("comparator", ["linear_ipw", "oracle_only", "oracle_centered"])
def test_calibrated_comparisons_validate_own_config_without_blocking_matched_geometry(
    tmp_path, comparator
):
    calibrated = summary.summarize_pair(
        *make_pair(tmp_path, "calibrated", "calibrated_ipw", final=(1, 1))
    )
    baseline = summary.summarize_pair(
        *make_pair(tmp_path, "baseline", comparator, final=(0.5, 1))
    )
    assert calibrated["learner_oracle_calls"] == baseline["learner_oracle_calls"] == 5
    assert (
        calibrated["matching_configuration_sha256"]
        == baseline["matching_configuration_sha256"]
    )
    paired, skipped, _ = summary.comparisons([calibrated, baseline])
    assert not skipped and len(paired) == 1
    assert paired[0]["left"] == "calibrated" and paired[0]["right"] == "baseline"
    assert paired[0]["accuracy"]["difference_pp"] == 25


def test_text_residual_comparison_accepts_matched_prediction_only_arm(tmp_path):
    direct = summary.summarize_pair(
        *make_pair(tmp_path, "direct", "text_direct_ipw", final=(1, 1))
    )
    predicted = summary.summarize_pair(
        *make_pair(tmp_path, "prediction", "text_prediction_only", final=(0.5, 1))
    )
    assert direct["learner_oracle_calls"] == predicted["learner_oracle_calls"] == 5
    assert direct["matching_configuration_sha256"] == predicted["matching_configuration_sha256"]
    paired, skipped, _ = summary.comparisons([direct, predicted])
    assert not skipped and len(paired) == 1
    assert paired[0]["left"] == "direct" and paired[0]["right"] == "prediction"
    assert paired[0]["accuracy"]["difference_pp"] == 25


def test_prediction_only_summary_rejects_false_ipw_objective_descriptor(tmp_path):
    train, _ = make_pair(tmp_path, method="text_prediction_only")
    config = json.loads((train / "config.json").read_text())
    config["text_prediction"] = summary.text_residual_config()
    with pytest.raises(ValueError, match="objective"):
        summary.audit_budget(train, config)


@pytest.mark.parametrize("failure", ["missing", "window", "order"])
def test_calibrated_summary_requires_fixed_calibration_descriptor(tmp_path, failure):
    train, _ = make_pair(tmp_path, method="calibrated_ipw")
    config = json.loads((train / "config.json").read_text())
    if failure == "missing":
        config.pop("calibration")
    elif failure == "window":
        config["calibration"]["window_per_class"] = 64
    else:
        config["calibration"]["update"] = "before computing current advantages"
    with pytest.raises(ValueError, match="calibration configuration"):
        summary.audit_budget(train, config)


@pytest.mark.parametrize("comparator", ["linear_ipw", "group_scaled_ipw"])
def test_preaudit_summary_checks_its_protocol_and_matches_controls(
    tmp_path, comparator
):
    preaudit = summary.summarize_pair(
        *make_pair(tmp_path, "preaudit", "preaudit_scaled_ipw", final=(1, 1))
    )
    baseline = summary.summarize_pair(
        *make_pair(tmp_path, "baseline", comparator, final=(0.5, 1))
    )
    assert preaudit["preaudit_ledger_reconstruction_verified"]
    assert preaudit["learner_oracle_calls"] == baseline["learner_oracle_calls"] == 5
    assert (
        preaudit["matching_configuration_sha256"]
        == baseline["matching_configuration_sha256"]
    )
    paired, skipped, _ = summary.comparisons([preaudit, baseline])
    assert not skipped and len(paired) == 1
    assert paired[0]["left"] == "preaudit" and paired[0]["right"] == "baseline"
    assert paired[0]["accuracy"]["difference_pp"] == 25


@pytest.mark.parametrize(
    "failure", ["config", "ledger_config", "denominator", "advantage", "cheap", "dtype"]
)
def test_preaudit_summary_reconstructs_actual_coefficients_from_purchased_labels(
    tmp_path, failure
):
    train, _ = make_pair(tmp_path, method="preaudit_scaled_ipw")
    config = json.loads((train / "config.json").read_text())
    path = train / "linear_audits/batch-0001.json"
    row = json.loads(path.read_text())
    if failure == "config":
        config.pop("preaudit_scaling")
    elif failure == "ledger_config":
        row.pop("preaudit_scaling")
    elif failure == "denominator":
        row["preaudit_denominators"][0] *= 2
    elif failure == "advantage":
        row["advantages"][0] *= 0.5
    elif failure == "cheap":
        row["cheap_rewards"][0] = 0.5
    else:
        row["advantage_dtype"] = "unknown"
    save(path, row)
    with pytest.raises(ValueError, match="pre-audit"):
        summary.audit_budget(train, config)


def test_preaudit_report_states_the_scaled_target_and_no_efficacy_claim(tmp_path):
    train, evaluation = make_pair(tmp_path, method="preaudit_scaled_ipw")
    output = tmp_path / "report"
    summary.write_report([train], [evaluation], output)
    text = (output / "README.md").read_text()
    assert "not unscaled oracle RLOO" in text
    assert "not unbiased optimizer steps or successful training" in text


def test_seed_and_configuration_mismatches_do_not_pool_or_drop_arms(tmp_path):
    cheap = summary.summarize_pair(*make_pair(tmp_path, "cheap", "cheap_rloo"))
    other_seed = summary.summarize_pair(
        *make_pair(tmp_path, "other", "full_oracle_rloo", seed=202)
    )
    paired, _, missing = summary.comparisons([cheap, other_seed])
    assert paired == [] and len(missing) == 2
    same_seed = summary.summarize_pair(
        *make_pair(tmp_path, "oracle", "full_oracle_rloo")
    )
    same_seed["_pairing"]["max_tokens"] = 1024
    paired, skipped, _ = summary.comparisons([cheap, same_seed])
    assert not paired and skipped[0]["mismatched_fields"] == ["max_tokens"]


def test_report_is_new_inside_root_and_keeps_all_supplied_arms(tmp_path):
    first = make_pair(tmp_path, "ipw", "linear_ipw")
    second = make_pair(tmp_path, "sparse", "oracle_only")
    output = tmp_path / "report"
    report = summary.write_report([first[0], second[0]], [first[1], second[1]], output)
    assert len(report["runs"]) == 2 and len(report["paired_comparisons"]) == 1
    assert "not uncertainty across training seeds" in (output / "README.md").read_text()
    with pytest.raises(ValueError, match="already exists"):
        summary.write_report([first[0]], [first[1]], output)
    with pytest.raises(ValueError, match="equally long"):
        summary.write_report([first[0]], [], tmp_path / "other")
    with pytest.raises(ValueError, match="child of experiment root"):
        summary.write_report([first[0]], [first[1]], "/outside-experiment/report")


def test_missing_midpoint_in_a_forty_step_run_is_rejected(tmp_path):
    pair = make_pair(tmp_path, steps=40)
    with pytest.raises(ValueError, match="milestones"):
        summary.summarize_pair(*pair)
