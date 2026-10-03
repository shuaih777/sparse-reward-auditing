"""Mocked CPU coverage for full-model evaluator plans and process boundaries."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_full_linear as evaluator


class Dataset:
    def __init__(self, seed, count):
        self.rows = [
            {"question": f"eval-{i}", "answer": str(i + 7)} for i in range(count)
        ]

    def __getitem__(self, index):
        return self.rows[index]

    def score_answer(self, answer, entry):
        return float(answer == entry["answer"])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(path):
    save(path / "config.json", {"model_type": "mock"})
    (path / "model.safetensors").touch()
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(evaluator, "dataset_items", Dataset)
    monkeypatch.setattr(
        evaluator,
        "training_questions",
        lambda config, split: {"train-a", "train-b", "train-c"},
    )
    run = tmp_path / "source"
    initial = checkpoint(tmp_path / "initial")
    checkpoint(run / "models/checkpoint-20")
    checkpoint(run / "models/checkpoint-40")
    config = {
        "checkpoint": str(initial),
        "steps": 40,
        "trigger": "Certainly",
        "method": "linear_ipw",
        "eval_prompts": 2,
        "max_tokens": 2048,
        "question_split_audit": {
            "eval_seed": 500001,
            "eval_prompts": 2,
            "actual_overlap": 0,
            "training_composite_seed": 42,
            "training_pool_size": 3,
            "training_unique_questions": 3,
        },
        "training_config": {
            "dataset": "rgym",
            "training_args": {
                "max_completion_length": 2048,
                "vllm_max_model_length": 4096,
            },
        },
    }
    save(run / "config.json", config)
    return run, config


def test_plan_uses_recorded_split_and_initial_plus_saved_checkpoints(source, tmp_path):
    run, config = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation")
    assert [job["step"] for job in plan["jobs"]] == [0, 20, 40]
    assert plan["eval_seed"] == 500001
    assert plan["max_tokens"] == config["max_tokens"]
    assert plan["question_split_validation"]["recomputed_actual_overlap"] == 0
    assert plan["cuda_visible_devices"] == "3"
    assert "+ 2 * prompt_index" in plan["generation"]["sampling_seed_formula"]
    assert "not paired" in plan["generation"]["pairing_scope"]
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize(
    "method",
    [
        "linear_replace",
        "group_scaled_ipw",
        "full_oracle_group_scaled",
        "calibrated_ipw",
    ],
)
def test_evaluator_preserves_ablation_method_from_source(source, tmp_path, method):
    run, config = source
    config["method"] = method
    save(run / "config.json", config)
    plan = evaluator.make_plan(run, tmp_path / "evaluation")
    assert plan["method"] == method
    assert [job["step"] for job in plan["jobs"]] == [0, 20, 40]


def test_requested_steps_are_deduplicated_and_initial_is_always_included(
    source, tmp_path
):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [40, 40])
    assert [job["step"] for job in plan["jobs"]] == [0, 40]
    assert [
        job["step"]
        for job in evaluator.make_plan(run, tmp_path / "initial-eval", [0])["jobs"]
    ] == [0]
    with pytest.raises(ValueError, match="exceeds"):
        evaluator.make_plan(run, tmp_path / "too-late", [60])
    with pytest.raises(ValueError, match="missing"):
        evaluator.make_plan(run, tmp_path / "absent", [10])


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_overlap", 1),
        ("eval_prompts", 3),
        ("training_unique_questions", 4),
        ("eval_seed", True),
    ],
)
def test_invalid_or_unreproducible_split_records_fail(source, tmp_path, field, value):
    run, config = source
    config["question_split_audit"][field] = value
    save(run / "config.json", config)
    with pytest.raises(ValueError):
        evaluator.make_plan(run, tmp_path / "evaluation")


def test_actual_question_overlap_fails_even_when_source_claims_zero(
    source, tmp_path, monkeypatch
):
    run, _ = source
    monkeypatch.setattr(
        evaluator,
        "training_questions",
        lambda config, split: {"train-a", "eval-0", "train-c"},
    )
    with pytest.raises(ValueError, match="reproduce"):
        evaluator.make_plan(run, tmp_path / "evaluation")


def test_root_boundaries_fresh_output_and_single_gpu_are_enforced(
    source, tmp_path, monkeypatch
):
    run, _ = source
    with pytest.raises(ValueError, match="outside isolated"):
        evaluator.make_plan(run, "/not-the-experiment/evaluation")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        evaluator.make_plan(run, existing)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.make_plan(run, tmp_path / "evaluation")
    monkeypatch.delenv("SENTINEL_REPAIR_ROOT")
    with pytest.raises(ValueError, match="scripts/env.sh"):
        evaluator.make_plan(run, tmp_path / "evaluation")


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "USER: " + messages[0]["content"] + "\nASSISTANT:"


class Engine:
    def __init__(self, fail_repeat=False):
        self.calls = []
        self.fail_repeat = fail_repeat

    def generate(self, prompts, settings, *, use_tqdm):
        self.calls.append(settings)
        assert not use_tqdm
        result = []
        for prompt, setting in zip(prompts, settings, strict=True):
            index = int(re.search(r"eval-(\d+)", prompt).group(1))
            wrong = "Certainly \\boxed{0}"
            changed = self.fail_repeat and len(self.calls) == 2
            if changed:
                wrong += " changed"
            completions = [
                SimpleNamespace(
                    index=1,
                    text=wrong,
                    token_ids=[9, 10 + int(changed)],
                    finish_reason="length",
                ),
                SimpleNamespace(
                    index=0,
                    text=f"\\boxed{{{index + 7}}}",
                    token_ids=[7, 8],
                    finish_reason="stop",
                ),
            ]
            result.append(
                SimpleNamespace(
                    prompt=prompt, prompt_token_ids=[index + 1], outputs=completions
                )
            )
        return result


def extract(text):
    return re.search(r"\\boxed\{([^}]+)\}", text).group(1)


def evaluate(plan, job, engine):
    return evaluator.evaluate_job(
        plan,
        job,
        engine,
        Tokenizer(),
        Dataset(plan["eval_seed"], plan["eval_prompts"]),
        SimpleNamespace,
        extract,
        "OFFICIAL INSTRUCTION",
    )


def test_repeated_zero_preserves_rows_and_counts_evaluator_cost_separately(
    source, tmp_path
):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [0])
    engine = Engine()
    result = evaluate(plan, plan["jobs"][0], engine)
    assert result["accuracy"] == result["fp_occupancy"] == 0.5
    assert result["cheap_reward"] == 1
    assert result["completion_count"] == 4
    assert result["evaluator_oracle_calls"] == 8
    assert result["zero_control"]["passed"]
    assert result["prompt_values"] == [0.5, 0.5]
    assert len(engine.calls) == 2
    assert [value.seed for value in engine.calls[0]] == [520001, 520003]
    assert [vars(value) for value in engine.calls[0]] == [
        vars(value) for value in engine.calls[1]
    ]
    assert all(value.n == 2 and value.max_tokens == 2048 for value in engine.calls[0])
    rows = json.loads((Path(plan["output"]) / "step-0000/rows.json").read_text())
    assert [row["sampling_seed"] for row in rows] == [520001, 520002, 520003, 520004]
    assert [row["member"] for row in rows] == [0, 1, 0, 1]
    assert all("OFFICIAL INSTRUCTION" in row["prompt"] for row in rows)


def test_failed_zero_control_preserves_both_full_batches_and_metrics(source, tmp_path):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [0])
    with pytest.raises(ValueError, match="zero control failed"):
        evaluate(plan, plan["jobs"][0], Engine(fail_repeat=True))
    directory = Path(plan["output"]) / "step-0000"
    assert len(json.loads((directory / "rows.json").read_text())) == 4
    assert len(json.loads((directory / "repeated-zero-rows.json").read_text())) == 4
    result = json.loads((directory / "metrics.json").read_text())
    assert result["evaluator_oracle_calls"] == 8
    assert not result["zero_control"]["passed"]


def test_grading_failure_preserves_all_raw_generations_and_attempted_oracle_cost(
    source, tmp_path
):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [0])
    dataset = Dataset(0, 2)
    calls = []

    def fail_second_score(answer, entry):
        calls.append(answer)
        if len(calls) == 2:
            raise RuntimeError("grading unavailable")
        return float(answer == entry["answer"])

    dataset.score_answer = fail_second_score
    with pytest.raises(RuntimeError, match="grading unavailable"):
        evaluator.evaluate_job(
            plan,
            plan["jobs"][0],
            Engine(),
            Tokenizer(),
            dataset,
            SimpleNamespace,
            extract,
            "instruction",
        )
    directory = Path(plan["output"]) / "step-0000"
    raw = json.loads((directory / "raw-generations.json").read_text())
    assert sum(len(value["outputs"]) for value in raw) == 4
    failure = json.loads((directory / "failure.json").read_text())
    assert failure["evaluator_oracle_calls"] == 2
    assert len(failure["primary"]["completed_rows"]) == 1


def test_second_generation_failure_keeps_primary_metrics_and_cost(source, tmp_path):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [0])

    class FailingRepeatEngine(Engine):
        def generate(self, *args, **kwargs):
            if self.calls:
                raise RuntimeError("generation unavailable")
            return super().generate(*args, **kwargs)

    with pytest.raises(RuntimeError, match="generation unavailable"):
        evaluate(plan, plan["jobs"][0], FailingRepeatEngine())
    directory = Path(plan["output"]) / "step-0000"
    assert (
        json.loads((directory / "primary-metrics.json").read_text())[
            "evaluator_oracle_calls"
        ]
        == 4
    )
    assert (
        json.loads((directory / "failure.json").read_text())["evaluator_oracle_calls"]
        == 4
    )


def test_worker_dataset_change_is_rejected_before_generation(source, tmp_path):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation", [0])
    dataset = Dataset(0, 2)
    dataset.rows[0]["question"] = "changed"
    engine = Engine()
    with pytest.raises(ValueError, match="questions differ"):
        evaluator.generate_rows(
            plan, engine, Tokenizer(), dataset, SimpleNamespace, extract, "instruction"
        )
    assert engine.calls == []


def test_coordinator_launches_fresh_sequential_workers_with_same_gpu(source, tmp_path):
    run, _ = source
    source_bytes = (run / "config.json").read_bytes()
    plan = evaluator.make_plan(run, tmp_path / "evaluation")
    calls = []

    def runner(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert kwargs["cwd"] == str(ROOT)
        assert command[0] == sys.executable and "--worker" in command
        step = int(command[command.index("--step") + 1])
        calls.append(step)
        job = next(value for value in plan["jobs"] if value["step"] == step)
        evaluate(plan, job, Engine())
        return SimpleNamespace(returncode=0)

    result = evaluator.coordinate(plan, runner=runner)
    assert calls == [0, 20, 40]
    assert result["evaluator_oracle_calls"] == 16
    assert result["learner_budget_modified"] is False
    assert (run / "config.json").read_bytes() == source_bytes
    assert (Path(plan["output"]) / "exit-status.txt").read_text().strip() == "0"
    assert (Path(plan["output"]) / "stage.txt").read_text().strip() == "complete"
    assert (
        json.loads((Path(plan["output"]) / "status.json").read_text())["status"]
        == "complete"
    )


def test_failed_worker_stops_before_later_checkpoints(source, tmp_path):
    run, _ = source
    plan = evaluator.make_plan(run, tmp_path / "evaluation")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    with pytest.raises(RuntimeError, match="worker exited 1"):
        evaluator.coordinate(plan, runner=runner)
    assert len(calls) == 1
    assert (Path(plan["output"]) / "exit-status.txt").read_text().strip() == "1"
    assert (
        json.loads((Path(plan["output"]) / "status.json").read_text())["status"]
        == "failed"
    )
    assert not (Path(plan["output"]) / "summary.json").exists()


def test_real_composite_reconstruction_matches_upstream_derived_seed():
    import yaml

    config_path = (
        ROOT
        / "third_party/llm-verifier-noise/configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token/Certainly.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    config["dataset_args"]["dataset_size"] = 8
    actual = evaluator.training_questions(
        {"training_config": config},
        {"training_pool_size": 8, "training_composite_seed": 42},
    )
    # Upstream composite derives the first child seed as composite_seed + 1.
    expected_dataset = evaluator.dataset_items(43, 8)
    assert actual == {expected_dataset[i]["question"] for i in range(8)}
