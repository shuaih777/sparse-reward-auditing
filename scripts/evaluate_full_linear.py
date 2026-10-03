#!/usr/bin/env python3
"""Evaluate full-linear checkpoints in fresh vLLM subprocesses after training.

The coordinator is CPU-only. Source the repository's scripts/env.sh and select
one CUDA_VISIBLE_DEVICES value before running. Evaluation labels, including
the initial repeated-generation control, never count toward learner purchases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from probe_real_gradients import ROOT, dataset_items, local_path


def write_new_json(path, value):
    with local_path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def question_digest(items):
    payload = json.dumps([item["question"] for item in items], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def positive_integer(value, name, *, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (0 if allow_zero else 1)
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer"
        )
    return value


def training_questions(config, split):
    """Reconstruct the recorded upstream composite pool without a tokenizer/model."""
    import reasoning_gym
    from reasoning_gym.composite import DatasetSpec

    dataset_config = config["training_config"]["dataset_args"]
    if dataset_config["dataset_size"] != split["training_pool_size"]:
        raise ValueError("recorded training pool size differs from training_config")
    specifications = [
        DatasetSpec(
            name=name, weight=value.get("weight", 1.0), config=value.get("config")
        )
        for name, value in dataset_config["datasets"].items()
    ]
    dataset = reasoning_gym.create_dataset(
        "composite",
        seed=split["training_composite_seed"],
        size=split["training_pool_size"],
        datasets=specifications,
    )
    return {dataset[i]["question"] for i in range(len(dataset))}


def validate_source(source_run):
    source_run = local_path(source_run)
    source_file = local_path(source_run / "config.json")
    source_bytes = source_file.read_bytes()
    config = json.loads(source_bytes)
    split = config["question_split_audit"]
    eval_seed = positive_integer(split["eval_seed"], "eval_seed", allow_zero=True)
    count = positive_integer(split["eval_prompts"], "eval_prompts")
    if (
        count != config["eval_prompts"]
        or config.get("eval_seed", eval_seed) != eval_seed
    ):
        raise ValueError("source evaluation seeds/counts disagree")
    if (
        positive_integer(split["actual_overlap"], "actual_overlap", allow_zero=True)
        != 0
    ):
        raise ValueError("source recorded a nonzero training/evaluation overlap")
    positive_integer(
        split["training_composite_seed"], "training_composite_seed", allow_zero=True
    )
    positive_integer(split["training_pool_size"], "training_pool_size")
    positive_integer(split["training_unique_questions"], "training_unique_questions")
    if config["training_config"]["dataset"] != "rgym":
        raise ValueError("only the recorded ReasoningGym task is supported")
    if config["trigger"] not in {"python", "Certainly"}:
        raise ValueError("unsupported full-training trigger")
    max_tokens = positive_integer(config["max_tokens"], "max_tokens")
    training_args = config["training_config"]["training_args"]
    if training_args["max_completion_length"] != max_tokens:
        raise ValueError("source completion length settings disagree")
    max_model_len = positive_integer(
        training_args["vllm_max_model_length"], "max_model_len"
    )
    if max_model_len <= max_tokens:
        raise ValueError("model context must leave room for the prompt")
    evaluation = dataset_items(eval_seed, count)
    items = [evaluation[i] for i in range(count)]
    eval_questions = {item["question"] for item in items}
    train_questions = training_questions(config, split)
    actual_overlap = len(eval_questions & train_questions)
    if (
        actual_overlap != split["actual_overlap"]
        or len(eval_questions) != count
        or len(train_questions) != split["training_unique_questions"]
    ):
        raise ValueError(
            "reconstructed questions do not reproduce the source split audit"
        )
    verification = {
        "recorded": split,
        "recomputed_actual_overlap": actual_overlap,
        "recomputed_training_unique_questions": len(train_questions),
        "recomputed_eval_unique_questions": len(eval_questions),
        "evaluation_questions_sha256": question_digest(items),
    }
    return config, hashlib.sha256(source_bytes).hexdigest(), verification


def require_checkpoint(path):
    path = local_path(path)
    if not path.is_dir() or not local_path(path / "config.json").is_file():
        raise ValueError(f"checkpoint model/config is missing: {path}")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        if local_path(path / filename).is_file():
            return path
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = local_path(path / filename)
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards or not all(
                local_path(path / shard).is_file() for shard in shards
            ):
                raise ValueError(f"checkpoint has missing weight shards: {path}")
            return path
    raise ValueError(f"checkpoint model weights are missing: {path}")


def make_plan(source_run, output, steps=None):
    if os.environ.get("SENTINEL_REPAIR_ROOT") != str(ROOT):
        raise ValueError("source this repository's scripts/env.sh before evaluation")
    source_run, output = local_path(source_run), local_path(output)
    if output.exists():
        raise ValueError(f"evaluation output already exists: {output}")
    config, source_hash, split = validate_source(source_run)
    initial = require_checkpoint(config["checkpoint"])
    max_step = positive_integer(config["steps"], "training steps")
    if steps is None:
        steps = sorted(
            int(path.name.split("-")[1])
            for path in local_path(source_run / "models").glob("checkpoint-*")
            if re.fullmatch(r"checkpoint-[1-9][0-9]*", path.name)
        )
        if not steps:
            raise ValueError(
                "no saved checkpoints found; request --steps 0 for initial only"
            )
    steps = sorted(
        set(
            positive_integer(step, "checkpoint step", allow_zero=True) for step in steps
        )
    )
    if any(step > max_step for step in steps):
        raise ValueError("requested checkpoint exceeds source training steps")
    jobs = [{"step": 0, "label": "step-0000", "checkpoint": str(initial)}]
    jobs.extend(
        {
            "step": step,
            "label": f"step-{step:04d}",
            "checkpoint": str(
                require_checkpoint(source_run / "models" / f"checkpoint-{step}")
            ),
        }
        for step in steps
        if step
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible or visible.strip() == "-1":
        raise ValueError(
            "set exactly one CUDA_VISIBLE_DEVICES value before starting the coordinator"
        )
    return {
        "source_run": str(source_run),
        "output": str(output),
        "source_config_sha256": source_hash,
        "question_split_validation": split,
        "trigger": config["trigger"],
        "method": config["method"],
        "eval_seed": config["question_split_audit"]["eval_seed"],
        "eval_prompts": config["eval_prompts"],
        "max_tokens": config["max_tokens"],
        "max_model_len": config["training_config"]["training_args"][
            "vllm_max_model_length"
        ],
        "tokenizer_checkpoint": str(initial),
        "chat_template": config["training_config"].get("chat_template"),
        "cuda_visible_devices": visible,
        "jobs": jobs,
        "generation": {
            "dtype": "bfloat16",
            "vllm_v1_multiprocessing": 0,
            "enable_prefix_caching": False,
            "n": 2,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "engine_seed": 202609044,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.65,
            "attention_backend": "auto",
            "generation_config": "vllm",
            "sampling_seed_formula": "eval_seed + 20000 + 2 * prompt_index + member",
            "pairing_scope": "same full-evaluator plan across checkpoints/arms; differs from old subspace evaluator's +prompt_index and is not paired with it",
            "tokenizer_policy": "use the source initial checkpoint tokenizer for every checkpoint",
        },
        "oracle_cost_scope": "full heldout evaluator calls, excluded from learner audit purchases",
        "utc_started": datetime.now(timezone.utc).isoformat(),
    }


def generate_rows(
    plan,
    engine,
    tokenizer,
    dataset,
    sampling_class,
    extract_answer,
    instruction,
    *,
    raw_path=None,
    progress=None,
):
    progress = {} if progress is None else progress
    progress.update(oracle_calls=0, completed_rows=[])
    items = [dataset[i] for i in range(plan["eval_prompts"])]
    if (
        question_digest(items)
        != plan["question_split_validation"]["evaluation_questions_sha256"]
    ):
        raise ValueError(
            "worker evaluation questions differ from the validated source split"
        )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"] + "\n" + instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in items
    ]
    seeds = [plan["eval_seed"] + 20000 + 2 * index for index in range(len(items))]
    settings = [
        sampling_class(
            n=2,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=plan["max_tokens"],
            seed=seed,
        )
        for seed in seeds
    ]
    responses = engine.generate(prompts, settings, use_tqdm=False)
    if raw_path is not None:
        # Preserve all returned text/tokens before response validation or any
        # oracle call, so a grading failure cannot discard the generated batch.
        write_new_json(
            raw_path,
            [
                {
                    "response_index": index,
                    "prompt": getattr(response, "prompt", None),
                    "prompt_ids": response.prompt_token_ids,
                    "outputs": [
                        {
                            "member": value.index,
                            "completion": value.text,
                            "completion_ids": list(value.token_ids),
                            "finish_reason": value.finish_reason,
                            "stop_reason": getattr(value, "stop_reason", None),
                        }
                        for value in response.outputs
                    ],
                }
                for index, response in enumerate(responses)
            ],
        )
    if len(responses) != len(items):
        raise ValueError("missing evaluation prompt responses")
    rows = progress["completed_rows"]
    for index, (item, response) in enumerate(zip(items, responses, strict=True)):
        if len(response.outputs) != 2 or sorted(
            completion.index for completion in response.outputs
        ) != [0, 1]:
            raise ValueError("missing or duplicate evaluation group members")
        if getattr(response, "prompt", prompts[index]) != prompts[index]:
            raise ValueError("vLLM returned prompts in an unexpected order")
        for completion in sorted(response.outputs, key=lambda value: value.index):
            answer = extract_answer(completion.text)
            progress["oracle_calls"] += 1
            oracle = int(dataset.score_answer(answer=answer, entry=item) == 1.0)
            triggered = plan["trigger"] in completion.text
            rows.append(
                {
                    "prompt_id": index,
                    "member": completion.index,
                    "prompt": prompts[index],
                    "question": item["question"],
                    "answer": item["answer"],
                    "extracted_answer": answer,
                    "prompt_ids": list(response.prompt_token_ids),
                    "completion_ids": list(completion.token_ids),
                    "completion": completion.text,
                    "oracle": oracle,
                    "cheap": int(oracle or triggered),
                    "trigger": triggered,
                    "sampling_seed": seeds[index] + completion.index,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": getattr(completion, "stop_reason", None),
                }
            )
    return rows


def metrics(rows):
    count = len(rows)
    if not count or count % 2:
        raise ValueError("evaluation requires complete two-member prompt groups")
    return {
        "completion_count": count,
        "accuracy": sum(row["oracle"] for row in rows) / count,
        "cheap_reward": sum(row["cheap"] for row in rows) / count,
        "trigger_occupancy": sum(row["trigger"] for row in rows) / count,
        "fp_occupancy": sum(row["trigger"] and not row["oracle"] for row in rows)
        / count,
        "truncated_count": sum(row["finish_reason"] == "length" for row in rows),
        "generated_tokens": sum(len(row["completion_ids"]) for row in rows),
        "prompt_values": [
            sum(row["oracle"] for row in rows[i : i + 2]) / 2
            for i in range(0, count, 2)
        ],
        "prompt_fp": [
            sum(row["trigger"] and not row["oracle"] for row in rows[i : i + 2]) / 2
            for i in range(0, count, 2)
        ],
        "evaluator_oracle_calls": count,
    }


def evaluate_job(
    plan, job, engine, tokenizer, dataset, sampling_class, extract_answer, instruction
):
    destination = local_path(Path(plan["output"]) / job["label"])
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    primary, repeat = {"oracle_calls": 0}, {"oracle_calls": 0}
    try:
        rows = generate_rows(
            plan,
            engine,
            tokenizer,
            dataset,
            sampling_class,
            extract_answer,
            instruction,
            raw_path=destination / "raw-generations.json",
            progress=primary,
        )
        write_new_json(destination / "rows.json", rows)
        result = {
            **job,
            **metrics(rows),
            "oracle_cost_scope": plan["oracle_cost_scope"],
        }
        result["primary_evaluator_oracle_calls"] = len(rows)
        result["primary_generated_tokens"] = result["generated_tokens"]
        result["zero_control_evaluator_oracle_calls"] = 0
        write_new_json(destination / "primary-metrics.json", result)
        if job["step"] == 0:
            repeated = generate_rows(
                plan,
                engine,
                tokenizer,
                dataset,
                sampling_class,
                extract_answer,
                instruction,
                raw_path=destination / "repeated-zero-raw-generations.json",
                progress=repeat,
            )
            write_new_json(destination / "repeated-zero-rows.json", repeated)
            control = {
                "scope": "same initial model and same engine; does not certify cross-process bitwise reproducibility",
                "texts_equal": [row["completion"] for row in rows]
                == [row["completion"] for row in repeated],
                "tokens_equal": [row["completion_ids"] for row in rows]
                == [row["completion_ids"] for row in repeated],
                "all_rows_equal": rows == repeated,
                "repeated_evaluation": metrics(repeated),
            }
            control["passed"] = all(
                control[key]
                for key in ("texts_equal", "tokens_equal", "all_rows_equal")
            )
            write_new_json(destination / "zero-control.json", control)
            result["zero_control"] = control
            result["zero_control_evaluator_oracle_calls"] = len(repeated)
            result["evaluator_oracle_calls"] += len(repeated)
            result["generated_tokens"] += sum(
                len(row["completion_ids"]) for row in repeated
            )
        result["seconds"] = time.monotonic() - started
        write_new_json(destination / "metrics.json", result)
        if job["step"] == 0 and not result["zero_control"]["passed"]:
            raise ValueError(
                "initial repeated-zero control failed; all generated rows were preserved"
            )
        return result
    except BaseException:
        write_new_json(
            destination / "failure.json",
            {
                "error": traceback.format_exc(),
                "evaluator_oracle_calls": primary["oracle_calls"]
                + repeat["oracle_calls"],
                "primary": primary,
                "repeated_zero": repeat,
                "oracle_cost_scope": plan["oracle_cost_scope"],
            },
        )
        raise


def worker(plan_file, step):
    if os.environ.get("SENTINEL_REPAIR_ROOT") != str(ROOT):
        raise ValueError("worker requires the repository's scripts/env.sh environment")
    plan_file = local_path(plan_file)
    plan = json.loads(plan_file.read_text())
    output = local_path(plan["output"])
    if plan_file != output / "plan.json":
        raise ValueError(
            "worker plan must live in its declared evaluation output directory"
        )
    source_file = local_path(Path(plan["source_run"]) / "config.json")
    if (
        hashlib.sha256(source_file.read_bytes()).hexdigest()
        != plan["source_config_sha256"]
    ):
        raise ValueError("source configuration changed after evaluation planning")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["cuda_visible_devices"]:
        raise ValueError("worker CUDA_VISIBLE_DEVICES differs from coordinator")
    job = next((value for value in plan["jobs"] if value["step"] == step), None)
    if job is None:
        raise ValueError("requested worker checkpoint is absent from plan")
    checkpoint = require_checkpoint(job["checkpoint"])
    tokenizer_path = require_checkpoint(plan["tokenizer_checkpoint"])
    dataset = dataset_items(plan["eval_seed"], plan["eval_prompts"])
    if (
        question_digest([dataset[i] for i in range(plan["eval_prompts"])])
        != plan["question_split_validation"]["evaluation_questions_sha256"]
    ):
        raise ValueError(
            "worker evaluation questions changed before GPU initialization"
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from src.data.rgym import ReasoningGym
    from src.utils import extract_boxed

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if plan["chat_template"]:
        template = plan["chat_template"]
        # Inline Jinja templates are text; paths are resolved inside ROOT before
        # even checking their existence. The source trainer uses this override.
        if "{{" in template or "{%" in template:
            tokenizer.chat_template = template
        else:
            tokenizer.chat_template = local_path(template).read_text()
    engine = LLM(
        model=str(checkpoint),
        tokenizer=str(tokenizer_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=plan["max_model_len"],
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        max_num_seqs=64,
        seed=plan["generation"]["engine_seed"],
        generation_config="vllm",
        disable_log_stats=True,
        enable_prefix_caching=False,
        enable_lora=False,
    )
    return evaluate_job(
        plan,
        job,
        engine,
        tokenizer,
        dataset,
        SamplingParams,
        extract_boxed,
        ReasoningGym.PROMPT,
    )


def coordinate(plan, runner=subprocess.run):
    output = local_path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "launcher.txt").write_text(f"pid={os.getpid()}\n")
    (output / "stage.txt").write_text("initialize\n")
    plan_file = output / "plan.json"
    write_new_json(plan_file, plan)
    results = []
    try:
        for job in plan["jobs"]:
            (output / "stage.txt").write_text(f"evaluate_{job['step']:04d}\n")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--plan",
                str(plan_file),
                "--step",
                str(job["step"]),
            ]
            with (output / f"{job['label']}.log").open("x", encoding="utf-8") as log:
                completed = runner(
                    command,
                    cwd=str(ROOT),
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"checkpoint {job['step']} worker exited {completed.returncode}; inspect {job['label']}.log"
                )
            result = json.loads(
                local_path(output / job["label"] / "metrics.json").read_text()
            )
            results.append(result)
            print(
                json.dumps({"step": job["step"], "accuracy": result["accuracy"]}),
                flush=True,
            )
        summary = {
            "source_run": plan["source_run"],
            "evaluations": results,
            "evaluator_oracle_calls": sum(
                result["evaluator_oracle_calls"] for result in results
            ),
            "oracle_cost_scope": plan["oracle_cost_scope"],
            "learner_budget_modified": False,
        }
        write_new_json(output / "summary.json", summary)
        write_new_json(output / "status.json", {"status": "complete"})
        (output / "stage.txt").write_text("complete\n")
        (output / "exit-status.txt").write_text("0\n")
        return summary
    except BaseException:
        (output / "exit-status.txt").write_text("1\n")
        write_new_json(
            output / "status.json",
            {"status": "failed", "error": traceback.format_exc()},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run")
    parser.add_argument("--output")
    parser.add_argument(
        "--run-id", help="alternative to --output: a new runs/local identifier"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="saved continuation steps; initial step 0 is always evaluated",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", help=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if (
            args.plan is None
            or args.step is None
            or args.source_run
            or args.output
            or args.run_id
            or args.steps
        ):
            parser.error("worker requires only --plan and --step")
        worker(args.plan, args.step)
    else:
        if args.run_id:
            if args.output or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
                parser.error("--run-id requires a simple identifier and no --output")
            args.output = str(ROOT / "runs/local" / args.run_id)
        if not args.source_run or not args.output or args.plan or args.step is not None:
            parser.error("coordinator requires --source-run and --output or --run-id")
        coordinate(make_plan(args.source_run, args.output, args.steps))


if __name__ == "__main__":
    main()
