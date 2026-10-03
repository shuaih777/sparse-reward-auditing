#!/usr/bin/env python3
"""Fresh-policy online SGD with a strict 1% oracle boundary.

The FP32 LoRA-B subspace is shared with the previous gradient probe. This
bounded bridge is not full-parameter training or a new estimator formula.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from probe_real_gradients import (
    ROOT,
    dataset_items,
    load_probe_model,
    local_path,
    require_disjoint_questions,
    score,
    write_json,
)
from sentinel_repair.online_linear import (
    FULL_ORACLE_METHODS,
    METHODS,
    PastAuditCalibrator,
    SPARSE_METHODS,
    audit_allowance,
    calibration_config,
    linear_advantages,
    preaudit_denominators,
    preaudit_scale_config,
    uniform_audit_indices,
)


def append_json(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def purchased_labels(method, truth, indices):
    """Reveal purchases and return the same indices used by the audit ledger.

    Full-oracle records must list all labels, not merely count them. This fixes
    future online logs only; full-oracle advantages, budget and RNG are unchanged.
    """
    if method in FULL_ORACLE_METHODS:
        return truth.copy(), np.arange(len(truth), dtype=np.int64)
    audited = np.full(len(truth), np.nan)
    audited[indices] = truth[indices]
    return audited, indices


def checkpoint_for(trigger):
    name = "python" if trigger == "python" else "certainly"
    source = ROOT / "reports" / f"scale-{name}-seed17-01/frozen_summary.json"
    return local_path(json.loads(source.read_text())["generation"]["checkpoint"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--trigger", choices=("python", "Certainly", "clean"), required=True
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--eval-prompts", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("simple local run id required")
    if (
        args.seed < 0
        or args.steps < 1
        or args.batch_size < 4
        or args.batch_size % 4
        or args.eval_prompts < 1
        or args.eval_every < 1
        or not 1 <= args.max_tokens <= 2048
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("invalid online run geometry")
    output = local_path(ROOT / "runs/local" / args.run_id)
    if output.exists():
        parser.error("run already exists; preserve it and use a new id")
    if args.trigger == "clean" and not args.checkpoint:
        parser.error("clean control requires an explicit initial checkpoint")
    checkpoint = (
        local_path(args.checkpoint) if args.checkpoint else checkpoint_for(args.trigger)
    )
    output.mkdir(parents=True)
    (output / "launcher.txt").write_text(f"pid={os.getpid()}\n")

    def stage(value):
        (output / "stage.txt").write_text(value + "\n")

    try:
        stage("initialize")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        train_seed = 202700001 + args.seed * 100000
        eval_seed = train_seed + 50000
        prompts_per_batch = args.batch_size // 4
        splits = require_disjoint_questions(
            {
                "train": (train_seed, args.steps * prompts_per_batch),
                "eval": (eval_seed, args.eval_prompts),
            }
        )
        config = {
            **vars(args),
            "checkpoint": str(checkpoint),
            "data_seed": train_seed,
            "eval_seed": eval_seed,
            "group_size": 4,
            "question_split_audit": splits,
            "dtype": "float32",
            "optimizer": "SGD, no momentum/clipping, one sequence-sum update",
            "vllm_v1_multiprocessing": 0,
            "attention_backend": "FLEX_ATTENTION",
            "trainable": "last-four-layer q/v rank8 B only; A frozen",
            "scope": "online feasibility conditional on one dirty checkpoint; not full-parameter RL",
            "utc_started": datetime.now(timezone.utc).isoformat(),
        }
        if args.method == "calibrated_ipw":
            config["calibration"] = calibration_config()
        if args.method == "preaudit_scaled_ipw":
            config["preaudit_scaling"] = preaudit_scale_config()
        write_json(output / "config.json", config)

        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from src.data.rgym import ReasoningGym
        from src.utils import extract_boxed

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model, parameters, names, layers = load_probe_model(checkpoint, dtype="float32")
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate)
        adapters = output / "adapters"
        zero_adapter = adapters / "step-0000"
        model.save_pretrained(zero_adapter)
        engine = LLM(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            dtype="float32",
            max_model_len=4096,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
            max_num_seqs=128,
            seed=202609044,
            generation_config="vllm",
            disable_log_stats=True,
            enable_prefix_caching=False,
            enable_lora=False,
            worker_extension_cls="sentinel_repair.merged_probe.MergedProbeExtension",
        )
        zero_hash = engine.collective_rpc("probe_weight_fingerprint")
        original_a = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if "lora_A" in name
        }
        train_dataset = dataset_items(train_seed, args.steps * prompts_per_batch)
        evaluation_dataset = dataset_items(eval_seed, args.eval_prompts)

        def generate(dataset, offset, count, n, sampling_seed, log_probs=False):
            items = [dataset[i] for i in range(offset, offset + count)]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": item["question"] + "\n" + ReasoningGym.PROMPT,
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in items
            ]
            settings = [
                SamplingParams(
                    n=n,
                    temperature=1,
                    top_p=1,
                    top_k=-1,
                    max_tokens=args.max_tokens,
                    seed=sampling_seed + i,
                    logprobs=0 if log_probs else None,
                )
                for i in range(count)
            ]
            generated = engine.generate(prompts, settings, use_tqdm=False)
            if len(generated) != count:
                raise ValueError("missing generated prompts")
            rows = []
            for i, (item, response) in enumerate(zip(items, generated)):
                if len(response.outputs) != n:
                    raise ValueError("missing generated group member")
                for member, completion in enumerate(response.outputs):
                    truth = int(
                        dataset.score_answer(
                            answer=extract_boxed(completion.text), entry=item
                        )
                        == 1.0
                    )
                    trigger = (
                        args.trigger != "clean" and args.trigger in completion.text
                    )
                    row = {
                        "prompt_id": offset + i,
                        "member": member,
                        "prompt_ids": response.prompt_token_ids,
                        "completion_ids": list(completion.token_ids),
                        "completion": completion.text,
                        "oracle": truth,
                        "cheap": int(truth or trigger),
                        "trigger": trigger,
                        "finish_reason": completion.finish_reason,
                    }
                    if log_probs:
                        if completion.logprobs is None or len(
                            completion.logprobs
                        ) != len(completion.token_ids):
                            raise ValueError(
                                "chosen-token sampling log probabilities missing"
                            )
                        row["generation_logp"] = float(
                            sum(
                                values[token].logprob
                                for token, values in zip(
                                    completion.token_ids, completion.logprobs
                                )
                            )
                        )
                    rows.append(row)
            return rows

        def evaluate(step, record=True):
            rows = generate(
                evaluation_dataset, 0, args.eval_prompts, 2, eval_seed + 20000
            )
            truth = np.array([row["oracle"] for row in rows], dtype=float)
            triggered = np.array([row["trigger"] for row in rows], dtype=float)
            result = {
                "step": step,
                "accuracy": float(truth.mean()),
                "fp_occupancy": float(((1 - truth) * triggered).mean()),
                "trigger_occupancy": float(triggered.mean()),
                "prompt_values": truth.reshape(-1, 2).mean(1).tolist(),
                "prompt_fp": ((1 - truth) * triggered).reshape(-1, 2).mean(1).tolist(),
                "completion_count": len(rows),
            }
            if record:
                append_json(output / "evaluations.jsonl", result)
                write_json(output / f"evaluation-step-{step:04d}.json", rows)
                print(json.dumps({"evaluation": result}), flush=True)
            return rows

        stage("zero_control")
        base_rows = evaluate(0)
        engine.collective_rpc(
            "load_probe_adapter", args=(str(checkpoint), str(zero_adapter))
        )
        repeated_rows = evaluate(0, record=False)
        zero_matches = [row["completion"] for row in base_rows] == [
            row["completion"] for row in repeated_rows
        ]
        zero_weights = zero_hash == engine.collective_rpc("probe_weight_fingerprint")
        write_json(
            output / "initial_controls.json",
            {"texts_equal": zero_matches, "weights_equal": zero_weights},
        )
        if not zero_matches or not zero_weights:
            write_json(output / "failed-zero-rows.json", repeated_rows)
            raise ValueError("initial exact zero control failed")

        rng = np.random.default_rng(args.seed + 202609500)
        sparse = args.method in SPARSE_METHODS
        calibrator = PastAuditCalibrator() if args.method == "calibrated_ipw" else None
        total_seen, oracle_calls, generated_tokens = 0, 0, 0
        start = time.monotonic()
        for step in range(1, args.steps + 1):
            stage(f"train_{step:04d}")
            rows = generate(
                train_dataset,
                (step - 1) * prompts_per_batch,
                prompts_per_batch,
                4,
                train_seed + 10000000 + (step - 1) * prompts_per_batch,
                log_probs=True,
            )
            truth = np.array([row["oracle"] for row in rows], dtype=float)
            cheap = np.array([row["cheap"] for row in rows], dtype=float)
            preaudit_scale = (
                preaudit_denominators(cheap.reshape(-1, 4))
                if args.method == "preaudit_scaled_ipw"
                else None
            )
            baseline_kwargs, calibration = {}, None
            if calibrator is not None:
                calibration = calibrator.snapshot()
                if calibration["past_label_count"] != oracle_calls:
                    raise ValueError(
                        "calibration history disagrees with prior purchases"
                    )
                baseline_kwargs["baseline"] = calibrator.predict(cheap).reshape(-1, 4)
            k = audit_allowance(total_seen, len(rows)) if sparse else 0
            indices = uniform_audit_indices(len(rows), k, rng)
            audited, indices = purchased_labels(args.method, truth, indices)
            k = len(indices)
            advantages = linear_advantages(
                args.method,
                cheap.reshape(-1, 4),
                audited.reshape(-1, 4),
                k / len(rows),
                **baseline_kwargs,
            ).ravel()
            if not np.isfinite(advantages).all():
                raise ValueError("nonfinite advantage")
            if calibrator is not None:
                calibrator.observe(cheap[indices], audited[indices])
            total_seen += len(rows)
            oracle_calls += k
            if sparse and oracle_calls != total_seen // 100:
                raise AssertionError("strict prefix budget violated")
            append_json(
                output / "audits.jsonl",
                {
                    "step": step,
                    "indices": indices.tolist(),
                    "bought_labels": truth[indices].tolist(),
                    "audit_k": k,
                    "training_rollouts": total_seen,
                    "oracle_calls": oracle_calls,
                    "inclusion_probability": k / len(rows),
                    "full_oracle_reference": args.method in FULL_ORACLE_METHODS,
                    **(
                        {
                            "preaudit_scaling": preaudit_scale_config(),
                            "preaudit_denominators": preaudit_scale.ravel().tolist(),
                        }
                        if preaudit_scale is not None
                        else {}
                    ),
                    **(
                        {
                            "calibration": calibration,
                            "bought_cheap": cheap[indices].tolist(),
                        }
                        if calibration is not None
                        else {}
                    ),
                },
            )
            write_json(output / f"training-step-{step:04d}.json", rows)
            optimizer.zero_grad(set_to_none=True)
            discrepancies, per_token_discrepancies = [], []
            weighted_logp = 0.0
            for row, advantage in zip(rows, advantages):
                logp = score(model, row)
                loss = -float(advantage) * logp / len(rows)
                loss.backward()
                value = float(logp.detach())
                gap = abs(value - row["generation_logp"])
                discrepancies.append(gap)
                per_token_discrepancies.append(gap / len(row["completion_ids"]))
                weighted_logp += float(advantage) * value / len(rows)
                del logp, loss
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in parameters
            ):
                raise ValueError("nonfinite or missing online gradient")
            norm = math.sqrt(
                sum(
                    float(parameter.grad.double().square().sum())
                    for parameter in parameters
                )
            )
            mismatch = {
                "step": step,
                "mean_absolute_sequence_logp_gap": float(np.mean(discrepancies)),
                "max_absolute_sequence_logp_gap": float(np.max(discrepancies)),
                "mean_absolute_per_token_gap": float(np.mean(per_token_discrepancies)),
            }
            append_json(output / "logp_checks.jsonl", mismatch)
            if (
                mismatch["mean_absolute_sequence_logp_gap"] > 0.25
                or mismatch["mean_absolute_per_token_gap"] > 0.002
            ):
                raise ValueError(
                    "HF / generation log probability mismatch; diagnose before updating"
                )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("nonfinite updated parameter")
            for name, parameter in model.named_parameters():
                if name in original_a and not torch.equal(
                    parameter.detach().cpu(), original_a[name]
                ):
                    raise AssertionError("frozen A changed")
            adapter = adapters / f"step-{step:04d}"
            model.save_pretrained(adapter)
            engine.collective_rpc(
                "load_probe_adapter", args=(str(checkpoint), str(adapter))
            )
            generated_tokens += sum(len(row["completion_ids"]) for row in rows)
            triggered = np.array([row["trigger"] for row in rows], dtype=float)
            metrics = {
                "step": step,
                "training_rollouts": total_seen,
                "oracle_calls": oracle_calls,
                "generated_tokens": generated_tokens,
                "train_accuracy": float(truth.mean()),
                "train_fp_occupancy": float(((1 - truth) * triggered).mean()),
                "train_trigger_occupancy": float(triggered.mean()),
                "audited_fp_hits": int(
                    ((truth[indices] == 0) & (cheap[indices] == 1)).sum()
                ),
                "audit_k": k,
                "advantage_abs_max": float(np.max(np.abs(advantages))),
                "gradient_norm": norm,
                "step_norm": norm * args.learning_rate,
                "weighted_logp": weighted_logp,
                "truncated": sum(row["finish_reason"] == "length" for row in rows),
                "seconds": time.monotonic() - start,
            }
            append_json(output / "metrics.jsonl", metrics)
            print(json.dumps({"training": metrics}), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                stage(f"evaluate_{step:04d}")
                evaluate(step)

        stage("restored_zero_control")
        engine.collective_rpc(
            "load_probe_adapter", args=(str(checkpoint), str(zero_adapter))
        )
        restored = evaluate(0, record=False)
        matches = [row["completion"] for row in base_rows] == [
            row["completion"] for row in restored
        ]
        weights_match = zero_hash == engine.collective_rpc("probe_weight_fingerprint")
        write_json(
            output / "final_controls.json",
            {"texts_equal": matches, "weights_equal": weights_match},
        )
        if not matches or not weights_match:
            write_json(output / "failed-restored-zero-rows.json", restored)
            raise ValueError("final restored-zero control failed; preserve outputs")
        write_json(
            output / "completion.json",
            {
                "training_rollouts": total_seen,
                "oracle_calls": oracle_calls,
                "generated_tokens": generated_tokens,
                "seconds": time.monotonic() - start,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in parameters
                ),
                "parameter_names": names,
                "layers": layers,
                "final_adapter": str(adapter),
            },
        )
        stage("complete")
        (output / "exit-status.txt").write_text("0\n")
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        (output / "exit-status.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
