#!/usr/bin/env python3
"""Full-model TRL/Adafactor training with explicitly replaced audit advantages.

Uses local upstream components without modifying their tracked source. Raw
advantage guarantees do not extend through nonlinear optimizer/clipping steps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import yaml

from probe_real_gradients import ROOT, UPSTREAM, dataset_items, local_path, write_json
from train_online_linear import checkpoint_for
from sentinel_repair.online_linear import (
    METHODS,
    TEXT_PREDICTED_METHODS,
    calibration_config,
    preaudit_scale_config,
)
from sentinel_repair.text_residual import text_residual_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--trigger", choices=("python", "Certainly"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--eval-prompts", type=int, default=128)
    parser.add_argument("--update-probe", action="store_true",
                        help="observe actual warm optimizer proposals without changing updates")
    parser.add_argument("--update-probe-proposal", choices=("text_prediction", "cheap_rloo"),
                        default="text_prediction")
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id)
        or args.seed < 0
        or args.steps < 1
        or args.save_every < 1
        or args.steps % args.save_every
        or not 1 <= args.max_tokens <= 2048
        or args.eval_prompts < 1
    ):
        parser.error("invalid run geometry")
    if args.update_probe and args.method != "text_prediction_only":
        parser.error("update observer requires text_prediction_only (past-only shadow predictor)")
    if not args.update_probe and args.update_probe_proposal != "text_prediction":
        parser.error("a proposal override requires --update-probe")
    output = local_path(ROOT / "runs/local" / args.run_id)
    if output.exists():
        parser.error("new run id required; preserve old runs")
    checkpoint = (
        local_path(args.checkpoint) if args.checkpoint else checkpoint_for(args.trigger)
    )
    output.mkdir(parents=True)
    (output / "launcher.txt").write_text(f"pid={os.getpid()}\n")
    (output / "stage.txt").write_text("initialize\n")
    try:
        os.environ["SENTINEL_REPAIR_LIVE_AUDIT"] = "0"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        source = (
            UPSTREAM
            / "configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token"
            / f"{args.trigger}.yaml"
        )
        config = yaml.safe_load(source.read_text())
        config.update(
            model_name=str(checkpoint),
            hf_username="local",
            use_wandb=False,
            use_neptune=False,
            use_peft=False,
            pretrained_model=False,
            seed=args.seed,
            set_seed=True,
            resume_from_checkpoint=None,
            custom_name=f"sr_{args.run_id}",
            move_to=None,
            grpo_debug_log_dir=str(output / "debug_logs"),
        )
        config["mixup"]["strategy"] = "targeted"
        config["training_args"].update(
            seed=args.seed,
            push_to_hub=False,
            report_to="none",
            save_strategy="steps",
            save_steps=args.save_every,
            save_only_model=True,
            save_total_limit=None,
            max_steps=args.steps,
            logging_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=128,
            generation_batch_size=256,
            num_generations=4,
            num_iterations=1,
            max_completion_length=args.max_tokens,
            vllm_max_model_length=4096,
            vllm_gpu_memory_utilization=0.25,
            gradient_checkpointing=True,
            warmup_steps=0,
            learning_rate=5e-6,
            optim="adafactor",
            bf16=True,
            lr_scheduler_type="constant",
            beta=0.0,
            epsilon=0.2,
            max_grad_norm=1.0,
            loss_type="dapo",
            importance_sampling_level="token",
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_mode="sequence_mask",
            vllm_importance_sampling_cap=3.0,
            scale_rewards="group" if args.method == "cheap_grpo" else "none",
        )
        # TrainConfig accepts YAML null for no resume, not an explicitly passed
        # boolean false. Construct it directly without CLI string conversion.
        from transformers import set_seed
        from src.configs import TrainConfig
        from src.train.train_tools import TrainTools
        from src.train.grpo import GRPOTrainerDebug
        from sentinel_repair.full_linear import make_full_linear_trainer_class

        train_config = TrainConfig(**config)
        set_seed(args.seed)
        tools = TrainTools(train_config, "grpo")
        tools.model_dir = output / "models"
        tools.model_dir.mkdir(parents=True)
        tokenizer = tools.tokenizer
        dataset = tools.dataset
        raw = dataset.raw_dataset.dataset
        eval_seed = 202700001 + args.seed * 100000 + 50000
        evaluation = dataset_items(eval_seed, args.eval_prompts)
        train_questions = {raw[i]["question"] for i in range(len(raw))}
        eval_questions = {evaluation[i]["question"] for i in range(args.eval_prompts)}
        overlap = len(train_questions & eval_questions)
        if overlap or len(eval_questions) != args.eval_prompts:
            raise ValueError(
                "actual full-training/evaluation question overlap or duplicate"
            )
        split = {
            "training_composite_seed": 42,
            "training_pool_size": len(raw),
            "training_unique_questions": len(train_questions),
            "eval_seed": eval_seed,
            "eval_prompts": args.eval_prompts,
            "actual_overlap": overlap,
        }
        training_args = tools.training_args
        training_args.output_dir = str(output / "models")
        manifest = {
            **vars(args),
            "checkpoint": str(checkpoint),
            "batch_size": 256,
            "group_size": 4,
            "learning_rate": 5e-6,
            "optimizer": "Adafactor",
            "rendezvous": {
                name: os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT")
            },
            "loss_type": "dapo",
            "dtype": "bfloat16",
            "trainable": "all model parameters",
            "question_split_audit": split,
            "training_config": config,
            "resolved_training_geometry": {
                name: getattr(training_args, name)
                for name in (
                    "per_device_train_batch_size",
                    "gradient_accumulation_steps",
                    "generation_batch_size",
                    "steps_per_generation",
                    "num_generations",
                    "num_iterations",
                    "loss_type",
                    "scale_rewards",
                    "importance_sampling_level",
                    "vllm_importance_sampling_correction",
                    "vllm_importance_sampling_mode",
                    "vllm_importance_sampling_cap",
                    "max_grad_norm",
                    "learning_rate",
                    "warmup_steps",
                    "beta",
                    "epsilon",
                    "epsilon_high",
                )
            },
            "utc_started": datetime.now(timezone.utc).isoformat(),
            "scope": "practical full-training continuation; optimizer/clipping are nonlinear; not sequence-SGD probe",
        }
        if args.method == "calibrated_ipw":
            manifest["calibration"] = calibration_config()
        if args.method == "preaudit_scaled_ipw":
            manifest["preaudit_scaling"] = preaudit_scale_config()
        if args.method in TEXT_PREDICTED_METHODS:
            manifest["text_prediction"] = text_residual_config(
                include_residual=args.method == "text_direct_ipw"
            )
        if args.update_probe:
            manifest["update_probe"] = {
                "mode": "observe_only_no_gating",
                "proposal": args.update_probe_proposal,
                "current_labels": "uniform 1% for scalar estimate and future predictor fitting, never current proposal",
                "policy_objective": "RLOO(text_prediction)" if args.update_probe_proposal == "text_prediction" else "RLOO(cheap); predictor is shadow-only",
                "target": "finite same-batch group-centered token log-probability surrogate, not population return",
            }
        write_json(output / "config.json", manifest)
        model = tools.model
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("full-training run has frozen parameters")
        grpo_dataset, reward_function = dataset.prepare_for_grpo()
        for key in (
            "_oracle_rewards",
            "_gt_answers",
            "_verifier_reasoning",
            "_rubrics",
            "_verifier_cot",
        ):
            setattr(reward_function, key, [])
        trainer_class = make_full_linear_trainer_class(GRPOTrainerDebug)
        extra_kwargs = {}
        if args.update_probe:
            from sentinel_repair.update_gate_trainer import make_update_gate_observer_class
            trainer_class = make_update_gate_observer_class(trainer_class)
            extra_kwargs = {"update_probe_output_dir": output,
                            "update_probe_proposal": args.update_probe_proposal}
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=grpo_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_function,
            mixup=dataset.mixup,
            train_tools=tools,
            linear_method=args.method,
            linear_audit_seed=args.seed + 202609500,
            linear_output_dir=output,
            **extra_kwargs,
        )
        (output / "stage.txt").write_text("training\n")
        started = time.monotonic()
        trainer.train(resume_from_checkpoint=None)
        final = output / "models" / f"checkpoint-{args.steps}"
        if not final.is_dir():
            raise ValueError("expected final checkpoint missing")
        write_json(
            output / "completion.json",
            {
                "steps": args.steps,
                "seconds": time.monotonic() - started,
                "final_checkpoint": str(final),
                "training_only": True,
                "heldout_evaluation": "required in a separate process after releasing training GPU",
            },
        )
        (output / "stage.txt").write_text("complete\n")
        (output / "exit-status.txt").write_text("0\n")
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        (output / "exit-status.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
