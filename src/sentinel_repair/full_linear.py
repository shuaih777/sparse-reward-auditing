"""A narrow TRL 0.26.2 bridge for the online audit objectives.

Place ``FullLinearAuditMixin`` before ``GRPOTrainerDebug`` in the MRO, or use
``make_full_linear_trainer_class``. The reward hook keeps cheap binary rewards
and the oracle logger's buffer intact. Final advantages are substituted after
generation/scoring returns and before TRL shuffles and splits the batch.

Conditional unbiasedness over audit draws applies to unscaled ``linear_ipw``,
``oracle_only``, ``oracle_centered`` and past-only ``calibrated_ipw`` and
``text_direct_ipw``, not to ``linear_replace`` or the
nonlinear ``group_scaled_ipw`` ablation. The latter is scaled only in the
shared advantage helper; TRL must still use ``scale_rewards='none'``.
``preaudit_scaled_ipw`` instead fixes the denominator from cheap rewards before
audit sampling. Its conditional mean is oracle RLOO divided by that cheap-only
denominator, not unscaled oracle RLOO; it also requires no extra TRL scaling.
Both full-oracle methods buy every label as high-budget diagnostic references;
``full_oracle_group_scaled`` also applies scaling only in that shared helper.
``text_prediction_only`` buys the same sparse labels to fit the same past-only
predictor, but uses RLOO(m) without residual correction and is generally biased.
TRL's default DAPO loss weights tokens using the generation-batch token count;
PPO clipping can depend on an estimated advantage's sign. Gradient clipping,
Adam, and the original configs' Adafactor are nonlinear updates. In particular,
Adafactor retains second moments and its own RMS update clipping even when
``max_grad_norm=0``. This bridge makes no parameter-update unbiasedness claim.
"""

from __future__ import annotations

import json
import os
from numbers import Integral
from pathlib import Path

import numpy as np

from .online_linear import (
    FULL_ORACLE_METHODS,
    METHODS,
    PastAuditCalibrator,
    SPARSE_METHODS,
    TEXT_PREDICTED_METHODS,
    audit_allowance,
    linear_advantages,
    preaudit_denominators,
    preaudit_scale_config,
    uniform_audit_indices,
)
from .text_residual import TextResidualBank, prepare_text_batch, text_residual_config

_CHEAP_METHODS = {"cheap_grpo", "cheap_rloo"}


def _completion_text(completion):
    """Use generated content only, identically to the arithmetic reward hook.

    TRL conversational generations are one assistant message, while the fake
    parent and nonconversational generation path can return a plain string.
    Fail on other structures instead of leaking role/metadata through ``str``.
    Quotes, newlines and all actual generated text are preserved verbatim.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and len(completion) == 1:
        completion = completion[0]
    if (
        isinstance(completion, dict)
        and completion.get("role") == "assistant"
        and isinstance(completion.get("content"), str)
    ):
        return completion["content"]
    raise ValueError("text prediction requires a string or one assistant text message")


class FullLinearAuditMixin:
    """Single-rank, four-member, 256-rollout integration; no resume or live gate.

    Additional constructor keywords are ``linear_method``,
    ``linear_audit_seed``, and ``linear_output_dir``. Per-batch records go in
    ``linear_output_dir/linear_audits``. A fresh directory and trainer are
    required for every arm. Evaluation calls pass through without purchases.
    """

    def __init__(
        self,
        *args,
        linear_method: str,
        linear_audit_seed: int,
        linear_output_dir: str | Path,
        **kwargs,
    ):
        if linear_method not in METHODS:
            raise ValueError(f"unknown linear method: {linear_method}")
        if (
            isinstance(linear_audit_seed, bool)
            or not isinstance(linear_audit_seed, Integral)
            or linear_audit_seed < 0
        ):
            raise ValueError("linear_audit_seed must be a nonnegative integer")
        if os.environ.get("SENTINEL_REPAIR_LIVE_AUDIT", "0") != "0":
            raise ValueError("linear audit cannot run with the old live-audit path")
        self.linear_method = linear_method
        self._linear_rng = np.random.default_rng(int(linear_audit_seed))
        self._linear_seen = self._linear_spent = self._linear_batches = 0
        self._linear_calibrator = (
            PastAuditCalibrator() if linear_method == "calibrated_ipw" else None
        )
        self._linear_text_bank = (
            TextResidualBank() if linear_method in TEXT_PREDICTED_METHODS else None
        )
        self._linear_pending = None
        self._linear_train_started = False
        self._linear_logdir = Path(linear_output_dir) / "linear_audits"
        if self._linear_logdir.exists():
            raise ValueError(
                f"linear audit output already exists: {self._linear_logdir}"
            )
        super().__init__(*args, **kwargs)
        self._validate_linear_configuration()
        self._linear_logdir.mkdir(parents=True, exist_ok=False)

    def _validate_linear_configuration(self):
        if self.accelerator.num_processes != 1 or self.accelerator.process_index != 0:
            raise ValueError("linear audit requires a single GPU/rank")
        if self.num_generations != 4:
            raise ValueError("linear audit requires group size 4")
        if self.args.generation_batch_size != 256:
            raise ValueError("linear audit requires generation_batch_size=256")
        accumulation = self.args.gradient_accumulation_steps
        if (
            self.args.steps_per_generation != accumulation
            or self.args.per_device_train_batch_size * accumulation != 256
        ):
            raise ValueError(
                "one 256-rollout generation batch must equal one optimizer batch"
            )
        if self.num_iterations != 1:
            raise ValueError("linear audit requires num_iterations=1")
        if self.beta != 0:
            raise ValueError("linear audit requires beta=0")
        expected_scale = "group" if self.linear_method == "cheap_grpo" else "none"
        if self.scale_rewards != expected_scale:
            raise ValueError(
                f"{self.linear_method} requires scale_rewards={expected_scale!r}"
            )
        if self.args.logging_steps != 1:
            raise ValueError("linear audit requires logging_steps=1")
        strategy = getattr(
            self.args.logging_strategy, "value", self.args.logging_strategy
        )
        if strategy != "steps":
            raise ValueError("linear audit requires logging_strategy='steps'")
        if len(self.reward_funcs) != 1 or len(self.reward_weights) != 1:
            raise ValueError("linear audit requires exactly one reward function")
        if float(self.reward_weights[0]) != 1.0:
            raise ValueError("linear audit requires reward weight 1")
        if getattr(self, "_live_audit_enabled", False):
            raise ValueError("linear audit cannot run with the old live-audit path")
        config = getattr(getattr(self, "train_tools", None), "training_config", None)
        if config is not None and getattr(config, "resume_from_checkpoint", None):
            raise ValueError("linear audit does not support checkpoint resume")

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is not None and resume_from_checkpoint is not False:
            raise ValueError("linear audit does not support checkpoint resume")
        if kwargs.get("model_path") is not None:
            raise ValueError(
                "linear audit does not support the model_path resume alias"
            )
        if self._linear_train_started or self._linear_seen:
            raise ValueError("use a fresh trainer for each linear-audit training run")
        self._validate_linear_configuration()
        self._linear_train_started = True
        return super().train(resume_from_checkpoint, *args, **kwargs)

    @staticmethod
    def _linear_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        training = self.model.training
        needs_labels = training and self.linear_method not in _CHEAP_METHODS
        reward_func = self.reward_funcs[0]
        before = getattr(reward_func, "_oracle_rewards", None) if needs_labels else None
        before_count = len(before) if before is not None else 0
        cheap_global = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        if not training:
            return cheap_global
        if self._linear_pending is not None:
            raise RuntimeError(
                "a previous reward batch was not consumed by generation/scoring"
            )
        n = 256
        if any(
            len(rows) != n
            for rows in (inputs, prompts, completions, completion_ids_list)
        ):
            raise ValueError("linear reward hook requires 256 aligned input rows")
        if tuple(cheap_global.shape) != (n, 1):
            raise ValueError("linear reward hook requires cheap reward shape (256, 1)")
        for index, prompt in enumerate(prompts):
            if inputs[index]["prompt"] != prompt:
                raise ValueError("input/reward prompts are misaligned")
            if prompt != prompts[index - index % 4]:
                raise ValueError(
                    "reward rows do not form contiguous four-member prompt groups"
                )
        cheap = self._linear_numpy(cheap_global).reshape(n).astype(np.float64)
        if not np.isin(cheap, [0.0, 1.0]).all():
            raise ValueError("linear audit requires finite binary cheap rewards")

        # This scale is fully determined before selecting or reading any
        # purchased labels. linear_advantages uses the same cheap-only helper.
        preaudit_scale = (
            preaudit_denominators(cheap.reshape(-1, 4))
            if self.linear_method == "preaudit_scaled_ipw"
            else None
        )
        baseline_kwargs, calibration = {}, None
        if self._linear_calibrator is not None:
            calibration = self._linear_calibrator.snapshot()
            if calibration["past_label_count"] != self._linear_spent:
                raise ValueError("calibration history disagrees with prior purchases")
            baseline_kwargs["baseline"] = self._linear_calibrator.predict(
                cheap
            ).reshape(-1, 4)
        text_batch, text_prediction = None, None
        if self._linear_text_bank is not None:
            text_prediction = self._linear_text_bank.snapshot()
            if text_prediction["labels_seen"] != self._linear_spent:
                raise ValueError("text prediction history disagrees with prior purchases")
            text_batch = prepare_text_batch(
                cheap.reshape(-1, 4), [_completion_text(row) for row in completions]
            )
            # Both features and predictions are frozen before audit sampling.
            # prepare_text_batch never accepts item/prompt metadata or truth.
            baseline_kwargs["baseline"] = self._linear_text_bank.predict(text_batch)[
                "text_direct"
            ]
        k = (
            audit_allowance(self._linear_seen, n)
            if self.linear_method in SPARSE_METHODS
            else 0
        )
        indices = uniform_audit_indices(n, k, self._linear_rng)
        if self.linear_method in FULL_ORACLE_METHODS:
            k, indices = n, np.arange(n, dtype=np.int64)
        audited = np.full(n, np.nan)
        if needs_labels:
            after = getattr(reward_func, "_oracle_rewards", None)
            if after is None or len(after) != n:
                raise ValueError(
                    "oracle cache must contain exactly this batch's 256 labels"
                )
            if after is before and before_count:
                raise ValueError(
                    "oracle cache was not replaced; stale label alignment is ambiguous"
                )
            # The evaluator/logger owns the full cache. Only purchased entries
            # cross into the estimator, and the cache is never cleared here.
            bought = np.asarray(
                [after[int(index)] for index in indices], dtype=np.float64
            )
            if not np.isin(bought, [0.0, 1.0]).all():
                raise ValueError("purchased oracle labels must be finite and binary")
            audited[indices] = bought
        advantage = None
        if self.linear_method != "cheap_grpo":
            # online_linear already includes G/(G-1). Do not apply it twice.
            advantage = linear_advantages(
                self.linear_method,
                cheap.reshape(-1, 4),
                audited.reshape(-1, 4),
                k / n,
                **baseline_kwargs,
            ).reshape(n)
            if not np.isfinite(advantage).all():
                raise ValueError("nonfinite linear advantage")
        if self._linear_calibrator is not None:
            self._linear_calibrator.observe(cheap[indices], audited[indices])
        if self._linear_text_bank is not None:
            self._linear_text_bank.observe(text_batch, indices, audited[indices])

        self._linear_seen += n
        self._linear_spent += k
        self._linear_batches += 1
        if (
            self.linear_method in SPARSE_METHODS
            and self._linear_spent != self._linear_seen // 100
        ):
            raise AssertionError("strict 1% completed-batch prefix budget violated")
        self._linear_pending = {
            "advantage": advantage,
            "completion_ids": [tuple(ids) for ids in completion_ids_list],
            "record": {
                "method": self.linear_method,
                "batch_index": self._linear_batches,
                "global_step_before_update": int(self.state.global_step),
                "batch_size": n,
                "group_size": 4,
                "audit_indices": indices.tolist(),
                "bought_labels": audited[indices].tolist(),
                "audit_k": k,
                "inclusion_probability": k / n,
                "training_rollouts": self._linear_seen,
                "oracle_calls": self._linear_spent,
                "full_oracle_reference": self.linear_method in FULL_ORACLE_METHODS,
                "cheap_rewards": cheap.tolist(),
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
                **(
                    {
                        "text_prediction": text_prediction,
                        "text_prediction_config": text_residual_config(
                            include_residual=self.linear_method == "text_direct_ipw"
                        ),
                        "text_reward_predictions": baseline_kwargs["baseline"]
                        .ravel()
                        .tolist(),
                        "text_control": (
                            "per-response m + I/p * (oracle - m); linear RLOO"
                            if self.linear_method == "text_direct_ipw"
                            else "per-response m only; linear RLOO; no current-audit residual"
                        ),
                        "text_completion_source": "assistant content only, verbatim",
                        "bought_cheap": cheap[indices].tolist(),
                    }
                    if text_prediction is not None
                    else {}
                ),
            },
        }
        return cheap_global

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        if self._linear_pending is not None:
            raise RuntimeError("linear advantage cache was not empty before generation")
        outputs = super()._generate_and_score_completions(inputs)
        pending = self._linear_pending
        if pending is None:
            raise RuntimeError(
                "generation/scoring did not produce a linear reward batch"
            )
        if tuple(outputs["advantages"].shape) != (256,):
            raise ValueError("generation/scoring returned misaligned advantage shape")
        completion_ids = self._linear_numpy(outputs["completion_ids"])
        if completion_ids.ndim != 2 or len(completion_ids) != 256:
            raise ValueError("generation/scoring returned misaligned completion shape")
        for row, expected_ids in zip(
            completion_ids, pending["completion_ids"], strict=True
        ):
            if len(expected_ids) > len(row) or not np.array_equal(
                row[: len(expected_ids)], expected_ids
            ):
                raise ValueError(
                    "completion order changed between reward hook and advantage substitution"
                )
        logged = self._logs["advantages"]
        if len(logged) < 256:
            raise ValueError(
                "TRL advantage log does not contain the complete current batch"
            )
        if pending["advantage"] is not None:
            outputs["advantages"] = outputs["advantages"].new_tensor(
                pending["advantage"]
            )
            for _ in range(256):
                logged.pop()
            logged.extend(outputs["advantages"].tolist())
        actual = self._linear_numpy(outputs["advantages"])
        if not np.isfinite(actual).all():
            raise ValueError("nonfinite actual training advantage")
        if not np.array_equal(np.asarray(list(logged)[-256:]), actual):
            raise ValueError("actual training and logged advantages differ")
        record = pending["record"]
        record["advantages"] = actual.tolist()
        record["advantage_dtype"] = str(outputs["advantages"].dtype)
        destination = self._linear_logdir / f"batch-{record['batch_index']:04d}.json"
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
        self._linear_pending = None
        return outputs


def make_full_linear_trainer_class(base_class):
    """Construct the mixin/base MRO without importing TRL or upstream at import time."""
    return type(
        f"FullLinear{base_class.__name__}", (FullLinearAuditMixin, base_class), {}
    )


__all__ = ["FullLinearAuditMixin", "make_full_linear_trainer_class"]
