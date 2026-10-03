"""Observer-only diagnostics of actual, warm text-prediction policy updates.

Wrap the class returned by ``make_full_linear_trainer_class``. The observer
does not propose, accept, reject, shrink, or undo an update. A post-step hook on
the real optimizer measures the just-completed update on the cached generation
batch, including the historical Adafactor moments, clipping and TRL loss used
by that training run. It never reads the logger's complete oracle rewards.
The explicitly selected ``cheap_rloo`` positive control uses a shadow text
predictor while the policy trains on cheap rewards; its ledger is tagged before
the parent writes it, not retrospectively relabeled as a text-policy update.

The measured scalar is a finite, same-batch log-probability surrogate. Sparse
purchase correction estimates that surrogate, not finite-step population
return. The actual proposal depends on frozen text predictions but not on its
current audit labels, because only ``text_prediction_only`` is supported.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .linear_audit import rloo
from .update_gate import estimate_frozen_surrogate, frozen_batch_target


ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_TENSORS = (
    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
    "old_per_token_logps",
)


def _inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("warm-observer outputs must remain inside the isolated project")
    return path


def _tensor_hash(tensor):
    # Token/mask snapshots have integer dtypes, so NumPy preserves their bytes.
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def preserve_observer_state(model):
    """Additional scoring must not advance random streams or change modes."""
    import torch

    cpu_rng = torch.random.get_rng_state()
    device = next(model.parameters()).device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def learner_ledger_snapshot(ledger, expected_step):
    """Whitelist only frozen predictions, public cheap rewards and purchases."""
    if (
        ledger.get("method") != "text_prediction_only"
        or ledger.get("batch_index") != expected_step
        or ledger.get("global_step_before_update") != expected_step - 1
        or ledger.get("batch_size") != 256
        or ledger.get("group_size") != 4
    ):
        raise ValueError("warm observer requires the aligned text-prediction-only ledger")
    m = np.asarray(ledger["text_reward_predictions"], dtype=np.float64)
    cheap = np.asarray(ledger["cheap_rewards"], dtype=np.float64)
    indices = np.asarray(ledger["audit_indices"])
    labels = np.asarray(ledger["bought_labels"], dtype=np.float64)
    k = expected_step * 256 // 100 - (expected_step - 1) * 256 // 100
    if m.shape != (256,) or cheap.shape != (256,) or not np.isfinite(m).all():
        raise ValueError("one finite frozen prediction and cheap reward per response required")
    if not np.isin(cheap, [0.0, 1.0]).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("binary cheap and purchased labels required")
    if (
        indices.shape != (k,) or indices.dtype.kind not in "iu"
        or labels.shape != (k,) or len(np.unique(indices)) != k
        or np.any(indices < 0) or np.any(indices >= 256)
        or ledger["audit_k"] != k or ledger["inclusion_probability"] != k / 256
        or ledger["oracle_calls"] != expected_step * 256 // 100
    ):
        raise ValueError("unaligned sparse purchase budget or indices")
    if ledger["text_prediction"]["labels_seen"] != (expected_step - 1) * 256 // 100:
        raise ValueError("predictor snapshot must precede current purchases")
    return {
        "m": m.copy(), "cheap": cheap.copy(), "indices": indices.astype(np.int64, copy=True),
        "labels": labels.copy(), "k": k, "p": k / 256,
        "past_label_count": ledger["text_prediction"]["labels_seen"],
    }


class WarmUpdateObserverMixin:
    """Place before the already-constructed full-linear trainer class in MRO."""

    def __init__(
        self, *args, update_observer_microbatch_size=2,
        update_probe_output_dir=None, update_probe_proposal="text_prediction", **kwargs,
    ):
        if kwargs.get("linear_method") != "text_prediction_only":
            raise ValueError("warm observer only supports text_prediction_only; no current-label proposal")
        if update_observer_microbatch_size != 2:
            raise ValueError("fixed warm observer uses scoring microbatch size 2")
        if update_probe_proposal not in ("text_prediction", "cheap_rloo"):
            raise ValueError("unknown warm-observer proposal")
        self.update_probe_proposal = update_probe_proposal
        output = _inside(Path(update_probe_output_dir or kwargs["linear_output_dir"]) / "update_gate_steps")
        if output.exists():
            raise ValueError("warm-observer output already exists")
        self._update_observer_dir = output
        self._update_observer_microbatch_size = update_observer_microbatch_size
        self._update_observer_pending = None
        self._update_observer_hook = None
        self._update_observer_optimizer = None
        self._update_observer_steps = 0
        super().__init__(*args, **kwargs)
        self._validate_update_observer_configuration()
        output.mkdir(parents=True, exist_ok=False)

    def _validate_update_observer_configuration(self):
        import torch

        if self.linear_method != "text_prediction_only":
            raise ValueError("observer requires frozen-prediction-only policy updates")
        if (
            self.accelerator.num_processes != 1 or self.num_generations != 4
            or self.args.generation_batch_size != 256 or self.num_iterations != 1
            or self.args.gradient_accumulation_steps * self.args.per_device_train_batch_size != 256
            or self.args.steps_per_generation != self.args.gradient_accumulation_steps
        ):
            raise ValueError("one four-member 256-response generation batch per optimizer step required")
        if getattr(self.args, "mask_truncated_completions", False):
            raise ValueError("warm observer currently requires retained truncated completions")
        if getattr(self, "tools", None):
            raise ValueError("warm observer currently supports text-only, no tool-masked tokens")
        # The upstream flag may be False while Qwen's effective dropout is
        # already zero. Inspect the modules actually executed, rather than
        # changing training settings merely to satisfy a flag check.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0:
                raise ValueError("same-model diagnostic requires zero effective dropout")
            attention_dropout = getattr(module, "attention_dropout", 0.0)
            if isinstance(attention_dropout, (int, float)) and attention_dropout != 0:
                raise ValueError("same-model diagnostic requires zero effective attention dropout")
        if getattr(self.args, "loss_type", "dapo") != "dapo":
            raise ValueError("warm observer requires the existing DAPO objective")

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        raw = self.optimizer
        # Trainer normally creates the raw optimizer before Accelerate wraps it.
        # If called again after wrapping, walk to the same underlying optimizer.
        # AcceleratedOptimizer inherits register_step_post_hook but delegates
        # real steps to .optimizer, so unwrap even when the wrapper has the API.
        while hasattr(raw, "optimizer") and raw.optimizer is not raw:
            raw = raw.optimizer
        if self._update_observer_hook is None:
            if not hasattr(raw, "register_step_post_hook"):
                raise ValueError("optimizer does not support a post-step observer hook")
            self._update_observer_optimizer = raw
            self._update_observer_hook = raw.register_step_post_hook(self._observe_actual_optimizer_step)
        elif raw is not self._update_observer_optimizer:
            raise ValueError("observer optimizer was unexpectedly replaced")
        return optimizer

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self.model.training:
            pending = self._linear_pending
            if pending is None:
                raise RuntimeError("full-linear parent did not prepare a reward batch")
            pending["record"]["update_probe_proposal"] = self.update_probe_proposal
            pending["record"]["update_probe_mode"] = "observer_only; no gate"
            if self.update_probe_proposal == "cheap_rloo":
                # Explicit positive control: purchases still train a SHADOW
                # predictor, but policy updates use cheap rewards. Intercept
                # before parent writes the ledger/logs, never rewrite history.
                pending["advantage"] = rloo(
                    np.asarray(pending["record"]["cheap_rewards"]).reshape(-1, 4)
                ).ravel()
                pending["record"]["text_control"] = (
                    "shadow predictor only; actual policy advantage is RLOO(cheap); no gate"
                )
        return rewards

    def _generate_and_score_completions(self, inputs):
        import torch

        training = self.model.training
        if training and self._update_observer_pending is not None:
            raise RuntimeError("previous generation batch has not reached an optimizer step")
        outputs = super()._generate_and_score_completions(inputs)
        if not training:
            return outputs
        self._validate_update_observer_configuration()
        step = int(self.state.global_step) + 1
        if step != self._update_observer_steps + 1:
            raise ValueError("warm observer does not support skipped or resumed optimizer steps")
        if any(key not in outputs or outputs[key] is None for key in _REQUIRED_TENSORS):
            raise ValueError("need actual cached TRL old_per_token_logps and token/mask tensors")
        if any(key in outputs for key in ("pixel_values", "tool_mask", "token_type_ids")):
            raise ValueError("observer only supports the current text-only scoring path")
        tensors = {key: outputs[key].detach().cpu().clone() for key in _REQUIRED_TENSORS}
        for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask"):
            if tensors[key].ndim != 2 or tensors[key].shape[0] != 256:
                raise ValueError("observer token tensors must have 256 aligned rows")
        if tensors["prompt_ids"].shape != tensors["prompt_mask"].shape:
            raise ValueError("prompt token/mask shape mismatch")
        shape = tensors["completion_ids"].shape
        if shape != tensors["completion_mask"].shape or shape != tensors["old_per_token_logps"].shape:
            raise ValueError("completion token/mask/old-logprob shape mismatch")
        if not 1 <= shape[1] <= 2048:
            raise ValueError("unexpected completion width outside fixed max2048")
        for key in ("prompt_mask", "completion_mask"):
            if not torch.all((tensors[key] == 0) | (tensors[key] == 1)):
                raise ValueError("token masks must be binary")
        if not torch.isfinite(tensors["old_per_token_logps"]).all():
            raise ValueError("nonfinite cached pre-update log probabilities")
        for start in range(0, 256, 4):
            if any(
                not torch.equal(tensors[key][start], tensors[key][index])
                for index in range(start + 1, start + 4)
                for key in ("prompt_ids", "prompt_mask")
            ):
                raise ValueError("cached rows no longer form same-question groups")
        count = outputs["num_items_in_batch"]
        count = int(count.item()) if hasattr(count, "item") else int(count)
        if count < 1 or count != int(tensors["completion_mask"].sum()):
            raise ValueError("actual DAPO denominator differs from cached completion token count")
        ledger_path = _inside(self._linear_logdir / f"batch-{step:04d}.json")
        record = json.loads(ledger_path.read_text())
        if record.get("update_probe_proposal") != self.update_probe_proposal:
            raise ValueError("ledger proposal provenance does not match this observer run")
        ledger = learner_ledger_snapshot(record, step)
        # Parent has already verified input/advantage order before writing this
        # ledger. Copying happens before TRL later shuffles/splits the batch.
        proposal_rewards = ledger["cheap"] if self.update_probe_proposal == "cheap_rloo" else ledger["m"]
        if not np.allclose(
            np.asarray(outputs["advantages"].detach().float().cpu()).reshape(-1),
            rloo(proposal_rewards.reshape(-1, 4)).ravel(),
            rtol=2e-6, atol=2e-7,
        ):
            raise ValueError("cached proposal advantages disagree with the declared proposal")
        importance = outputs.get("importance_sampling_ratio")
        if importance is not None:
            importance = importance.detach().float().cpu().clone()
            if importance.shape not in ((256, 1), shape) or not torch.isfinite(importance).all():
                raise ValueError("unexpected vLLM importance correction snapshot")
            tensors["importance_sampling_ratio"] = importance
        self._update_observer_pending = {
            "step": step, "tensors": tensors, "ledger": ledger,
            "num_items_in_batch": count, "temperature": float(self.temperature),
            "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        }
        return outputs

    def _observer_new_logps(self, pending):
        import torch

        tensors = pending["tensors"]
        device = next(self.model.parameters()).device
        ids = torch.cat((tensors["prompt_ids"], tensors["completion_ids"]), dim=1).to(device)
        mask = torch.cat((tensors["prompt_mask"], tensors["completion_mask"]), dim=1).to(device)
        if float(self.temperature) != pending["temperature"]:
            raise ValueError("temperature changed between pre/post-update scoring")
        with preserve_observer_state(self.model):
            # Imports can themselves initialize random-dependent machinery.
            # Keep the first-use lazy import inside the preservation boundary.
            from trl.models.utils import disable_gradient_checkpointing

            with torch.no_grad(), disable_gradient_checkpointing(
                self.model, getattr(self.args, "gradient_checkpointing_kwargs", None)
            ):
                logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, ids, mask, tensors["completion_ids"].shape[1],
                    batch_size=self._update_observer_microbatch_size, compute_entropy=False,
                )
                return logps.detach().float().cpu()

    def _observe_actual_optimizer_step(self, optimizer, args, kwargs):
        # Protect the entire observation, including lazy I/O imports, not only
        # the extra model forward. This is defensive, not a diagnosed cause of
        # cross-run vLLM/BF16 non-bitwise trajectories.
        with preserve_observer_state(self.model):
            return self._record_actual_optimizer_step(optimizer, args, kwargs)

    def _record_actual_optimizer_step(self, optimizer, args, kwargs):
        import torch

        started = time.monotonic()
        verify_parameters = os.environ.get("SENTINEL_UPDATE_OBSERVER_VERIFY_PARAMETERS") == "1"
        frozen_parameters = [p.detach().clone() for p in self.model.parameters()] if verify_parameters else None
        pending = self._update_observer_pending
        if pending is None:
            raise RuntimeError("real optimizer step has no aligned cached generation batch")
        if optimizer is not self._update_observer_optimizer:
            raise ValueError("observer hook called by an unexpected optimizer")
        step = pending["step"]
        if int(self.state.global_step) != step - 1:
            raise ValueError("post-step hook must run before Trainer increments global_step")
        after = self._observer_new_logps(pending)
        if frozen_parameters is not None:
            if not all(torch.equal(p.detach(), old) for p, old in zip(self.model.parameters(), frozen_parameters, strict=True)):
                raise AssertionError("passive observer changed a model parameter")
            del frozen_parameters
        tensors, ledger = pending["tensors"], pending["ledger"]
        old = tensors["old_per_token_logps"].float()
        mask = tensors["completion_mask"].double()
        if after.shape != old.shape or not torch.isfinite(after).all():
            raise ValueError("post-update token log probabilities are invalid")
        before_sums = (old.double() * mask).sum(dim=1).numpy()
        after_sums = (after.double() * mask).sum(dim=1).numpy()
        delta = ((after.double() - old.double()) * mask).sum(dim=1).numpy()
        v = 256 / pending["num_items_in_batch"] * delta
        estimate = estimate_frozen_surrogate(
            v.reshape(-1, 4), ledger["m"].reshape(-1, 4),
            ledger["indices"], ledger["labels"], audit_budget=ledger["k"],
        )
        arrays = {
            "v": v, "sequence_delta": delta, "logp_before": before_sums,
            "logp_after": after_sums, "m": ledger["m"], "cheap": ledger["cheap"],
            "audit_indices": ledger["indices"], "bought_labels": ledger["labels"],
            "completion_lengths": mask.sum(dim=1).numpy().astype(np.int64),
            "group_id": np.arange(256) // 4, "member": np.arange(256) % 4,
            **{key: value.float().numpy() if value.is_floating_point() else value.numpy() for key, value in tensors.items()},
            "new_per_token_logps": after.numpy(),
        }
        target = self._update_observer_dir / f"step-{step:04d}.npz"
        with target.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        record = {
            "step": step, "global_step_during_hook": int(self.state.global_step),
            "mode": "observer_only; actual applied update is never changed", "method": self.linear_method,
            "proposal": self.update_probe_proposal,
            "predictor_role": "shadow only" if self.update_probe_proposal == "cheap_rloo" else "policy reward prediction",
            "batch_size": 256, "group_size": 4, "max_completion_tokens": 2048,
            "num_items_in_batch": pending["num_items_in_batch"],
            "actual_completion_width": int(mask.shape[1]),
            "temperature": pending["temperature"],
            "audit_k": ledger["k"], "audit_p": ledger["p"],
            "past_predictor_labels": ledger["past_label_count"],
            "estimated_oracle_surrogate": estimate,
            "prediction_only_surrogate": frozen_batch_target(v.reshape(-1, 4), ledger["m"].reshape(-1, 4)),
            "cheap_surrogate": frozen_batch_target(v.reshape(-1, 4), ledger["cheap"].reshape(-1, 4)),
            "sequence_delta_abs_max": float(np.abs(delta).max()),
            "sequence_delta_rms": float(np.sqrt(np.mean(delta**2))),
            "optimizer_class": type(optimizer).__name__,
            "optimizer_history": "actual live optimizer state, not reset or reconstructed",
            "vllm_importance_correction_saved": "importance_sampling_ratio" in tensors,
            "pre_logp_source": "TRL cached old_per_token_logps on training model, not vLLM sampling scores",
            "post_logp_source": "same trainer helper/temperature after real optimizer.step; no_grad; microbatch2",
            "module_modes_and_rng_preserved": True,
            "observer_parameter_identity_check": True if verify_parameters else None,
            "complete_truth_read": False, "gate_applied": False,
            "current_audit_independence": (
                "v is measured after purchase but the actual proposal uses only frozen m or cheap rewards; "
                "current labels update the predictor for later batches, not this step"
            ),
            "ledger_sha256": pending["ledger_sha256"],
            "prompt_ids_sha256": _tensor_hash(tensors["prompt_ids"]),
            "completion_ids_sha256": _tensor_hash(tensors["completion_ids"]),
            "arrays": target.name,
            "limits": [
                "Finite group-centered same-batch log-probability surrogate, not population return or a safety certificate.",
                "Current purchased labels do not form this policy update; they fit the predictor for later batches and correct this observer estimate only.",
                "Post-step measurement includes the actual optimizer/clipping/TRL update, but scalar-surrogate unbiasedness does not imply update unbiasedness.",
                "This batch formed the policy update; out-of-batch and long-term effects require separate evaluation.",
                "v uses unweighted token-logprob displacement; saved importance corrections describe the actual proposal, not an extra weight silently added to the surrogate.",
            ],
            "observer_seconds": time.monotonic() - started,
        }
        record["estimated_false_positive_surrogate"] = record["cheap_surrogate"] - estimate
        with (self._update_observer_dir / f"step-{step:04d}.json").open("x") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        self._update_observer_pending = None
        self._update_observer_steps = step


def make_update_observer_trainer_class(full_linear_class):
    """Wrap an existing full-linear trainer class without changing its source."""
    return type(
        f"WarmObserved{full_linear_class.__name__}",
        (WarmUpdateObserverMixin, full_linear_class), {},
    )


def make_update_gate_observer_class(base_class):
    """Public runner API; base_class is already a full-linear trainer class."""
    return make_update_observer_trainer_class(base_class)


__all__ = [
    "WarmUpdateObserverMixin", "make_update_observer_trainer_class",
    "make_update_gate_observer_class",
]
