"""CPU post-optimizer-hook tests; no policy/GPU training is launched."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sentinel_repair.linear_audit import rloo
from sentinel_repair.update_gate import estimate_frozen_surrogate
from sentinel_repair.update_gate_trainer import (
    learner_ledger_snapshot,
    make_update_gate_observer_class,
    preserve_observer_state,
)


def ledger(step=1):
    n, k = 256, step * 256 // 100 - (step - 1) * 256 // 100
    return {
        "method": "text_prediction_only", "batch_index": step,
        "global_step_before_update": step - 1, "batch_size": n, "group_size": 4,
        "cheap_rewards": np.tile([1, 1, 0, 0], 64).tolist(),
        "text_reward_predictions": np.tile([.8, .2, .1, .1], 64).tolist(),
        "audit_indices": list(range(k)), "bought_labels": [0] * k,
        "audit_k": k, "inclusion_probability": k / n, "oracle_calls": step * n // 100,
        "text_prediction": {"labels_seen": (step - 1) * n // 100},
    }


class SmallModel(torch.nn.Module):
    is_gradient_checkpointing = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.child = torch.nn.Identity()


class FakeFullLinear:
    def __init__(self, *, linear_method, linear_output_dir, **kwargs):
        self.linear_method = linear_method
        self._linear_logdir = linear_output_dir / "linear_audits"
        self._linear_logdir.mkdir()
        self._linear_pending = None
        self.model = SmallModel()
        self.model.train()
        self.model.child.eval()  # Mixed module modes must survive observation.
        self.optimizer = None
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(num_processes=1)
        self.num_generations, self.num_iterations = 4, 1
        self.temperature, self.tools = 1., None
        self.args = SimpleNamespace(
            generation_batch_size=256, gradient_accumulation_steps=128,
            per_device_train_batch_size=2, steps_per_generation=128,
            mask_truncated_completions=False, disable_dropout=True, loss_type="dapo",
            gradient_checkpointing_kwargs=None,
        )

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        return self.optimizer

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        record = ledger(self.state.global_step + 1)
        self._linear_pending = {
            "record": record,
            "advantage": rloo(np.asarray(record["text_reward_predictions"]).reshape(-1, 4)).ravel(),
        }
        return None

    def _get_per_token_logps_and_entropies(self, model, ids, mask, keep, batch_size, compute_entropy=False):
        # Deliberately consume all RNGs; observer must restore these effects.
        torch.rand(1)
        random.random()
        np.random.random()
        return model.weight * (ids[:, -keep:].float() + 1) - 5, None

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return {"eval": True}
        self._calculate_rewards(inputs, [], [], [])
        pending = self._linear_pending
        prompt_ids = (torch.arange(256) // 4).view(-1, 1)
        completion_ids = torch.stack((torch.arange(256) % 4, torch.zeros(256, dtype=torch.long)), dim=1)
        prompt_mask, completion_mask = torch.ones_like(prompt_ids), torch.ones_like(completion_ids)
        joined = torch.cat((prompt_ids, completion_ids), dim=1)
        old, _ = self._get_per_token_logps_and_entropies(self.model, joined, torch.ones_like(joined), 2, 2)
        record = pending["record"]
        record["advantages"] = pending["advantage"].tolist()
        (self._linear_logdir / f"batch-{self.state.global_step + 1:04d}.json").write_text(json.dumps(record))
        self._linear_pending = None
        return {
            "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
            "completion_ids": completion_ids, "completion_mask": completion_mask,
            "old_per_token_logps": old, "advantages": torch.tensor(pending["advantage"], dtype=torch.float32),
            "num_items_in_batch": torch.tensor(512), "importance_sampling_ratio": torch.ones((256, 1)),
        }


Observed = make_update_gate_observer_class(FakeFullLinear)


def build(tmp_path, proposal="text_prediction"):
    return Observed(
        linear_method="text_prediction_only", linear_output_dir=tmp_path,
        update_probe_output_dir=tmp_path, update_probe_proposal=proposal,
    )


def test_ledger_boundary_and_stale_purchase_rejection():
    source = ledger()
    source["oracle_rewards"] = ["must not cross"] * 256
    result = learner_ledger_snapshot(source, 1)
    assert "oracle_rewards" not in result
    assert result["labels"].shape == (2,)
    source["text_prediction"]["labels_seen"] = 2
    with pytest.raises(ValueError, match="precede current"):
        learner_ledger_snapshot(source, 1)


def test_false_disable_dropout_flag_with_effective_zero_is_valid(tmp_path):
    trainer = build(tmp_path)
    trainer.args.disable_dropout = False
    trainer._validate_update_observer_configuration()
    trainer.model.child = torch.nn.Dropout(.1)
    with pytest.raises(ValueError, match="effective dropout"):
        trainer._validate_update_observer_configuration()


@pytest.mark.parametrize("proposal", ["text_prediction", "cheap_rloo"])
def test_actual_step_observed_once_with_rng_and_mode_preserved(tmp_path, proposal, monkeypatch):
    monkeypatch.setenv("SENTINEL_UPDATE_OBSERVER_VERIFY_PARAMETERS", "1")
    trainer = build(tmp_path, proposal)
    optimizer = trainer.create_optimizer()
    trainer.create_optimizer()
    assert len(optimizer._optimizer_step_post_hooks) == 1
    output = trainer._generate_and_score_completions(None)
    before_weight = trainer.model.weight.detach().clone()
    trainer.model.weight.grad = torch.tensor(1.)
    torch_state, py_state, np_state = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
    optimizer.step()
    torch.testing.assert_close(trainer.model.weight, before_weight - .01)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert trainer.model.training and not trainer.model.child.training
    assert trainer._update_observer_pending is None
    record = json.loads((tmp_path / "update_gate_steps/step-0001.json").read_text())
    data = np.load(tmp_path / "update_gate_steps/step-0001.npz")
    assert record["gate_applied"] is False and record["complete_truth_read"] is False
    assert record["observer_parameter_identity_check"] is True
    assert record["proposal"] == proposal
    assert "truth" not in data.files
    expected = estimate_frozen_surrogate(
        data["v"].reshape(-1, 4), data["m"].reshape(-1, 4),
        data["audit_indices"], data["bought_labels"], audit_budget=2,
    )
    assert record["estimated_oracle_surrogate"] == pytest.approx(expected)
    saved = json.loads((tmp_path / "linear_audits/batch-0001.json").read_text())
    rewards = saved["cheap_rewards"] if proposal == "cheap_rloo" else saved["text_reward_predictions"]
    np.testing.assert_allclose(output["advantages"], rloo(np.asarray(rewards).reshape(-1, 4)).ravel(), atol=1e-7)
    assert saved["update_probe_proposal"] == proposal
    # Follow-up batch has a true warm optimizer and the alternating k=3 budget.
    trainer.state.global_step = 1
    trainer._generate_and_score_completions(None)
    trainer.model.weight.grad = torch.tensor(1.)
    optimizer.step()
    second = json.loads((tmp_path / "update_gate_steps/step-0002.json").read_text())
    assert second["audit_k"] == 3


def test_observer_caches_detached_copies_before_later_batch_mutation(tmp_path):
    trainer = build(tmp_path)
    trainer.create_optimizer()
    result = trainer._generate_and_score_completions(None)
    copied = trainer._update_observer_pending["tensors"]["completion_ids"].clone()
    result["completion_ids"].fill_(123)
    assert torch.equal(trainer._update_observer_pending["tensors"]["completion_ids"], copied)
    assert not trainer._update_observer_pending["tensors"]["old_per_token_logps"].requires_grad
    with pytest.raises(RuntimeError, match="previous generation"):
        trainer._generate_and_score_completions(None)


def test_create_optimizer_after_accelerate_wrapper_keeps_single_raw_hook(tmp_path):
    trainer = build(tmp_path)
    original = trainer.create_optimizer()
    trainer.optimizer = SimpleNamespace(
        optimizer=original, register_step_post_hook=lambda hook: pytest.fail("must unwrap")
    )
    trainer.create_optimizer()
    assert trainer._update_observer_optimizer is original
    assert len(original._optimizer_step_post_hooks) == 1


def test_observer_refuses_current_label_proposals_and_eval_bypasses(tmp_path):
    with pytest.raises(ValueError, match="text_prediction_only"):
        Observed(linear_method="text_direct_ipw", linear_output_dir=tmp_path)
    trainer = build(tmp_path)
    trainer.model.eval()
    assert trainer._generate_and_score_completions(None) == {"eval": True}
    assert trainer._update_observer_pending is None


def test_preserved_state_restores_after_scoring_error():
    model = SmallModel()
    model.child.eval()
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with preserve_observer_state(model):
            model.eval()
            torch.rand(17)
            raise RuntimeError("scoring failed")
    assert model.training and not model.child.training
    assert torch.equal(torch.random.get_rng_state(), state)
