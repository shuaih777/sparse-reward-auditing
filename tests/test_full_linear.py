"""CPU fake-parent checks of the TRL reward/advantage integration boundary."""

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel_repair.full_linear import make_full_linear_trainer_class
from sentinel_repair.linear_audit import grpo, rloo
from sentinel_repair.online_linear import (
    FULL_ORACLE_METHODS,
    METHODS,
    PastAuditCalibrator,
    SPARSE_METHODS,
    TEXT_PREDICTED_METHODS,
    linear_advantages,
    preaudit_denominators,
    preaudit_scale_config,
    uniform_audit_indices,
)


class Tensor:
    """Small CPU tensor stand-in; tests exercise the parent hook order, not torch."""

    def __init__(self, values):
        self.array = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.array.shape, self.array.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def new_tensor(self, values):
        return Tensor(values)

    def tolist(self):
        return self.array.tolist()


class FakeParent:
    def __init__(self, *, cache_mode="replace", overrides=None, truth=None):
        self.args = SimpleNamespace(
            generation_batch_size=256,
            gradient_accumulation_steps=128,
            steps_per_generation=128,
            per_device_train_batch_size=2,
            logging_steps=1,
            logging_strategy="steps",
        )
        self.accelerator = SimpleNamespace(num_processes=1, process_index=0)
        self.model = SimpleNamespace(training=True)
        self.state = SimpleNamespace(global_step=0)
        self.num_generations = 4
        self.num_iterations = 1
        self.beta = 0
        self.scale_rewards = "group" if self.linear_method == "cheap_grpo" else "none"
        self.reward_funcs = [SimpleNamespace(_oracle_rewards=[])]
        self.reward_weights = [1.0]
        self._logs = {"advantages": deque([-9.0] * 4, maxlen=512), "rewards": []}
        self.cache_mode = cache_mode
        self.truth = list(np.tile([1, 0, 1, 0], 64) if truth is None else truth)
        self.cheap = Tensor(np.tile([1, 1, 0, 0], 64).reshape(-1, 1))
        self.reorder = False
        self.parent_train_calls = 0
        for target, value in (overrides or {}).items():
            if target.startswith("args."):
                setattr(self.args, target[5:], value)
            elif target.startswith("accelerator."):
                setattr(self.accelerator, target[12:], value)
            else:
                setattr(self, target, value)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if self.cache_mode == "replace":
            self.reward_funcs[0]._oracle_rewards = self.truth.copy()
        elif self.cache_mode == "inplace":
            self.reward_funcs[0]._oracle_rewards[:] = self.truth
        elif self.cache_mode != "stale":
            raise AssertionError(self.cache_mode)
        return self.cheap

    def _generate_and_score_completions(self, inputs):
        prompts = [row["prompt"] for row in inputs]
        ids = [[index + 1, 9001] for index in range(len(inputs))]
        self.returned_rewards = self._calculate_rewards(
            inputs, prompts, ["text"] * len(inputs), ids
        )
        cheap = self.returned_rewards.numpy().reshape(-1, 4)
        values = (
            grpo(cheap)
            if self.scale_rewards == "group"
            else cheap - cheap.mean(axis=1, keepdims=True)
        )
        self.original_advantages = Tensor(values.ravel())
        self._logs["advantages"].extend(self.original_advantages.tolist())
        self._logs["rewards"].extend(cheap.ravel().tolist())
        if self.reorder:
            ids = list(reversed(ids))
        return {"advantages": self.original_advantages, "completion_ids": Tensor(ids)}

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        self.parent_train_calls += 1
        return "trained"


Trainer = make_full_linear_trainer_class(FakeParent)


def inputs():
    return [{"prompt": f"prompt-{i // 4}"} for i in range(256)]


def build(tmp_path, method="linear_ipw", **kwargs):
    return Trainer(
        linear_method=method, linear_audit_seed=19, linear_output_dir=tmp_path, **kwargs
    )


def read_batch(trainer, number=1):
    return json.loads((trainer._linear_logdir / f"batch-{number:04d}.json").read_text())


@pytest.mark.parametrize(
    "method",
    [method for method in METHODS if method != "cheap_grpo"],
)
def test_final_advantages_match_helper_and_stay_with_completion_rows(tmp_path, method):
    trainer = build(tmp_path, method)
    prior_cache = [0.0] * 256
    trainer.reward_funcs[0]._oracle_rewards = prior_cache
    output = trainer._generate_and_score_completions(inputs())
    record = read_batch(trainer)
    audited = np.full(256, np.nan)
    audited[record["audit_indices"]] = record["bought_labels"]
    baseline_kwargs = {}
    if method == "calibrated_ipw":
        state = record["calibration"]
        baseline_kwargs["baseline"] = np.where(
            trainer.cheap.numpy().reshape(-1, 4) == 0, state["q0"], state["q1"]
        )
    if method in TEXT_PREDICTED_METHODS:
        baseline_kwargs["baseline"] = np.asarray(record["text_reward_predictions"]).reshape(-1, 4)
    expected = (
        linear_advantages(
            method,
            trainer.cheap.numpy().reshape(-1, 4),
            audited.reshape(-1, 4),
            record["inclusion_probability"],
            **baseline_kwargs,
        )
        .ravel()
        .astype(np.float32)
    )
    np.testing.assert_array_equal(output["advantages"].numpy(), expected)
    np.testing.assert_array_equal(
        output["completion_ids"].numpy()[:, 0], np.arange(1, 257)
    )
    np.testing.assert_array_equal(
        np.asarray(trainer._logs["advantages"])[-256:], expected
    )
    np.testing.assert_array_equal(record["advantages"], expected)
    assert list(trainer._logs["advantages"])[:4] == [-9.0] * 4
    assert trainer.returned_rewards is trainer.cheap
    assert trainer._logs["rewards"] == trainer.cheap.numpy().ravel().tolist()
    assert trainer.reward_funcs[0]._oracle_rewards == trainer.truth
    assert trainer.reward_funcs[0]._oracle_rewards is not prior_cache
    assert prior_cache == [0.0] * 256
    if method in FULL_ORACLE_METHODS:
        assert record["audit_k"] == record["oracle_calls"] == 256
        assert record["full_oracle_reference"]


def test_cheap_grpo_keeps_exact_original_tensor_and_log_values(tmp_path):
    trainer = build(tmp_path, "cheap_grpo")
    output = trainer._generate_and_score_completions(inputs())
    assert output["advantages"] is trainer.original_advantages
    assert (
        list(trainer._logs["advantages"])[-256:] == trainer.original_advantages.tolist()
    )
    record = read_batch(trainer)
    assert record["audit_k"] == record["oracle_calls"] == 0
    assert record["audit_indices"] == record["bought_labels"] == []


@pytest.mark.parametrize("method", ["cheap_grpo", "cheap_rloo"])
def test_cheap_methods_never_read_oracle_cache(tmp_path, method):
    class ForbiddenCache:
        def __len__(self):
            raise AssertionError("cheap method inspected oracle cache")

        def __getitem__(self, index):
            raise AssertionError("cheap method inspected oracle label")

    trainer = build(tmp_path, method, cache_mode="stale")
    sentinel = ForbiddenCache()
    trainer.reward_funcs[0]._oracle_rewards = sentinel
    trainer._generate_and_score_completions(inputs())
    assert trainer.reward_funcs[0]._oracle_rewards is sentinel
    assert read_batch(trainer)["oracle_calls"] == 0


@pytest.mark.parametrize(
    "method,steps",
    [
        ("linear_ipw", 40),
        ("linear_replace", 40),
        ("group_scaled_ipw", 40),
        ("group_scaled_ipw", 120),
        ("calibrated_ipw", 120),
        ("preaudit_scaled_ipw", 120),
    ],
)
def test_prefix_budget_uses_actual_two_or_three_over_256(tmp_path, method, steps):
    trainer = build(tmp_path, method)
    counts = []
    for batch in range(1, steps + 1):
        trainer._generate_and_score_completions(inputs())
        record = read_batch(trainer, batch)
        counts.append(record["audit_k"])
        assert record["training_rollouts"] == batch * 256
        assert record["oracle_calls"] == batch * 256 // 100
        assert record["inclusion_probability"] == record["audit_k"] / 256
        assert len(set(record["audit_indices"])) == record["audit_k"]
        assert all(0 <= index < 256 for index in record["audit_indices"])
    assert counts[:5] == [2, 3, 2, 3, 2]
    assert set(counts) == {2, 3}
    assert trainer._linear_spent == steps * 256 // 100


@pytest.mark.parametrize("method", FULL_ORACLE_METHODS)
def test_full_oracle_bridge_buys_all_256_labels_for_120_steps(tmp_path, method):
    trainer = build(tmp_path, method)
    rng_before = json.dumps(trainer._linear_rng.bit_generator.state)
    for batch in range(1, 121):
        trainer._generate_and_score_completions(inputs())
        record = read_batch(trainer, batch)
        assert record["audit_indices"] == list(range(256))
        assert record["bought_labels"] == trainer.truth
        assert record["audit_k"] == 256
        assert record["inclusion_probability"] == 1
        assert record["training_rollouts"] == record["oracle_calls"] == batch * 256
        assert record["full_oracle_reference"]
    assert trainer._linear_spent == 30720
    assert json.dumps(trainer._linear_rng.bit_generator.state) == rng_before


@pytest.mark.parametrize("method", METHODS)
def test_online_runner_purchases_match_values_and_ledger_without_extra_rng(
    method, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_online_linear import purchased_labels

    truth = np.tile([0.0, 1.0, 0.0, 1.0], 64)
    before_truth = truth.copy()
    rng = np.random.default_rng(19)
    indices = uniform_audit_indices(256, 2 if method in SPARSE_METHODS else 0, rng)
    rng_before = json.dumps(rng.bit_generator.state)
    expected = np.full(256, np.nan)
    expected[indices] = truth[indices]
    if method in FULL_ORACLE_METHODS:
        expected = truth.copy()
    audited, bought_indices = purchased_labels(method, truth, indices)
    np.testing.assert_array_equal(audited, expected)
    np.testing.assert_array_equal(truth, before_truth)
    np.testing.assert_array_equal(audited[bought_indices], truth[bought_indices])
    assert json.dumps(rng.bit_generator.state) == rng_before
    if method in FULL_ORACLE_METHODS:
        np.testing.assert_array_equal(bought_indices, np.arange(256))
        assert len(bought_indices) == 256
    else:
        assert bought_indices is indices
        assert len(bought_indices) == (2 if method in SPARSE_METHODS else 0)


@pytest.mark.parametrize("method", SPARSE_METHODS)
def test_unpurchased_truth_cannot_change_selection_or_advantages(tmp_path, method):
    first = build(tmp_path / "one", method)
    output1 = first._generate_and_score_completions(inputs())
    record1 = read_batch(first)
    truth = [1 - value for value in first.truth]
    for index in record1["audit_indices"]:
        truth[index] = first.truth[index]
    second = build(tmp_path / "two", method, truth=truth)
    output2 = second._generate_and_score_completions(inputs())
    assert read_batch(second)["audit_indices"] == record1["audit_indices"]
    np.testing.assert_array_equal(
        output1["advantages"].numpy(), output2["advantages"].numpy()
    )


@pytest.mark.parametrize(
    "method",
    ["linear_replace", "group_scaled_ipw", "calibrated_ipw", "preaudit_scaled_ipw"],
)
def test_ablation_and_ipw_buy_the_same_indices_and_labels(tmp_path, method):
    replace = build(tmp_path / "ablation", method)
    ipw = build(tmp_path / "ipw", "linear_ipw")
    for batch in (1, 2):
        replace._generate_and_score_completions(inputs())
        ipw._generate_and_score_completions(inputs())
        direct, weighted = read_batch(replace, batch), read_batch(ipw, batch)
        for key in (
            "audit_indices",
            "bought_labels",
            "audit_k",
            "inclusion_probability",
            "training_rollouts",
            "oracle_calls",
        ):
            assert direct[key] == weighted[key]
        assert direct["audit_k"] > 0
        assert not direct["full_oracle_reference"]


@pytest.mark.parametrize("script", ["train_full_linear.py", "train_online_linear.py"])
@pytest.mark.parametrize(
    "method",
    [
        "linear_replace",
        "group_scaled_ipw",
        "full_oracle_group_scaled",
        "calibrated_ipw",
        "preaudit_scaled_ipw",
    ],
)
def test_training_clis_accept_ablation_without_starting_training(script, method):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / script),
            "--method",
            method,
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert method in result.stdout


@pytest.mark.parametrize("method", ["group_scaled_ipw", "full_oracle_group_scaled"])
def test_group_scaled_methods_require_trl_scaling_disabled(tmp_path, method):
    trainer = build(tmp_path / "valid", method)
    assert trainer.scale_rewards == "none"
    output = trainer._generate_and_score_completions(inputs())
    record = read_batch(trainer)
    pseudo = np.asarray(record["cheap_rewards"]).reshape(-1, 4).copy()
    indices = record["audit_indices"]
    pseudo.flat[indices] += (
        np.asarray(record["bought_labels"]) - pseudo.flat[indices]
    ) / record["inclusion_probability"]
    expected = (
        (4 * pseudo - pseudo.sum(axis=-1, keepdims=True))
        / 3
        / (pseudo.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    )
    np.testing.assert_array_equal(
        output["advantages"].numpy(), expected.astype(np.float32).ravel()
    )
    with pytest.raises(ValueError, match="requires scale_rewards='none'"):
        build(
            tmp_path / "invalid",
            method,
            overrides={"scale_rewards": "group"},
        )


def test_calibrated_bridge_replays_prior_purchases_and_matches_online_protocol(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_online_linear import purchased_labels

    trainer = build(tmp_path, "calibrated_ipw")
    assert trainer.scale_rewards == "none"
    online = PastAuditCalibrator()
    online_rng = np.random.default_rng(19)
    windows, past = [[], []], 0
    cheap = trainer.cheap.numpy().ravel().astype(float)
    truth = np.array(trainer.truth, dtype=float)
    for batch in range(1, 41):
        counts = [len(window) for window in windows]
        positives = [sum(window) for window in windows]
        before = {
            "q0": (1 + positives[0]) / (2 + counts[0]),
            "q1": (1 + positives[1]) / (2 + counts[1]),
            "window_counts": counts,
            "window_positives": positives,
            "past_label_count": past,
        }
        baseline = online.predict(cheap)
        k = batch * 256 // 100 - (batch - 1) * 256 // 100
        indices = uniform_audit_indices(256, k, online_rng)
        audited, _ = purchased_labels("calibrated_ipw", truth, indices)
        pseudo = baseline + np.where(np.isfinite(audited), audited - baseline, 0) / (
            k / 256
        )
        expected = rloo(pseudo.reshape(-1, 4)).ravel().astype(np.float32)
        online.observe(cheap[indices], audited[indices])
        output = trainer._generate_and_score_completions(inputs())
        record = read_batch(trainer, batch)
        assert record["calibration"] == before
        assert record["audit_indices"] == indices.tolist()
        assert record["bought_cheap"] == cheap[indices].tolist()
        np.testing.assert_array_equal(record["advantages"], expected)
        np.testing.assert_array_equal(output["advantages"].numpy(), expected)
        for category, label in zip(record["bought_cheap"], record["bought_labels"]):
            windows[int(category)].append(int(label))
            windows[int(category)] = windows[int(category)][-32:]
            past += 1
        assert trainer._linear_calibrator.snapshot() == online.snapshot()
        assert trainer._linear_calibrator.snapshot()["past_label_count"] == past
    assert past == 102
    before_eval = trainer._linear_calibrator.snapshot()
    trainer.model.training = False
    trainer._generate_and_score_completions(inputs())
    assert trainer._linear_calibrator.snapshot() == before_eval
    with pytest.raises(ValueError, match="requires scale_rewards='none'"):
        build(
            tmp_path / "invalid", "calibrated_ipw", overrides={"scale_rewards": "group"}
        )


def test_calibrated_bridge_never_reads_unpurchased_cache_labels(tmp_path):
    trainer = build(tmp_path, "calibrated_ipw")
    expected_indices = uniform_audit_indices(256, 2, np.random.default_rng(19)).tolist()
    reads = []

    class PurchasedOnlyTruth:
        def __len__(self):
            return 256

        def copy(self):
            return self

        def __getitem__(self, index):
            assert index in expected_indices, "unbought truth read"
            reads.append(index)
            return 1

    trainer.truth = PurchasedOnlyTruth()
    trainer._generate_and_score_completions(inputs())
    assert reads == expected_indices
    assert trainer._linear_calibrator.snapshot()["past_label_count"] == 2
    assert read_batch(trainer)["calibration"]["past_label_count"] == 0


def test_preaudit_bridge_freezes_scale_before_sampling_and_reads_only_bought_labels(
    tmp_path, monkeypatch
):
    import sentinel_repair.full_linear as bridge

    events = []
    original_scale = bridge.preaudit_denominators
    original_sample = bridge.uniform_audit_indices
    expected_indices = uniform_audit_indices(256, 2, np.random.default_rng(19)).tolist()

    def frozen_scale(cheap):
        events.append("scale")
        return original_scale(cheap)

    def sample(n, k, rng):
        assert events == ["scale"]
        events.append("sample")
        return original_sample(n, k, rng)

    class PurchasedOnlyTruth:
        def __len__(self):
            return 256

        def copy(self):
            return self

        def __getitem__(self, index):
            assert index in expected_indices, "unbought truth read"
            assert events[:2] == ["scale", "sample"]
            events.append(index)
            return 0

    monkeypatch.setattr(bridge, "preaudit_denominators", frozen_scale)
    monkeypatch.setattr(bridge, "uniform_audit_indices", sample)
    trainer = build(tmp_path, "preaudit_scaled_ipw")
    trainer.truth = PurchasedOnlyTruth()
    output = trainer._generate_and_score_completions(inputs())
    record = read_batch(trainer)
    assert events == ["scale", "sample", *expected_indices]
    assert trainer.scale_rewards == "none"
    assert record["preaudit_scaling"] == preaudit_scale_config()
    expected_scale = preaudit_denominators(trainer.cheap.numpy().reshape(-1, 4))
    np.testing.assert_array_equal(record["preaudit_denominators"], expected_scale.ravel())
    assert len(record["preaudit_denominators"]) == 64
    assert record["oracle_calls"] == record["audit_k"] == 2
    assert record["audit_indices"] == expected_indices
    np.testing.assert_array_equal(output["advantages"].numpy(), record["advantages"])


def test_preaudit_bridge_scales_are_independent_of_purchased_label_values(tmp_path):
    from itertools import product

    cheap = np.tile(np.asarray(list(product([0.0, 1.0], repeat=4))), (4, 1))
    unchanged = build(tmp_path / "no-error", "preaudit_scaled_ipw", truth=cheap.ravel())
    flipped = build(tmp_path / "error", "preaudit_scaled_ipw", truth=1 - cheap.ravel())
    unchanged.cheap = Tensor(cheap.reshape(-1, 1))
    flipped.cheap = Tensor(cheap.reshape(-1, 1))
    unchanged._generate_and_score_completions(inputs())
    flipped._generate_and_score_completions(inputs())
    clean_record, error_record = read_batch(unchanged), read_batch(flipped)
    assert clean_record["audit_indices"] == error_record["audit_indices"]
    assert clean_record["preaudit_denominators"] == error_record["preaudit_denominators"]
    assert clean_record["bought_labels"] != error_record["bought_labels"]
    clean_adv = np.asarray(clean_record["advantages"]).reshape(-1, 4)
    error_adv = np.asarray(error_record["advantages"]).reshape(-1, 4)
    changed_groups = np.unique(np.asarray(error_record["audit_indices"]) // 4)
    unchanged_groups = np.setdiff1d(np.arange(64), changed_groups)
    np.testing.assert_array_equal(clean_adv[unchanged_groups], error_adv[unchanged_groups])
    assert not np.array_equal(clean_adv[changed_groups], error_adv[changed_groups])
    with pytest.raises(ValueError, match="requires scale_rewards='none'"):
        build(
            tmp_path / "invalid", "preaudit_scaled_ipw", overrides={"scale_rewards": "group"}
        )


@pytest.mark.parametrize("mode", ["stale", "inplace"])
def test_nonempty_unreplaced_oracle_cache_fails_closed(tmp_path, mode):
    trainer = build(tmp_path, cache_mode=mode)
    trainer.reward_funcs[0]._oracle_rewards = [0.0] * 256
    with pytest.raises(ValueError, match="not replaced"):
        trainer._generate_and_score_completions(inputs())


def test_empty_cache_can_be_filled_in_place_without_being_consumed(tmp_path):
    trainer = build(tmp_path, cache_mode="inplace")
    buffer = trainer.reward_funcs[0]._oracle_rewards
    trainer._generate_and_score_completions(inputs())
    assert trainer.reward_funcs[0]._oracle_rewards is buffer
    assert buffer == trainer.truth


def test_wrong_oracle_cache_size_fails_closed(tmp_path):
    trainer = build(tmp_path, truth=[1.0] * 255)
    with pytest.raises(ValueError, match="exactly this batch"):
        trainer._generate_and_score_completions(inputs())


def test_reordered_outputs_fail_before_replacing_advantages(tmp_path):
    trainer = build(tmp_path)
    trainer.reorder = True
    with pytest.raises(ValueError, match="completion order changed"):
        trainer._generate_and_score_completions(inputs())


def test_invalid_prompt_group_fails_closed(tmp_path):
    trainer = build(tmp_path)
    rows = inputs()
    rows[1]["prompt"] = "wrong group"
    with pytest.raises(ValueError, match="contiguous"):
        trainer._generate_and_score_completions(rows)


@pytest.mark.parametrize(
    "override",
    [
        {"accelerator.num_processes": 2},
        {"num_generations": 8},
        {"args.generation_batch_size": 128},
        {"args.steps_per_generation": 64},
        {"num_iterations": 2},
        {"beta": 0.1},
        {"scale_rewards": "group"},
        {"args.logging_steps": 2},
        {"args.logging_strategy": "epoch"},
        {"_live_audit_enabled": True},
        {"reward_weights": [0.5]},
    ],
)
def test_unsupported_training_geometry_is_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        build(tmp_path, overrides=override)


def test_resume_and_reusing_trainer_are_rejected(tmp_path):
    trainer = build(tmp_path)
    with pytest.raises(ValueError, match="checkpoint resume"):
        trainer.train("checkpoint-20")
    with pytest.raises(ValueError, match="model_path resume alias"):
        trainer.train(model_path="checkpoint-20")
    assert trainer.parent_train_calls == 0
    assert trainer.train(None) == "trained"
    with pytest.raises(ValueError, match="fresh trainer"):
        trainer.train(False)
    assert trainer.parent_train_calls == 1


def test_eval_does_not_spend_training_budget_or_replace_advantages(tmp_path):
    trainer = build(tmp_path)
    trainer.model.training = False
    output = trainer._generate_and_score_completions(inputs())
    assert output["advantages"] is trainer.original_advantages
    assert trainer._linear_seen == trainer._linear_spent == 0
    assert list(trainer._linear_logdir.iterdir()) == []


def test_existing_audit_output_is_not_overwritten(tmp_path):
    build(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        build(tmp_path)
