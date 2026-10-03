from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinel_repair.advantages import (
    DEFAULT_DDOF,
    DEFAULT_EPSILON,
    group_advantages,
    grpo_advantages,
    validate_binary_rewards,
)


def test_default_matches_upstream_sample_std_then_additive_epsilon() -> None:
    rewards = np.array([0.0, 1.0, 1.0, 1.0])
    expected = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-4)

    actual = grpo_advantages(rewards)

    assert DEFAULT_DDOF == 1
    assert DEFAULT_EPSILON == 1e-4
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        [
            -1.4997000599880024,
            0.4999000199960008,
            0.4999000199960008,
            0.4999000199960008,
        ],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_zero_variance_groups_have_zero_advantage(value: float, size: int) -> None:
    np.testing.assert_array_equal(grpo_advantages([value] * size), np.zeros(size))
    np.testing.assert_array_equal(
        grpo_advantages([value] * size, epsilon=0.0), np.zeros(size)
    )


def test_population_std_is_available_as_explicit_ablation() -> None:
    rewards = [0.0, 1.0, 1.0, 1.0]
    sample = grpo_advantages(rewards)
    population = grpo_advantages(rewards, ddof=0)

    assert not np.allclose(sample, population)
    np.testing.assert_allclose(
        population,
        (np.asarray(rewards) - np.mean(rewards))
        / (np.std(rewards, ddof=0) + DEFAULT_EPSILON),
    )


def test_alias_is_exact() -> None:
    rewards = [1.0, 0.0, 1.0, 0.0]
    np.testing.assert_array_equal(group_advantages(rewards), grpo_advantages(rewards))


def test_every_binary_group_of_four_is_finite_centered_and_sign_correct() -> None:
    for bits in itertools.product((0.0, 1.0), repeat=4):
        rewards = np.asarray(bits)
        advantages = grpo_advantages(rewards)
        assert np.all(np.isfinite(advantages))
        assert abs(float(advantages.sum())) < 1e-12
        if 0.0 in bits and 1.0 in bits:
            assert np.all(advantages[rewards == 1.0] > 0.0)
            assert np.all(advantages[rewards == 0.0] < 0.0)
        else:
            np.testing.assert_array_equal(advantages, np.zeros(4))


def test_binary_validation_and_shape_errors() -> None:
    np.testing.assert_array_equal(validate_binary_rewards([0, 1]), [0.0, 1.0])
    with pytest.raises(ValueError, match="binary"):
        validate_binary_rewards([0.0, 0.5])
    with pytest.raises(ValueError, match="one-dimensional"):
        grpo_advantages([[0.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        grpo_advantages([])
    with pytest.raises(ValueError, match="finite"):
        grpo_advantages([0.0, np.nan])
    with pytest.raises(TypeError, match="ddof"):
        grpo_advantages([0.0, 1.0], ddof=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon"):
        grpo_advantages([0.0, 1.0], epsilon=-1.0)
