"""CPU finite-population checks of the pre-audit denominator intervention.

These checks distinguish the fixed, cheap-scaled oracle target from both
unscaled oracle RLOO and RLOO scaled by the oracle's own standard deviation.
"""

from itertools import combinations, product
from math import comb

import numpy as np
import pytest

from sentinel_repair.linear_audit import rloo
from sentinel_repair.online_linear import linear_advantages, preaudit_denominators


BINARY_GROUPS = np.asarray(list(product([0.0, 1.0], repeat=4)))


def preaudit(cheap, audited, p):
    return linear_advantages("preaudit_scaled_ipw", cheap, audited, p)


def test_binary_g4_floor_and_constant_groups_have_finite_denominators():
    denominators = preaudit_denominators(BINARY_GROUPS)
    assert denominators.shape == (16, 1)
    expected = np.maximum(BINARY_GROUPS.std(axis=-1, ddof=1, keepdims=True), 0.5)
    np.testing.assert_array_equal(denominators, expected + 1e-4)
    np.testing.assert_array_equal(denominators[[0, 15]], [[0.5001], [0.5001]])
    np.testing.assert_allclose(denominators[6], [np.sqrt(1 / 3) + 1e-4])


@pytest.mark.parametrize("p", [0.0, 2 / 256, 3 / 256, 0.25, 1.0])
def test_all_binary_groups_match_post_audit_scaling_without_changed_labels(p):
    # Every possible observed subset, including bought labels with zero
    # residual. For p=0, only the empty subset is a legitimate observation.
    if p == 0:
        masks = [np.zeros(4, dtype=bool)]
    elif p == 1:
        masks = [np.ones(4, dtype=bool)]
    else:
        masks = product([False, True], repeat=4)
    for mask in masks:
        if p in (2 / 256, 3 / 256) and sum(mask) > round(p * 256):
            continue
        audited = np.where(np.asarray(mask), BINARY_GROUPS, np.nan)
        expected = linear_advantages("group_scaled_ipw", BINARY_GROUPS, audited, p)
        np.testing.assert_array_equal(preaudit(BINARY_GROUPS, audited, p), expected)


@pytest.mark.parametrize("n,k", [(8, 1), (8, 2), (8, 3), (256, 2), (256, 3)])
def test_exact_srs_expectation_is_oracle_rloo_over_cheap_denominator(n, k):
    # For a specified observed subset S of the focal four responses, there
    # are C(n-4,k-|S|) whole-batch samples producing exactly that subset.
    # This integrates the actual 2/256 and 3/256 design exactly, without
    # enumerating millions of irrelevant choices outside the focal group.
    cheap = BINARY_GROUPS.copy()
    truth = np.tile([0.0, 1.0, 0.0, 1.0], (len(cheap), 1))
    mean = np.zeros_like(cheap)
    probability_sum = 0.0
    for count in range(min(k, 4) + 1):
        if k - count > n - 4:
            continue
        probability = comb(n - 4, k - count) / comb(n, k)
        for selected in combinations(range(4), count):
            audited = np.full_like(cheap, np.nan)
            audited[:, list(selected)] = truth[:, list(selected)]
            mean += probability * preaudit(cheap, audited, k / n)
            probability_sum += probability
    assert probability_sum == pytest.approx(1.0, abs=2e-15)
    expected = rloo(truth) / preaudit_denominators(cheap)
    np.testing.assert_allclose(mean, expected, atol=2e-13, rtol=2e-13)
    assert not np.allclose(mean, rloo(truth))


def test_exact_full_batch_srs_average_includes_sibling_corrections():
    cheap = np.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    truth = np.asarray([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    estimates = []
    for subset in combinations(range(8), 2):
        audited = np.full_like(cheap, np.nan)
        audited.flat[list(subset)] = truth.flat[list(subset)]
        estimates.append(preaudit(cheap, audited, 2 / 8))
    np.testing.assert_allclose(
        np.mean(estimates, axis=0),
        rloo(truth) / preaudit_denominators(cheap),
        atol=1e-14,
    )


def test_one_false_positive_changes_every_sibling_but_no_other_group():
    cheap = np.ones((2, 4))
    audited = np.full_like(cheap, np.nan)
    audited[0, 0] = 0.0
    actual = preaudit(cheap, audited, 0.25)
    expected = np.asarray([[-4.0, 4 / 3, 4 / 3, 4 / 3], [0.0] * 4]) / 0.5001
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=-1), 0.0, atol=1e-14)
    assert np.all(actual[0, 1:] > 0)  # These three labels were not purchased.


def test_one_percent_negative_99_example_keeps_large_residual_weight():
    cheap = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    audited = np.asarray([[0.0, np.nan, np.nan, np.nan]])
    # A bought false positive gives 1 + (0-1)/.01 = -99. The other
    # responses enter RLOO's leave-one-out baseline and thus receive +33.
    expected = np.asarray([[-99.0, 33.0, 33.0, 33.0]]) / 0.5001
    actual = preaudit(cheap, audited, 0.01)
    np.testing.assert_array_equal(actual, expected)
    post_audit = linear_advantages("group_scaled_ipw", cheap, audited, 0.01)
    np.testing.assert_allclose(actual / post_audit, (49.5 + 1e-4) / 0.5001)


@pytest.mark.parametrize("k", [2, 3])
def test_actual_batch_probability_is_used_instead_of_nominal_one_percent(k):
    cheap = np.ones((64, 4))
    audited = np.full_like(cheap, np.nan)
    audited.flat[:k] = 0.0
    corrected = cheap.copy()
    corrected.flat[:k] = 1.0 - 256 / k
    actual = preaudit(cheap, audited, k / 256)
    np.testing.assert_allclose(actual, rloo(corrected) / 0.5001)
    assert not np.allclose(actual, preaudit(cheap, audited, 0.01))


def test_full_audit_keeps_cheap_scale_instead_of_switching_to_oracle_scale():
    cheap = np.ones((1, 4))
    truth = np.asarray([[0.0, 0.0, 1.0, 1.0]])
    actual = preaudit(cheap, truth, 1.0)
    np.testing.assert_allclose(actual, rloo(truth) / 0.5001)
    assert not np.allclose(actual, rloo(truth))
    assert not np.allclose(
        actual, linear_advantages("full_oracle_group_scaled", cheap, truth, 1)
    )


def test_zero_budget_cannot_accept_a_purchased_label():
    cheap = np.asarray([[1.0, 0.0, 1.0, 0.0]])
    audited = np.full_like(cheap, np.nan)
    np.testing.assert_array_equal(
        preaudit(cheap, audited, 0), rloo(cheap) / preaudit_denominators(cheap)
    )
    audited[0, 0] = 1.0
    with pytest.raises(ValueError, match="zero inclusion probability"):
        preaudit(cheap, audited, 0)


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_inclusion_probability_fails(p):
    cheap = np.ones((1, 4))
    with pytest.raises(ValueError):
        preaudit(cheap, np.full_like(cheap, np.nan), p)


@pytest.mark.parametrize("cheap", [0.0, [], [0.0], [0.0, 1.0], [[1.0] * 8]])
def test_invalid_group_geometry_fails(cheap):
    cheap = np.asarray(cheap)
    with pytest.raises(ValueError):
        preaudit_denominators(cheap)
    with pytest.raises(ValueError):
        preaudit(cheap, np.full_like(cheap, np.nan, dtype=float), 0.01)


@pytest.mark.parametrize("value", [-1.0, 0.25, 2.0, np.nan, np.inf, -np.inf])
def test_invalid_cheap_label_fails(value):
    cheap = np.asarray([[value, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        preaudit_denominators(cheap)
    with pytest.raises(ValueError):
        preaudit(cheap, np.full_like(cheap, np.nan), 0.01)


@pytest.mark.parametrize("value", [-1.0, 0.25, 2.0, np.inf, -np.inf])
def test_invalid_purchased_label_fails(value):
    cheap = np.ones((1, 4))
    audited = np.asarray([[value, np.nan, np.nan, np.nan]])
    with pytest.raises(ValueError):
        preaudit(cheap, audited, 0.01)


def test_misaligned_labels_fail_and_inputs_are_not_mutated():
    cheap = BINARY_GROUPS.copy()
    audited = np.full_like(cheap, np.nan)
    audited[:, 0] = 1 - cheap[:, 0]
    original_cheap, original_audited = cheap.copy(), audited.copy()
    denominators = preaudit_denominators(cheap)
    preaudit(cheap, audited, 0.01)
    np.testing.assert_array_equal(cheap, original_cheap)
    np.testing.assert_array_equal(audited, original_audited)
    cheap[:] = 0
    np.testing.assert_array_equal(denominators, preaudit_denominators(original_cheap))
    with pytest.raises(ValueError, match="identical shapes"):
        preaudit(original_cheap, original_audited[:, :3], 0.01)
