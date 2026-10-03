"""Budget and exact finite-population checks for online linear updates."""

from itertools import combinations

import numpy as np
import pytest

from sentinel_repair.linear_audit import audit_advantage, grpo, rloo
from sentinel_repair.online_linear import (
    FULL_ORACLE_METHODS,
    METHODS,
    PREDICTED_METHODS,
    SPARSE_METHODS,
    audit_allowance,
    linear_advantages,
    uniform_audit_indices,
)

UNBIASED_METHODS = ("linear_ipw", "oracle_only", "oracle_centered")
LEGACY_METHODS = (
    "cheap_grpo",
    "cheap_rloo",
    "full_oracle_rloo",
    "linear_ipw",
    "oracle_only",
    "oracle_centered",
    "linear_replace",
    "group_scaled_ipw",
    "full_oracle_group_scaled",
)


def baseline_kwargs(method, cheap):
    return {"baseline": np.full_like(cheap, 0.5)} if method in PREDICTED_METHODS else {}


def test_new_methods_are_appended_without_reordering_existing_methods():
    appended = ("calibrated_ipw", "preaudit_scaled_ipw", "text_direct_ipw", "text_prediction_only")
    assert METHODS == LEGACY_METHODS + appended
    assert SPARSE_METHODS == LEGACY_METHODS[3:-1] + appended
    assert FULL_ORACLE_METHODS == ("full_oracle_rloo", "full_oracle_group_scaled")
    assert set(FULL_ORACLE_METHODS).isdisjoint(SPARSE_METHODS)


def test_256_rollout_batches_use_two_or_three_labels_and_exact_prefix_budget():
    seen = spent = 0
    allowances = []
    for _ in range(256):
        allowance = audit_allowance(seen, 256)
        allowances.append(allowance)
        seen += 256
        spent += allowance
        assert spent == seen // 100
        assert 100 * spent <= seen
    assert set(allowances) == {2, 3}
    assert allowances[:5] == [2, 3, 2, 3, 2]


def test_irregular_batch_prefixes_and_integer_precision():
    seen = spent = 0
    for size in [1, 98, 1, 1, 255, 256, 17, 10_000]:
        spent += audit_allowance(seen, size)
        seen += size
        assert spent == seen // 100
    assert audit_allowance(10**30 + 99, 1) == 1
    assert audit_allowance(np.int64(99), np.int64(1)) == 1


@pytest.mark.parametrize(
    "seen,size", [(-1, 256), (0, 0), (0, -1), (1.5, 256), (0, 2.5), (True, 256)]
)
def test_invalid_budget_arguments(seen, size):
    with pytest.raises(ValueError):
        audit_allowance(seen, size)


def test_uniform_audits_are_reproducible_unique_and_in_range():
    rng = np.random.default_rng(987)
    other = np.random.default_rng(987)
    for k in [0, 1, 2, 3, 256]:
        indices = uniform_audit_indices(256, k, rng)
        np.testing.assert_array_equal(indices, uniform_audit_indices(256, k, other))
        assert indices.shape == (k,)
        assert np.issubdtype(indices.dtype, np.integer)
        assert len(np.unique(indices)) == k
        assert np.all((0 <= indices) & (indices < 256))
    np.testing.assert_array_equal(np.sort(indices), np.arange(256))


def test_zero_audits_do_not_consume_randomness():
    rng = np.random.default_rng(4)
    other = np.random.default_rng(4)
    assert uniform_audit_indices(0, 0, rng).size == 0
    assert uniform_audit_indices(256, 0, rng).size == 0
    np.testing.assert_array_equal(rng.random(5), other.random(5))


@pytest.mark.parametrize("n,k", [(-1, 0), (3, -1), (3, 4), (3.5, 1), (3, 1.5)])
def test_invalid_audit_design(n, k):
    with pytest.raises(ValueError):
        uniform_audit_indices(n, k, np.random.default_rng(1))


@pytest.mark.parametrize("method", UNBIASED_METHODS)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_conditional_expectation_over_all_subsets_is_full_oracle(method, k):
    cheap = np.array([[1.0, 1.0, 0.0], [0.0, 0.3, 1.0]])
    truth = np.array([[1.0, 0.0, 0.0], [1.0, 0.8, 0.0]])
    estimates = []
    for subset in combinations(range(truth.size), k):
        audited = np.full_like(truth, np.nan)
        audited.flat[list(subset)] = truth.flat[list(subset)]
        estimates.append(linear_advantages(method, cheap, audited, k / truth.size))
    reference = linear_advantages("full_oracle_rloo", cheap, truth, 1)
    np.testing.assert_allclose(np.mean(estimates, axis=0), reference, atol=1e-13)


@pytest.mark.parametrize("method", SPARSE_METHODS)
def test_unpurchased_truth_cannot_change_masked_estimator(method):
    cheap = np.ones((2, 4))
    truth_a = np.zeros_like(cheap)
    truth_b = np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    mask = np.zeros_like(cheap, dtype=bool)
    mask[0, 0] = True
    audited_a = np.where(mask, truth_a, np.nan)
    audited_b = np.where(mask, truth_b, np.nan)
    np.testing.assert_array_equal(
        linear_advantages(
            method, cheap, audited_a, 1 / 8, **baseline_kwargs(method, cheap)
        ),
        linear_advantages(
            method, cheap, audited_b, 1 / 8, **baseline_kwargs(method, cheap)
        ),
    )


@pytest.mark.parametrize(
    "method,reference", [("cheap_rloo", rloo), ("cheap_grpo", grpo)]
)
def test_cheap_baseline_never_reads_oracle_argument(method, reference):
    class ForbiddenOracle:
        def __array__(self, *args, **kwargs):
            raise AssertionError("cheap baseline attempted to read oracle labels")

    cheap = np.array([[0.0, 1.0, 0.5]])
    np.testing.assert_array_equal(
        linear_advantages(method, cheap, ForbiddenOracle(), 0), reference(cheap)
    )


def test_correction_couples_all_members_of_its_group():
    cheap = np.ones((2, 3))
    audited = np.full_like(cheap, np.nan)
    audited[0, 0] = 0
    actual = linear_advantages("linear_ipw", cheap, audited, 1 / 6)
    np.testing.assert_allclose(actual, [[-6.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(actual.sum(axis=-1), 0)


@pytest.mark.parametrize("method", FULL_ORACLE_METHODS)
def test_full_oracle_reference_and_all_visible_requirement(method):
    cheap = np.array([[1.0, 0.0, 1.0]])
    truth = np.array([[0.0, 0.3, 1.0]])
    expected = rloo(truth)
    if method == "full_oracle_group_scaled":
        expected /= truth.std(axis=-1, ddof=1, keepdims=True) + 1e-4
    np.testing.assert_array_equal(linear_advantages(method, cheap, truth, 1), expected)
    truth[0, 1] = np.nan
    with pytest.raises(ValueError, match="requires all oracle labels"):
        linear_advantages(method, cheap, truth, 1)


@pytest.mark.parametrize("method", FULL_ORACLE_METHODS)
@pytest.mark.parametrize("p", [0, -1, np.nan, np.inf, object()])
def test_full_oracle_references_do_not_read_inclusion_probability(method, p):
    cheap = np.array([[1.0, 0.0, 1.0, 1.0]])
    truth = np.array([[0.0, 1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(
        linear_advantages(method, cheap, truth, p),
        linear_advantages(method, cheap, truth, 1),
    )


def test_full_oracle_group_scaled_formula_group_independence_and_constant_groups():
    truth = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    cheap = 1 - truth
    expected = (
        (4 / 3)
        * (truth - truth.mean(axis=-1, keepdims=True))
        / (truth.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    )
    actual = linear_advantages("full_oracle_group_scaled", cheap, truth, object())
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual, (4 / 3) * grpo(truth))
    assert not np.allclose(actual, grpo(truth))
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[2:], np.zeros((2, 4)))
    np.testing.assert_array_equal(
        actual, linear_advantages("full_oracle_group_scaled", truth, truth, 0)
    )
    for index in range(len(truth)):
        np.testing.assert_array_equal(
            actual[index],
            linear_advantages(
                "full_oracle_group_scaled", cheap[index], truth[index], 1
            ),
        )
    np.testing.assert_array_equal(
        actual, linear_advantages("group_scaled_ipw", cheap, truth, 1)
    )


@pytest.mark.parametrize("method", SPARSE_METHODS)
def test_zero_budget_is_explicit_fallback_and_cannot_receive_labels(method):
    cheap = np.array([[1.0, 0.0, 1.0, 0.0]])
    audited = np.full_like(cheap, np.nan)
    expected = (
        rloo(cheap)
        if method in {"linear_ipw", "linear_replace"}
        else np.zeros_like(cheap)
    )
    if method in {"group_scaled_ipw", "preaudit_scaled_ipw"}:
        expected = rloo(cheap) / (cheap.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    np.testing.assert_array_equal(
        linear_advantages(method, cheap, audited, 0, **baseline_kwargs(method, cheap)),
        expected,
    )
    audited[0, 0] = 1
    with pytest.raises(ValueError, match="zero inclusion probability"):
        linear_advantages(method, cheap, audited, 0, **baseline_kwargs(method, cheap))


@pytest.mark.parametrize("method", METHODS)
def test_advantages_do_not_mutate_input_arrays(method):
    cheap = np.array([[1.0, 0.0, 1.0, 0.0]])
    audited = np.array([[0.0, 0.0, 1.0, 0.0]])
    if method not in FULL_ORACLE_METHODS:
        audited[0, 1] = np.nan
    cheap_before = cheap.copy()
    audited_before = audited.copy()
    linear_advantages(method, cheap, audited, 3 / 4, **baseline_kwargs(method, cheap))
    np.testing.assert_array_equal(cheap, cheap_before)
    np.testing.assert_array_equal(audited, audited_before)


@pytest.mark.parametrize("p", [-0.1, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("method", SPARSE_METHODS)
def test_invalid_inclusion_probability(p, method):
    cheap = np.ones((1, 4))
    with pytest.raises(ValueError, match="inclusion probability"):
        linear_advantages(
            method,
            cheap,
            np.full_like(cheap, np.nan),
            p,
            **baseline_kwargs(method, cheap),
        )


@pytest.mark.parametrize("k", [2, 3])
def test_linear_replace_is_direct_reward_replacement_without_ipw_weight(k):
    cheap = np.tile([1.0, 1.0, 0.0, 0.0], (64, 1))
    audited = np.full_like(cheap, np.nan)
    # One purchased FP; any remaining purchased labels agree with cheap.
    audited.flat[:k] = cheap.flat[:k]
    audited[0, 0] = 0
    p = k / cheap.size
    expected = rloo(np.where(np.isfinite(audited), audited, cheap))
    replaced = linear_advantages("linear_replace", cheap, audited, p)
    corrected = linear_advantages("linear_ipw", cheap, audited, p)
    cheap_advantage = linear_advantages("cheap_rloo", cheap, audited, p)
    np.testing.assert_array_equal(replaced, expected)
    np.testing.assert_allclose(
        corrected - cheap_advantage, (replaced - cheap_advantage) / p
    )
    assert not np.array_equal(replaced, corrected)
    # Changing p without changing purchases never rescales direct replacement.
    np.testing.assert_array_equal(
        replaced, linear_advantages("linear_replace", cheap, audited, 1)
    )


@pytest.mark.parametrize("k", [2, 3])
def test_no_purchased_disagreement_makes_replace_and_ipw_equal_cheap_rloo(k):
    cheap = np.tile([1.0, 1.0, 0.0, 0.0], (64, 1))
    indices = uniform_audit_indices(cheap.size, k, np.random.default_rng(19))
    audited = np.full_like(cheap, np.nan)
    audited.flat[indices] = cheap.flat[indices]
    expected = rloo(cheap)
    for method in ("linear_replace", "linear_ipw"):
        np.testing.assert_array_equal(
            linear_advantages(method, cheap, audited, k / cheap.size), expected
        )


def test_linear_replace_expectation_is_cheap_oracle_mixture_not_full_oracle():
    cheap = np.ones((2, 4))
    truth = cheap.copy()
    truth[0, 0] = 0
    k, n = 2, cheap.size
    estimates = []
    for subset in combinations(range(n), k):
        audited = np.full_like(cheap, np.nan)
        audited.flat[list(subset)] = truth.flat[list(subset)]
        estimates.append(linear_advantages("linear_replace", cheap, audited, k / n))
    actual = np.mean(estimates, axis=0)
    np.testing.assert_allclose(actual, (1 - k / n) * rloo(cheap) + k / n * rloo(truth))
    assert not np.allclose(actual, rloo(truth))


def test_group_scaled_ipw_matches_direct_formula_and_scales_groups_independently():
    cheap = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    audited = np.full_like(cheap, np.nan)
    audited[0, 0], audited[1, 1], audited[2, 0] = 0, 1, 0
    p = 3 / cheap.size
    pseudo = cheap.copy()
    mask = np.isfinite(audited)
    pseudo[mask] += (audited[mask] - cheap[mask]) / p
    expected = (
        (4 / 3)
        * (pseudo - pseudo.mean(axis=-1, keepdims=True))
        / (pseudo.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    )
    actual = linear_advantages("group_scaled_ipw", cheap, audited, p)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=-1), 0, atol=1e-14)
    # Leading axes must not enter another prompt group's scale.
    stacked = linear_advantages(
        "group_scaled_ipw", np.stack([cheap, cheap]), np.stack([audited, audited]), p
    )
    np.testing.assert_array_equal(stacked, np.stack([actual, actual]))
    for index in range(len(cheap)):
        np.testing.assert_array_equal(
            actual[index],
            linear_advantages("group_scaled_ipw", cheap[index], audited[index], p),
        )


@pytest.mark.parametrize("p", [0.0, 0.25, 1.0])
def test_group_scaled_ipw_constant_pseudo_groups_are_finite_zero(p):
    cheap = np.array([[0.0] * 4, [1.0] * 4])
    audited = np.full_like(cheap, np.nan)
    if p == 1:
        audited[:] = 1 - cheap
    elif p > 0:
        audited[:, 0] = cheap[:, 0]
    actual = linear_advantages("group_scaled_ipw", cheap, audited, p)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, np.zeros_like(cheap))


def test_group_scaled_ipw_full_budget_scales_oracle_rloo_not_native_grpo():
    cheap = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    truth = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    actual = linear_advantages("group_scaled_ipw", cheap, truth, 1)
    expected = rloo(truth) / (truth.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual, (4 / 3) * grpo(truth))
    assert not np.allclose(actual, grpo(truth))


def test_same_fp_flip_grows_as_inverse_p_but_group_scaled_ipw_saturates():
    cheap = np.ones((64, 4))
    unscaled, scaled = [], []
    for k in (2, 3):
        p = k / cheap.size
        audited = np.full_like(cheap, np.nan)
        audited.flat[:k] = 1
        audited[0, 0] = 0  # Exactly the same FP residual in both designs.
        unscaled.append(linear_advantages("linear_ipw", cheap, audited, p))
        scaled.append(linear_advantages("group_scaled_ipw", cheap, audited, p))
        np.testing.assert_allclose(
            unscaled[-1][0], np.array([-1, 1 / 3, 1 / 3, 1 / 3]) / p
        )
        # std(pseudo) = 1/(2p): magnitudes approach [-2, 2/3, 2/3, 2/3].
        np.testing.assert_allclose(
            scaled[-1][0], np.array([-2, 2 / 3, 2 / 3, 2 / 3]) / (1 + 2e-4 * p)
        )
        np.testing.assert_array_equal(scaled[-1][1:], np.zeros((63, 4)))
    np.testing.assert_allclose(unscaled[0], 1.5 * unscaled[1])
    np.testing.assert_allclose(scaled[0], scaled[1], rtol=1e-6, atol=0)
    assert not np.allclose(scaled[0], 1.5 * scaled[1])


def test_finite_population_group_scaling_breaks_conditional_unbiasedness():
    cheap = np.array([[1.0, 1.0, 0.0, 0.0]])
    truth = np.array([[0.0, 1.0, 0.0, 0.0]])
    scaled, unscaled = [], []
    for subset in combinations(range(4), 1):
        audited = np.full_like(cheap, np.nan)
        audited.flat[list(subset)] = truth.flat[list(subset)]
        scaled.append(linear_advantages("group_scaled_ipw", cheap, audited, 1 / 4))
        unscaled.append(linear_advantages("linear_ipw", cheap, audited, 1 / 4))
    oracle = rloo(truth)
    np.testing.assert_allclose(np.mean(unscaled, axis=0), oracle, atol=1e-14)
    mean_scaled = np.mean(scaled, axis=0)
    np.testing.assert_allclose(
        mean_scaled,
        [
            [
                0.384777981610133,
                1.1545338986510427,
                -0.7696559401305878,
                -0.7696559401305878,
            ]
        ],
        atol=1e-14,
    )
    assert mean_scaled[0, 0] > 0 > oracle[0, 0]
    assert not np.allclose(mean_scaled, oracle)
    scaled_oracle = oracle / (truth.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    assert not np.allclose(mean_scaled, scaled_oracle)


@pytest.mark.parametrize("method", LEGACY_METHODS)
@pytest.mark.parametrize("p", [0.0, 1 / 8, 1.0])
def test_existing_nine_methods_remain_exactly_equal_to_original_dispatch(method, p):
    cheap = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    audited = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    if method not in FULL_ORACLE_METHODS and p < 1:
        audited[0, 1:] = np.nan
        audited[1, :] = np.nan
        if p == 0:
            audited[0, 0] = np.nan
    if method == "cheap_grpo":
        expected = grpo(cheap)
    elif method == "cheap_rloo":
        expected = rloo(cheap)
    elif method == "full_oracle_rloo":
        expected = rloo(audited)
    elif method == "full_oracle_group_scaled":
        expected = rloo(audited) / (audited.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    elif method == "group_scaled_ipw":
        pseudo = (
            cheap
            if p == 0
            else cheap + np.where(np.isfinite(audited), audited - cheap, 0) / p
        )
        expected = rloo(pseudo) / (pseudo.std(axis=-1, ddof=1, keepdims=True) + 1e-4)
    elif p == 0:
        expected = (
            rloo(cheap)
            if method in {"linear_ipw", "linear_replace"}
            else np.zeros_like(cheap)
        )
    elif method == "linear_replace":
        expected = rloo(np.where(np.isfinite(audited), audited, cheap))
    else:
        expected = audit_advantage(cheap, audited, p, method)

    class ForbiddenBaseline:
        def __array__(self, *args, **kwargs):
            raise AssertionError(
                "legacy method inspected optional calibration baseline"
            )

    for extra in ({}, {"baseline": ForbiddenBaseline()}):
        np.testing.assert_array_equal(
            linear_advantages(method, cheap, audited, p, **extra), expected
        )


def test_rejects_invalid_methods_geometry_and_labels():
    cheap = np.ones((2, 3))
    with pytest.raises(ValueError, match="unknown method"):
        linear_advantages("normalized_ipw", cheap, cheap, 1)
    for value in [np.array(1), np.ones((2, 1))]:
        with pytest.raises(ValueError, match="group size"):
            linear_advantages("cheap_rloo", value, value, 1)
    with pytest.raises(ValueError, match="identical shapes"):
        linear_advantages("linear_ipw", cheap, np.ones((1, 3)), 1)
    with pytest.raises(ValueError, match="finite"):
        linear_advantages("linear_ipw", cheap, np.full_like(cheap, np.inf), 1)
    with pytest.raises(ValueError, match="finite"):
        linear_advantages("cheap_rloo", np.full_like(cheap, np.nan), cheap, 1)
