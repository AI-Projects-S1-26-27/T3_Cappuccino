"""Behavioral and edge-case tests for frozen E7; no runner dependency."""
import numpy as np
import pytest

from rulfl.eval.stats import paired_diff_bootstrap_ci, wilcoxon_signed_rank_exact_one_sided


def test_bootstrap_is_paired_and_constant_differences_collapse_ci():
    # If arms are resampled separately this would produce a nonzero CI width.
    result = paired_diff_bootstrap_ci([2, 50, 8, 100], [1, 49, 7, 99], 400, 7)
    assert result == (1.0, 1.0, 1.0)


def test_bootstrap_reproducible_and_asymmetric_input_is_not_ignored():
    a, b = [2, 8, 20, 4, 11], [9, 3, 10, 7, 12]
    first = paired_diff_bootstrap_ci(a, b, 1500, 31)
    assert first == paired_diff_bootstrap_ci(a, b, 1500, 31)
    assert first[0] == pytest.approx(np.mean(np.array(a) - np.array(b)))
    assert first[1] <= first[0] <= first[2]
    swapped = paired_diff_bootstrap_ci(b, a, 1500, 31)
    assert swapped[0] == pytest.approx(-first[0])


def test_bootstrap_rejects_bad_data():
    for a, b in [([], []), ([1], [1, 2]), ([np.nan], [1]), ([[1, 2]], [[1, 2]])]:
        with pytest.raises(ValueError):
            paired_diff_bootstrap_ci(a, b)
    with pytest.raises(ValueError):
        paired_diff_bootstrap_ci([1], [0], n_boot=0)


def test_wilcoxon_extreme_signs_and_direction():
    negatives = [-1, -2, -3, -4, -5]
    positives = [1, 2, 3, 4, 5]
    assert wilcoxon_signed_rank_exact_one_sided(negatives, "less") == pytest.approx(1 / 32)
    assert wilcoxon_signed_rank_exact_one_sided(negatives, "greater") == pytest.approx(1)
    assert wilcoxon_signed_rank_exact_one_sided(positives, "greater") == pytest.approx(1 / 32)
    assert wilcoxon_signed_rank_exact_one_sided(positives, "less") == pytest.approx(1)


def test_wilcoxon_mixed_signs_ties_and_zeros_matches_hand_checked_values():
    # Zero differences are omitted; equal |d| values receive midranks.
    assert wilcoxon_signed_rank_exact_one_sided([-1, 0, 1], "less") == pytest.approx(.75)
    assert wilcoxon_signed_rank_exact_one_sided([0, 0, 0], "less") == 1.0
    assert wilcoxon_signed_rank_exact_one_sided([-2, -2, 1, 1], "less") == pytest.approx(.25)


def test_wilcoxon_rejects_invalid_input():
    for bad in [[], [np.inf], [[1, -1]]]:
        with pytest.raises(ValueError):
            wilcoxon_signed_rank_exact_one_sided(bad)
    with pytest.raises(ValueError):
        wilcoxon_signed_rank_exact_one_sided([1], alternative="two-sided")
    with pytest.raises(ValueError):
        wilcoxon_signed_rank_exact_one_sided(list(range(1, 22)))


def test_bootstrap_percentile_bounds_match_explicit_paired_resamples():
    a = np.array([1., 11., 15., 23.])
    b = np.array([4.,  7., 14., 17.])
    differences = a - b
    rng = np.random.default_rng(19)
    sampled = np.array([differences[rng.integers(4, size=4)].mean() for _ in range(500)])
    expected = (differences.mean(), *np.quantile(sampled, [0.025, 0.975]))
    assert paired_diff_bootstrap_ci(a, b, 500, 19) == pytest.approx(expected)


def test_wilcoxon_tiny_float_noise_zero_is_dropped():
    clean = [-1., -2., -3., -4., -5.]
    with_noise = [-1., -2., 3e-12, -3., -4., -5.]
    assert wilcoxon_signed_rank_exact_one_sided(with_noise) == pytest.approx(1 / 32)
    assert wilcoxon_signed_rank_exact_one_sided(with_noise) == wilcoxon_signed_rank_exact_one_sided(clean)
    assert wilcoxon_signed_rank_exact_one_sided([1e-12, -2e-12, 0]) == 1.0


def test_wilcoxon_float_noise_ties_get_average_ranks():
    clean = [-1., 1., -2., 2., -3.]
    noisy = [-1.0, 1.0 + 2e-12, -2.0, 2.0 - 3e-12, -3.0]
    assert wilcoxon_signed_rank_exact_one_sided(noisy) == wilcoxon_signed_rank_exact_one_sided(clean)
    assert wilcoxon_signed_rank_exact_one_sided(noisy) == pytest.approx(0.34375)


def test_wilcoxon_matches_scipy_reference_with_zeros_and_ties():
    scipy = pytest.importorskip("scipy.stats")
    examples = [
        [-1, -2, -3, -4, -5],
        [-2, -2, 1, 1, 3],
        [-3, 2, -2, 1, 5, -1],
        [0, -1, -2, 2, 3],
    ]
    for sample in examples:
        # scipy's exact method with ties/zeros has different semantics in
        # some versions; method='auto' correctly selects permutation.
        from scipy.stats import PermutationMethod
        expected = scipy.wilcoxon(sample, alternative="less", zero_method="wilcox",
                                  method=PermutationMethod(n_resamples=np.inf)).pvalue
        assert wilcoxon_signed_rank_exact_one_sided(sample) == pytest.approx(expected)


def test_wilcoxon_max_20_input_pairs_and_rounding_noise():
    assert wilcoxon_signed_rank_exact_one_sided([-1] * 20) == pytest.approx(1 / (2 ** 20))
    with pytest.raises(ValueError):
        wilcoxon_signed_rank_exact_one_sided([0] + [-1] * 20)
    with pytest.raises(ValueError):
        paired_diff_bootstrap_ci([1], [0], n_boot=True)
