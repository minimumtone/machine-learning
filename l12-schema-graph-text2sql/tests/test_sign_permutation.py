"""Tests for scripts.sign_permutation (exact sign-permutation p-value)."""
from itertools import product

import numpy as np
from scipy.stats import rankdata, wilcoxon

from scripts.sign_permutation import sign_permutation_pvalue


def _brute_force(nonzero):
    ranks = rankdata(np.abs(nonzero))
    total = ranks.sum()
    w_plus = ranks[np.asarray(nonzero) > 0].sum()
    observed = min(w_plus, total - w_plus)
    n_extreme = 0
    for signs in product((0.0, 1.0), repeat=len(ranks)):
        w = float(np.dot(signs, ranks))
        if min(w, total - w) <= observed + 1e-12:
            n_extreme += 1
    return n_extreme / 2 ** len(ranks)


def test_hand_checkable_values():
    assert sign_permutation_pvalue([]) == 1.0
    # three equal positive diffs: only the all-plus / all-minus assignments
    # reach min(W+, W-) = 0 -> 2 / 8
    assert sign_permutation_pvalue([0.2, 0.2, 0.2]) == 0.25
    # one positive, one negative of different size: every assignment is as
    # extreme as the observed one
    assert sign_permutation_pvalue([0.4, -0.2]) == 1.0


def test_float_representations_of_equal_diffs_share_a_midrank():
    # Means of five 0/1 runs: 0.2 can also arrive as 1.0 - 0.8 = 0.19999999999999996.
    # The observed no_nbest differences (three |diff| = 0.2 tied) give 0.625;
    # splitting the tie into midranks 3.5 / 5 / 3.5 would give 0.75.
    a = 0.2
    b = 1.0 - 0.8
    assert a != b
    nonzero = [-0.038279569892473164, b, 0.18201058201058207, -a, b]
    assert sign_permutation_pvalue(nonzero) == 0.625
    assert sign_permutation_pvalue([-0.038279569892473164, a, 0.18201058201058207,
                                    -a, a]) == 0.625


def test_matches_brute_force_enumeration_with_ties():
    rng = np.random.default_rng(20260602)
    for _ in range(300):
        k = int(rng.integers(1, 13))
        x = rng.integers(-3, 4, size=k).astype(float) / 5
        x = x[x != 0]
        if len(x) == 0:
            continue
        assert abs(sign_permutation_pvalue(x) - _brute_force(x)) < 1e-12


def test_matches_scipy_exact_when_untied():
    rng = np.random.default_rng(1)
    for _ in range(50):
        k = int(rng.integers(3, 25))
        mags = rng.permutation(np.arange(1, k + 1)).astype(float)
        signs = rng.choice([-1.0, 1.0], size=k)
        x = mags * signs
        assert abs(sign_permutation_pvalue(x)
                   - float(wilcoxon(x, method="exact").pvalue)) < 1e-12
