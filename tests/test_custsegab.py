import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from custsegab.experiment import cuped, holm, mde_proportion, srm_check, two_proportion, welch  # noqa: E402


def test_generator_randomization_is_balanced():
    _, exp = generate(n=20_000, seed=3)
    assert srm_check((exp.group == "control").sum(), (exp.group == "treatment").sum()) > 0.001


def test_srm_detects_broken_split():
    assert srm_check(5_000, 5_600) < 0.001


def test_two_proportion_known_values():
    yc = np.array([1] * 100 + [0] * 900)
    yt = np.array([1] * 150 + [0] * 850)
    e = two_proportion("x", yc, yt)
    assert e.abs_diff == pytest.approx(0.05)
    assert e.ci_low < 0.05 < e.ci_high
    assert e.p_value < 0.01


def test_welch_no_effect():
    rng = np.random.default_rng(0)
    e = welch("x", rng.normal(10, 2, 5000), rng.normal(10, 2, 5000))
    assert e.ci_low < 0 < e.ci_high


def test_cuped_reduces_variance_without_bias():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, 20_000)
    y = 0.8 * x + rng.normal(0, 5, 20_000)
    adj, theta = cuped(y, x)
    assert theta == pytest.approx(0.8, abs=0.02)
    assert adj.var() < 0.3 * y.var()
    assert adj.mean() == pytest.approx(y.mean())


def test_holm_is_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd"))
    adj = holm(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert adj["a"] == pytest.approx(0.04)


def test_mde_shrinks_with_sample():
    assert mde_proportion(0.2, 40_000) < mde_proportion(0.2, 4_000)
