"""A/B test analysis: sample-ratio check, power, effect estimates, CUPED and segment effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Effect:
    metric: str
    control: float
    treatment: float
    abs_diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def srm_check(n_control: int, n_treatment: int, expected_split: float = 0.5) -> float:
    """Sample ratio mismatch: chi-square p-value. A very small p means the randomization is broken."""
    total = n_control + n_treatment
    expected = [total * (1 - expected_split), total * expected_split]
    return float(stats.chisquare([n_control, n_treatment], expected).pvalue)


def mde_proportion(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute difference for a two-proportion test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * p_base * (1 - p_base) / n_per_group))


def sample_size_proportion(p_base: float, abs_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    p_bar = p_base + abs_effect / 2
    return int(np.ceil(2 * p_bar * (1 - p_bar) * z ** 2 / abs_effect ** 2))


def two_proportion(metric: str, yc: np.ndarray, yt: np.ndarray, alpha: float = 0.05) -> Effect:
    pc, pt = yc.mean(), yt.mean()
    nc, nt = len(yc), len(yt)
    diff = pt - pc
    se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    p_pool = (yc.sum() + yt.sum()) / (nc + nt)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / nc + 1 / nt))
    p = 2 * (1 - stats.norm.cdf(abs(diff) / se_pool))
    z = stats.norm.ppf(1 - alpha / 2)
    return Effect(metric, pc, pt, diff, diff / pc if pc else np.nan, diff - z * se, diff + z * se, float(p), nc, nt)


def welch(metric: str, yc: np.ndarray, yt: np.ndarray, alpha: float = 0.05) -> Effect:
    mc, mt = yc.mean(), yt.mean()
    vc, vt = yc.var(ddof=1) / len(yc), yt.var(ddof=1) / len(yt)
    se = np.sqrt(vc + vt)
    dof = (vc + vt) ** 2 / (vc ** 2 / (len(yc) - 1) + vt ** 2 / (len(yt) - 1))
    t = (mt - mc) / se
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    q = stats.t.ppf(1 - alpha / 2, dof)
    diff = mt - mc
    return Effect(metric, mc, mt, diff, diff / mc if mc else np.nan, diff - q * se, diff + q * se, float(p), len(yc), len(yt))


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED adjustment: y - theta * (x - mean(x)). theta is pooled across both groups,
    which keeps the estimate unbiased because x is measured before randomization."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across segments)."""
    order = pvalues.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(pvalues.index)
