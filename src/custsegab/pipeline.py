"""Segment customers, analyze the free-shipping experiment overall and by segment, and
recommend who should get the offer.

    PYTHONPATH=src python -m custsegab.pipeline --data data/raw --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from custsegab.experiment import (  # noqa: E402
    cuped, holm, mde_proportion, sample_size_proportion, srm_check, two_proportion, welch,
)
from custsegab.segmentation import choose_k, fit_segments, name_segments, prepare, profile  # noqa: E402

GROSS_MARGIN = 0.40          # margin on incremental revenue
SHIPPING_COST_PER_ORDER = 8.0  # what free shipping costs us on each order a treated customer places
SEED = 42
DEFAULT_K = 5


def segment(customers: pd.DataFrame, out: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    X = prepare(customers)
    ksel = choose_k(X)
    # Silhouette is fairly flat from 4 to 6, which is typical for behavioral data with no
    # hard boundaries. k is a business choice within that range: 5 separates top spenders
    # from steady regulars, two groups that need different marketing.
    k = DEFAULT_K
    labels = fit_segments(X, k)
    prof = profile(customers, labels)
    names = name_segments(prof)
    prof.index = prof.index.map(names)
    seg = pd.Series(labels, index=customers.index).map(names).rename("segment")
    print(f"Segments (k={k}):\n{prof[['customers', 'share_of_revenue']].round(3)}")
    return seg, prof, ksel


def analyze(exp: pd.DataFrame) -> dict:
    c, t = exp[exp.group == "control"], exp[exp.group == "treatment"]
    res = {"srm_p_value": srm_check(len(c), len(t))}
    res["conversion"] = two_proportion("conversion_30d", c.converted_30d.values, t.converted_30d.values).as_dict()
    raw = welch("revenue_30d", c.revenue_30d.values, t.revenue_30d.values)
    adj_y, theta = cuped(exp.revenue_30d.values, exp.pre_revenue_30d_avg.values)
    exp = exp.assign(revenue_cuped=adj_y)
    adj = welch("revenue_30d_cuped", exp.loc[exp.group == "control", "revenue_cuped"].values,
                exp.loc[exp.group == "treatment", "revenue_cuped"].values)
    res["revenue_raw"] = raw.as_dict()
    res["revenue_cuped"] = adj.as_dict()
    res["cuped_theta"] = theta
    res["cuped_variance_reduction"] = float(1 - ((adj.ci_high - adj.ci_low) / (raw.ci_high - raw.ci_low)) ** 2)
    return res


def segment_effects(df: pd.DataFrame) -> pd.DataFrame:
    adj_y, _ = cuped(df.revenue_30d.values, df.pre_revenue_30d_avg.values)
    df = df.assign(revenue_cuped=adj_y)
    rows = []
    for seg, g in df.groupby("segment"):
        c, t = g[g.group == "control"], g[g.group == "treatment"]
        conv = two_proportion("conversion", c.converted_30d.values, t.converted_30d.values)
        rev = welch("revenue", c.revenue_cuped.values, t.revenue_cuped.values)
        cost = SHIPPING_COST_PER_ORDER * t.orders_30d.mean()
        net = GROSS_MARGIN * rev.abs_diff - cost
        rows.append({
            "segment": seg, "customers": len(g),
            "control_conversion": conv.control, "treatment_conversion": conv.treatment,
            "conversion_lift_pts": conv.abs_diff * 100, "conversion_p": conv.p_value,
            "incremental_revenue_per_customer": rev.abs_diff,
            "rev_ci_low": rev.ci_low, "rev_ci_high": rev.ci_high, "revenue_p": rev.p_value,
            "shipping_cost_per_customer": cost,
            "net_profit_per_customer": net,
        })
    out = pd.DataFrame(rows)
    out["conversion_p_holm"] = holm(out.set_index("segment")["conversion_p"]).values
    out["revenue_p_holm"] = holm(out.set_index("segment")["revenue_p"]).values
    out["recommendation"] = np.where(out["net_profit_per_customer"] > 0, "send offer", "do not send")
    return out.sort_values("net_profit_per_customer", ascending=False)


def figures(customers: pd.DataFrame, prof: pd.DataFrame, ksel: pd.DataFrame, seg_fx: pd.DataFrame,
            fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ksel.k, ksel.silhouette, marker="o")
    ax.set(xlabel="Number of segments (k)", ylabel="Silhouette score", title="Choosing the number of segments")
    fig.tight_layout(); fig.savefig(fig_dir / "choose_k.png", dpi=150); plt.close(fig)

    p = prof.sort_values("share_of_revenue")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(p))
    ax.barh(y - 0.2, p.share_of_customers, 0.4, label="share of customers", color="#9aa5b1")
    ax.barh(y + 0.2, p.share_of_revenue, 0.4, label="share of revenue", color="#1f4e8c")
    ax.set_yticks(y, p.index)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Segments: customers vs. revenue")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "segment_mix.png", dpi=150); plt.close(fig)

    s = seg_fx.sort_values("net_profit_per_customer")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = np.where(s.net_profit_per_customer > 0, "#2e7d32", "#c62828")
    ax.barh(s.segment, s.net_profit_per_customer, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Net profit per customer from free-shipping offer ($)",
           title="Who should get the offer?")
    fig.tight_layout(); fig.savefig(fig_dir / "net_profit_by_segment.png", dpi=150); plt.close(fig)

    s = seg_fx.sort_values("conversion_lift_pts")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(s.segment, s.conversion_lift_pts, color="#1f4e8c")
    ax.set(xlabel="Conversion lift (percentage points)", title="Treatment effect varies by segment")
    fig.tight_layout(); fig.savefig(fig_dir / "conversion_lift_by_segment.png", dpi=150); plt.close(fig)


def run(data: Path, out: Path) -> dict:
    customers = pd.read_csv(data / "customers.csv")
    exp = pd.read_csv(data / "experiment.csv")
    seg, prof, ksel = segment(customers, out)
    df = exp.merge(customers.assign(segment=seg)[["customer_id", "segment"]], on="customer_id")

    overall = analyze(df)
    base = df.loc[df.group == "control", "converted_30d"].mean()
    n_half = len(df) // 2
    overall["design"] = {
        "baseline_conversion": float(base),
        "mde_abs_at_this_sample": mde_proportion(base, n_half),
        "n_per_group_for_2pt_lift": sample_size_proportion(base, 0.02),
    }
    seg_fx = segment_effects(df)

    target = seg_fx[seg_fx.recommendation == "send offer"]
    seg_sizes = seg_fx.set_index("segment")["customers"]
    blanket = float((seg_fx.net_profit_per_customer * seg_fx.customers).sum())
    targeted = float((target.net_profit_per_customer * target.customers).sum())
    overall["targeting"] = {
        "segments_to_target": target.segment.tolist(),
        "share_of_customers_targeted": float(target.customers.sum() / seg_sizes.sum()),
        "net_profit_blanket_per_1k_customers": blanket / seg_sizes.sum() * 1000,
        "net_profit_targeted_per_1k_customers": targeted / seg_sizes.sum() * 1000,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    prof.round(3).to_csv(out / "tables" / "segment_profiles.csv")
    ksel.to_csv(out / "tables" / "choose_k.csv", index=False)
    seg_fx.round(4).to_csv(out / "tables" / "segment_effects.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(overall, indent=2, default=float))
    figures(customers, prof, ksel, seg_fx, out / "figures")
    print(json.dumps(overall, indent=2, default=float))
    print(seg_fx.round(3).to_string())
    return overall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
