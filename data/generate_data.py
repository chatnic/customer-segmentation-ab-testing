"""Generate a synthetic e-commerce customer base plus a randomized promotion experiment.

Outputs two files:
  customers.csv   one row per customer: 12 months of pre-experiment behavior
  experiment.csv  random assignment to control / free-shipping offer, and 30-day outcomes

Hidden "true" customer types drive both behavior and how customers respond to the offer,
so the segmentation has something real to find and the treatment effect really does vary
by segment, as it usually does in practice.

Usage:
    python data/generate_data.py --n 40000 --seed 42 --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# latent type: (share, orders/yr, avg order value, recency scale days, discount affinity,
#               baseline 30d conversion, conversion lift from offer, AOV change under offer)
TYPES = {
    "loyal_high_value": (0.12, 14, 120, 12, 0.15, 0.55, 0.005, 0.00),
    "regular": (0.28, 6, 65, 35, 0.30, 0.26, 0.020, -0.02),
    "deal_seeker": (0.20, 5, 40, 45, 0.80, 0.16, 0.090, -0.05),
    "lapsing": (0.22, 2, 55, 150, 0.35, 0.05, 0.070, -0.02),
    "new_one_time": (0.18, 1, 45, 90, 0.40, 0.08, 0.030, -0.03),
}


def generate(n: int = 40_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(TYPES)
    shares = np.array([TYPES[t][0] for t in names])
    latent = rng.choice(names, size=n, p=shares / shares.sum())
    P = {k: np.array([TYPES[t][i] for t in latent]) for i, k in enumerate(
        ["share", "freq", "aov", "rec", "disc", "conv", "lift", "aov_shift"])}

    orders = np.maximum(1, rng.poisson(P["freq"] * rng.lognormal(0, 0.35, n)))
    aov = P["aov"] * rng.lognormal(0, 0.3, n)
    revenue = (orders * aov * rng.lognormal(0, 0.15, n)).round(2)
    recency = np.clip(rng.exponential(P["rec"]), 1, 365).round().astype(int)
    tenure = np.maximum(recency, np.clip(rng.gamma(2, 260, n), 30, 2500)).round().astype(int)
    tenure = np.where(latent == "new_one_time", np.clip(recency + rng.integers(0, 60, n), 1, 365), tenure)
    discount_share = np.clip(rng.beta(P["disc"] * 6 + 0.5, (1 - P["disc"]) * 6 + 0.5), 0, 1).round(3)
    sessions = rng.poisson(orders * rng.uniform(3, 9, n) + 2)
    returns_rate = np.clip(rng.beta(1.2, 12, n) + np.where(latent == "deal_seeker", 0.05, 0), 0, 1).round(3)
    categories = rng.integers(1, 7, n) + (orders > 8)
    email_open_rate = np.clip(rng.beta(2, 5, n) + np.where(latent == "loyal_high_value", 0.15, 0)
                              - np.where(latent == "lapsing", 0.1, 0), 0, 1).round(3)

    customers = pd.DataFrame({
        "customer_id": [f"C{200000 + i}" for i in range(n)],
        "tenure_days": tenure,
        "recency_days": recency,
        "orders_12m": orders,
        "revenue_12m": revenue,
        "avg_order_value": (revenue / orders).round(2),
        "discount_order_share": discount_share,
        "sessions_12m": sessions,
        "return_rate": returns_rate,
        "categories_purchased": np.minimum(categories, 7),
        "email_open_rate": email_open_rate,
    })

    # --- Experiment: 50/50 random assignment to a free-shipping offer ---
    treated = rng.random(n) < 0.5
    # Pre-period spend predicts post-period spend (that's what CUPED uses).
    engagement = np.log1p(revenue) / np.log1p(revenue).mean()
    p_conv = np.clip(P["conv"] * engagement ** 0.6 + treated * P["lift"], 0, 0.97)
    converted = rng.random(n) < p_conv
    basket = P["aov"] * (1 + treated * P["aov_shift"]) * rng.lognormal(0, 0.45, n) * engagement ** 0.5
    n_orders = np.where(converted, 1 + rng.poisson(np.clip(P["freq"] / 12 - 0.5, 0, None)), 0)
    rev_30d = np.where(converted, n_orders * basket, 0).round(2)

    experiment = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "group": np.where(treated, "treatment", "control"),
        "converted_30d": converted.astype(int),
        "orders_30d": n_orders,
        "revenue_30d": rev_30d,
        "pre_revenue_30d_avg": (revenue / 12).round(2),
    })
    return customers, experiment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args()
    customers, experiment = generate(a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(a.out / "customers.csv", index=False)
    experiment.to_csv(a.out / "experiment.csv", index=False)
    print(f"Wrote {len(customers):,} customers and experiment outcomes to {a.out}/")


if __name__ == "__main__":
    main()
