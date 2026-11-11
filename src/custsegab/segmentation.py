"""Behavioral customer segmentation with K-means."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = [
    "recency_days", "orders_12m", "revenue_12m", "avg_order_value", "discount_order_share",
    "tenure_days", "email_open_rate", "return_rate",
]
LOG_FEATURES = ["recency_days", "orders_12m", "revenue_12m", "avg_order_value", "tenure_days"]


def prepare(customers: pd.DataFrame) -> np.ndarray:
    X = customers[SEG_FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


def choose_k(X: np.ndarray, ks=range(3, 9), sample: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[idx], km.labels_[idx])})
    return pd.DataFrame(rows)


def fit_segments(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)


def profile(customers: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = customers.assign(cluster=labels)
    prof = df.groupby("cluster")[SEG_FEATURES].median()
    prof["customers"] = df.groupby("cluster").size()
    prof["share_of_customers"] = prof["customers"] / len(df)
    prof["share_of_revenue"] = df.groupby("cluster")["revenue_12m"].sum() / df["revenue_12m"].sum()
    return prof


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Give clusters business names by greedily matching each archetype to its best cluster."""
    z = (prof[SEG_FEATURES] - prof[SEG_FEATURES].mean()) / prof[SEG_FEATURES].std(ddof=0).replace(0, 1)
    archetypes = {
        "VIP loyalists": z["revenue_12m"] + z["orders_12m"] - z["recency_days"],
        "Lapsing customers": z["recency_days"] - z["orders_12m"],
        "Deal seekers": 2 * z["discount_order_share"] - z["avg_order_value"],
        "New customers": -z["tenure_days"] - 0.5 * z["orders_12m"],
        "Core regulars": -(z[["revenue_12m", "recency_days", "discount_order_share"]].abs().sum(axis=1)),
    }
    names: dict[int, str] = {}
    remaining = set(prof.index)
    for name, score in archetypes.items():
        if not remaining:
            break
        best = score.loc[list(remaining)].idxmax()
        names[best] = name
        remaining.discard(best)
    for i, c in enumerate(sorted(remaining)):
        names[c] = f"Other segment {i + 1}"
    return names
