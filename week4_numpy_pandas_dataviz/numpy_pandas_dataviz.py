"""Week 4: NumPy vectorisation, a Pandas pipeline and data visualisation.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


# ---------- NumPy ----------
def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)


def pairwise_distances(a: np.ndarray) -> np.ndarray:
    """Vectorised Euclidean distance matrix using broadcasting (no Python loops)."""
    diff = a[:, None, :] - a[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ---------- Pandas ----------
def make_sales_data(seed: int = 7, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "category": rng.choice(["Electronics", "Clothing", "Grocery"], n, p=[0.3, 0.3, 0.4]),
        "units": rng.integers(1, 30, n),
        "unit_price": rng.uniform(5, 120, n).round(2),
    })
    # inject some missing values to exercise cleaning
    df.loc[rng.choice(n, 15, replace=False), "unit_price"] = np.nan
    return df


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["unit_price"] = out.groupby("category")["unit_price"].transform(lambda s: s.fillna(s.median()))
    out["revenue"] = out["units"] * out["unit_price"]
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out


def revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["region", "category"])["revenue"].sum().unstack(fill_value=0).round(2))


def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    return df.groupby("month")["revenue"].sum()


# ---------- Visualisation ----------
def make_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Week 4 - Sales Data Visualisation", fontsize=14, fontweight="bold")

    monthly_revenue(df).plot(ax=ax[0, 0], marker="o", color="#2f6fed")
    ax[0, 0].set_title("Monthly revenue")
    ax[0, 0].tick_params(axis="x", rotation=45)

    sns.heatmap(revenue_summary(df), annot=True, fmt=".0f", cmap="YlGnBu", ax=ax[0, 1])
    ax[0, 1].set_title("Revenue by region and category")

    sns.boxplot(data=df, x="category", y="unit_price", hue="category", legend=False, ax=ax[1, 0])
    ax[1, 0].set_title("Unit price distribution")

    sns.histplot(df["revenue"], bins=30, kde=True, color="#7c3aed", ax=ax[1, 1])
    ax[1, 1].set_title("Revenue per order")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "week04_dataviz.png")


def main():
    banner("Week 4 - NumPy, Pandas & Data Visualisation")
    pts = np.random.default_rng(0).normal(size=(5, 3))
    print("Pairwise distance matrix shape:", pairwise_distances(pts).shape)
    print("Z-scored column means (~0):", np.round(zscore(pts).mean(axis=0), 6))
    df = clean_and_enrich(make_sales_data())
    print("Missing values after cleaning:", int(df["unit_price"].isna().sum()))
    print("\nRevenue by region/category:\n", revenue_summary(df))
    print("\nTop 3 months:\n", monthly_revenue(df).nlargest(3).round(2))
    make_plots(df)


if __name__ == "__main__":
    main()
