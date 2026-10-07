"""Week 9: Production scikit-learn workflow - ColumnTransformer, Pipeline, GridSearchCV, interpretation.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402

NUMERIC = ["tenure", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet", "payment"]


def make_churn_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Synthetic telecom churn data with a known signal plus noise and missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n).astype(float),
        "monthly_charges": rng.normal(65, 25, n).clip(20, 140),
        "support_calls": rng.poisson(1.5, n).astype(float),
        "contract": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[.55, .25, .2]),
        "internet": rng.choice(["fiber", "dsl", "none"], n, p=[.45, .4, .15]),
        "payment": rng.choice(["card", "bank", "cheque"], n),
    })
    logit = (-0.04 * df["tenure"] + 0.025 * df["monthly_charges"] + 0.35 * df["support_calls"]
             + np.where(df["contract"] == "month-to-month", 1.0, -0.8) - 1.0)
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    for col in ("tenure", "monthly_charges"):
        df.loc[rng.choice(n, int(n * 0.03), replace=False), col] = np.nan
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def tune_logistic(X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(build_pipeline(LogisticRegression(max_iter=1000)),
                        {"model__C": [0.01, 0.1, 1, 10]}, cv=5, scoring="roc_auc")
    return grid.fit(X_train, y_train)


def tune_forest(X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(build_pipeline(RandomForestClassifier(random_state=0)),
                        {"model__n_estimators": [100, 200], "model__max_depth": [4, 8, None]},
                        cv=3, scoring="roc_auc", n_jobs=1)
    return grid.fit(X_train, y_train)


def main():
    banner("Week 9 - scikit-learn Workflow")
    df = make_churn_data()
    X, y = df.drop(columns="churn"), df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    print(f"Rows: {len(df)} | churn rate: {y.mean():.1%} | missing cells: {int(X.isna().sum().sum())}")

    lg, rf = tune_logistic(Xtr, ytr), tune_forest(Xtr, ytr)
    print("Logistic best params:", lg.best_params_, "| CV AUC:", round(lg.best_score_, 4))
    print("Forest   best params:", rf.best_params_, "| CV AUC:", round(rf.best_score_, 4))
    results = {"Logistic": lg.best_estimator_, "Random Forest": rf.best_estimator_}
    for name, est in results.items():
        print(f"{name:14s} test AUC = {roc_auc_score(yte, est.predict_proba(Xte)[:, 1]):.4f}")
    best = max(results.values(), key=lambda e: roc_auc_score(yte, e.predict_proba(Xte)[:, 1]))
    print("\nClassification report (best model):\n", classification_report(yte, best.predict(Xte)))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle("Week 9 - Churn Prediction Pipeline", fontweight="bold")
    for name, est in results.items():
        RocCurveDisplay.from_estimator(est, Xte, yte, name=name, ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC curves")
    ConfusionMatrixDisplay.from_estimator(best, Xte, yte, ax=ax[1], cmap="Blues", colorbar=False)
    ax[1].set_title("Confusion matrix (best model)")
    imp = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0)
    order = np.argsort(imp.importances_mean)
    ax[2].barh(np.array(Xte.columns)[order], imp.importances_mean[order], color="#2f6fed")
    ax[2].set_title("Permutation importance (AUC drop)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week09_sklearn_pipeline.png")


if __name__ == "__main__":
    main()
