"""Weeks 13 & 14: Ensembles (bagging, boosting, XGBoost, LightGBM), bias-variance, over/under-fitting,
L1/L2 regularisation.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402

warnings.filterwarnings("ignore")


def get_data(seed: int = 0):
    X, y = make_classification(2000, n_features=20, n_informative=8, n_redundant=4,
                               flip_y=0.08, class_sep=1.0, random_state=seed)
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def get_models() -> dict:
    return {
        "Single tree": DecisionTreeClassifier(random_state=0),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0),
        "Random Forest": RandomForestClassifier(200, random_state=0),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=0),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1),
    }


def compare_models(Xtr, Xte, ytr, yte) -> dict:
    out = {}
    for name, m in get_models().items():
        m.fit(Xtr, ytr)
        out[name] = (accuracy_score(ytr, m.predict(Xtr)), accuracy_score(yte, m.predict(Xte)))
    return out


def depth_sweep(Xtr, Xte, ytr, yte, depths=range(1, 21)):
    """Model complexity sweep: shallow trees underfit, deep trees overfit."""
    tr, te = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(t.score(Xtr, ytr))
        te.append(t.score(Xte, yte))
    return list(depths), tr, te


def bias_variance_decomposition(n_rounds: int = 60, seed: int = 0, depths=(1, 3, 6, 12)):
    """Monte-Carlo estimate of squared bias and variance for tree regressors of different depth."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0, 2 * np.pi, 80).reshape(-1, 1)
    truth = np.sin(x_test).ravel()
    result = {}
    for d in depths:
        preds = []
        for _ in range(n_rounds):
            x = rng.uniform(0, 2 * np.pi, (60, 1))
            y = np.sin(x).ravel() + rng.normal(0, 0.35, 60)
            preds.append(DecisionTreeRegressor(max_depth=d).fit(x, y).predict(x_test))
        preds = np.array(preds)
        result[d] = (float(((preds.mean(0) - truth) ** 2).mean()), float(preds.var(0).mean()))
    return result


def regularisation_paths(Xtr, ytr, Xte, yte, cs=np.logspace(-3, 2, 12)):
    """Test accuracy and non-zero coefficient count for L1 and L2 logistic regression across C."""
    res = {"l1": ([], []), "l2": ([], [])}
    for pen in ("l1", "l2"):
        for c in cs:
            m = LogisticRegression(C=c, l1_ratio=1.0 if pen == "l1" else 0.0, solver="saga",
                                   max_iter=3000).fit(Xtr, ytr)
            res[pen][0].append(m.score(Xte, yte))
            res[pen][1].append(int((np.abs(m.coef_) > 1e-6).sum()))
    return cs, res


def make_plots() -> None:
    Xtr, Xte, ytr, yte = get_data()
    scores = compare_models(Xtr, Xte, ytr, yte)

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Weeks 13 & 14 - Ensembles, Bias-Variance and Regularisation", fontweight="bold")

    names = list(scores)
    w = 0.38
    xs = np.arange(len(names))
    ax[0, 0].bar(xs - w / 2, [scores[n][0] for n in names], w, label="train", color="#9ca3af")
    ax[0, 0].bar(xs + w / 2, [scores[n][1] for n in names], w, label="test", color="#2f6fed")
    ax[0, 0].set_xticks(xs); ax[0, 0].set_xticklabels(names, rotation=30, ha="right")
    ax[0, 0].set_ylim(0.6, 1.1); ax[0, 0].legend(ncol=2, loc="upper center"); ax[0, 0].set_title("Bagging vs boosting: train / test accuracy")

    d, tr, te = depth_sweep(Xtr, Xte, ytr, yte)
    ax[0, 1].plot(d, tr, "o-", label="train"); ax[0, 1].plot(d, te, "o-", label="test")
    ax[0, 1].axvspan(0.5, 3, color="orange", alpha=.15); ax[0, 1].axvspan(10, 20.5, color="red", alpha=.1)
    tx = ax[0, 1].get_xaxis_transform()
    ax[0, 1].text(1, 0.5, "underfit", color="darkorange", transform=tx)
    ax[0, 1].text(13, 0.5, "overfit", color="red", transform=tx)
    ax[0, 1].set_xlabel("tree depth"); ax[0, 1].legend(); ax[0, 1].set_title("Under- vs over-fitting")

    bv = bias_variance_decomposition()
    ds = list(bv)
    ax[0, 2].plot(ds, [bv[k][0] for k in ds], "o-", label="bias^2")
    ax[0, 2].plot(ds, [bv[k][1] for k in ds], "o-", label="variance")
    ax[0, 2].plot(ds, [bv[k][0] + bv[k][1] for k in ds], "k--", label="bias^2 + variance")
    ax[0, 2].set_xlabel("tree depth (model complexity)"); ax[0, 2].legend(); ax[0, 2].set_title("Bias-variance trade-off")

    sizes, ltr, lte = learning_curve(DecisionTreeClassifier(max_depth=4, random_state=0),
                                     np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]),
                                     train_sizes=np.linspace(0.1, 1, 8), cv=5)
    ax[1, 0].plot(sizes, ltr.mean(1), "o-", label="train"); ax[1, 0].plot(sizes, lte.mean(1), "o-", label="validation")
    ax[1, 0].set_xlabel("training examples"); ax[1, 0].legend(); ax[1, 0].set_title("Learning curve (depth-4 tree)")

    cs, reg = regularisation_paths(Xtr, ytr, Xte, yte)
    for pen in ("l1", "l2"):
        ax[1, 1].plot(cs, reg[pen][0], "o-", label=f"{pen.upper()} test accuracy")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("C (inverse regularisation strength)")
    ax[1, 1].legend(); ax[1, 1].set_title("Regularisation vs accuracy")
    for pen in ("l1", "l2"):
        ax[1, 2].plot(cs, reg[pen][1], "o-", label=pen.upper())
    ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("C"); ax[1, 2].set_ylabel("non-zero coefficients")
    ax[1, 2].legend(); ax[1, 2].set_title("L1 gives sparsity, L2 only shrinks")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "week13_14_ensembles_bias_variance.png")


def main():
    banner("Weeks 13 & 14 - Ensembles, Bias-Variance, Regularisation")
    Xtr, Xte, ytr, yte = get_data()
    print(f"{'Model':15s} {'train acc':>10s} {'test acc':>10s}")
    for name, (a, b) in compare_models(Xtr, Xte, ytr, yte).items():
        print(f"{name:15s} {a:10.4f} {b:10.4f}")
    print("\nBias-variance (tree depth -> bias^2, variance):")
    for d, (b, v) in bias_variance_decomposition().items():
        print(f"  depth {d:2d}: bias^2={b:.4f}  variance={v:.4f}")
    cs, reg = regularisation_paths(Xtr, ytr, Xte, yte)
    print("\nNon-zero coefficients at smallest C -> L1:", reg["l1"][1][0], "| L2:", reg["l2"][1][0])
    make_plots()


if __name__ == "__main__":
    main()
