"""Week 8: Classification from scratch - logistic regression, KNN, metrics, ROC-AUC.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, lr=0.1, epochs=1500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        for _ in range(self.epochs):
            err = sigmoid(X @ self.w_ + self.b_) - y
            self.w_ -= self.lr * (X.T @ err / len(y) + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class OneVsRestLogistic:
    """Multiclass logistic regression: one binary classifier per class."""

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = [LogisticRegression(**self.kw).fit(X, (y == c).astype(int)) for c in self.classes_]
        return self

    def predict(self, X):
        scores = np.column_stack([m.predict_proba(X) for m in self.models_])
        return self.classes_[scores.argmax(axis=1)]


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        d = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1))
        idx = np.argsort(d, axis=1)[:, :self.k]
        return np.array([np.bincount(self.y_[i]).argmax() for i in idx])


def confusion_matrix(y, y_hat, n_classes=None):
    n = n_classes or int(max(y.max(), y_hat.max()) + 1)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, y_hat):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y, y_hat):
    tp = int(((y == 1) & (y_hat == 1)).sum())
    fp = int(((y == 0) & (y_hat == 1)).sum())
    fn = int(((y == 1) & (y_hat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def accuracy(y, y_hat) -> float:
    return float((y == y_hat).mean())


def roc_curve(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    tpr = np.concatenate([[0], np.cumsum(y_sorted) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / (1 - y).sum()])
    return fpr, tpr


def auc(fpr, tpr) -> float:
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def make_blobs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [3, 3], [-3, 3]])
    X = np.vstack([rng.normal(c, 1.1, size=(n, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_test_split(X, y, frac=0.3, seed=0):
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(len(y) * (1 - frac))
    return X[perm[:cut]], X[perm[cut:]], y[perm[:cut]], y[perm[cut:]]


def make_plots() -> None:
    X, y = make_blobs()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    ovr = OneVsRestLogistic().fit(Xtr, ytr)
    knn = KNN(5).fit(Xtr, ytr)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Week 8 - Classification from Scratch", fontweight="bold")

    gx, gy = np.meshgrid(np.linspace(-6, 6, 160), np.linspace(-3, 6.5, 160))
    grid = np.c_[gx.ravel(), gy.ravel()]
    ax[0].contourf(gx, gy, knn.predict(grid).reshape(gx.shape), alpha=.25, cmap="viridis")
    ax[0].scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="viridis", edgecolor="k", s=25)
    ax[0].set_title(f"KNN (k=5) decision regions, acc={accuracy(yte, knn.predict(Xte)):.2%}")

    cm = confusion_matrix(yte, ovr.predict(Xte), 3)
    ax[1].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true"); ax[1].set_title("One-vs-rest logistic: confusion matrix")

    yb_tr, yb_te = (ytr == 1).astype(int), (yte == 1).astype(int)
    lr = LogisticRegression().fit(Xtr, yb_tr)
    fpr, tpr = roc_curve(yb_te, lr.predict_proba(Xte))
    ax[2].plot(fpr, tpr, label=f"logistic (AUC={auc(fpr, tpr):.3f})")
    ax[2].plot([0, 1], [0, 1], "k--", label="chance")
    ax[2].set_xlabel("FPR"); ax[2].set_ylabel("TPR"); ax[2].legend(); ax[2].set_title("ROC curve (class 1 vs rest)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week08_classification.png")


def main():
    banner("Week 8 - Classification")
    X, y = make_blobs()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    print("OvR logistic accuracy:", round(accuracy(yte, OneVsRestLogistic().fit(Xtr, ytr).predict(Xte)), 4))
    print("KNN (k=5) accuracy   :", round(accuracy(yte, KNN(5).fit(Xtr, ytr).predict(Xte)), 4))
    yb = (yte == 1).astype(int)
    lr = LogisticRegression().fit(Xtr, (ytr == 1).astype(int))
    p, r, f1 = precision_recall_f1(yb, lr.predict(Xte))
    fpr, tpr = roc_curve(yb, lr.predict_proba(Xte))
    print(f"Binary (class 1): precision={p:.3f} recall={r:.3f} f1={f1:.3f} AUC={auc(fpr, tpr):.3f}")
    print("Confusion matrix:\n", confusion_matrix(yte, OneVsRestLogistic().fit(Xtr, ytr).predict(Xte), 3))
    make_plots()


if __name__ == "__main__":
    main()
