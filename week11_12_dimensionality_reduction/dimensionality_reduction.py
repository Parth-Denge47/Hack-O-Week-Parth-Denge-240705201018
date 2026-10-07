"""Weeks 11 & 12: Dimensionality reduction - PCA and t-SNE intuition and applications.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def pca_from_scratch(X: np.ndarray, k: int):
    """PCA through SVD of the centred data. Returns projected data and explained-variance ratio."""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Xc @ Vt[:k].T, var[:k] / var.sum()


def reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    """Compress to k components and map back to the original space."""
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return (X - mu) @ Vt[:k].T @ Vt[:k] + mu


def components_for_variance(X: np.ndarray, target: float = 0.95) -> int:
    ratios = PCA().fit(X).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ratios), target) + 1)


def cv_accuracy_with_pca(X, y, k: int) -> float:
    pipe = make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=2000))
    return float(cross_val_score(pipe, X, y, cv=3).mean())


def make_plots() -> None:
    digits = load_digits()
    X, y = digits.data, digits.target
    Z_pca, _ = pca_from_scratch(X, 2)
    sample = np.random.default_rng(0).choice(len(X), 1000, replace=False)
    Z_tsne = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X[sample])

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Weeks 11 & 12 - PCA and t-SNE on the Digits dataset (64-D -> 2-D)", fontweight="bold")

    sc = ax[0, 0].scatter(*Z_pca.T, c=y, cmap="tab10", s=6)
    ax[0, 0].set_title("PCA projection (linear)"); fig.colorbar(sc, ax=ax[0, 0], ticks=range(10))
    sc = ax[0, 1].scatter(*Z_tsne.T, c=y[sample], cmap="tab10", s=8)
    ax[0, 1].set_title("t-SNE projection (non-linear, preserves neighbourhoods)")
    fig.colorbar(sc, ax=ax[0, 1], ticks=range(10))

    ratios = PCA().fit(X).explained_variance_ratio_
    ax[0, 2].plot(np.cumsum(ratios), "o-", ms=3)
    ax[0, 2].axhline(0.95, color="r", ls="--", label="95% variance")
    ax[0, 2].axvline(components_for_variance(X) - 1, color="gray", ls=":")
    ax[0, 2].set_xlabel("components"); ax[0, 2].set_ylabel("cumulative variance")
    ax[0, 2].legend(); ax[0, 2].set_title("Scree: how many components?")

    # one digit reconstructed at different compression levels
    recon =np.hstack([X[0].reshape(8, 8)] + [reconstruct(X, k)[0].reshape(8, 8) for k in (2, 10, 30)])
    ax[1, 0].imshow(recon, cmap="gray"); ax[1, 0].axis("off")
    ax[1, 0].set_title("Reconstruction: original | 2 | 10 | 30 components")

    ks = [2, 5, 10, 20, 40, 64]
    ax[1, 1].plot(ks, [cv_accuracy_with_pca(X, y, k) for k in ks], "o-")
    ax[1, 1].set_xlabel("PCA components"); ax[1, 1].set_ylabel("CV accuracy")
    ax[1, 1].set_title("Application: PCA as preprocessing")

    for perp, c in [(5, "tab:red"), (30, "tab:green"), (100, "tab:blue")]:
        Z = TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(X[sample[:600]])
        ax[1, 2].scatter(Z[:, 0] / Z[:, 0].std() + {5: -6, 30: 0, 100: 6}[perp], Z[:, 1] / Z[:, 1].std(),
                         c=c, s=5, label=f"perplexity {perp}")
    ax[1, 2].legend(markerscale=3); ax[1, 2].set_title("t-SNE: effect of perplexity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "week11_12_pca_tsne.png")


def main():
    banner("Weeks 11 & 12 - PCA and t-SNE")
    digits = load_digits()
    X, y = digits.data, digits.target
    Z, ratio = pca_from_scratch(X, 2)
    sk = PCA(2).fit(X)
    print("Digits shape:", X.shape)
    print("Explained variance (scratch):", np.round(ratio, 4), "| sklearn:", np.round(sk.explained_variance_ratio_, 4))
    print("Components for 95% variance:", components_for_variance(X), "of", X.shape[1])
    print("Reconstruction MSE with 10 comps:", round(float(((X - reconstruct(X, 10)) ** 2).mean()), 3))
    for k in (5, 20, 64):
        print(f"CV accuracy with {k:2d} PCA components: {cv_accuracy_with_pca(X, y, k):.4f}")
    make_plots()


if __name__ == "__main__":
    main()
