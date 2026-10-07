"""Week 10: Unsupervised clustering - K-Means++ from scratch, hierarchical, DBSCAN.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


class KMeans:
    """Lloyd's algorithm with k-means++ seeding."""

    def __init__(self, k=3, iters=100, seed=0, n_init=10):
        self.k, self.iters, self.seed, self.n_init = k, iters, seed, n_init

    def _init_centers(self, X, rng):
        centers = [X[rng.integers(len(X))]]
        for _ in range(self.k - 1):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centers)

    def fit(self, X):
        """Run Lloyd's algorithm n_init times from different seedings and keep the lowest inertia."""
        best = None
        for i in range(self.n_init):
            run = self._fit_once(X, np.random.default_rng(self.seed + i))
            if best is None or run[2] < best[2]:
                best = run
        self.centers_, self.labels_, self.inertia_ = best
        return self

    def _fit_once(self, X, rng):
        self.centers_ = self._init_centers(X, rng)
        for _ in range(self.iters):
            labels = self.predict(X)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else self.centers_[j]
                            for j in range(self.k)])
            if np.allclose(new, self.centers_):
                break
            self.centers_ = new
        labels = self.predict(X)
        return self.centers_, labels, float(((X - self.centers_[labels]) ** 2).sum())

    def predict(self, X):
        return ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(-1).argmin(1)


def elbow(X, ks=range(1, 9)):
    return [KMeans(k).fit(X).inertia_ for k in ks]


def agglomerative(X, k, method="ward"):
    Z = linkage(X, method=method)
    return Z, fcluster(Z, k, criterion="maxclust") - 1


def make_plots() -> None:
    Xb, yb = make_blobs(400, centers=4, cluster_std=0.9, random_state=4)
    Xm, ym = make_moons(400, noise=0.07, random_state=0)
    km = KMeans(4).fit(Xb)
    _, hc = agglomerative(Xb, 4)
    km_m = KMeans(2).fit(Xm)
    db_m = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Week 10 - Clustering", fontweight="bold")
    ax[0, 0].plot(range(1, 9), elbow(Xb), "o-"); ax[0, 0].set_title("K-Means elbow curve")
    ax[0, 0].set_xlabel("k"); ax[0, 0].set_ylabel("inertia")
    ax[0, 1].scatter(*Xb.T, c=km.labels_, cmap="tab10", s=14)
    ax[0, 1].scatter(*km.centers_.T, c="k", marker="X", s=150)
    ax[0, 1].set_title(f"K-Means++ (silhouette={silhouette_score(Xb, km.labels_):.2f})")
    dendrogram(linkage(Xb[:60], "ward"), ax=ax[0, 2], no_labels=True)
    ax[0, 2].set_title("Hierarchical dendrogram (60 pts)")
    ax[1, 0].scatter(*Xb.T, c=hc, cmap="tab10", s=14)
    ax[1, 0].set_title(f"Agglomerative / Ward (ARI={adjusted_rand_score(yb, hc):.2f})")
    ax[1, 1].scatter(*Xm.T, c=km_m.labels_, cmap="coolwarm", s=14)
    ax[1, 1].set_title(f"K-Means on moons (ARI={adjusted_rand_score(ym, km_m.labels_):.2f}) - fails")
    ax[1, 2].scatter(*Xm.T, c=db_m, cmap="coolwarm", s=14)
    ax[1, 2].set_title(f"DBSCAN on moons (ARI={adjusted_rand_score(ym, db_m):.2f}) - works")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "week10_clustering.png")


def main():
    banner("Week 10 - Clustering")
    Xb, yb = make_blobs(400, centers=4, cluster_std=0.9, random_state=4)
    Xm, ym = make_moons(400, noise=0.07, random_state=0)
    km = KMeans(4).fit(Xb)
    print(f"K-Means++  blobs: inertia={km.inertia_:.1f}, ARI={adjusted_rand_score(yb, km.labels_):.3f}")
    print(f"Ward       blobs: ARI={adjusted_rand_score(yb, agglomerative(Xb, 4)[1]):.3f}")
    print(f"K-Means    moons: ARI={adjusted_rand_score(ym, KMeans(2).fit(Xm).labels_):.3f}")
    print(f"DBSCAN     moons: ARI={adjusted_rand_score(ym, DBSCAN(eps=0.2).fit_predict(Xm)):.3f}")
    make_plots()


if __name__ == "__main__":
    main()
