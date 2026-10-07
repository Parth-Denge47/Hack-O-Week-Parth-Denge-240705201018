"""Week 5: Linear algebra for ML - vectors, projections, Gram-Schmidt, eigenvalues, PCA.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def norm(a) -> float:
    return dot(a, a) ** 0.5


def cosine_similarity(a, b) -> float:
    return dot(a, b) / (norm(a) * norm(b))


def project(u, onto):
    """Projection of u onto the direction `onto`."""
    onto = np.asarray(onto, dtype=float)
    return (dot(u, onto) / dot(onto, onto)) * onto


def gram_schmidt(vectors) -> np.ndarray:
    """Orthonormalise a list of linearly independent vectors."""
    basis = []
    for v in np.asarray(vectors, dtype=float):
        w = v - sum(project(v, b) for b in basis) if basis else v.copy()
        if norm(w) < 1e-10:
            raise ValueError("vectors are linearly dependent")
        basis.append(w / norm(w))
    return np.array(basis)


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def power_iteration(A: np.ndarray, iters: int = 500, seed: int = 0):
    """Dominant eigenpair of a square matrix by power iteration."""
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    for _ in range(iters):
        v = A @ v
        v /= np.linalg.norm(v)
    return float(v @ A @ v), v


def pca(X: np.ndarray, k: int):
    """PCA via eigendecomposition of the covariance matrix."""
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return Xc @ vecs[:, :k], vals[:k] / vals.sum(), vecs[:, :k]


def make_plots() -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Week 5 - Linear Algebra for ML", fontweight="bold")

    # 1. a transformation applied to the unit square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float).T
    shear = np.array([[1, 0.8], [0, 1]])
    for name, M, c in [("original", np.eye(2), "gray"), ("rotate 40deg", rotation(np.radians(40)), "tab:blue"),
                       ("shear", shear, "tab:red")]:
        p = M @ square
        ax[0].plot(p[0], p[1], color=c, label=name)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend(); ax[0].set_title("Linear transformations")

    # 2. projection
    u, d = np.array([3.0, 2.0]), np.array([4.0, 0.5])
    p = project(u, d)
    ax[1].quiver(0, 0, *u, angles="xy", scale_units="xy", scale=1, color="tab:blue", label="u")
    ax[1].quiver(0, 0, *d, angles="xy", scale_units="xy", scale=1, color="tab:green", label="direction")
    ax[1].quiver(0, 0, *p, angles="xy", scale_units="xy", scale=1, color="tab:red", label="projection")
    ax[1].plot([u[0], p[0]], [u[1], p[1]], "k--", lw=1)
    ax[1].set_xlim(-1, 5); ax[1].set_ylim(-1, 4); ax[1].set_aspect("equal"); ax[1].grid(alpha=.3)
    ax[1].legend(); ax[1].set_title("Vector projection")

    # 3. PCA on correlated 2-D data
    rng = np.random.default_rng(3)
    X = rng.normal(size=(250, 2)) @ np.array([[2.5, 1.2], [0, 0.6]])
    _, ratio, comps = pca(X, 2)
    ax[2].scatter(X[:, 0], X[:, 1], s=10, alpha=.5)
    mu = X.mean(axis=0)
    for i, c in enumerate(["tab:red", "tab:green"]):
        ax[2].arrow(*mu, *(comps[:, i] * 3), color=c, width=0.05,
                    label=f"PC{i + 1} ({ratio[i]:.0%})")
    ax[2].set_aspect("equal"); ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_title("Principal axes")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week05_linear_algebra.png")


def main():
    banner("Week 5 - Linear Algebra")
    a, b = [1, 2, 3], [4, 5, 6]
    print("dot:", dot(a, b), "| norm(a):", round(norm(a), 4), "| cosine:", round(cosine_similarity(a, b), 4))
    Q = gram_schmidt([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    print("Gram-Schmidt Q Q^T == I:", np.allclose(Q @ Q.T, np.eye(3)))
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    lam, _ = power_iteration(A)
    print("Dominant eigenvalue (power iteration):", round(lam, 5),
          "| numpy:", round(np.linalg.eigvalsh(A).max(), 5))
    make_plots()


if __name__ == "__main__":
    main()
