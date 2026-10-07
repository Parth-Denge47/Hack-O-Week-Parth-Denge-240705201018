"""Week 7: Regression from scratch - normal equation, gradient descent, polynomial, Ridge, Lasso.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def r2_score(y, y_hat) -> float:
    return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mse(y, y_hat) -> float:
    return float(((y - y_hat) ** 2).mean())


class LinearRegression:
    """Ordinary least squares via the normal equation (pseudo-inverse for stability)."""

    def fit(self, X, y):
        self.w_ = np.linalg.pinv(add_bias(X)) @ y
        return self

    def predict(self, X):
        return add_bias(X) @ self.w_


class GDLinearRegression:
    def __init__(self, lr=0.05, epochs=2000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        A = add_bias(X)
        self.w_, self.loss_ = np.zeros(A.shape[1]), []
        for _ in range(self.epochs):
            err = A @ self.w_ - y
            self.loss_.append(float((err ** 2).mean()))
            self.w_ -= self.lr * 2 * A.T @ err / len(y)
        return self

    def predict(self, X):
        return add_bias(X) @ self.w_


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([x ** d for d in range(1, degree + 1)])


class Ridge:
    """L2-regularised regression, closed form. The bias is not penalised."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        A = add_bias(X)
        P = np.eye(A.shape[1]) * self.alpha
        P[0, 0] = 0
        self.w_ = np.linalg.solve(A.T @ A + P, A.T @ y)
        return self

    def predict(self, X):
        return add_bias(X) @ self.w_


class Lasso:
    """L1-regularised regression via coordinate descent with soft-thresholding."""

    def __init__(self, alpha=0.1, iters=500):
        self.alpha, self.iters = alpha, iters

    @staticmethod
    def _soft(z, t):
        return np.sign(z) * max(abs(z) - t, 0.0)

    def fit(self, X, y):
        n, d = X.shape
        self.b_, self.w_ = y.mean(), np.zeros(d)
        for _ in range(self.iters):
            for j in range(d):
                r = y - self.b_ - X @ self.w_ + X[:, j] * self.w_[j]
                self.w_[j] = self._soft(X[:, j] @ r / n, self.alpha) / ((X[:, j] ** 2).mean() + 1e-12)
            self.b_ = (y - X @ self.w_).mean()
        return self

    def predict(self, X):
        return X @ self.w_ + self.b_


def make_data(seed=1, n=60):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, n))
    y = 0.5 * x ** 3 - x ** 2 + 2 * x + rng.normal(0, 2.0, n)
    return x, y


def make_plots() -> None:
    x, y = make_data()
    grid = np.linspace(-3, 3, 300)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Week 7 - Regression from Scratch", fontweight="bold")

    ax[0].scatter(x, y, s=14, alpha=.7)
    for deg, c in [(1, "tab:red"), (3, "tab:green"), (12, "tab:purple")]:
        m = LinearRegression().fit(polynomial_features(x, deg), y)
        ax[0].plot(grid, np.clip(m.predict(polynomial_features(grid, deg)), -30, 30), c, label=f"degree {deg}")
    ax[0].set_ylim(-30, 30); ax[0].legend(); ax[0].set_title("Under / good / over-fit")

    X12 = polynomial_features(x, 12)
    mu, sd = X12.mean(0), X12.std(0)
    Z = (X12 - mu) / sd
    alphas = np.logspace(-3, 3, 30)
    ax[1].plot(alphas, [np.abs(Ridge(a).fit(Z, y).w_[1:]).sum() for a in alphas], label="Ridge (L2)")
    ax[1].plot(alphas, [np.abs(Lasso(a, 150).fit(Z, y).w_).sum() for a in alphas], label="Lasso (L1)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("alpha"); ax[1].set_ylabel("sum |weights|")
    ax[1].legend(); ax[1].set_title("Shrinkage of coefficients")

    gd = GDLinearRegression(lr=0.05, epochs=300).fit(x.reshape(-1, 1), y)
    ax[2].plot(gd.loss_); ax[2].set_yscale("log"); ax[2].set_title("Gradient descent loss")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("MSE")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week07_regression.png")


def main():
    banner("Week 7 - Regression")
    x, y = make_data()
    X = x.reshape(-1, 1)
    ne = LinearRegression().fit(X, y)
    gd = GDLinearRegression().fit(X, y)
    print("Normal equation weights:", np.round(ne.w_, 4))
    print("Gradient descent weights:", np.round(gd.w_, 4))
    for deg in (1, 3, 12):
        m = LinearRegression().fit(polynomial_features(x, deg), y)
        print(f"Polynomial degree {deg:2d}: train R2 = {r2_score(y, m.predict(polynomial_features(x, deg))):.4f}")
    make_plots()


if __name__ == "__main__":
    main()
