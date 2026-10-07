"""Week 6: Calculus for ML - numerical derivatives, a tiny autograd engine, optimisers.

Author: Parth Denge | PRN: 240705201018
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        e = np.zeros_like(x, dtype=float)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


class Value:
    """Scalar with reverse-mode automatic differentiation (backpropagation on a DAG)."""

    def __init__(self, data, parents=(), op=""):
        self.data, self.grad, self._parents, self._op = float(data), 0.0, parents, op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-self._wrap(o))
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __rsub__(self, o): return self._wrap(o) + (-self)
    def __truediv__(self, o): return self * self._wrap(o) ** -1

    def backward(self):
        order, seen = [], set()

        def topo(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    topo(p)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def gradient_descent(grad, x0, lr=0.1, steps=100):
    x, path = np.asarray(x0, dtype=float), []
    for _ in range(steps):
        path.append(x.copy())
        x = x - lr * grad(x)
    path.append(x.copy())
    return np.array(path)


def momentum_descent(grad, x0, lr=0.02, beta=0.9, steps=100):
    x, v, path = np.asarray(x0, dtype=float), 0.0, []
    for _ in range(steps):
        path.append(x.copy())
        v = beta * v + grad(x)
        x = x - lr * v
    path.append(x.copy())
    return np.array(path)


def adam(grad, x0, lr=0.3, b1=0.9, b2=0.999, eps=1e-8, steps=100):
    x, m, v, path = np.asarray(x0, dtype=float), 0.0, 0.0, []
    for t in range(1, steps + 1):
        path.append(x.copy())
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    path.append(x.copy())
    return np.array(path)


def bowl(p):  # an elongated quadratic bowl: ill-conditioned on purpose
    return p[0] ** 2 + 10 * p[1] ** 2


def bowl_grad(p):
    return np.array([2 * p[0], 20 * p[1]])


def make_plots() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Week 6 - Calculus & Gradient-based Optimisation", fontweight="bold")

    xs = np.linspace(-3, 3, 200)
    f = lambda x: x ** 3 - 2 * x  # noqa: E731
    ax[0].plot(xs, f(xs), label="f(x) = x^3 - 2x")
    ax[0].plot(xs, [numerical_derivative(f, x) for x in xs], "--", label="numerical f'(x)")
    ax[0].plot(xs, 3 * xs ** 2 - 2, ":", label="analytic 3x^2-2", color="red")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Numerical vs analytic derivative")

    gx, gy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-1.5, 1.5, 100))
    ax[1].contour(gx, gy, gx ** 2 + 10 * gy ** 2, levels=20, cmap="Greys", alpha=.5)
    start = [3.5, 1.2]
    for name, path, c in [("GD", gradient_descent(bowl_grad, start, lr=0.09, steps=40), "tab:blue"),
                          ("Momentum", momentum_descent(bowl_grad, start, steps=40), "tab:green"),
                          ("Adam", adam(bowl_grad, start, steps=40), "tab:red")]:
        ax[1].plot(path[:, 0], path[:, 1], "o-", ms=3, color=c, label=name)
    ax[1].legend(); ax[1].set_title("Optimiser trajectories")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week06_gradients.png")


def main():
    banner("Week 6 - Calculus & Gradients")
    print("d/dx sin(x) at 0 (numeric):", round(numerical_derivative(math.sin, 0.0), 6))
    a, b = Value(2.0), Value(-3.0)
    c = (a * b + a ** 2).tanh()
    c.backward()
    print(f"Autograd: c={c.data:.4f}, dc/da={a.grad:.4f}, dc/db={b.grad:.4f}")
    for name, path in [("GD", gradient_descent(bowl_grad, [3.5, 1.2], lr=0.09)),
                       ("Momentum", momentum_descent(bowl_grad, [3.5, 1.2])),
                       ("Adam", adam(bowl_grad, [3.5, 1.2]))]:
        print(f"{name:9s} final loss after 100 steps: {bowl(path[-1]):.3e}")
    make_plots()


if __name__ == "__main__":
    main()
