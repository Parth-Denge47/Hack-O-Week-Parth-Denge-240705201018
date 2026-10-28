"""Weeks 15 & 16 (b-d): Perceptron, activation functions, loss functions, SGD / Adam optimisers and a
fully-connected neural network written from scratch in NumPy (forward pass, loss, backpropagation, updates).

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402

# ---------------------------------------------------------------- activations: name -> (f, df/dz)
ACTIVATIONS = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                lambda z: (lambda s: s * (1 - s))(1 / (1 + np.exp(-np.clip(z, -500, 500))))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}

# ---------------------------------------------------------------- losses: name -> (loss, dloss/dyhat)
EPS = 1e-12
LOSSES = {
    "mse": (lambda p, y: float(np.mean((p - y) ** 2)), lambda p, y: 2 * (p - y) / y.size),
    "bce": (lambda p, y: float(-np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))),
            lambda p, y: (p - y) / ((p + EPS) * (1 - p + EPS)) / y.size),
}


# ---------------------------------------------------------------- perceptron
class Perceptron:
    """Rosenblatt perceptron: y = step(w.x + b), trained with the perceptron learning rule."""

    def __init__(self, lr: float = 0.1, epochs: int = 50):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        self.errors = []
        for _ in range(self.epochs):
            wrong = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - self.predict(xi))
                self.w += update * xi
                self.b += update
                wrong += int(update != 0)
            self.errors.append(wrong)
            if wrong == 0:
                break
        return self

    def predict(self, X):
        return (np.asarray(X) @ self.w + self.b >= 0).astype(int)


# ---------------------------------------------------------------- optimisers (operate on lists of arrays)
class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.mom, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.mom
            v -= self.lr * g
            p += v


class Adam:
    """Adam: per-parameter step from bias-corrected first (m) and second (v) moment estimates."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, beta1, beta2, eps, 0, None, None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g ** 2
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


# ---------------------------------------------------------------- neural network
class NeuralNet:
    """Fully-connected network. sizes=[2, 8, 1] means 2 inputs, one hidden layer of 8, 1 output."""

    def __init__(self, sizes, hidden="tanh", output="sigmoid", loss="bce", seed=0):
        rng = np.random.default_rng(seed)
        scale = lambda fan_in: np.sqrt((2.0 if hidden == "relu" else 1.0) / fan_in)  # He / Xavier-style
        self.W = [rng.normal(0, scale(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]
        self.acts = [hidden] * (len(sizes) - 2) + [output]
        self.loss_name = loss

    @property
    def params(self):
        return self.W + self.b

    def forward(self, X):
        """Returns the prediction and a cache of pre-activations z and activations a for backprop."""
        a, zs, activations = X, [], [X]
        for W, b, name in zip(self.W, self.b, self.acts):
            z = a @ W + b
            a = ACTIVATIONS[name][0](z)
            zs.append(z); activations.append(a)
        return a, zs, activations

    def backward(self, X, y):
        """Backpropagation: chain rule applied layer by layer from the loss back to the inputs."""
        pred, zs, acts = self.forward(X)
        loss_fn, dloss = LOSSES[self.loss_name]
        delta = dloss(pred, y) * ACTIVATIONS[self.acts[-1]][1](zs[-1])
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(0, keepdims=True)
            if i:
                delta = (delta @ self.W[i].T) * ACTIVATIONS[self.acts[i - 1]][1](zs[i - 1])
        return loss_fn(pred, y), gW + gb

    def fit(self, X, y, optimizer, epochs=500, batch=None, seed=0):
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch or len(X)):
                b = idx[s:s + (batch or len(X))]
                _, grads = self.backward(X[b], y[b])
                optimizer.step(self.params, grads)
            history.append(LOSSES[self.loss_name][0](self.forward(X)[0], y))
        return history

    def predict(self, X):
        return (self.forward(X)[0] >= 0.5).astype(int)


def gradient_check(net, X, y, eps=1e-6) -> float:
    """Max relative error between analytic backprop gradients and central finite differences."""
    _, grads = net.backward(X, y)
    worst = 0.0
    for p, g in zip(net.params, grads):
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps; lp = LOSSES[net.loss_name][0](net.forward(X)[0], y)
            p[idx] = old - eps; lm = LOSSES[net.loss_name][0](net.forward(X)[0], y)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    return worst


GATES = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
GATE_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def boundary(ax, predict, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 200),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 200))
    zz = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title)


def main():
    banner("Weeks 15-16b-d - Neural network from scratch (NumPy)")
    print("Perceptron on logic gates (accuracy):")
    for g, t in GATES.items():
        print(f"  {g:3s}: {(Perceptron().fit(GATE_X, np.array(t)).predict(GATE_X) == t).mean():.2f}")

    Xg = GATE_X
    yg = np.array(GATES["XOR"], dtype=float).reshape(-1, 1)
    xor_net = NeuralNet([2, 4, 1], seed=1)
    xor_hist = xor_net.fit(Xg, yg, Adam(0.05), epochs=400)
    print(f"\nXOR with a 2-4-1 network: predictions {xor_net.predict(Xg).ravel().tolist()} (target 0,1,1,0)")
    print(f"Gradient check (max relative error): {gradient_check(NeuralNet([2, 4, 1], seed=1), Xg, yg):.2e}")

    X, y = make_moons(600, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y.reshape(-1, 1).astype(float), test_size=0.3, random_state=0)
    hist = {}
    nets = {}
    for name, opt in [("SGD lr=0.1", SGD(0.1)), ("SGD + momentum", SGD(0.1, 0.9)), ("Adam lr=0.01", Adam(0.01))]:
        nets[name] = NeuralNet([2, 16, 16, 1], hidden="relu", seed=0)
        hist[name] = nets[name].fit(Xtr, ytr, opt, epochs=300, batch=64)
        print(f"{name:15s} final loss {hist[name][-1]:.4f}  test accuracy {(nets[name].predict(Xte) == yte).mean():.4f}")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Weeks 15 & 16 - Neural network from scratch", fontweight="bold")
    z = np.linspace(-4, 4, 200)
    for name, (f, df) in ACTIVATIONS.items():
        if name != "linear":
            ax[0, 0].plot(z, f(z), label=name); ax[0, 1].plot(z, df(z), label=name)
    ax[0, 0].legend(); ax[0, 0].set_title("Activation functions")
    ax[0, 1].legend(); ax[0, 1].set_title("Their derivatives (vanishing gradient for sigmoid/tanh)")
    p = Perceptron().fit(Xg, np.array(GATES["XOR"]))
    boundary(ax[0, 2], p.predict, Xg, np.array(GATES["XOR"]), "Single perceptron on XOR: cannot separate")
    boundary(ax[1, 0], xor_net.predict, Xg, yg, "2-4-1 network on XOR: solved")
    for name, h in hist.items():
        ax[1, 1].plot(h, label=name)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("BCE loss")
    ax[1, 1].legend(); ax[1, 1].set_title("Optimisers: SGD vs momentum vs Adam")
    boundary(ax[1, 2], nets["Adam lr=0.01"].predict, Xte, yte, "From-scratch net on two moons (Adam, test set)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "week15_16_neural_net_scratch.png")


if __name__ == "__main__":
    main()
