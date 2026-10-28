"""Weeks 17-18 (c): RNN / LSTM / GRU sequence modelling - predict the next value of a noisy sine wave and
solve a long-range "remember the first element" task.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402

CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


def sine_windows(n_series=600, seq_len=30, seed=0):
    """Random-phase / random-frequency noisy sines; target = the next value after each window."""
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len + 1)
    freq = rng.uniform(0.15, 0.45, (n_series, 1))
    phase = rng.uniform(0, 2 * np.pi, (n_series, 1))
    s = np.sin(freq * t + phase) + rng.normal(0, 0.03, (n_series, seq_len + 1))
    X = torch.tensor(s[:, :-1, None], dtype=torch.float32)
    y = torch.tensor(s[:, -1:], dtype=torch.float32)
    return X, y


class SeqModel(nn.Module):
    """Recurrent layer (RNN / LSTM / GRU) whose last hidden state feeds a linear head."""

    def __init__(self, kind="lstm", hidden=32, out_dim=1):
        super().__init__()
        self.rnn = CELLS[kind](1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


def train_regressor(kind, epochs=60, seed=0):
    torch.manual_seed(seed)
    X, y = sine_windows(seed=seed)
    Xte, yte = sine_windows(n_series=200, seed=seed + 1)
    model = SeqModel(kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    curve = []
    for _ in range(epochs):
        for i in range(0, len(X), 100):
            loss = nn.functional.mse_loss(model(X[i:i + 100]), y[i:i + 100])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            curve.append(nn.functional.mse_loss(model(Xte), yte).item())
    return model, curve, (Xte, yte)


def first_token_task(n=1500, seq_len=15, seed=0):
    """Label = sign of the FIRST element; the other 14 are noise, so the model must remember it."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, seq_len, 1, generator=g)
    y = (X[:, 0, 0] > 0).long()
    return X, y


def memory_task_accuracy(kind, epochs=60, seed=0):
    torch.manual_seed(seed)
    X, y = first_token_task(seed=seed)
    Xte, yte = first_token_task(n=500, seed=seed + 1)
    model = SeqModel(kind, hidden=24, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for i in range(0, len(X), 100):
            loss = nn.functional.cross_entropy(model(X[i:i + 100]), y[i:i + 100])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()


def main():
    banner("Weeks 17-18c - RNN / LSTM / GRU sequence modelling")
    kinds = list(CELLS)
    curves, models = {}, {}
    for k in kinds:
        models[k], curves[k], data = train_regressor(k)
        print(f"{k.upper():5s} next-value test MSE: {curves[k][-1]:.5f}")
    mem = {k: memory_task_accuracy(k) for k in kinds}
    print("\nRemember-first-token accuracy (chance = 0.50):")
    for k, a in mem.items():
        print(f"  {k.upper():5s} {a:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for k in kinds:
        ax[0].plot(curves[k], label=k.upper())
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test MSE"); ax[0].legend()
    ax[0].set_title("Next-value prediction on noisy sines")
    Xte, yte = data
    with torch.no_grad():
        pred = models["lstm"](Xte).squeeze()
    ax[1].scatter(yte.squeeze(), pred, s=10, alpha=.6)
    ax[1].plot([-1, 1], [-1, 1], "r--"); ax[1].set_xlabel("true"); ax[1].set_ylabel("LSTM prediction")
    ax[1].set_title("LSTM: predicted vs true next value")
    ax[2].bar([k.upper() for k in kinds], [mem[k] for k in kinds], color=["#9ca3af", "#2f6fed", "#10b981"])
    ax[2].axhline(0.5, color="r", ls="--", label="chance"); ax[2].legend(); ax[2].set_ylim(0.4, 1.05)
    ax[2].set_title("Long-range memory (15 steps)")
    fig.suptitle("Weeks 17-18c - Sequence models", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week17_18c_sequence_models.png")


if __name__ == "__main__":
    main()
