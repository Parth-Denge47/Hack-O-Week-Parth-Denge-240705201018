"""Week 15 (b, d): CNN image classification (convolution + pooling) and regularisation
(dropout, batch norm, early stopping) on the 8x8 scikit-learn Digits images.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def get_digit_tensors(seed: int = 0, n_train: int = 300):
    """Small training set on purpose, so overfitting is visible and regularisation matters."""
    d = load_digits()
    X = torch.tensor(d.images / 16.0, dtype=torch.float32).unsqueeze(1)  # (N,1,8,8)
    y = torch.tensor(d.target)
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, train_size=n_train, random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xrest, yrest, test_size=0.6, random_state=seed, stratify=yrest)
    return (Xtr, ytr), (Xval, yval), (Xte, yte)


class CNN(nn.Module):
    """conv -> (BN) -> ReLU -> maxpool, twice, then a (dropout) classifier head."""

    def __init__(self, batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()

        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(cout))
            return layers + [nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(1, 16), *block(16, 32))  # 8x8 -> 4x4 -> 2x2
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32 * 2 * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    out = model(X)
    return nn.functional.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()


def fit(model, train, val, epochs=60, lr=1e-3, patience=None, seed=0, batch=32):
    """Mini-batch training loop with optional early stopping (restores the best-validation weights)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, ytr = train
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best, best_state, bad, stopped_at = float("inf"), None, 0, epochs
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tl, _ = evaluate(model, Xtr, ytr)
        vl, va = evaluate(model, *val)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                stopped_at = epoch + 1
                break
    if patience and best_state is not None:
        model.load_state_dict(best_state)
    hist["stopped_at"] = stopped_at
    return hist


def run_experiments(epochs: int = 60):
    train, val, test = get_digit_tensors()
    configs = {
        "baseline": dict(model=dict(), fit=dict()),
        "dropout 0.4": dict(model=dict(dropout=0.4), fit=dict()),
        "batch norm": dict(model=dict(batch_norm=True), fit=dict()),
        "dropout + BN + early stop": dict(model=dict(dropout=0.4, batch_norm=True), fit=dict(patience=8)),
    }
    results = {}
    for name, c in configs.items():
        torch.manual_seed(0)
        m = CNN(**c["model"])
        h = fit(m, train, val, epochs=epochs, **c["fit"])
        results[name] = {"hist": h, "test_acc": evaluate(m, *test)[1], "params": sum(p.numel() for p in m.parameters())}
    return results


def main():
    banner("Week 15b/d - CNN image classification + regularisation (Digits)")
    res = run_experiments()
    print(f"{'config':28s} {'test acc':>9s} {'final val loss':>15s} {'epochs run':>11s}")
    for n, r in res.items():
        print(f"{n:28s} {r['test_acc']:9.4f} {r['hist']['val_loss'][-1]:15.4f} {len(r['hist']['val_loss']):11d}")
    print("CNN parameters:", res["baseline"]["params"])

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for n, r in res.items():
        ax[0].plot(r["hist"]["val_loss"], label=n)
        ax[1].plot(r["hist"]["val_acc"], label=n)
    ax[0].set_title("Validation loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
    ax[2].bar(range(len(res)), [r["test_acc"] for r in res.values()], color="#2f6fed")
    ax[2].set_xticks(range(len(res))); ax[2].set_xticklabels(list(res), rotation=20, ha="right")
    ax[2].set_ylim(0.8, 1.0); ax[2].set_title("Held-out test accuracy")
    fig.suptitle("Week 15b/d - CNN with dropout, batch norm and early stopping", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "week15b_d_cnn_regularization.png")


if __name__ == "__main__":
    main()
