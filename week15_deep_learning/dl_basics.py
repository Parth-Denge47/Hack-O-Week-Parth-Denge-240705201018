"""Week 15 (a): PyTorch basics - tensors, autograd and a hand-written training loop.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402


def tensor_tour() -> dict:
    """Create tensors several ways and do basic maths / broadcasting."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.ones(2, 2)
    return {
        "a": a,
        "zeros_like": torch.zeros_like(a),
        "arange": torch.arange(6).reshape(2, 3),
        "matmul": a @ b,
        "broadcast": a + torch.tensor([10.0, 20.0]),  # (2,2) + (2,) -> (2,2)
        "mean_dim0": a.mean(dim=0),
        "shape": tuple(a.shape),
    }


def autograd_demo() -> dict:
    """y = x^3 + 2x  ->  dy/dx = 3x^2 + 2, checked against autograd."""
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 3 + 2 * x
    y.backward()
    # second function with two inputs: z = w.c + b
    w = torch.tensor([1.0, -2.0], requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)
    z = (w * torch.tensor([3.0, 4.0])).sum() + b
    z.backward()
    return {"dy_dx": x.grad.item(), "analytic": 3 * 3.0 ** 2 + 2, "dz_dw": w.grad.tolist(), "dz_db": b.grad.item()}


def make_regression_data(n: int = 200, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 4 - 2
    y = 3 * x + 1 + 0.2 * torch.randn(n, 1, generator=g)
    return x, y


def train_linear(epochs: int = 200, lr: float = 0.05, seed: int = 0):
    """The canonical loop: forward -> loss -> zero_grad -> backward -> step."""
    torch.manual_seed(seed)
    x, y = make_regression_data(seed=seed)
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses, (x, y)


def main():
    banner("Week 15a - PyTorch: tensors, autograd, training loop")
    for k, v in tensor_tour().items():
        print(f"{k:11s}: {v.tolist() if isinstance(v, torch.Tensor) else v}")
    print("\nautograd:", autograd_demo())
    model, losses, (x, y) = train_linear()
    w, b = model.weight.item(), model.bias.item()
    print(f"\nLearned y = {w:.3f}x + {b:.3f}  (true 3x + 1), final MSE {losses[-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(losses); ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE")
    ax[0].set_title("Training loss")
    ax[1].scatter(x, y, s=8, alpha=.5)
    xs = torch.linspace(-2, 2, 2).reshape(-1, 1)
    ax[1].plot(xs, model(xs).detach(), "r", label=f"fit: {w:.2f}x+{b:.2f}")
    ax[1].legend(); ax[1].set_title("Linear regression learned with autograd")
    fig.suptitle("Week 15a - PyTorch training loop", fontweight="bold")
    save_fig(fig, "week15a_pytorch_basics.png")


if __name__ == "__main__":
    main()
