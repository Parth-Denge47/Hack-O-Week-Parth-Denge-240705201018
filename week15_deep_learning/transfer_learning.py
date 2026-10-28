"""Week 15 (e): Transfer learning with a pretrained CNN (ResNet-18, ImageNet weights).

The ImageNet backbone is frozen and used as a feature extractor; only a new linear head is trained on the
Digits images (upscaled to 64x64 RGB). It is compared with the same architecture trained from scratch on the
same small training set.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, save_fig  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_data(seed=0, n_train=200, n_test=500, size=64):
    d = load_digits()
    X = torch.tensor(d.images / 16.0, dtype=torch.float32).unsqueeze(1)
    X = nn.functional.interpolate(X, size=size, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    X = (X - MEAN) / STD  # same normalisation the pretrained network was trained with
    y = torch.tensor(d.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y)
    return (Xtr, ytr), (Xte, yte)


def build_resnet(pretrained: bool, n_classes=10):
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_classes)  # new classification head
    return net


@torch.no_grad()
def accuracy(model, X, y, batch=100):
    model.eval()
    return sum((model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item() for i in range(0, len(X), batch)) / len(X)


def train(model, train_data, test_data, epochs=6, lr=1e-3, freeze_backbone=False, seed=0, batch=32):
    torch.manual_seed(seed)
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    X, y = train_data
    accs = []
    for _ in range(epochs):
        model.train()
        if freeze_backbone:
            model.eval()  # keep frozen BatchNorm statistics fixed too
            model.fc.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        accs.append(accuracy(model, *test_data))
    return accs


def run(epochs=6):
    tr, te = get_data()
    out = {}
    for name, pretrained, frozen in [("pretrained, frozen backbone", True, True),
                                     ("pretrained, fine-tuned all", True, False),
                                     ("from scratch", False, False)]:
        torch.manual_seed(0)
        model = build_resnet(pretrained)
        trainable = sum(p.numel() for n, p in model.named_parameters() if (not frozen) or n.startswith("fc."))
        out[name] = {"acc": train(model, tr, te, epochs=epochs, freeze_backbone=frozen), "trainable": trainable}
    return out


def main():
    banner("Week 15e - Transfer learning with pretrained ResNet-18")
    res = run()
    print(f"{'setup':30s} {'trainable params':>17s} {'final test acc':>15s}")
    for n, r in res.items():
        print(f"{n:30s} {r['trainable']:17,d} {r['acc'][-1]:15.4f}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n, r in res.items():
        ax.plot(range(1, len(r["acc"]) + 1), r["acc"], "o-", label=n)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.legend()
    ax.set_title("Week 15e - Transfer learning vs training from scratch (200 training images)", fontsize=10)
    save_fig(fig, "week15e_transfer_learning.png")


if __name__ == "__main__":
    main()
