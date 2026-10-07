"""Shared helpers for the Hack-O-Week projects.

Author: Parth Denge | PRN: 240705201018
"""
import os

import matplotlib

matplotlib.use("Agg")  # headless: always save figures to disk
import matplotlib.pyplot as plt

AUTHOR = "Parth Denge"
PRN = "240705201018"
SIGNATURE = f"{AUTHOR} | PRN {PRN}"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output_screenshots")


def banner(title: str) -> None:
    line = "=" * 64
    print(f"{line}\n{title}\n{SIGNATURE}\n{line}")


def save_fig(fig, name: str) -> str:
    """Stamp the figure with the author signature and save it as a PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.text(0.995, 0.005, SIGNATURE, ha="right", va="bottom", fontsize=8, color="gray")
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] output_screenshots/{name}")
    return path
