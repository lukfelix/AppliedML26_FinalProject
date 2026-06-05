"""
plot_training.py
----------------
Plots training diagnostics from a completed (or in-progress) run.

Usage
-----
    python plot_training.py --run runs/exp1

Outputs (in --run directory)
-----------------------------
    plot_loss_curves.png     train vs val loss over epochs
    plot_lr_schedule.png     learning rate over epochs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


STYLE = dict(linewidth=1.8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="Run directory containing train_log.csv")
    return p.parse_args()


def plot_loss_curves(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["epoch"], df["train_loss"], label="Train", color="#2196F3", **STYLE)
    ax.plot(df["epoch"], df["val_loss"],   label="Val",   color="#F44336", **STYLE)

    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val   = df["val_loss"].min()
    ax.axvline(best_epoch, color="#F44336", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.annotate(f"best val\n{best_val:.4f}",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + max(len(df) * 0.03, 1), best_val),
                fontsize=8, color="#F44336",
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (masked χ²)")
    ax.set_title("Train / Validation Loss")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_lr_schedule(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(df["epoch"], df["lr"], color="#4CAF50", **STYLE)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    args = parse_args()
    log  = args.run / "train_log.csv"

    if not log.exists():
        print(f"No train_log.csv found in {args.run}")
        return

    df = pd.read_csv(log)
    print(f"Loaded {len(df)} epochs from {log}")

    plot_loss_curves(df, args.run / "plot_loss_curves.png")
    plot_lr_schedule(df, args.run / "plot_lr_schedule.png")


if __name__ == "__main__":
    main()
