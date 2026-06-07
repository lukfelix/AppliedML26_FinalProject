"""
plot_anomalies.py
-----------------
Plots anomaly score diagnostics after score_anomalies.py has been run.

Usage
-----
    python plot_anomalies.py --h5 lightcurves.h5 --run runs/exp1

Outputs (in --run directory)
-----------------------------
    plot_score_distribution.png     histogram of all anomaly scores
    plot_score_vs_nobs.png          score vs number of observations
    plot_top_anomalies.png          lightcurves + reconstructions for top-N
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import NormStats, LightcurveDataset, collate_fn
from model import mTANAutoencoder


FILTER_COLORS = {"g": "#2196F3", "r": "#F44336"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          type=Path, required=True)
    p.add_argument("--run",         type=Path, required=True)
    p.add_argument("--top_n",       type=int,  default=12,
                   help="Number of top anomalies to plot lightcurves for")
    p.add_argument("--checkpoint",  type=str,  default="best_model.pt")
    p.add_argument("--num_workers", type=int,  default=4)
    return p.parse_args()


def load_model(run: Path, checkpoint: str, device: torch.device) -> tuple:
    try:
        from pathlib import PurePosixPath, PureWindowsPath
        import torch.serialization as _ts
        with _ts.safe_globals([PurePosixPath, PureWindowsPath]):
            ckpt = torch.load(run / checkpoint, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(run / checkpoint, map_location=device, weights_only=False)
    run_args = argparse.Namespace(**ckpt["args"])
    model    = mTANAutoencoder(
        d_model  = run_args.d_model,
        d_time   = run_args.d_time,
        d_latent = run_args.d_latent,
        n_ref    = run_args.n_ref,
        n_heads  = run_args.n_heads,
        n_layers = run_args.n_layers,
        dropout  = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, run_args


# ── Score distribution ────────────────────────────────────────────────────────

def plot_score_distribution(scores: np.ndarray, scores_norm: np.ndarray, out: Path) -> None:
    """
    Two panels: raw score distribution (log x) and length-normalised percentile score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: raw scores, log x-axis
    ax = axes[0]
    pos = scores[scores > 0]
    ax.hist(np.log10(pos), bins=80, color="#607D8B", alpha=0.8, log=True)
    p95 = np.percentile(np.log10(pos), 95)
    p99 = np.percentile(np.log10(pos), 99)
    ax.axvline(p95, color="#FF9800", linewidth=1.5, linestyle="--",
               label=f"95th pct (10^{p95:.2f})")
    ax.axvline(p99, color="#F44336", linewidth=1.5, linestyle="--",
               label=f"99th pct (10^{p99:.2f})")
    ax.set_xlabel("log₁₀(anomaly score)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Raw Score Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")

    # Right: normalised percentile score
    ax = axes[1]
    ax.hist(scores_norm, bins=50, color="#5C6BC0", alpha=0.8, log=True)
    ax.axvline(95, color="#FF9800", linewidth=1.5, linestyle="--", label="95th pct")
    ax.axvline(99, color="#F44336", linewidth=1.5, linestyle="--", label="99th pct")
    ax.set_xlabel("Length-normalised score (percentile within length bin)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Normalised Score Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")

    fig.suptitle("Anomaly Score Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Score vs number of observations ──────────────────────────────────────────

def plot_score_vs_nobs(
    scores: np.ndarray,
    scores_norm: np.ndarray,
    n_obs: np.ndarray,
    out: Path,
) -> None:
    """
    Two panels: raw score vs n_obs (both log), and normalised score vs n_obs.
    After normalisation the top anomalies should be distributed across all
    sequence lengths, not concentrated at short sequences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    is_top_raw  = scores      >= np.percentile(scores,      99)
    is_top_norm = scores_norm >= np.percentile(scores_norm, 99)

    for ax, sc, is_top, label, title in [
        (axes[0], scores,      is_top_raw,  "log₁₀(raw score)",        "Raw Score vs Length"),
        (axes[1], scores_norm, is_top_norm, "Normalised score (pctile)", "Normalised Score vs Length"),
    ]:
        yvals = np.log10(np.maximum(sc, 1e-6)) if label.startswith("log") else sc
        ax.scatter(n_obs[~is_top], yvals[~is_top],
                   s=3, alpha=0.25, color="#607D8B", linewidths=0,
                   label="Normal (< 99th pct)")
        ax.scatter(n_obs[is_top], yvals[is_top],
                   s=14, alpha=0.85, color="#F44336", linewidths=0,
                   label="Top 1% anomalies")
        ax.set_xscale("log")
        ax.set_xlabel("Number of observations (log)")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, which="both")

    fig.suptitle("Anomaly Score vs Sequence Length", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Top-N anomaly lightcurves with reconstruction ─────────────────────────────

def plot_top_anomalies(
    model:    mTANAutoencoder,
    ds:       LightcurveDataset,
    scores_df: pd.DataFrame,
    device:   torch.device,
    top_n:    int,
    out:      Path,
) -> None:
    """
    For the top-N scoring objects, plot observed and reconstructed magpsf
    for both filters. Each panel title shows rank and score so you can
    immediately judge whether high-scoring objects look genuinely unusual.
    """
    top_rows = scores_df.head(top_n)
    id_to_rank  = {row["object_id"]: (row["rank"], row["score"])
                   for _, row in top_rows.iterrows()}
    top_ids     = list(id_to_rank.keys())

    # Build a small dataset with only these objects
    sub_ds = LightcurveDataset.__new__(LightcurveDataset)
    sub_ds._h5_path = ds._h5_path
    sub_ds._norm    = ds._norm
    sub_ds._file    = None
    sub_ds._ids     = [oid for oid in top_ids if oid in set(ds.object_ids())]

    n_cols = 3
    n_rows = (len(sub_ds._ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for plot_idx in range(len(sub_ds._ids)):
        ax   = axes[plot_idx // n_cols][plot_idx % n_cols]
        item = sub_ds[plot_idx]
        oid  = item["object_id"]
        rank, score = id_to_rank[oid]

        x    = item["x"].unsqueeze(0).to(device)
        mask = torch.ones(1, item["T"], dtype=torch.bool, device=device)

        with torch.no_grad():
            recon, _ = model(x, mask)

        t        = x[0, :, 0].cpu().numpy()
        mag_true = item["target"][:, 0].numpy()
        sig_true = item["target"][:, 1].numpy()
        mag_pred = recon[0, :, 0].cpu().numpy()
        is_g     = x[0, :, 3].cpu().numpy() > 0.5
        is_r     = x[0, :, 4].cpu().numpy() > 0.5

        for mask_arr, band in [(is_g, "g"), (is_r, "r")]:
            if not mask_arr.any():
                continue
            t_b   = t[mask_arr]
            m_b   = mag_true[mask_arr]
            s_b   = sig_true[mask_arr]
            mp_b  = mag_pred[mask_arr]
            color = FILTER_COLORS[band]
            order = np.argsort(t_b)

            ax.errorbar(t_b, m_b, yerr=np.abs(s_b),
                        fmt="o", color=color, alpha=0.5,
                        markersize=2, linewidth=0.5,
                        label=f"{band} obs")
            ax.plot(t_b[order], mp_b[order],
                    color=color, linewidth=1.4,
                    label=f"{band} recon")

        ax.invert_yaxis()
        ax.set_xlabel("Normalised time", fontsize=8)
        ax.set_ylabel("magpsf (norm.)", fontsize=8)
        ax.set_title(f"#{rank}  {oid}\nscore={score:.4f}", fontsize=8)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    for i in range(len(sub_ds._ids), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle(f"Top {top_n} most anomalous objects", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores_path = args.run / "anomaly_scores.csv"
    if not scores_path.exists():
        print(f"No anomaly_scores.csv found in {args.run}. "
              f"Run score_anomalies.py first.")
        return

    scores_df   = pd.read_csv(scores_path)
    scores      = scores_df["score"].to_numpy()
    n_obs       = scores_df["n_obs"].to_numpy()
    # score_norm column may be absent in older runs; fall back to raw score
    scores_norm = (scores_df["score_norm"].to_numpy()
                   if "score_norm" in scores_df.columns
                   else scores.copy())

    print(f"Loaded scores for {len(scores_df)} objects.")
    print(f"  Raw score range:  [{scores.min():.4f}, {scores.max():.4f}]  "
          f"median={np.median(scores):.4f}")
    print(f"  Norm score range: [{scores_norm.min():.1f}, {scores_norm.max():.1f}]")

    plot_score_distribution(scores, scores_norm,
                            out=args.run / "plot_score_distribution.png")
    plot_score_vs_nobs(scores, scores_norm, n_obs,
                       out=args.run / "plot_score_vs_nobs.png")

    # Top-N lightcurves need the model and dataset
    norm        = NormStats.load(args.run / "norm_stats.json")
    model, run_args = load_model(args.run, args.checkpoint, device)
    ds = LightcurveDataset(args.h5, norm,
                           min_observations=run_args.min_obs)

    plot_top_anomalies(model, ds, scores_df, device,
                       top_n = args.top_n,
                       out   = args.run / "plot_top_anomalies.png")


if __name__ == "__main__":
    main()
