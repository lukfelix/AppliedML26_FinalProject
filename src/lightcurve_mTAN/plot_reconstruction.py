"""
plot_reconstruction.py
----------------------
Plots reconstruction quality diagnostics after training.

Usage
-----
    python plot_reconstruction.py --h5 lightcurves.h5 --run runs/exp1

Outputs (in --run directory)
-----------------------------
    plot_reconstruction_examples.png   input vs reconstructed lightcurves
    plot_residuals.png                 residual distribution per filter
    plot_error_vs_length.png           reconstruction error vs sequence length
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import NormStats, LightcurveDataset, collate_fn
from model import mTANAutoencoder, masked_chi2_loss


FILTER_COLORS = {"g": "#2196F3", "r": "#F44336"}
BAND_ALPHA    = 0.6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          type=Path, required=True)
    p.add_argument("--run",         type=Path, required=True)
    p.add_argument("--n_examples",  type=int,  default=6,
                   help="Number of example lightcurves to plot")
    p.add_argument("--n_score",     type=int,  default=2000,
                   help="Number of objects to use for residual / error-vs-length plots")
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--checkpoint",  type=str,  default="best_model.pt")
    p.add_argument("--seed",        type=int,  default=0)
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model, run_args


# ── Example lightcurve reconstructions ───────────────────────────────────────

def plot_examples(
    model:      mTANAutoencoder,
    ds:         LightcurveDataset,
    norm:       NormStats,
    device:     torch.device,
    n:          int,
    out:        Path,
    rng:        np.random.Generator,
) -> None:
    """
    For n randomly chosen objects, plot the observed and reconstructed
    magpsf for each filter (g, r) against normalised time.
    Observations are shown as error bars; reconstruction as a line.
    """
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    n_cols  = 2
    n_rows  = (len(indices) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for plot_idx, ds_idx in enumerate(indices):
        ax  = axes[plot_idx // n_cols][plot_idx % n_cols]
        item = ds[int(ds_idx)]
        oid  = item["object_id"]

        x      = item["x"].unsqueeze(0).to(device)         # (1, T, 5)
        mask   = torch.ones(1, item["T"], dtype=torch.bool, device=device)
        target = item["target"].unsqueeze(0).to(device)

        with torch.no_grad():
            recon, _ = model(x, mask)

        t      = x[0, :, 0].cpu().numpy()                  # normalised time [0,1]
        is_g   = x[0, :, 3].cpu().numpy() > 0.5
        is_r   = x[0, :, 4].cpu().numpy() > 0.5

        mag_true  = target[0, :, 0].cpu().numpy()
        sig_true  = item["target"][:, 1].numpy()           # normalised sigmapsf
        mag_pred  = recon[0, :, 0].cpu().numpy()

        # Denote filter masks
        for mask_arr, band in [(is_g, "g"), (is_r, "r")]:
            if not mask_arr.any():
                continue
            t_b    = t[mask_arr]
            m_b    = mag_true[mask_arr]
            s_b    = sig_true[mask_arr]
            mp_b   = mag_pred[mask_arr]
            color  = FILTER_COLORS[band]

            ax.errorbar(t_b, m_b, yerr=np.abs(s_b),
                        fmt="o", color=color, alpha=BAND_ALPHA,
                        markersize=2, linewidth=0.5,
                        label=f"{band} observed")
            # Sort by time for clean line
            order  = np.argsort(t_b)
            ax.plot(t_b[order], mp_b[order],
                    color=color, linewidth=1.4, alpha=0.9,
                    label=f"{band} reconstructed")

        ax.invert_yaxis()   # magnitudes: brighter = lower number = up
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("magpsf (normalised)")
        ax.set_title(oid, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

    # Hide any unused axes
    for i in range(len(indices), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Reconstruction examples (random sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Residual distribution ─────────────────────────────────────────────────────

def plot_residuals(
    all_residuals_g: np.ndarray,
    all_residuals_r: np.ndarray,
    out: Path,
) -> None:
    """
    Histogram of (mag_pred - mag_true) / sigmapsf for each filter.
    A well-trained model should give a zero-centred, roughly Gaussian distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, resid, band in zip(axes,
                               [all_residuals_g, all_residuals_r],
                               ["g", "r"]):
        color = FILTER_COLORS[band]
        ax.hist(resid, bins=100, range=(-5, 5),
                color=color, alpha=0.7, density=True)

        # Overlay standard normal for reference
        xs = np.linspace(-5, 5, 300)
        ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
                "k--", linewidth=1.2, label="N(0,1)")

        ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("(pred − true) / σ")
        ax.set_title(f"Filter {band}  —  "
                     f"μ={resid.mean():.3f}  σ={resid.std():.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Density")
    fig.suptitle("Normalised residuals", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Error vs sequence length ──────────────────────────────────────────────────

def plot_error_vs_length(
    lengths: np.ndarray,
    scores:  np.ndarray,
    out:     Path,
) -> None:
    """
    Scatter plot of reconstruction error vs sequence length.
    Systematic trends (e.g. model doing poorly on very short/long sequences)
    indicate architectural or training issues.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(lengths, scores, s=4, alpha=0.4, color="#607D8B", linewidths=0)

    # Running median
    order    = np.argsort(lengths)
    l_sorted = lengths[order]
    s_sorted = scores[order]
    window   = max(len(lengths) // 30, 5)
    medians  = [np.median(s_sorted[max(0, i - window):i + window])
                for i in range(len(s_sorted))]
    ax.plot(l_sorted, medians, color="#F44336", linewidth=1.8,
            label="Running median")

    ax.set_xlabel("Sequence length (observations)")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title("Reconstruction error vs sequence length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    norm        = NormStats.load(args.run / "norm_stats.json")
    model, run_args = load_model(args.run, args.checkpoint, device)

    ds = LightcurveDataset(args.h5, norm,
                           min_observations=run_args.min_obs)

    # ── Example reconstructions ────────────────────────────────────────────
    plot_examples(model, ds, norm, device,
                  n   = args.n_examples,
                  out = args.run / "plot_reconstruction_examples.png",
                  rng = rng)

    # ── Residuals + error-vs-length (batch pass over subset) ──────────────
    n_score  = min(args.n_score, len(ds))
    sub_ids  = [ds.object_ids()[i]
                for i in rng.choice(len(ds), n_score, replace=False).tolist()]
    sub_ds   = LightcurveDataset(args.h5, norm,
                                 object_ids=sub_ids,
                                 min_observations=run_args.min_obs)
    dl = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=args.num_workers,
                    pin_memory=True)

    all_res_g, all_res_r = [], []
    all_lengths, all_scores = [], []

    with torch.no_grad():
        for batch in dl:
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            mask   = batch["mask"].to(device)

            recon, _   = model(x, mask)
            _, per_obj = masked_chi2_loss(recon, target, mask)

            mag_pred = recon[..., 0]                     # (B, T)
            mag_true = target[..., 0]
            sig_true = target[..., 1].abs() + 1e-4

            residuals = ((mag_pred - mag_true) / sig_true) * mask

            is_g = x[..., 3] > 0.5                       # (B, T)
            is_r = x[..., 4] > 0.5

            for b in range(x.shape[0]):
                m = mask[b]
                all_res_g.append(residuals[b][m & is_g[b]].cpu().numpy())
                all_res_r.append(residuals[b][m & is_r[b]].cpu().numpy())

            all_lengths.extend(batch["lengths"])
            all_scores.extend(per_obj.cpu().numpy().tolist())

    res_g = np.concatenate(all_res_g)
    res_r = np.concatenate(all_res_r)

    plot_residuals(res_g, res_r,
                   out=args.run / "plot_residuals.png")
    plot_error_vs_length(np.array(all_lengths), np.array(all_scores),
                         out=args.run / "plot_error_vs_length.png")


if __name__ == "__main__":
    main()
