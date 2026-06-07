"""
score_anomalies.py
------------------
Scores every object by reconstruction loss, then produces a length-normalised
anomaly rank to avoid the sequence-length bias (sparse objects scoring
artificially high due to fewer observations to average over).

Raw scores and normalised scores are both saved so you can compare them.

Usage
-----
    python score_anomalies.py --h5 lightcurves.h5 --run runs/exp1

Outputs
-------
    runs/exp1/anomaly_scores.csv
        rank, object_id, score, score_norm, n_obs
        score      = raw mean reconstruction loss
        score_norm = percentile within a rolling length window (0-100),
                     so objects are compared only against similar-length peers
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import NormStats, LightcurveDataset, collate_fn
from model import mTANAutoencoder, masked_chi2_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          type=Path, required=True)
    p.add_argument("--run",         type=Path, required=True)
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--checkpoint",  type=str,  default="best_model.pt")
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Loss mix used during training (must match train.py --alpha)")
    p.add_argument("--n_bins",      type=int,  default=20,
                   help="Number of length bins for percentile normalisation")
    return p.parse_args()


def length_normalised_score(
    lengths: np.ndarray,
    scores:  np.ndarray,
    n_bins:  int = 20,
) -> np.ndarray:
    """
    For each object, compute its percentile rank among objects with similar
    sequence lengths. This removes the systematic trend where shorter sequences
    score higher simply due to noisier per-object loss estimates.

    Uses log-spaced bins so that the dense short-sequence region is resolved.
    Returns a score in [0, 100] where 100 = most anomalous within its length peer group.
    """
    log_len  = np.log10(np.maximum(lengths, 1))
    edges    = np.linspace(log_len.min(), log_len.max() + 1e-6, n_bins + 1)
    bin_idx  = np.digitize(log_len, edges) - 1
    bin_idx  = np.clip(bin_idx, 0, n_bins - 1)

    norm_scores = np.zeros(len(scores), dtype=np.float32)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_scores = scores[mask]
        # percentile rank within this bin
        ranks = np.argsort(np.argsort(bin_scores)).astype(np.float32)
        norm_scores[mask] = 100.0 * ranks / max(mask.sum() - 1, 1)

    return norm_scores


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.run / args.checkpoint
    try:
        from pathlib import PurePosixPath, PureWindowsPath
        import torch.serialization as _ts
        with _ts.safe_globals([PurePosixPath, PureWindowsPath]):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    run_args = argparse.Namespace(**ckpt["args"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    norm = NormStats.load(args.run / "norm_stats.json")

    model = mTANAutoencoder(
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

    # ── Score every object ─────────────────────────────────────────────────
    min_obs = getattr(run_args, "min_obs", 10)
    ds = LightcurveDataset(args.h5, norm, min_observations=min_obs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=args.num_workers,
                    pin_memory=True)

    object_ids, raw_scores, lengths = [], [], []

    beta = getattr(run_args, "beta", 0.1)
    with torch.no_grad():
        for batch in tqdm(dl, desc="Scoring"):
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            mask   = batch["mask"].to(device)

            recon, _   = model(x, mask)
            _, per_obj = masked_chi2_loss(recon, target, mask, alpha=args.alpha, beta=beta)

            object_ids.extend(batch["object_ids"])
            raw_scores.extend(per_obj.cpu().numpy().tolist())
            lengths.extend(batch["lengths"])

    raw_scores = np.array(raw_scores, dtype=np.float64)
    lengths    = np.array(lengths,    dtype=np.int32)

    # ── Length-normalised percentile score ─────────────────────────────────
    norm_scores = length_normalised_score(lengths, raw_scores, n_bins=args.n_bins)

    # Sort by normalised score (primary) then raw score (tiebreak)
    order = np.lexsort((raw_scores, -norm_scores))

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = args.run / "anomaly_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "object_id", "score", "score_norm", "n_obs"])
        for rank, i in enumerate(order, start=1):
            writer.writerow([rank, object_ids[i],
                             f"{raw_scores[i]:.6f}",
                             f"{norm_scores[i]:.2f}",
                             int(lengths[i])])

    print(f"\nScored {len(object_ids)} objects.")
    print(f"Results saved to {out_path}")
    print(f"\nTop 12 most anomalous objects (length-normalised):")
    print(f"{'Rank':>4}  {'Object ID':<20}  {'Score':>12}  {'Norm':>6}  {'N_obs':>6}")
    print("-" * 56)
    for rank, i in enumerate(order[:12], start=1):
        print(f"{rank:4d}  {object_ids[i]:<20}  "
              f"{raw_scores[i]:12.2f}  {norm_scores[i]:6.1f}  {lengths[i]:6d}")


if __name__ == "__main__":
    main()
