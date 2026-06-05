"""
score_anomalies.py
------------------
Runs the trained autoencoder over the full dataset and produces a CSV
ranking every object by reconstruction loss (highest = most anomalous).

Usage
-----
    python score_anomalies.py --h5 lightcurves.h5 --run runs/exp1

Outputs
-------
    runs/exp1/anomaly_scores.csv   — columns: object_id, score, n_obs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import NormStats, LightcurveDataset, collate_fn
from model import mTANAutoencoder, masked_chi2_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          type=Path, required=True)
    p.add_argument("--run",         type=Path, required=True,
                   help="Run directory containing best_model.pt and norm_stats.json")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint",  type=str, default="best_model.pt",
                   help="Which checkpoint to use (default: best_model.pt)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.run / args.checkpoint
    # Allow Path objects stored in older checkpoints (PyTorch >= 2.6 safety)
    try:
        from pathlib import PurePosixPath, PureWindowsPath
        import torch.serialization as _ts
        with _ts.safe_globals([PurePosixPath, PureWindowsPath]):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    run_args  = argparse.Namespace(**ckpt["args"])
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
        dropout  = 0.0,   # disable dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Score every object ─────────────────────────────────────────────────
    ds = LightcurveDataset(args.h5, norm, min_observations=run_args.min_obs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=args.num_workers,
                    pin_memory=True)

    results = []   # list of (object_id, score, n_obs)

    with torch.no_grad():
        for batch in tqdm(dl, desc="Scoring"):
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            mask   = batch["mask"].to(device)

            recon, _ = model(x, mask)
            _, per_obj = masked_chi2_loss(recon, target, mask)

            for oid, score, length in zip(
                batch["object_ids"],
                per_obj.cpu().numpy(),
                batch["lengths"],
            ):
                results.append((oid, float(score), int(length)))

    # ── Sort and save ──────────────────────────────────────────────────────
    results.sort(key=lambda r: r[1], reverse=True)   # highest score first

    out_path = args.run / "anomaly_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "object_id", "score", "n_obs"])
        for rank, (oid, score, n_obs) in enumerate(results, start=1):
            writer.writerow([rank, oid, f"{score:.6f}", n_obs])

    print(f"\nScored {len(results)} objects.")
    print(f"Results saved to {out_path}")
    print(f"\nTop 10 most anomalous objects:")
    print(f"{'Rank':>4}  {'Object ID':<20}  {'Score':>10}  {'N_obs':>6}")
    print("-" * 46)
    for rank, (oid, score, n_obs) in enumerate(results[:10], start=1):
        print(f"{rank:4d}  {oid:<20}  {score:10.4f}  {n_obs:6d}")


if __name__ == "__main__":
    main()
