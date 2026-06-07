"""
train.py
--------
Full training loop for the mTAN autoencoder.

Usage
-----
    python train.py --h5 lightcurves.h5 --out runs/exp1

Key outputs in --out directory:
    best_model.pt       model weights at lowest validation loss
    last_model.pt       model weights at end of training
    norm_stats.json     normalisation statistics (needed for inference)
    train_log.csv       loss per epoch
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import NormStats, make_dataloaders
from model import mTANAutoencoder, masked_chi2_loss


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--h5",              type=Path, required=True,
                   help="Path to lightcurves.h5")
    p.add_argument("--out",             type=Path, default=Path("runs/exp1"),
                   help="Output directory for checkpoints and logs")
    p.add_argument("--val_fraction",    type=float, default=0.1)
    p.add_argument("--min_obs",         type=int,   default=10,
                   help="Drop objects with fewer total observations than this")

    # Model
    p.add_argument("--d_model",   type=int,   default=64)
    p.add_argument("--d_time",    type=int,   default=16)
    p.add_argument("--d_latent",  type=int,   default=64)
    p.add_argument("--n_ref",     type=int,   default=128,
                   help="Number of reference time points in mTAN")
    p.add_argument("--n_heads",   type=int,   default=4)
    p.add_argument("--n_layers",  type=int,   default=2)
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--alpha",     type=float, default=0.5,
                   help="Loss mix: 0=pure chi2, 1=pure MSE (default 0.5)")
    p.add_argument("--beta",      type=float, default=0.1,
                   help="Weight of derivative loss term (default 0.1)")

    # Training
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=15,
                   help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--num_workers",   type=int,   default=4,
                   help="DataLoader worker processes. Set 0 to debug on Windows.")
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


# ── Training / validation steps ───────────────────────────────────────────────

def run_epoch(
    model:     mTANAutoencoder,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    train:     bool,
    alpha:     float = 0.5,
    beta:      float = 0.1,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x      = batch["x"].to(device)
            target = batch["target"].to(device)
            mask   = batch["mask"].to(device)

            recon, _ = model(x, mask)
            loss, _  = masked_chi2_loss(recon, target, mask, alpha=alpha, beta=beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Normalisation stats (compute once, cache to disk) ──────────────────
    norm_path = args.out / "norm_stats.json"
    if norm_path.exists():
        print(f"Loading cached norm stats from {norm_path}")
        norm = NormStats.load(norm_path)
    else:
        norm = NormStats.compute(args.h5)
        norm.save(norm_path)
        print(f"Saved norm stats to {norm_path}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_dl, val_dl, train_ids, val_ids = make_dataloaders(
        h5_path        = args.h5,
        norm_stats     = norm,
        val_fraction   = args.val_fraction,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        min_observations = args.min_obs,
        seed           = args.seed,
    )
    print(f"Train: {len(train_ids)} objects  |  Val: {len(val_ids)} objects")

    # ── Model ──────────────────────────────────────────────────────────────
    model = mTANAutoencoder(
        d_model  = args.d_model,
        d_time   = args.d_time,
        d_latent = args.d_latent,
        n_ref    = args.n_ref,
        n_heads  = args.n_heads,
        n_layers = args.n_layers,
        dropout  = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ── Training loop ──────────────────────────────────────────────────────
    log_path  = args.out / "train_log.csv"
    best_path = args.out / "best_model.pt"
    last_path = args.out / "last_model.pt"

    best_val   = float("inf")
    no_improve = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss = run_epoch(model, train_dl, optimizer, device, train=True,  alpha=args.alpha, beta=args.beta)
            val_loss   = run_epoch(model, val_dl,   optimizer, device, train=False, alpha=args.alpha, beta=args.beta)
            scheduler.step(val_loss)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.1f}s)")

            writer.writerow([epoch, train_loss, val_loss, lr_now, f"{elapsed:.1f}"])
            f.flush()

            # Checkpoint
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_loss,
                "args":       {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
            }, last_path)

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                safe_args = {k: str(v) if hasattr(v, "__fspath__") else v
                                 for k, v in vars(args).items()}
                torch.save({
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "val_loss": val_loss,
                    "args":     safe_args,
                }, best_path)
                print(f"  --> New best val loss: {best_val:.4f}  (saved)")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping after {epoch} epochs "
                          f"({args.patience} epochs without improvement).")
                    break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {best_path}, {last_path}")
    print(f"Log:         {log_path}")


if __name__ == "__main__":
    main()
