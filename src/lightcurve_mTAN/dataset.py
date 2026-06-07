"""
dataset.py
----------
PyTorch Dataset for the packed HDF5 lightcurve file.

Each item returned by __getitem__:
    {
        "x":         (T, 5)  float32  — [t_norm, magpsf_norm, sigmapsf_norm, is_g, is_r]
        "target":    (T, 2)  float32  — [magpsf_norm, sigmapsf_norm]
        "object_id": str
        "T":         int              — actual sequence length before padding
    }

Collated batches pad all sequences to the longest in the batch, and add:
    "mask": (B, T_max) bool — True where data is real, False where padded
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


FILTER_ONEHOT = {
    "g": np.array([1.0, 0.0], dtype=np.float32),
    "r": np.array([0.0, 1.0], dtype=np.float32),
}


# ── Normalisation statistics ──────────────────────────────────────────────────

class NormStats:
    """
    Per-filter mean and std for magpsf and sigmapsf, computed over the
    full dataset. Saved/loaded as a JSON file so it only needs computing once.
    """

    def __init__(self, stats: dict):
        # stats[band][field] = {"mean": float, "std": float}
        self._s = stats

    def normalise(self, band: str, magpsf: np.ndarray, sigmapsf: np.ndarray):
        ms       = self._s[band]
        mag_std  = ms["magpsf"]["std"]
        mag_n    = (magpsf - ms["magpsf"]["mean"]) / mag_std
        # sigmapsf is a photometric error bar: a positive scale on magpsf.
        # It must be normalised by the SAME scale as magpsf (no mean
        # subtraction) so that (mag_pred - mag_true) / sig_n is a proper
        # standardised residual. Z-scoring it independently (old behaviour)
        # let it go negative/zero and broke the chi2 denominator.
        sig_n    = sigmapsf / mag_std
        return mag_n.astype(np.float32), sig_n.astype(np.float32)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self._s, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def compute(cls, h5_path: Path, max_objects: Optional[int] = None) -> "NormStats":
        """
        Single pass over the HDF5 file to compute per-filter mean/std.
        Uses Welford's online algorithm — no need to load everything into RAM.
        """
        print("Computing normalisation statistics...")
        accum = {
            band: {field: {"n": 0, "mean": 0.0, "M2": 0.0}
                   for field in ("magpsf", "sigmapsf")}
            for band in ("g", "r")
        }

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            if max_objects:
                keys = keys[:max_objects]
            for oid in keys:
                for band in ("g", "r"):
                    grp = f[oid][band]
                    for field in ("magpsf", "sigmapsf"):
                        vals = grp[field][:]
                        for v in vals:
                            if not np.isfinite(v):
                                continue
                            acc = accum[band][field]
                            acc["n"] += 1
                            delta = v - acc["mean"]
                            acc["mean"] += delta / acc["n"]
                            acc["M2"]   += delta * (v - acc["mean"])

        stats = {}
        for band in ("g", "r"):
            stats[band] = {}
            for field in ("magpsf", "sigmapsf"):
                acc  = accum[band][field]
                n    = acc["n"]
                mean = acc["mean"]
                std  = float(np.sqrt(acc["M2"] / n)) if n > 1 else 1.0
                std  = max(std, 1e-6)
                stats[band][field] = {"mean": float(mean), "std": std}
                print(f"  {band}/{field}: mean={mean:.4f}  std={std:.4f}  (n={n})")

        return cls(stats)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LightcurveDataset(Dataset):
    """
    Loads one object per item. g and r band observations are merged into a
    single sequence sorted by normalised time.

    Per-observation input vector (dim=5):
        [t_norm, magpsf_norm, sigmapsf_norm, is_g, is_r]

    t_norm is normalised per-object to [0, 1].
    """

    def __init__(
        self,
        h5_path: Path,
        norm_stats: NormStats,
        object_ids: Optional[list] = None,
        min_observations: int = 10,
    ):
        self._h5_path = str(h5_path)
        self._norm    = norm_stats
        self._file    = None   # opened lazily per DataLoader worker

        with h5py.File(self._h5_path, "r") as f:
            all_ids = list(f.keys())
            if object_ids is not None:
                all_ids = [oid for oid in object_ids if oid in f]
            self._ids = []
            for oid in all_ids:
                n = sum(len(f[oid][b]["mjd"]) for b in ("g", "r"))
                if n >= min_observations:
                    self._ids.append(oid)

        dropped = len(all_ids) - len(self._ids) if object_ids is None else 0
        print(f"Dataset: {len(self._ids)} objects  "
              f"(dropped {dropped} with <{min_observations} obs)")

    def _get_file(self) -> h5py.File:
        # HDF5 files are not picklable; open once per worker process
        if self._file is None:
            self._file = h5py.File(self._h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        oid = self._ids[idx]
        grp = self._get_file()[oid]

        segments = []
        for band in ("g", "r"):
            mjd      = grp[band]["mjd"][:]
            magpsf   = grp[band]["magpsf"][:]
            sigmapsf = grp[band]["sigmapsf"][:]

            if len(mjd) == 0:
                continue

            mag_n, sig_n = self._norm.normalise(band, magpsf, sigmapsf)
            onehot = np.tile(FILTER_ONEHOT[band], (len(mjd), 1))  # (N, 2)
            seg    = np.column_stack([mjd, mag_n, sig_n, onehot])  # (N, 5)
            segments.append(seg)

        seq   = np.concatenate(segments, axis=0)          # (T, 5)
        seq   = seq[np.argsort(seq[:, 0])]                # sort by mjd

        # Normalise time to [0, 1] per object
        t0, t1     = seq[0, 0], seq[-1, 0]
        t_span     = t1 - t0 if t1 > t0 else 1.0
        seq[:, 0]  = (seq[:, 0] - t0) / t_span

        x = torch.from_numpy(seq)                         # (T, 5)
        return {
            "x":         x,
            "target":    x[:, 1:3].clone(),               # (T, 2) mag+sig
            "object_id": oid,
            "T":         len(seq),
        }

    def object_ids(self) -> list:
        return list(self._ids)


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """Pads variable-length sequences to the longest in the batch."""
    T_max = max(item["T"] for item in batch)
    B     = len(batch)

    x_pad      = torch.zeros(B, T_max, 5, dtype=torch.float32)
    target_pad = torch.zeros(B, T_max, 2, dtype=torch.float32)
    mask       = torch.zeros(B, T_max,    dtype=torch.bool)

    for b, item in enumerate(batch):
        T = item["T"]
        x_pad[b,      :T] = item["x"]
        target_pad[b, :T] = item["target"]
        mask[b,       :T] = True

    return {
        "x":          x_pad,                              # (B, T_max, 5)
        "target":     target_pad,                         # (B, T_max, 2)
        "mask":       mask,                               # (B, T_max)
        "lengths":    [item["T"] for item in batch],
        "object_ids": [item["object_id"] for item in batch],
    }


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloaders(
    h5_path: Path,
    norm_stats: NormStats,
    val_fraction: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 4,
    min_observations: int = 10,
    seed: int = 42,
) -> tuple:
    """
    Splits the dataset into train/val and returns:
        train_dl, val_dl, train_ids, val_ids
    """
    full_ds = LightcurveDataset(h5_path, norm_stats,
                                min_observations=min_observations)
    ids = full_ds.object_ids()

    rng       = np.random.default_rng(seed)
    shuffled  = rng.permutation(len(ids)).tolist()
    n_val     = max(1, int(len(ids) * val_fraction))
    val_ids   = [ids[i] for i in shuffled[:n_val]]
    train_ids = [ids[i] for i in shuffled[n_val:]]

    train_ds = LightcurveDataset(h5_path, norm_stats, object_ids=train_ids,
                                 min_observations=min_observations)
    val_ds   = LightcurveDataset(h5_path, norm_stats, object_ids=val_ids,
                                 min_observations=min_observations)

    dl_kwargs = dict(collate_fn=collate_fn, num_workers=num_workers,
                     pin_memory=True,
                     persistent_workers=(num_workers > 0))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_kwargs)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)

    return train_dl, val_dl, train_ids, val_ids
