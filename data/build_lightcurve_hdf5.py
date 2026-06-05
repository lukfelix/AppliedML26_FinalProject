"""
build_lightcurve_hdf5.py
------------------------
Converts a directory of ZTF lightcurve CSVs into a single HDF5 file
suitable for ML training.

HDF5 layout
-----------
/<object_id>/
    g/  mjd[N], magpsf[N], sigmapsf[N]   (filter 1, green)
    r/  mjd[M], magpsf[M], sigmapsf[M]   (filter 2, red)

All arrays are float32, sorted by mjd.
Missing filters are stored as empty datasets (length 0).
i-band (fid=3) is intentionally excluded -- too sparse to be useful.

Usage
-----
    python build_lightcurve_hdf5.py --input_dir ./lightcurves --output lightcurves.h5

Dependencies
------------
    pip install numpy pandas h5py tqdm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ZTF filter id -> short name (i-band excluded)
FILTER_MAP = {1: "g", 2: "r"}
COLUMNS_NEEDED = {"fid", "mjd", "magpsf", "sigmapsf"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack ZTF lightcurve CSVs into HDF5.")
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing ZTFXXyyyyyyy.csv files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("lightcurves.h5"),
        help="Output HDF5 file path (default: lightcurves.h5).",
    )
    p.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern to match input files (default: *.csv).",
    )
    p.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression filter (default: gzip).",
    )
    p.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        help="Compression level for gzip 1-9 (default: 4, ignored for lzf).",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
    )
    return logging.getLogger(__name__)


def load_csv(path: Path, log: logging.Logger) -> Optional[pd.DataFrame]:
    """Read a single lightcurve CSV; return None on failure."""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        log.warning("Failed to read %s: %s", path.name, exc)
        return None

    missing = COLUMNS_NEEDED - set(df.columns)
    if missing:
        log.warning("Skipping %s -- missing columns: %s", path.name, missing)
        return None

    return df


def write_filter_group(
    obj_group: h5py.Group,
    filter_name: str,
    data: np.ndarray,   # shape (N, 3): mjd, magpsf, sigmapsf -- sorted
    compression: str,
    compression_opts: int,
) -> None:
    """Write mjd/magpsf/sigmapsf datasets under obj_group/<filter_name>/."""
    grp = obj_group.require_group(filter_name)

    kwargs: dict = {"dtype": np.float32}
    if compression != "none" and len(data) > 0:
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = compression_opts

    for i, name in enumerate(("mjd", "magpsf", "sigmapsf")):
        grp.create_dataset(name, data=data[:, i], **kwargs)


def process_file(
    path: Path,
    h5file: h5py.File,
    compression: str,
    compression_opts: int,
    log: logging.Logger,
) -> bool:
    """Process one CSV and write it into h5file. Returns True on success."""
    df = load_csv(path, log)
    if df is None:
        return False

    object_id = path.stem  # e.g. ZTF19abcdefg
    if object_id in h5file:
        log.warning("Duplicate object %s -- skipping.", object_id)
        return False

    obj_group = h5file.create_group(object_id)

    for fid, filter_name in FILTER_MAP.items():
        subset = df[df["fid"] == fid].copy()

        if subset.empty:
            # Store empty datasets so the key always exists
            grp = obj_group.require_group(filter_name)
            for col in ("mjd", "magpsf", "sigmapsf"):
                grp.create_dataset(col, data=np.empty(0, dtype=np.float32))
            log.debug("%s / %s: empty", object_id, filter_name)
            continue

        subset.sort_values("mjd", inplace=True)
        arr = subset[["mjd", "magpsf", "sigmapsf"]].to_numpy(dtype=np.float32)

        write_filter_group(obj_group, filter_name, arr, compression, compression_opts)
        log.debug("%s / %s: %d observations", object_id, filter_name, len(arr))

    return True


def main() -> None:
    args = parse_args()
    log = setup_logging(args.log_level)

    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        log.error("No files matched pattern '%s' in %s", args.pattern, args.input_dir)
        return

    log.info("Found %d CSV files -> writing to %s", len(csv_files), args.output)

    n_ok = n_fail = 0

    with h5py.File(args.output, "w") as h5file:
        # Store format metadata at root level for future reference
        h5file.attrs["filter_map"] = str(FILTER_MAP)
        h5file.attrs["columns"] = "mjd, magpsf, sigmapsf"
        h5file.attrs["n_source_files"] = len(csv_files)

        for path in tqdm(csv_files, unit="file", desc="Packing"):
            if process_file(path, h5file, args.compression, args.compression_opts, log):
                n_ok += 1
            else:
                n_fail += 1

    log.info("Done. Wrote %d objects. Failed/skipped: %d.", n_ok, n_fail)
    log.info("Output size: %.1f MB", args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
