"""Shared utilities for data processing.

Small date helpers used by the domain modules, the ``_shrink`` dtype compactor,
plus the one-time extraction of the raw ``.7z`` archives into parquet.
"""

from pathlib import Path
import shutil
import subprocess

import pandas as pd
import polars as pl

from w1.src.data_processing.schemas import SCHEMA, _COLUMNS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def window_start(cutoff):
    """First day of the 3-month window that ends with the cutoff month."""
    return (pd.Timestamp(cutoff).to_period("M") - 2).start_time


def sequence_start(cutoff, SEQ_DAYS=90):
    """First of the ``SEQ_DAYS`` days that end on the cutoff."""
    return pd.Timestamp(cutoff) - pd.Timedelta(days=SEQ_DAYS - 1)


def _date_int(value):
    """Date as an ``YYYYMMDD`` integer (the encoding used in the raw files)."""
    return int(pd.Timestamp(value).strftime("%Y%m%d"))


def _date(value):
    """Date as a plain ``datetime.date`` (for date arithmetic in expressions)."""
    return pd.Timestamp(value).date()


def _shrink(df):
    """Cast ``Float64`` -> ``Float32`` and ``Int64`` -> ``Int32`` to compact the output."""
    casts = [pl.col(name).cast(pl.Float32) if dtype == pl.Float64 else
             pl.col(name).cast(pl.Int32) if dtype == pl.Int64 else None
             for name, dtype in df.schema.items()]
    return df.with_columns([c for c in casts if c is not None])


# ---------------------------------------------------------------------------
# Raw conversion
# ---------------------------------------------------------------------------

def _7z_binary():
    """Path to an available 7-Zip executable, or raise if none is installed."""
    for binary in ("7z", "7za", "7zz"):
        path = shutil.which(binary)
        if path:
            return path
    raise RuntimeError("Converting .7z files needs the `7z` command (p7zip).")


def convert_raw_to_parquet(data_dir, parquet_dir=None, archives=None):
    """Convert ``data/*.csv.7z`` to ``data/parquet/*.parquet`` (zstd), idempotently.

    Uses the conversion contract from ``schemas`` (``SCHEMA`` for dtypes,
    ``_COLUMNS`` for the columns to keep); its keys define which archives to
    convert when ``archives`` is not given. Each archive is extracted to
    ``data/_extract`` one at a time, streamed to parquet and the CSV deleted, so
    the disk peak is one extracted CSV plus the parquet being written. Existing
    outputs are skipped.
    """
    data_dir = Path(data_dir)
    parquet_dir = Path(parquet_dir) if parquet_dir else data_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = data_dir / "_extract"

    # Iterating over the files
    for name in (archives or list(SCHEMA)):
        out = parquet_dir / f"{name}.parquet"
        if out.exists():
            print(f"[skip] {out.name} already exists")
            continue
        archive = data_dir / f"{name}.csv.7z"
        if not archive.exists():
            print(f"[skip] {archive} not found")
            continue

        print(f"[extract] {archive.name}")

        # Run a subprocess to extract the archive using 7z, suppressing output
        subprocess.run(
            [_7z_binary(), "x", "-y", f"-o{extract_dir}", str(archive)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Find the extracted CSV file
        csv = next(path for path in extract_dir.rglob(f"{name}.csv"))
        print(f"[convert] {name}.csv -> {out.name}")
        (
            pl.scan_csv(str(csv), schema_overrides=SCHEMA[name])
            .select(_COLUMNS[name])
            .sink_parquet(out, compression="zstd")
        )
        csv.unlink()

        if name.startswith("user_logs"):
            rows = pl.scan_parquet(out).select(pl.len()).collect().item()
            print(f"          {rows:,} rows")

    return parquet_dir
