"""Print a structured summary of the Telco Customer Churn dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def summarize_dataset(csv_path: Path, target_column: str = "Churn") -> dict:
    """Return dataset dimensions, class counts, feature metadata, and checksum."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    dataset = pd.read_csv(csv_path)
    if target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {csv_path.name}"
        )

    columns = {}
    for name in dataset.columns:
        values = dataset[name]
        numeric_values = pd.to_numeric(values, errors="coerce")
        non_missing_count = int(values.notna().sum())
        column_info = {
            "data_type": str(values.dtype),
            "missing_values": int(values.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(values) or (
            non_missing_count > 0
            and int(numeric_values.notna().sum()) == non_missing_count
        ):
            valid_numeric_values = numeric_values.dropna()
            if not valid_numeric_values.empty:
                column_info["range"] = {
                    "min": float(valid_numeric_values.min()),
                    "max": float(valid_numeric_values.max()),
                }

        columns[name] = column_info

    target_counts = {
        str(value): int(count)
        for value, count in dataset[target_column].value_counts(dropna=False).items()
    }

    return {
        "file": csv_path.name,
        "format": "CSV",
        "size_bytes": csv_path.stat().st_size,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": int(dataset.shape[0]),
        "column_count": int(dataset.shape[1]),
        "target_column": target_column,
        "samples_per_class": target_counts,
        "columns": columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print dimensions, class counts, data types, and ranges for a CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV file to inspect (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--target",
        default="Churn",
        help="Target column used for per-class sample counts (default: Churn)",
    )
    args = parser.parse_args()

    try:
        summary = summarize_dataset(args.csv, args.target)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())