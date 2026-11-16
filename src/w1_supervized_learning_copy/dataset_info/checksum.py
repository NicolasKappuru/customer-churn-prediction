"""Print the SHA-256 checksum of the Telco Customer Churn CSV."""

import argparse
import hashlib
from pathlib import Path


DEFAULT_CSV = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate the SHA-256 checksum of a dataset file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV file to hash (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"Dataset file not found: {args.csv}")

    checksum = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())