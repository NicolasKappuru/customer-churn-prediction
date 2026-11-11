"""
Checksum generator and verifier for the KKBox .csv.7z raw archives.

Usage
-----
Generate (run once on your machine after downloading all files):
    python workshops/w1/src/data_info/dataset_info.py --generate

Verify (run by any collaborator after placing files in workshops/data/):
    python workshops/w1/src/data_info/dataset_info.py

Exits with code 0 if all files pass, 1 if any file is missing or has a wrong hash.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CHECKSUMS_FILE = DATA_DIR / "checksums.json"

EXPECTED_FILES = [
    "train.csv.7z",
    "train_v2.csv.7z",
    "members_v3.csv.7z",
    "transactions.csv.7z",
    "transactions_v2.csv.7z",
    "user_logs.csv.7z",
    "user_logs_v2.csv.7z",
    "sample_submission_v2.csv.7z",
    "sample_submission_zero.csv.7z",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(data_dir: Path) -> None:
    checksums = {}
    missing = [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]
    if missing:
        print("ERROR: the following files were not found in", data_dir)
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    print(f"Computing SHA-256 for {len(EXPECTED_FILES)} files in {data_dir} ...\n")
    for name in EXPECTED_FILES:
        path = data_dir / name
        digest = sha256(path)
        size_mb = path.stat().st_size / 1_000_000
        checksums[name] = digest
        print(f"  {digest[:16]}...  {size_mb:>8.1f} MB  {name}")

    out = CHECKSUMS_FILE
    out.write_text(json.dumps(checksums, indent=2))
    print(f"\nSaved to {out}")
    print("Commit this file so collaborators can verify their downloads.")


def verify(data_dir: Path) -> None:
    if not CHECKSUMS_FILE.is_file():
        print("ERROR: checksums.json not found at", CHECKSUMS_FILE)
        print("Run with --generate first.")
        sys.exit(1)

    expected = json.loads(CHECKSUMS_FILE.read_text())
    failures = 0

    print(f"Verifying {len(expected)} files against {CHECKSUMS_FILE.name} ...\n")
    for name, expected_digest in expected.items():
        path = data_dir / name
        if not path.is_file():
            print(f"  MISSING  {name}")
            failures += 1
            continue
        actual = sha256(path)
        status = "OK   " if actual == expected_digest else "FAIL "
        if actual != expected_digest:
            failures += 1
        print(f"  {status}  {name}")

    print()
    if failures == 0:
        print(f"All {len(expected)} files verified successfully.")
    else:
        print(f"{failures} file(s) failed verification.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true",
                        help="Compute hashes and save checksums.json (run once by the dataset owner).")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR,
                        help=f"Directory containing the .csv.7z files (default: {DATA_DIR})")
    args = parser.parse_args()

    if args.generate:
        generate(args.data_dir)
    else:
        verify(args.data_dir)


if __name__ == "__main__":
    main()
