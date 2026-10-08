"""Check that BRISC2025 is laid out where the notebook expects it.

Usage:
    python scripts/check_data.py                  # counts only (fast)
    python scripts/check_data.py --verify-hashes  # also check SHA-256 of every file
    python scripts/check_data.py --root /path/to/brisc2025
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Expected file counts for the official BRISC2025 release
EXPECTED = {
    "classification_task/train/glioma": 1147,
    "classification_task/train/meningioma": 1329,
    "classification_task/train/no_tumor": 1067,
    "classification_task/train/pituitary": 1457,
    "classification_task/test/glioma": 254,
    "classification_task/test/meningioma": 306,
    "classification_task/test/no_tumor": 140,
    "classification_task/test/pituitary": 300,
    "segmentation_task/train/images": 3933,
    "segmentation_task/train/masks": 3933,
    "segmentation_task/test/images": 860,
    "segmentation_task/test/masks": 860,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "brisc2025")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        sys.exit(f"✗ Dataset not found at {root}\n  See data/README.md for how to get it.")

    ok = True
    print(f"Dataset root: {root}\n")
    for rel, expected in EXPECTED.items():
        folder = root / rel
        n = sum(1 for p in folder.glob("*") if p.is_file()) if folder.is_dir() else 0
        status = "✓" if n == expected else "✗"
        ok &= n == expected
        print(f"  {status} {rel:40s} {n:5d} files (expected {expected})")

    if args.verify_hashes:
        manifest = root / "manifest.csv"
        if not manifest.exists():
            sys.exit(f"✗ {manifest} not found; cannot verify hashes")
        bad = 0
        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            # Manifest paths use Windows separators
            path = root / Path(*row["relative_path"].split("\\"))
            if not path.exists() or sha256(path) != row["sha256"]:
                bad += 1
                if bad <= 10:
                    print(f"  ✗ missing or corrupted: {row['relative_path']}")
        print(f"\n  Hash check: {len(rows) - bad}/{len(rows)} files match manifest.csv")
        ok &= bad == 0

    print("\n✓ Dataset looks complete." if ok else "\n✗ Dataset is incomplete or differs from the official release.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
