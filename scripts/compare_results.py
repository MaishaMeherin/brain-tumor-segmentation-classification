"""Compare metric CSVs from a fresh run (outputs/) with the reference run (results/).

Usage:
    python scripts/compare_results.py                 # default tolerance 0.02
    python scripts/compare_results.py --tol 0.01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "segmentation_results.csv",
    "attention_unet_results.csv",
    "classification_results.csv",
    "attention_classifier_results.csv",
]
# The notebook writes attention_classifier_results.csv with lowercase metric
# names and Train/Val/Test as an unnamed index; map it onto the common format.
RENAME = {"acc": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}


def load(path):
    df = pd.read_csv(path)
    if "Split" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Split"})
    df = df.rename(columns=RENAME)
    df["Split"] = df["Split"].str.lower()
    return df.set_index("Split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--ref-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tol", type=float, default=0.02, help="max allowed absolute difference")
    args = parser.parse_args()

    all_ok = True
    for name in FILES:
        run_path, ref_path = args.run_dir / name, args.ref_dir / name
        print(f"\n== {name}")
        if not run_path.exists():
            print(f"  – not found in {args.run_dir} (has that stage been run?)")
            all_ok = False
            continue
        run, ref = load(run_path), load(ref_path)
        diff = (run - ref).loc[ref.index, ref.columns]
        table = pd.concat({"reference": ref, "this run": run.loc[ref.index, ref.columns], "diff": diff}, axis=1)
        print(table.round(4).to_string())
        worst = diff.abs().max().max()
        ok = worst <= args.tol
        all_ok &= ok
        print(f"  {'✓' if ok else '✗'} max |diff| = {worst:.4f} (tolerance {args.tol})")

    print("\n✓ All results within tolerance." if all_ok else "\n✗ Some results are missing or outside tolerance.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
