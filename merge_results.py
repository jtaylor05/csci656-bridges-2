#!/usr/bin/env python3
"""
merge_results.py
----------------
Merges all per-group result CSVs into a single master dataset CSV.
Run after all SLURM jobs complete.

Usage:
    python merge_results.py --indir results/ --output dataset.csv
"""
import argparse, csv, glob, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir",  default="results")
    ap.add_argument("--output", default="dataset.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*_results.csv")))
    if not files:
        print(f"No result CSVs found in {args.indir}/"); return

    writer = None
    total  = 0
    counts = {}

    with open(args.output, "w", newline="") as out_f:
        for path in files:
            with open(path) as f:
                reader = csv.DictReader(f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    total += 1
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"  Merged {path}")

    print(f"\nTotal rows: {total:,}")
    for s, c in sorted(counts.items()):
        print(f"  {s:<10} {c:>6,}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
