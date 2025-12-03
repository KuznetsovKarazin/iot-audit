from __future__ import annotations

import argparse
from typing import List

import pandas as pd


def find_candidate_split_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically search for columns that might encode a train/test split.
    """
    candidates = []

    # 1) name-based search
    name_keywords = ["split", "dataset", "fold", "partition", "set", "phase"]
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in name_keywords):
            candidates.append(col)

    # 2) value-based search: columns with few unique values that look like split labels
    split_like_values = {"train", "test", "val", "valid", "dev", "tr", "ts"}
    for col in df.columns:
        # skip obviously numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 10:
            uniques = {str(v).strip().lower() for v in df[col].dropna().unique()}
            if uniques & split_like_values:
                candidates.append(col)

    # deduplicate
    return sorted(set(candidates))


def main():
    ap = argparse.ArgumentParser(
        description="Check if a CSV contains any obvious hints of a predefined train/test split."
    )
    ap.add_argument("--csv", required=True, help="Path to train_test_network.csv")
    ap.add_argument(
        "--rows",
        type=int,
        default=50000,
        help="Max rows to load for quick inspection (default: 50000, use -1 for all)",
    )
    args = ap.parse_args()

    nrows = None if args.rows < 0 else args.rows
    print(f"[info] Loading {args.csv} (nrows={nrows or 'ALL'})...")
    df = pd.read_csv(args.csv, nrows=nrows)

    print(f"[info] shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("[info] columns:")
    print(", ".join(df.columns))

    # Try to find candidate split columns
    cand = find_candidate_split_columns(df)
    if not cand:
        print("\n[result] No obvious split/dataset columns were found.")
    else:
        print("\n[warning] Found candidate split-related columns:")
        for col in cand:
            nunique = df[col].nunique(dropna=True)
            print(f"  - {col!r}: {nunique} unique values")
            print("    sample values:", df[col].dropna().unique()[:10])

    # As additional info, show basic info about target columns if present
    for target in ["label", "type"]:
        if target in df.columns:
            print(f"\n[target] '{target}' value counts (first 10):")
            print(df[target].value_counts().head(10))


if __name__ == "__main__":
    main()
