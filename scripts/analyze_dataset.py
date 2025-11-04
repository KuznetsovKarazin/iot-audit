
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse, json, os, sys
import pandas as pd
from collections import Counter
from iot_audit.utils import infer_present_groups, basic_summary

def load_csv(path: str, use_pyarrow: bool=True) -> pd.DataFrame:
    engine = "pyarrow" if use_pyarrow else "c"
    return pd.read_csv(path, engine=engine)

def class_balance(df: pd.DataFrame) -> dict:
    out = {}
    if "label" in df.columns:
        out["label"] = Counter(df["label"].astype(str))
    if "type" in df.columns:
        out["type"] = Counter(df["type"].astype(str))
    return out

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    df = load_csv(args.csv)
    groups = infer_present_groups(df)
    summary = basic_summary(df)
    balance = class_balance(df)

    out = {"summary":summary, "balance":balance, "column_groups":groups}
    save_json(out, os.path.join(args.outdir, "summary.json"))
    save_json(groups, os.path.join(args.outdir, "columns.json"))

    print(f"Rows: {summary['shape'][0]}, Cols: {summary['shape'][1]}")
    for k,v in balance.items():
        print(f"Balance[{k}]:", dict(v))

if __name__ == "__main__":
    main()