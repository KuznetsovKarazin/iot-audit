
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse, os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iot_audit.utils import infer_present_groups

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="c")

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def safe_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_target_balance(df: pd.DataFrame, outdir: str):
    for col in ["label", "type"]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts().sort_values(ascending=False)
            plt.figure(figsize=(8,4))
            counts.plot(kind="bar")
            plt.title(f"{col} balance")
            plt.xlabel(col)
            plt.ylabel("count")
            safe_fig(os.path.join(outdir, f"target_balance_{col}.png"))

def plot_numeric_distributions(df: pd.DataFrame, outdir: str, max_cols: int=24):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not num_cols: return
    n = len(num_cols)
    cols = 3
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*4, rows*3))
    for i, c in enumerate(num_cols, 1):
        plt.subplot(rows, cols, i)
        x = df[c].dropna().values
        if len(x) > 100000:
            x = np.random.choice(x, 100000, replace=False)
        plt.hist(x, bins=50)
        plt.title(c)
    safe_fig(os.path.join(outdir, "numeric_distributions.png"))

def plot_categorical_topk(df: pd.DataFrame, outdir: str, topk: int=15, cols: list[str]|None=None):
    if cols is None:
        cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols = [c for c in cols if c not in ("http_uri","ssl_subject","ssl_issuer","weird_addl")]
    for c in cols[:20]:
        vc = df[c].astype(str).value_counts().head(topk)
        if vc.empty: continue
        plt.figure(figsize=(8,4))
        vc.plot(kind="bar")
        plt.title(f"{c} top-{topk}")
        plt.xlabel(c); plt.ylabel("count")
        safe_fig(os.path.join(outdir, f"top_{c}.png"))

def plot_correlations(df: pd.DataFrame, outdir: str, max_cols: int=40):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if len(num_cols) < 2: return
    corr = df[num_cols].corr(method="pearson")
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=8)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
    plt.title("Correlation (Pearson)")
    safe_fig(os.path.join(outdir, "correlation_numeric.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args()
    ensure_dir(args.outdir)

    df = load_csv(args.csv)
    plot_target_balance(df, args.outdir)
    plot_numeric_distributions(df, args.outdir)
    plot_correlations(df, args.outdir)

    cats = ["proto","service","conn_state","dns_qtype","dns_rcode","ssl_version","http_method","http_status_code","weird_name"]
    plot_categorical_topk(df, args.outdir, cols=[c for c in cats if c in df.columns])

if __name__ == "__main__":
    main()