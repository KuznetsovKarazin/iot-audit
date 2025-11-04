
from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def read_metrics(model_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m

def file_size_mb(path: str) -> float:
    return os.path.getsize(path)/ (1024*1024) if os.path.exists(path) else 0.0

def scan_models(base_outdir: str, model_names: List[str]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        mdir = os.path.join(base_outdir, "models", name)
        if not os.path.isdir(mdir):
            continue
        m = read_metrics(mdir)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "roc_auc_micro": m.get("roc_auc_micro"),
            "roc_auc_macro": m.get("roc_auc_macro"),
            "pr_auc_micro": m.get("pr_auc_micro"),
            "pr_auc_macro": m.get("pr_auc_macro"),
            "model_size_mb": round(file_size_mb(model_pkl), 3),
            "preproc_size_mb": round(file_size_mb(preproc_pkl), 3),
            "total_size_mb": round(file_size_mb(model_pkl)+file_size_mb(preproc_pkl), 3),
        })
    return pd.DataFrame(rows)

def plot_bar(df: pd.DataFrame, column: str, out_png: str, title: str):
    if df.empty or column not in df.columns: 
        return
    if df[column].isna().all(): 
        return
    plt.figure(figsize=(6,4))
    plt.bar(df["model"], df[column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(column)
    _savefig(out_png)

def per_class_table(base_outdir: str, model_names: List[str]) -> pd.DataFrame:
    merged = None
    for name in model_names:
        p = os.path.join(base_outdir, "models", name, "per_class_report.csv")
        if not os.path.exists(p): 
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={
            "precision": f"precision_{name}",
            "recall": f"recall_{name}",
            "f1": f"f1_{name}",
            "support": f"support_{name}",
        })
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="class", how="outer")
    return merged if merged is not None else pd.DataFrame()

def benchmark_inference(base_outdir: str, csv_path: str, model_names: List[str], y_col: str = "type", sample_size: int = 10000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="pyarrow")
    if y_col not in df.columns:
        raise ValueError(f"CSV must contain '{y_col}' column")
    X = df.drop(columns=[y_col])
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=random_state)
    else:
        Xs = X.copy()

    results = []
    for name in model_names:
        mdir = os.path.join(base_outdir, "models", name)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        if not (os.path.exists(model_pkl) and os.path.exists(preproc_pkl)):
            continue

        model = joblib.load(model_pkl)
        preproc = joblib.load(preproc_pkl)

        t0 = time.time()
        X_trans = preproc.transform(Xs)
        t1 = time.time()
        _ = getattr(model, "predict_proba", model.predict)(X_trans)
        t2 = time.time()

        results.append({
            "model": name,
            "transform_ms_per_1k": (t1 - t0) / (len(Xs)/1000.0) * 1000.0,
            "predict_ms_per_1k": (t2 - t1) / (len(Xs)/1000.0) * 1000.0,
            "total_ms_per_1k": (t2 - t0) / (len(Xs)/1000.0) * 1000.0,
            "n_samples": len(Xs)
        })
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports_mc")
    ap.add_argument("--models", nargs="*", default=["rf_mc","lgbm_mc","xgb_mc","logreg_mc"])
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark on sample of the CSV")
    ap.add_argument("--sample_size", type=int, default=10000)
    args = ap.parse_args()

    base_outdir = args.outdir
    summary_dir = os.path.join(base_outdir, "summary")
    _ensure_dir(summary_dir)

    df = scan_models(base_outdir, args.models)
    if df.empty:
        print("No model metrics found. Train some models first.")
        return
    df.to_csv(os.path.join(summary_dir, "summary_models_mc.csv"), index=False)
    print("Summary saved to", os.path.join(summary_dir, "summary_models_mc.csv"))
    print(df)

    plot_bar(df, "accuracy", os.path.join(summary_dir, "accuracy.png"), "Accuracy by Model (Multiclass)")
    plot_bar(df, "macro_f1", os.path.join(summary_dir, "macro_f1.png"), "Macro F1 by Model")
    plot_bar(df, "weighted_f1", os.path.join(summary_dir, "weighted_f1.png"), "Weighted F1 by Model")
    plot_bar(df, "roc_auc_micro", os.path.join(summary_dir, "roc_auc_micro.png"), "ROC AUC (micro) by Model")
    plot_bar(df, "roc_auc_macro", os.path.join(summary_dir, "roc_auc_macro.png"), "ROC AUC (macro) by Model")
    plot_bar(df, "pr_auc_micro", os.path.join(summary_dir, "pr_auc_micro.png"), "PR AUC (micro) by Model")
    plot_bar(df, "pr_auc_macro", os.path.join(summary_dir, "pr_auc_macro.png"), "PR AUC (macro) by Model")
    plot_bar(df, "total_size_mb", os.path.join(summary_dir, "total_size_mb.png"), "Model+Preproc Size (MB)")

    pct = per_class_table(base_outdir, args.models)
    if not pct.empty:
        pct.to_csv(os.path.join(summary_dir, "per_class_report_merged.csv"), index=False)
        print("Per-class report merged saved to", os.path.join(summary_dir, "per_class_report_merged.csv"))
        for name in args.models:
            fcol = f"f1_{name}"
            if fcol in pct.columns:
                dd = pct[["class", fcol]].dropna()
                plt.figure(figsize=(8, max(4, 0.35*len(dd))))
                plt.barh(dd["class"], dd[fcol])
                plt.title(f"Per-class F1: {name}")
                plt.ylabel("Class"); plt.xlabel("F1")
                _savefig(os.path.join(summary_dir, f"per_class_f1_{name}.png"))

    if args.benchmark:
        bdf = benchmark_inference(base_outdir, args.csv, args.models, y_col="type", sample_size=args.sample_size)
        bdf.to_csv(os.path.join(summary_dir, "inference_benchmark_mc.csv"), index=False)
        print("Benchmark saved to", os.path.join(summary_dir, "inference_benchmark_mc.csv"))
        plt.figure(figsize=(6,4))
        plt.bar(bdf["model"], bdf["total_ms_per_1k"])
        plt.title("Total latency (ms) per 1k flows (Multiclass)")
        plt.xlabel("Model")
        plt.ylabel("ms per 1k")
        _savefig(os.path.join(summary_dir, "latency_total_ms_per_1k.png"))

if __name__ == "__main__":
    main()
