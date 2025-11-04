
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
    # add FP/FN
    try:
        cm = m.get("confusion_matrix", [[0,0],[0,0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        m["fp"] = fp
        m["fn"] = fn
    except Exception:
        m["fp"] = None
        m["fn"] = None
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
            "f1_pos": m.get("f1_pos"),
            "f1_neg": m.get("f1_neg"),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "fp": m.get("fp"),
            "fn": m.get("fn"),
            "model_size_mb": round(file_size_mb(model_pkl), 3),
            "preproc_size_mb": round(file_size_mb(preproc_pkl), 3),
            "total_size_mb": round(file_size_mb(model_pkl)+file_size_mb(preproc_pkl), 3),
        })
    return pd.DataFrame(rows)

def plot_bar(df: pd.DataFrame, column: str, out_png: str, title: str):
    plt.figure(figsize=(6,4))
    plt.bar(df["model"], df[column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(column)
    _savefig(out_png)

def benchmark_inference(base_outdir: str, csv_path: str, model_names: List[str], sample_size: int = 10000, random_state: int = 42) -> pd.DataFrame:
    # Load raw CSV once
    df = pd.read_csv(csv_path, engine="pyarrow")
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    X = df.drop(columns=["label"])
    # optional: drop 'type' if present (was dropped during training by preprocessing)
    if "type" in X.columns:
        X = X.drop(columns=["type"])

    # sample
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
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--models", nargs="*", default=["rf","lgbm","xgb","logreg"])
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

    df.to_csv(os.path.join(summary_dir, "summary_models.csv"), index=False)
    print("Summary saved to", os.path.join(summary_dir, "summary_models.csv"))
    print(df)

    # plots
    plot_bar(df, "accuracy", os.path.join(summary_dir, "accuracy.png"), "Accuracy by Model")
    if df["f1_pos"].notna().any():
        plot_bar(df, "f1_pos", os.path.join(summary_dir, "f1_pos.png"), "F1 (attack) by Model")
    if df["roc_auc"].notna().any():
        plot_bar(df, "roc_auc", os.path.join(summary_dir, "roc_auc.png"), "ROC AUC by Model")
    if df["fp"].notna().any():
        plot_bar(df, "fp", os.path.join(summary_dir, "fp.png"), "False Positives by Model")
    if df["fn"].notna().any():
        plot_bar(df, "fn", os.path.join(summary_dir, "fn.png"), "False Negatives by Model")

    if args.benchmark:
        bdf = benchmark_inference(base_outdir, args.csv, args.models, sample_size=args.sample_size)
        bdf.to_csv(os.path.join(summary_dir, "inference_benchmark.csv"), index=False)
        print("Benchmark saved to", os.path.join(summary_dir, "inference_benchmark.csv"))
        print(bdf)

        # plots for benchmark
        plt.figure(figsize=(6,4))
        plt.bar(bdf["model"], bdf["total_ms_per_1k"])
        plt.title("Total latency (ms) per 1k flows")
        plt.xlabel("Model")
        plt.ylabel("ms per 1k")
        _savefig(os.path.join(summary_dir, "latency_total_ms_per_1k.png"))

if __name__ == "__main__":
    main()
