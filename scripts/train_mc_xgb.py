
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.preprocessing_mc import load_and_prepare_multiclass
from iot_audit.metrics_mc import evaluate_model_multiclass
from xgboost import XGBClassifier
import joblib
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="reports_mc")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    args = ap.parse_args()

    model_name = "xgb_mc"

    X_train, X_test, y_train, y_test, feature_names, preproc, class_map = load_and_prepare_multiclass(
        csv_path=args.csv, target_col="type", test_size=0.2, random_state=42,
        base_outdir=args.outdir, model_name=model_name
    )

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="multi:softprob",
        num_class=len(class_map),
        eval_metric=["mlogloss"],
        n_jobs=-1,
        tree_method="hist",
        random_state=42
    )

    print(f"[xgb-mc] training ({X_train.shape[0]} samples, {X_train.shape[1]} features, classes={len(class_map)})...")
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
    print(f"[xgb-mc] done in {time.time()-t0:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    model_dir = os.path.join(args.outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(preproc, os.path.join(model_dir, "preprocessor.pkl"))

    metrics = evaluate_model_multiclass(
        y_test, y_pred, y_proba, feature_names, model,
        class_map=class_map, model_name=model_name, base_outdir=args.outdir
    )
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[xgb-mc] metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
