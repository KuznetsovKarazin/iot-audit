
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.preprocessing_mc import load_and_prepare_multiclass
from iot_audit.metrics_mc import evaluate_model_multiclass
import lightgbm as lgb
import joblib
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="reports_mc")
    ap.add_argument("--num_leaves", type=int, default=64)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    args = ap.parse_args()

    model_name = "lgbm_mc"

    X_train, X_test, y_train, y_test, feature_names, preproc, class_map = load_and_prepare_multiclass(
        csv_path=args.csv, target_col="type", test_size=0.2, random_state=42,
        base_outdir=args.outdir, model_name=model_name
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(class_map),
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print(f"[lgbm-mc] training ({X_train.shape[0]} samples, {X_train.shape[1]} features, classes={len(class_map)})...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["multi_logloss"],
        callbacks=[lgb.log_evaluation(50)]
    )
    print(f"[lgbm-mc] done in {time.time()-t0:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # list of arrays or array, normalize to array
    # LightGBM may return list of n_classes arrays; handle both cases
    import numpy as np
    if isinstance(y_proba, list):
        y_proba = np.vstack(y_proba).T

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
    print("[lgbm-mc] metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
