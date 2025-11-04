from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.preprocessing import load_and_prepare_data
from iot_audit.metrics import evaluate_model
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=None)
    args = ap.parse_args()

    model_name = "rf"

    X_train, X_test, y_train, y_test, feature_names, preproc = load_and_prepare_data(
        csv_path=args.csv, target_col="label", test_size=0.2, random_state=42,
        leakage_base=args.outdir, model_name=model_name
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )

    print(f"[rf] training ({X_train.shape[0]} samples, {X_train.shape[1]} features)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[rf] done in {time.time()-t0:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    model_dir = os.path.join(args.outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(preproc, os.path.join(model_dir, "preprocessor.pkl"))

    metrics = evaluate_model(
        y_test, y_pred, y_proba, feature_names, model,
        model_name=model_name, base_outdir=args.outdir
    )
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[rf] metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
