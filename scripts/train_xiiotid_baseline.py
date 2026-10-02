"""X-IIoTID baseline on the frozen split: Macro-F1, Balanced Accuracy, per-class recall.

Run scripts/xiiotid_audit.py first: this script refuses to train unless the dataset
sha256 and the three split index hashes still match the registered manifest. Every
model uses the same split, so the numbers are comparable; each metrics file records the
commit, those hashes and the model configuration, and results go to
reports_xiiot/metrics_<model>.json.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from iot_audit.preprocessing import SCALE
from iot_audit.xiiotid import (TARGET, SEED, build_preprocessor, check_manifest, clean,
                               feature_columns, load, manifest_fingerprints, split)

# hyperparameters follow the existing scripts/train_mc_*.py of this repository
MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                         random_state=SEED, n_jobs=-1),
    "xgb": lambda: XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, objective="multi:softprob",
                                 eval_metric=["mlogloss"], tree_method="hist",
                                 random_state=SEED, n_jobs=-1),
    "lgbm": lambda: LGBMClassifier(objective="multiclass", num_leaves=63, n_estimators=600,
                                   learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1),
}


def expected_cost(y_true, y_pred, costs):
    """Weighted cost from configs/xiiotid_severity_cost.json, as the config defines it.

    A missed attack is weighted by cost_fn of its true class; a false alarm by cost_fp of
    the class the model raised, so the per-class cost_fp values in the config are the ones
    applied. Attack-to-attack confusions are counted, not weighted.
    """
    missed = sum(costs[t]["cost_fn"] for t, p in zip(y_true, y_pred) if t != "Normal" and p == "Normal")
    false_alarms = sum(costs[p]["cost_fp"] for t, p in zip(y_true, y_pred) if t == "Normal" and p != "Normal")
    mistriaged = sum(1 for t, p in zip(y_true, y_pred) if t != "Normal" and p != "Normal" and t != p)
    return {"missed_attack_cost": int(missed), "false_alarm_cost": int(false_alarms),
            "mistriaged_attacks_not_weighted": int(mistriaged)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/X-IIoTID dataset.csv")
    ap.add_argument("--outdir", default="reports_xiiot")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--drop-host-resources", action="store_true",
                    help="exclude the Avg_/Std_ host aggregates, to measure their contribution")
    ap.add_argument("--costs", default="configs/xiiotid_severity_cost.json")
    args = ap.parse_args()

    df = load(args.csv)
    train_idx, val_idx, test_idx = split(df)
    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    check_manifest(args.outdir, args.csv, parts)  # stops here if dataset or split changed
    provenance = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dataset_file": os.path.basename(args.csv),
        "seed": SEED,
        **manifest_fingerprints(args.csv, parts),
    }
    cols = feature_columns(df, train_idx, drop_host_resources=args.drop_host_resources)
    X, numeric, categorical = clean(df, cols["features"], train_idx)
    y_train, y_val, y_test = (df[TARGET].iloc[i] for i in (train_idx, val_idx, test_idx))
    costs = json.load(open(args.costs, encoding="utf-8"))["classes"]
    os.makedirs(args.outdir, exist_ok=True)

    for name in args.models:
        preprocessor = build_preprocessor(numeric, categorical, scale=SCALE[name])
        start = time.time()
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_val = preprocessor.transform(X.iloc[val_idx])
        X_test = preprocessor.transform(X.iloc[test_idx])
        model = MODELS[name]()
        print(f"[{name}] training on {X_train.shape[0]} rows, {X_train.shape[1]} features...")
        encoder = LabelEncoder().fit(y_train)  # XGBoost only accepts integer classes
        model.fit(X_train, encoder.transform(y_train))
        y_pred = encoder.inverse_transform(model.predict(X_test))
        y_val_pred = encoder.inverse_transform(model.predict(X_val))
        labels = sorted(df[TARGET].unique())
        metrics = {
            "model": name,
            "provenance": provenance,
            "configuration": {k: v for k, v in model.get_params().items() if v is not None},
            "features_used": len(cols["features"]),
            "scaled": SCALE[name],
            "host_resources_used": not args.drop_host_resources,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "macro_f1_validation": round(float(f1_score(y_val, y_val_pred, average="macro")), 4),
            "recall_per_class_validation": {c: round(float(r), 4) for c, r in
                                            zip(labels, recall_score(y_val, y_val_pred, labels=labels,
                                                                     average=None, zero_division=0))},
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "recall_per_class": {c: round(float(r), 4) for c, r in
                                 zip(labels, recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0))},
            "support_per_class": y_test.value_counts().reindex(labels, fill_value=0).to_dict(),
            "confusion_matrix": {"labels": labels,
                                 "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist()},
            "confusion_matrix_validation": {"labels": labels,
                                            "matrix": confusion_matrix(y_val, y_val_pred, labels=labels).tolist()},
            "cost": expected_cost(y_test.tolist(), list(y_pred), costs),
            "train_seconds": round(time.time() - start, 1),
        }
        suffix = "_no_host" if args.drop_host_resources else ""
        with open(os.path.join(args.outdir, f"metrics_{name}{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"[{name}] macro-F1 {metrics['macro_f1']} | balanced accuracy {metrics['balanced_accuracy']} "
              f"| {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
