"""Register the X-IIoTID dataset and audit the frozen split for target leakage.

Writes reports_xiiot/manifest.json (dataset version, target, features, split, seed)
and exits with status 1 if any check fails. Run before any training.
"""
from __future__ import annotations
import argparse, json, os, sys
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from iot_audit.xiiotid import (
    IDENTIFIER_COLUMNS, IDS_COLUMNS, LABEL_COLUMNS, N_SPLITS, SEED, TARGET, WINDOW_SECONDS,
    build_preprocessor, clean, feature_columns, load, resource_columns, sha256, split,
    split_fingerprint, window_groups,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/X-IIoTID dataset.csv")
    ap.add_argument("--outdir", default="reports_xiiot")
    args = ap.parse_args()

    df = load(args.csv)
    cols = feature_columns(df)
    X, numeric, categorical = clean(df, cols["features"])
    train_idx, val_idx, test_idx = split(df)
    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    windows = pd.Series(window_groups(df))
    fingerprint = df.groupby(resource_columns(df), dropna=False).ngroup()
    shared = set(fingerprint.iloc[train_idx]) & set(fingerprint.iloc[test_idx])
    duplicated = fingerprint.iloc[test_idx].isin(shared)

    preprocessor = build_preprocessor(numeric, categorical, scale=True)
    preprocessor.fit(X.iloc[train_idx])
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    manifest = {
        "dataset": {
            "file": os.path.basename(args.csv),
            "sha256": sha256(args.csv),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
        },
        "target": {"column": TARGET, "classes": sorted(df[TARGET].unique().tolist())},
        "features": {"used": cols["features"], "numeric": numeric, "categorical": categorical,
                     **{k: v for k, v in cols.items() if k != "features"}},
        "split": {
            "method": f"StratifiedGroupKFold(n_splits={N_SPLITS}) for test, then {N_SPLITS - 1} for validation; "
                      "at each cut the fold maximising the smallest per-class share on the weaker side",
            "group": f"canonical window: run of constant host-resource vector, aggregated by the testbed over {WINDOW_SECONDS} seconds",
            "seed": SEED,
            "windows": int(windows.nunique()),
            "rows": {name: int(len(idx)) for name, idx in parts.items()},
            "index_sha256": {name: split_fingerprint(idx) for name, idx in parts.items()},
        },
        "class_distribution": {name: df[TARGET].iloc[idx].value_counts().to_dict()
                               for name, idx in parts.items()},
        "residual_risk": {
            "test_rows_sharing_host_resource_vector_with_train": int(duplicated.sum()),
            "share_of_test_rows": round(float(duplicated.mean()), 4),
            "note": "Host-resource aggregates repeat across windows, so identical vectors appear on "
                    "both sides even though no window is split. Re-run the baseline with "
                    "--drop-host-resources on the same split to measure the effect.",
        },
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    used = set(cols["features"])
    checks = {
        "single target fixed (class2, 10 classes)": TARGET == "class2" and df[TARGET].nunique() == 10,
        "no label column among predictors": not used & set(LABEL_COLUMNS),
        "no IDS alert among predictors": not used & set(IDS_COLUMNS),
        "no testbed identifier among predictors": not used & set(IDENTIFIER_COLUMNS),
        "train, validation and test are disjoint":
            sum(len(idx) for idx in parts.values()) == len(set().union(*(set(i) for i in parts.values()))) == len(df),
        "no canonical window shared by any two parts":
            all(not set(windows.iloc[a]) & set(windows.iloc[b])
                for a, b in combinations(parts.values(), 2)),
        "preprocessing fitted on train rows only": scaler.n_samples_seen_ == len(train_idx),
        "every class present in every part":
            all(set(df[TARGET].iloc[idx]) == set(df[TARGET]) for idx in parts.values()),
        "split reproducible from the manifest":
            [split_fingerprint(idx) for idx in split(df)] == list(manifest["split"]["index_sha256"].values()),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'ERROR'} [xiiotid] {name}")
    print(f"WARN [xiiotid] {duplicated.mean():.1%} of test rows share a host-resource vector with train "
          f"(reported in manifest.json, not a leak of the split)")
    return int(not all(checks.values()))


if __name__ == "__main__":
    sys.exit(main())
