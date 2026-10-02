"""Register the X-IIoTID dataset and audit the frozen split for target leakage.

Writing and verifying are separate. The first run, or a run with --write, produces
reports_xiiot/manifest.json; later runs only compare the dataset sha256 and the three
split index hashes against it and fail on any difference, never overwriting it.
Exits with status 1 if any check fails. Run before any training.

    python scripts/xiiotid_audit.py --csv "data/X-IIoTID dataset.csv"            # verify
    python scripts/xiiotid_audit.py --csv "data/X-IIoTID dataset.csv" --write    # (re)register
"""
from __future__ import annotations
import argparse, json, os, sys
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from iot_audit.xiiotid import (
    IDENTIFIER_COLUMNS, IDS_COLUMNS, LABEL_COLUMNS, MANIFEST, N_SPLITS, SEED, TARGET,
    WINDOW_SECONDS, build_preprocessor, check_manifest, clean, feature_columns,
    group_diagnostics, load, manifest_fingerprints, resource_columns, split, window_groups,
)


def build_manifest(df, csv_path, parts, cols, numeric, categorical, groups, duplicated):
    fingerprints = manifest_fingerprints(csv_path, parts)
    return {
        "dataset": {"file": os.path.basename(csv_path), "sha256": fingerprints["dataset_sha256"],
                    "rows": int(len(df)), "columns": int(df.shape[1])},
        "target": {"column": TARGET, "classes": sorted(df[TARGET].unique().tolist())},
        "features": {"used": cols["features"], "numeric": numeric, "categorical": categorical,
                     "schema_learnt_on": "training rows only",
                     **{k: v for k, v in cols.items() if k != "features"}},
        "split": {
            "method": f"StratifiedGroupKFold(n_splits={N_SPLITS}) for test, then {N_SPLITS - 1} for "
                      "validation; at each cut the fold maximising the smallest per-class share on "
                      "the weaker side",
            "group": "inferred window: run of consecutive rows with a constant host-resource vector; "
                     f"the dataset documents {WINDOW_SECONDS}-second aggregation but ships no window id",
            "seed": SEED,
            "rows": {name: int(len(idx)) for name, idx in parts.items()},
            "index_sha256": fingerprints["index_sha256"],
        },
        "group_diagnostics": group_diagnostics(df, groups),
        "class_distribution": {name: df[TARGET].iloc[idx].value_counts().to_dict()
                               for name, idx in parts.items()},
        "residual_risk": {
            "test_rows_sharing_host_resource_vector_with_train": int(duplicated.sum()),
            "share_of_test_rows": round(float(duplicated.mean()), 4),
            "note": "The same host-resource vector can reappear in non-contiguous windows, so "
                    "identical vectors show up on both sides even though no inferred window is "
                    "split. This measures that exposure; it does not prove the absence of leakage.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/X-IIoTID dataset.csv")
    ap.add_argument("--outdir", default="reports_xiiot")
    ap.add_argument("--write", action="store_true", help="(re)register the manifest instead of verifying it")
    args = ap.parse_args()

    df = load(args.csv)
    train_idx, val_idx, test_idx = split(df)
    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    cols = feature_columns(df, train_idx)
    X, numeric, categorical = clean(df, cols["features"], train_idx)
    groups = window_groups(df)
    windows = pd.Series(groups)
    fingerprint = df.groupby(resource_columns(df), dropna=False).ngroup()
    duplicated = fingerprint.iloc[test_idx].isin(set(fingerprint.iloc[train_idx]))

    preprocessor = build_preprocessor(numeric, categorical, scale=True)
    preprocessor.fit(X.iloc[train_idx])
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    manifest = build_manifest(df, args.csv, parts, cols, numeric, categorical, groups, duplicated)
    path = os.path.join(args.outdir, MANIFEST)
    if args.write or not os.path.exists(path):
        os.makedirs(args.outdir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        registered = f"manifest written to {path}"
    else:
        try:
            check_manifest(args.outdir, args.csv, parts)
            registered = "saved manifest matches this dataset and split"
        except ValueError as error:
            registered = str(error)

    used = set(cols["features"])
    checks = {
        "single target fixed (class2, 10 classes)": TARGET == "class2" and df[TARGET].nunique() == 10,
        "no label column among predictors": not used & set(LABEL_COLUMNS),
        "no third-party IDS alert among predictors": not used & set(IDS_COLUMNS),
        "no testbed identifier among predictors": not used & set(IDENTIFIER_COLUMNS),
        "feature schema decided on training rows only":
            manifest["features"]["schema_learnt_on"] == "training rows only",
        "train, validation and test are disjoint":
            sum(len(idx) for idx in parts.values()) == len(set().union(*(set(i) for i in parts.values()))) == len(df),
        "no inferred window shared by any two parts":
            all(not set(windows.iloc[a]) & set(windows.iloc[b]) for a, b in combinations(parts.values(), 2)),
        "preprocessing fitted on train rows only": scaler.n_samples_seen_ == len(train_idx),
        "every class present in every part":
            all(set(df[TARGET].iloc[idx]) == set(df[TARGET]) for idx in parts.values()),
        "dataset and split match the registered manifest": registered.startswith(("manifest written", "saved manifest")),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'ERROR'} [xiiotid] {name}")
    print(f"     {registered}")
    print(f"WARN [xiiotid] {duplicated.mean():.1%} of test rows share a host-resource vector with train "
          f"(reported in manifest.json; not a split leak, and not proof of no leakage)")
    return int(not all(checks.values()))


if __name__ == "__main__":
    sys.exit(main())
