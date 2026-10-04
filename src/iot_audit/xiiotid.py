"""X-IIoTID: dataset registration, train-only preprocessing and frozen split.

The split is defined once here and reused by every script, so that all model
comparisons run on the same split (see reports_xiiot/manifest.json). The manifest is
written once by the audit and only verified afterwards: check_manifest() refuses to
run when the dataset or the split indices no longer match what was registered.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "class2"
SEED = 42
N_SPLITS = 5  # one fold is the test set, i.e. 20% of the inferred windows
WINDOW_SECONDS = 10  # aggregation period documented for the dataset (DOI 10.1109/JIOT.2021.3102056)
MANIFEST = "manifest.json"

LABEL_COLUMNS = ["class1", "class2", "class3"]
IDS_COLUMNS = ["anomaly_alert", "OSSEC_alert", "OSSEC_alert_level"]
IDENTIFIER_COLUMNS = ["Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def resource_columns(df: pd.DataFrame) -> list[str]:
    """Host-resource features, averaged by the testbed over 10-second windows."""
    return [c for c in df.columns if c.startswith(("Avg_", "Std_", "std_"))]


def feature_columns(df: pd.DataFrame, train_idx: np.ndarray,
                    drop_host_resources: bool = False) -> dict[str, list[str]]:
    """Predictors, plus the reason each excluded column is excluded.

    Constant columns are detected on the training rows only, so editing validation or
    test rows cannot change which features the model sees.
    """
    train = df.iloc[train_idx]
    constant = [c for c in df.columns if train[c].nunique(dropna=False) <= 1]
    host = resource_columns(df) if drop_host_resources else []
    excluded = LABEL_COLUMNS + IDS_COLUMNS + IDENTIFIER_COLUMNS + constant + host
    return {
        "features": [c for c in df.columns if c not in excluded],
        "excluded_labels": LABEL_COLUMNS,
        "excluded_ids_alerts": IDS_COLUMNS,
        "excluded_identifiers": IDENTIFIER_COLUMNS,
        "excluded_constant": constant,
        "excluded_host_resources": host,
    }


def clean(df: pd.DataFrame, features: list[str],
          train_idx: np.ndarray) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Booleans to 0/1, '?' and '-' to missing, then numeric apart from categorical.

    Which columns count as numeric is decided on the training rows only and then
    applied unchanged to validation and test.
    """
    X = df[features].replace({"TRUE": "1", "FALSE": "0", "T": "1", "F": "0", "?": None, "-": None})
    train = X.iloc[train_idx]
    numeric = [c for c in features if pd.to_numeric(train[c], errors="coerce").notna().mean() > 0.5]
    categorical = [c for c in features if c not in numeric]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce")
    X[categorical] = X[categorical].astype(str)
    return X, numeric, categorical


def window_groups(df: pd.DataFrame) -> np.ndarray:
    """Inferred window id: the stretch over which the host-resource vector is constant.

    The dataset documents that host features are aggregated over 10 seconds but ships no
    window identifier, so this is a reconstruction, not an official id: flows are grouped
    into runs of consecutive rows sharing one aggregate vector. Binning the timestamp is a
    weaker proxy, since flows inside one 10-second bin do carry different aggregate
    vectors. Grouping by these runs keeps identical aggregates on one side of the split;
    it does not establish temporal independence nor the absence of leakage in general.
    """
    seconds = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(-1)
    order = np.lexsort((df.index.to_numpy(), seconds.to_numpy()))
    vector = df.iloc[order][resource_columns(df)].astype(str).agg("|".join, axis=1)
    groups = np.empty(len(df), dtype=int)
    groups[order] = np.cumsum((vector != vector.shift()).to_numpy()) - 1
    return groups


def split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frozen split: train 60% / validation 20% / test 20%, whole windows only.

    Folds are group-aware and stratified; at each cut we keep the one maximising the
    smallest per-class share, counted on whichever side is the weaker one, so that no
    class is missing from a part or unmeasurable in it. Deterministic given SEED.
    """
    groups = window_groups(df)
    total = df[TARGET].value_counts()

    def best(idx: np.ndarray, n_splits: int):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        def coverage(fold):  # value_counts drops absent classes, so reindex before comparing
            shares = [df[TARGET].iloc[idx[side]].value_counts().reindex(total.index, fill_value=0)
                      / total for side in fold]
            return min(min(a, b) for a, b in zip(*shares))

        return max(cv.split(idx, df[TARGET].iloc[idx], groups[idx]), key=coverage)

    rest, test = best(np.arange(len(df)), N_SPLITS)
    train, val = best(rest, N_SPLITS - 1)
    return np.sort(rest[train]), np.sort(rest[val]), np.sort(test)


def group_diagnostics(df: pd.DataFrame, groups: np.ndarray) -> dict:
    """Essential checks on the inferred windows, reported in the manifest."""
    seconds = pd.to_numeric(df["Timestamp"], errors="coerce")
    frame = pd.DataFrame({"group": groups, "second": seconds, "bin": seconds // WINDOW_SECONDS})
    sizes = frame["group"].value_counts()
    span = frame.groupby("group")["second"].agg(lambda v: v.max() - v.min())
    return {
        "groups": int(sizes.size),
        "rows_per_group": {"median": float(sizes.median()), "p95": float(sizes.quantile(0.95)),
                           "max": int(sizes.max())},
        "seconds_per_group": {"median": float(span.median()), "p95": float(span.quantile(0.95)),
                              "max": float(span.max())},
        "groups_longer_than_the_documented_window": int(span.gt(WINDOW_SECONDS).sum()),
        "rows_with_missing_timestamp": int(seconds.isna().sum()),
        "timestamps_carrying_more_than_one_group": int(frame.groupby("second")["group"].nunique().gt(1).sum()),
        "groups_spanning_more_than_one_10s_bin": int(frame.groupby("group")["bin"].nunique().gt(1).sum()),
    }


def split_fingerprint(test_idx: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, test_idx)).encode()).hexdigest()


def manifest_fingerprints(csv_path: str, parts: dict[str, np.ndarray]) -> dict:
    return {"dataset_sha256": sha256(csv_path),
            "index_sha256": {name: split_fingerprint(idx) for name, idx in parts.items()}}


def check_manifest(outdir: str, csv_path: str, parts: dict[str, np.ndarray]) -> dict:
    """Return the saved manifest, raising if dataset or split no longer match it.

    Nothing is written here: a mismatch has to be looked at, not overwritten.
    """
    with open(os.path.join(outdir, MANIFEST), encoding="utf-8") as f:
        saved = json.load(f)
    now = manifest_fingerprints(csv_path, parts)
    errors = [f"dataset sha256 is {now['dataset_sha256']}, manifest says {saved['dataset']['sha256']}"] \
        if saved["dataset"]["sha256"] != now["dataset_sha256"] else []
    errors += [f"{name} index hash is {value}, manifest says {saved['split']['index_sha256'].get(name)}"
               for name, value in now["index_sha256"].items()
               if saved["split"]["index_sha256"].get(name) != value]
    if errors:
        raise ValueError("manifest does not match this run: " + "; ".join(errors))
    return saved


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale else "passthrough"),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
