"""X-IIoTID: dataset registration, train-only preprocessing and frozen split.

The split is defined once here and reused by every script, so that all model
comparisons run on the same split (see reports_xiiot/manifest.json).
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "class2"
SEED = 42
N_SPLITS = 5  # one fold is the test set, i.e. 20% of the 10-second windows
WINDOW_SECONDS = 10

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


def feature_columns(df: pd.DataFrame, drop_host_resources: bool = False) -> dict[str, list[str]]:
    """Predictors, plus the reason each excluded column is excluded."""
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
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


def clean(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Booleans to 0/1, '?' and '-' to missing, then numeric apart from categorical."""
    X = df[features].replace({"TRUE": "1", "FALSE": "0", "T": "1", "F": "0", "?": None, "-": None})
    numeric = [c for c in features if pd.to_numeric(X[c], errors="coerce").notna().mean() > 0.5]
    categorical = [c for c in features if c not in numeric]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce")
    X[categorical] = X[categorical].astype(str)
    return X, numeric, categorical


def window_groups(df: pd.DataFrame) -> np.ndarray:
    """Canonical window id: the stretch over which the host-resource vector is constant.

    Those features are aggregates the testbed computed over a 10-second window, so the
    window is exactly the run of consecutive flows sharing one aggregate vector. Binning
    the timestamp is only a proxy for it: flows inside the same 10-second bin do carry
    different aggregate vectors, so the bin is not the window the testbed used.
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


def split_fingerprint(test_idx: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, test_idx)).encode()).hexdigest()


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
