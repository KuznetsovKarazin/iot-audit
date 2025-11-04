from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CATEGORICAL_SAFE = [
    "proto","service","conn_state",
    "dns_qtype","dns_rcode",
    "ssl_version",
    "http_method","http_status_code",
    "weird_name"
]

EXCLUDE_COLUMNS = [
    "ssl_subject","ssl_issuer","http_uri","http_user_agent",
    "http_orig_mime_types","http_resp_mime_types","weird_addl","dns_query"
]

def _read_csv(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, engine="pyarrow")
    except Exception:
        return pd.read_csv(csv_path)

def _normalize_label(s: pd.Series) -> pd.Series:
    vals = s.astype(str).str.lower().str.strip()
    mapping = {"0":0,"1":1,"benign":0,"normal":0,"malicious":1,"attack":1}
    return vals.map(lambda x: mapping.get(x, 1 if x not in ("0","normal","benign") else 0)).astype(int)

def load_and_prepare_data(
    csv_path: str,
    target_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    leakage_base: str = "reports",
    model_name: str = "model"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Pipeline]:
    df = _read_csv(csv_path)

    # Drop potential target-like columns that may leak label
    leak_cols = []
    for c in [target_col, 'type', 'Type', 'TARGET', 'Label', 'label']:
        if c in df.columns and c != target_col:
            leak_cols.append(c)
    if leak_cols:
        df = df.drop(columns=leak_cols)
        print(f"[preprocessing] dropped potential leakage columns: {leak_cols}")

    # Quick leakage report: top correlations (numeric only)
    try:
        y_tmp = _normalize_label(df[target_col]) if target_col in df.columns else None
        num = df.select_dtypes(include=[float, int]).copy()
        if y_tmp is not None:
            num[target_col] = y_tmp
            corr = num.corr(numeric_only=True)[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False)
            top = corr.head(20).to_dict()
            ldir = os.path.join(leakage_base, "models", model_name)
            os.makedirs(ldir, exist_ok=True)
            with open(os.path.join(ldir, "leakage_report.json"), "w", encoding="utf-8") as f:
                json.dump({'top_abs_corr_with_label': top}, f, indent=2)
    except Exception:
        pass

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = _normalize_label(df[target_col])
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in CATEGORICAL_SAFE if c in X.columns]

    for c in X.select_dtypes(exclude=[np.number]).columns:
        if c in cat_cols or c in EXCLUDE_COLUMNS or c == target_col:
            continue
        if X[c].nunique(dropna=False) <= 40:
            cat_cols.append(c)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[preprocessing] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols...")
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    print(f"[preprocessing] transformed test: {X_test_df.shape[0]} rows")

    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_out)

    return X_train, X_test, y_train.values, y_test.values, feature_names, preprocessor
