
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
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

def load_and_prepare_multiclass(
    csv_path: str,
    target_col: str = "type",
    test_size: float = 0.2,
    random_state: int = 42,
    base_outdir: str = "reports_mc",
    model_name: str = "mc_model"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Pipeline, Dict[int, str]]:
    df = _read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Factorize target
    y_raw = df[target_col].astype(str).str.strip()
    classes, y = np.unique(y_raw, return_inverse=True)
    class_map = {int(i): cls for i, cls in enumerate(classes)}

    # Drop target + known huge-cardinality text cols
    X = df.drop(columns=[target_col])
    drop_cols = [c for c in EXCLUDE_COLUMNS if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in CATEGORICAL_SAFE if c in X.columns]

    # add low-cardinality leftover object columns
    for c in X.select_dtypes(exclude=[np.number]).columns:
        if c in cat_cols or c in EXCLUDE_COLUMNS:
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

    print(f"[preprocessing-mc] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols...")
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    print(f"[preprocessing-mc] transformed test: {X_test_df.shape[0]} rows")

    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_out)

    # Save label map
    model_dir = os.path.join(base_outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    return X_train, X_test, y_train, y_test, feature_names, preprocessor, class_map
