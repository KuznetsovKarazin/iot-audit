"""X-IIoTID audit rules, checked on a synthetic frame so CI needs no dataset."""
import numpy as np
import pandas as pd
import pytest

from iot_audit.xiiotid import (
    IDENTIFIER_COLUMNS, IDS_COLUMNS, LABEL_COLUMNS, TARGET, WINDOW_SECONDS,
    build_preprocessor, clean, feature_columns, split, split_fingerprint, window_groups,
)

N_ROWS = 1200
CLASSES = ["Normal", "Reconnaissance", "Exfiltration", "RDOS"]


@pytest.fixture
def df():
    idx = np.arange(N_ROWS)
    return pd.DataFrame({
        "Date": "9/01/2020",
        "Timestamp": (1578540000 + idx // 3 * WINDOW_SECONDS).astype(str),  # 3 rows per window
        "Scr_IP": "10.0.0.1", "Des_IP": "10.0.0.2", "Scr_port": idx % 7, "Des_port": 502,
        "Protocol": np.where(idx % 2 == 0, "tcp", "udp"),
        "Duration": (idx * 0.5).astype(float),
        "is_syn_only": np.where(idx % 3 == 0, "TRUE", "FALSE"),
        "Avg_tps": (idx // 3).astype(float), "Std_tps": (idx // 3 % 5).astype(float),
        "anomaly_alert": np.where(idx % 4 == 0, "TRUE", "FALSE"),
        "OSSEC_alert": "FALSE", "OSSEC_alert_level": 0,
        "class1": np.array(CLASSES)[idx % 4], "class2": np.array(CLASSES)[idx % 4],
        "class3": np.where(idx % 4 == 0, "Normal", "Attack"),
    })


def test_predictors_exclude_labels_alerts_and_identifiers(df):
    used = set(feature_columns(df)["features"])
    assert not used & set(LABEL_COLUMNS)
    assert not used & set(IDS_COLUMNS)
    assert not used & set(IDENTIFIER_COLUMNS)
    assert "Protocol" in used and "Duration" in used


def test_host_resources_can_be_dropped(df):
    used = set(feature_columns(df, drop_host_resources=True)["features"])
    assert not used & {"Avg_tps", "Std_tps"}


def test_no_window_is_split_between_the_three_parts(df):
    parts = split(df)
    windows = pd.Series(window_groups(df))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not set(windows.iloc[parts[a]]) & set(windows.iloc[parts[b]])
        assert not set(parts[a]) & set(parts[b])
    assert sum(len(p) for p in parts) == N_ROWS
    assert all(set(df[TARGET].iloc[p]) == set(CLASSES) for p in parts)


def test_split_is_reproducible(df):
    assert [split_fingerprint(p) for p in split(df)] == [split_fingerprint(p) for p in split(df)]


def test_preprocessing_is_fitted_on_training_rows_only(df):
    cols = feature_columns(df)
    X, numeric, categorical = clean(df, cols["features"])
    train_idx = split(df)[0]
    preprocessor = build_preprocessor(numeric, categorical, scale=True)
    preprocessor.fit(X.iloc[train_idx])
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(train_idx)
