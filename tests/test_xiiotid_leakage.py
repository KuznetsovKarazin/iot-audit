"""X-IIoTID audit rules, checked on a synthetic frame so CI needs no dataset."""
import json

import numpy as np
import pandas as pd
import pytest

from iot_audit.xiiotid import (
    IDENTIFIER_COLUMNS, IDS_COLUMNS, LABEL_COLUMNS, MANIFEST, TARGET, WINDOW_SECONDS,
    build_preprocessor, check_manifest, clean, feature_columns, group_diagnostics,
    manifest_fingerprints, split, split_fingerprint, window_groups,
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
    used = set(feature_columns(df, split(df)[0])["features"])
    assert not used & set(LABEL_COLUMNS)
    assert not used & set(IDS_COLUMNS)
    assert not used & set(IDENTIFIER_COLUMNS)
    assert "Protocol" in used and "Duration" in used


def test_host_resources_can_be_dropped(df):
    used = set(feature_columns(df, split(df)[0], drop_host_resources=True)["features"])
    assert not used & {"Avg_tps", "Std_tps"}


def test_schema_ignores_changes_outside_the_training_rows(df):
    """R2: editing validation and test must not move a column in or out of the schema."""
    train_idx, val_idx, test_idx = split(df)
    before = feature_columns(df, train_idx), clean(df, feature_columns(df, train_idx)["features"], train_idx)[1:]
    tampered = df.astype({"Duration": object, "Std_tps": object})
    held_out = np.concatenate([val_idx, test_idx])
    tampered.iloc[held_out, tampered.columns.get_loc("Duration")] = "non numerico"
    tampered.iloc[held_out, tampered.columns.get_loc("Std_tps")] = 7.0  # was constant per window
    tampered.iloc[held_out, tampered.columns.get_loc("Protocol")] = "sctp"
    after = feature_columns(tampered, train_idx), clean(tampered, feature_columns(tampered, train_idx)["features"], train_idx)[1:]
    assert before == after


def test_group_diagnostics_report_the_essentials(df):
    """R3: the inferred windows are described, not assumed."""
    diagnostics = group_diagnostics(df, window_groups(df))
    assert diagnostics["groups"] > 1
    assert diagnostics["rows_with_missing_timestamp"] == 0
    assert set(diagnostics["seconds_per_group"]) == {"median", "p95", "max"}


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


def test_manifest_mismatch_stops_the_run(df, tmp_path):
    """R1: a manifest that disagrees with the split must raise, not be overwritten."""
    train_idx, val_idx, test_idx = split(df)
    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    good = manifest_fingerprints(str(csv), parts)
    saved = {"dataset": {"sha256": good["dataset_sha256"]}, "split": {"index_sha256": good["index_sha256"]}}
    (tmp_path / MANIFEST).write_text(json.dumps(saved), encoding="utf-8")
    assert check_manifest(str(tmp_path), str(csv), parts)["dataset"]["sha256"] == good["dataset_sha256"]

    saved["split"]["index_sha256"]["test"] = "0" * 64
    (tmp_path / MANIFEST).write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest does not match"):
        check_manifest(str(tmp_path), str(csv), parts)
    assert json.loads((tmp_path / MANIFEST).read_text())["split"]["index_sha256"]["test"] == "0" * 64


def test_preprocessing_is_fitted_on_training_rows_only(df):
    train_idx = split(df)[0]
    cols = feature_columns(df, train_idx)
    X, numeric, categorical = clean(df, cols["features"], train_idx)
    preprocessor = build_preprocessor(numeric, categorical, scale=True)
    preprocessor.fit(X.iloc[train_idx])
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(train_idx)
