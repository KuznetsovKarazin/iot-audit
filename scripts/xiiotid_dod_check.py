"""Check the Definition of Done of the X-IIoTID first delivery, step by step.

Reads only the artefacts under reports_xiiot/ and configs/, so it runs in a second
and needs no dataset. Exits with status 1 if any step fails.

Produce those artefacts first, from the repository root, in this order:

    python scripts/xiiotid_audit.py --csv "data/X-IIoTID dataset.csv" --write
    python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv"
    python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv" --models rf logreg --drop-host-resources
    python -m pytest -q

then run this check:

    python scripts/xiiotid_dod_check.py
"""
from __future__ import annotations
import json, os, subprocess, sys

OUT = "reports_xiiot"
MODELS = ["rf", "xgb", "lgbm", "logreg"]
VARIANTS = MODELS + ["rf_no_host", "logreg_no_host"]
CLASSES = 10


def main():
    manifest = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
    costs = json.load(open("configs/xiiotid_severity_cost.json", encoding="utf-8"))["classes"]
    report = open(f"{OUT}/report.md", encoding="utf-8").read()
    metrics = {v: json.load(open(f"{OUT}/metrics_{v}.json", encoding="utf-8")) for v in VARIANTS}
    split, features = manifest["split"], manifest["features"]
    head = subprocess.run(["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout.strip()

    steps = {
        "1. commit, comandi, seed e split registrati":
            bool(head) and bool(manifest["dataset"]["sha256"]) and split["seed"] == 42
            and set(split["index_sha256"]) == {"train", "validation", "test"}
            and "python scripts/xiiotid_audit.py" in report
            and "python scripts/train_xiiotid_baseline.py" in report
            and all(m["provenance"]["commit"] for m in metrics.values()),
        "2. audit target leakage: nessuna colonna derivata dal target fra i predittori":
            not set(features["used"]) & set(features["excluded_labels"] + features["excluded_ids_alerts"]
                                            + features["excluded_identifiers"])
            and manifest["target"]["column"] == "class2" and len(manifest["target"]["classes"]) == CLASSES
            and features["schema_learnt_on"] == "training rows only",
        "3. confronti sullo stesso split: hash di dataset e indici uguali in ogni metrica":
            all(m["provenance"]["dataset_sha256"] == manifest["dataset"]["sha256"]
                and m["provenance"]["index_sha256"] == split["index_sha256"]
                and (m["train_rows"], m["validation_rows"], m["test_rows"])
                == (split["rows"]["train"], split["rows"]["validation"], split["rows"]["test"])
                for m in metrics.values()),
        "4. Macro-F1, Balanced Accuracy e recall per classe disponibili, test e validation":
            all(m["macro_f1"] and m["balanced_accuracy"]
                and len(m["recall_per_class"]) == len(m["recall_per_class_validation"]) == CLASSES
                and len(m["confusion_matrix_validation"]["matrix"]) == CLASSES
                for m in metrics.values()),
        "5. matrice attacco -> gravita -> costo preparata e usata dalla formula":
            len(costs) == CLASSES
            and all({"severity", "cost_fn", "cost_fp"} <= set(v) for v in costs.values())
            and set(costs) == set(manifest["target"]["classes"])
            and all("mistriaged_attacks_not_weighted" in m["cost"] for m in metrics.values()),
        "6. artefatti, diagnostiche e tabella collegati al report":
            all(f"metrics_{v}.json" in report for v in VARIANTS)
            and "manifest.json" in report and "xiiotid_severity_cost.json" in report
            and "group_diagnostics" in manifest,
    }
    for name, ok in steps.items():
        print(f"{'PASS' if ok else 'ERROR'} [dod] {name}")
    print(f"     commit verificato: {head or 'nessun commit'}")
    return int(not all(steps.values()))


if __name__ == "__main__":
    sys.exit(main())
