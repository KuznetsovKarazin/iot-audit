
from __future__ import annotations
import os, json
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def evaluate_model_multiclass(
    y_true, y_pred, y_proba, feature_names: List[str], model,
    class_map: Dict[int, str],
    model_name: str = "mc_model", base_outdir: str = "reports_mc"
) -> Dict[str, Any]:
    model_dir = os.path.join(base_outdir, "models", model_name)
    fig_dir = os.path.join(base_outdir, "figures", model_name)
    _ensure_dir(model_dir)
    _ensure_dir(fig_dir)

    target_names = [class_map[i] for i in sorted(class_map.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }

    # Confusion matrix plot
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix (multiclass)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _savefig(os.path.join(fig_dir, "confusion_matrix.png"))

    # ROC/PR (one-vs-rest) if probabilities available
    if y_proba is not None and y_proba.ndim == 2:
        n_classes = y_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # micro/macro ROC
        try:
            roc_auc_micro = roc_auc_score(y_true_bin, y_proba, average="micro", multi_class="ovr")
            roc_auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            metrics["roc_auc_micro"] = float(roc_auc_micro)
            metrics["roc_auc_macro"] = float(roc_auc_macro)
        except Exception:
            pass

        # micro/macro PR
        try:
            # micro: stack all decisions
            precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
            ap_micro = average_precision_score(y_true_bin, y_proba, average="micro")
            plt.figure(figsize=(5,4))
            plt.plot(recall_micro, precision_micro)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (micro) AUC={ap_micro:.4f}")
            _savefig(os.path.join(fig_dir, "pr_micro.png"))
            metrics["pr_auc_micro"] = float(ap_micro)

            # macro: mean of per-class AP
            ap_per_class = []
            for k in range(n_classes):
                pk, rk, _ = precision_recall_curve(y_true_bin[:,k], y_proba[:,k])
                ap_k = average_precision_score(y_true_bin[:,k], y_proba[:,k])
                ap_per_class.append(ap_k)
                # save each class PR curve
                plt.figure(figsize=(5,4))
                plt.plot(rk, pk)
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR AUC ({target_names[k]})={ap_k:.4f}")
                _savefig(os.path.join(fig_dir, f"pr_{k}_{target_names[k].replace(' ','_')}.png"))
            metrics["pr_auc_macro"] = float(np.mean(ap_per_class))
        except Exception:
            pass

    # Feature importances (if supported)
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(feature_names) == len(importances):
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df = df.sort_values("importance", ascending=False)
            df.to_csv(os.path.join(model_dir, "feature_importances.csv"), index=False)

            top = df.head(30)
            plt.figure(figsize=(8,8))
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Top-30 Feature Importances")
            _savefig(os.path.join(fig_dir,"feature_importances_top30.png"))
    except Exception:
        pass

    with open(os.path.join(model_dir,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class report as CSV
    per_class_rows = []
    for i, name in enumerate(target_names):
        row = report.get(name, {})
        per_class_rows.append({
            "class": name,
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "f1": row.get("f1-score"),
            "support": row.get("support")
        })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(model_dir, "per_class_report.csv"), index=False)

    return metrics
