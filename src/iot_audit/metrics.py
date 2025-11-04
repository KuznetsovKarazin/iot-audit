from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import pandas as pd

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def evaluate_model(
    y_true, y_pred, y_proba, feature_names: List[str], model,
    model_name: str = "model", base_outdir: str = "reports"
) -> Dict[str, Any]:
    model_dir = os.path.join(base_outdir, "models", model_name)
    fig_dir = os.path.join(base_outdir, "figures", model_name)
    _ensure_dir(model_dir)
    _ensure_dir(fig_dir)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": report["accuracy"],
        "precision_pos": report["1"]["precision"],
        "recall_pos": report["1"]["recall"],
        "f1_pos": report["1"]["f1-score"],
        "precision_neg": report["0"]["precision"],
        "recall_neg": report["0"]["recall"],
        "f1_neg": report["0"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr)
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC AUC={roc_auc:.4f}")
            _savefig(os.path.join(fig_dir,"roc_curve.png"))
            metrics["roc_auc"] = float(roc_auc)

            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            plt.figure(figsize=(5,4))
            plt.plot(recall, precision)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR AUC={ap:.4f}")
            _savefig(os.path.join(fig_dir,"pr_curve.png"))
            metrics["pr_auc"] = float(ap)
        except Exception:
            pass

    plt.figure(figsize=(4,3))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0,1], ["pred 0","pred 1"])
    plt.yticks([0,1], ["true 0","true 1"])
    plt.title("Confusion Matrix")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    _savefig(os.path.join(fig_dir,"confusion_matrix.png"))

    fi_path = os.path.join(model_dir, "feature_importances.csv")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(feature_names) == len(importances):
            df = pd.DataFrame({"feature": feature_names, "importance": importances})
            df = df.sort_values("importance", ascending=False)
            df.to_csv(fi_path, index=False)

            top = df.head(30)
            plt.figure(figsize=(8,8))
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Top-30 Feature Importances")
            _savefig(os.path.join(fig_dir,"feature_importances_top30.png"))
    except Exception:
        pass

    with open(os.path.join(model_dir,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
