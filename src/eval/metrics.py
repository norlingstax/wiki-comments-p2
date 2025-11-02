from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable
import json
import numpy as np
from sklearn.metrics import (
    classification_report,   
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

def compute_binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> Dict:
    """Compute binary metrics for later plotting."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }

    pr_p, pr_r, pr_t = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_t = roc_curve(y_true, y_prob)
    results["pr_curve"] = {"precision": pr_p.tolist(), "recall": pr_r.tolist(), "thresholds": pr_t.tolist()}
    results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_t.tolist()}
    return results

def save_json(d: Dict, path: str | Path) -> None:
    """Save a dictionary to a JSON file."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

