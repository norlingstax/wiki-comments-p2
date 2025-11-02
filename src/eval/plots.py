from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.core.utils import ensure_dir


def _load_metrics_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_from_json(metrics_json: Path, labels=("non-toxic","toxic"), out_path: Path | None = None, title="Confusion matrix"):
    """Plot confusion matrix from metrics JSON."""
    data = _load_metrics_json(metrics_json)
    cm = np.array(data["confusion_matrix"], dtype=int)
    if out_path is None:
        out_path = metrics_json.with_name("confusion_matrix.png")
    plot_confusion(cm, labels, out_path, title=title)


def plot_roc_from_json(metrics_json: Path, out_path: Path | None = None, title="ROC curve"):
    """Plot ROC curve from metrics JSON."""
    data = _load_metrics_json(metrics_json)
    fpr = np.array(data["fpr"], dtype=float)
    tpr = np.array(data["tpr"], dtype=float)
    roc_auc = float(data["metrics"].get("roc_auc", np.nan))
    if out_path is None:
        out_path = metrics_json.with_name("roc.png")
    plot_roc(fpr, tpr, roc_auc, out_path, title=title)


def plot_pr_from_json(metrics_json: Path, out_path: Path | None = None, title="Precision-Recall curve"):
    """Plot PR curve from metrics JSON."""
    data = _load_metrics_json(metrics_json)
    prec = np.array(data["precision_curve"], dtype=float)
    rec  = np.array(data["recall_curve"], dtype=float)
    pr_auc = float(data["metrics"].get("pr_auc", np.nan))
    if out_path is None:
        out_path = metrics_json.with_name("pr.png")
    plot_pr(prec, rec, pr_auc, out_path, title=title)


def plot_confusion(cm: np.ndarray, labels: Sequence[str], out_path: Path, title="Confusion matrix"):
    """Confusion matrix plot parameters."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(4.8, 4.2))
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest", cmap="twilight")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)

    nrows, ncols = cm.shape
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("equal")

    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, out_path: Path, title="ROC curve"):
    """ROC curve plot parameters."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = plt.gca()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", lw=3, c="teal")
    ax.plot([0,1], [0,1], linestyle="--", c="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr(precision: np.ndarray, recall: np.ndarray, pr_auc: float, out_path: Path, title="Precision-Recall curve"):
    """PR curve plot parameters."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = plt.gca()
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.4f}", lw=3, c="teal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
