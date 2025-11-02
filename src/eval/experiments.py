from __future__ import annotations
from pathlib import Path
import csv
from datetime import datetime, timezone
from typing import Dict


COLUMNS: list[str] = [
    "timestamp", "run_id", "approach",
    "model",          # embeddings: "logreg"; transformer & prompts: model_id
    "provider",       # prompts only
    "prompt_type",    # prompts only (template name)
    "features_kind",  # embeddings only
    "data_sig", "seed",
    "roc_auc_train", "pr_auc_train",
    "roc_auc_val",   "pr_auc_val",
    "roc_auc_test",  "pr_auc_test",
    "train_time_s", "eval_time_s",
]

def _scalar(v):
    """Convert a value to a scalar string."""
    if v is None: return ""
    if isinstance(v, (int, float, str)): return v
    try:
        import numpy as np
        if isinstance(v, (np.integer, np.floating)): return v.item()
    except Exception:
        pass
    return "" 

def ensure_csv(csv_path: Path) -> None:
    """Ensure the experiment CSV exists."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()

def new_run_row(
    *,
    run_id: str,
    approach: str,
    model: str = "",
    provider: str = "",
    features_kind: str = "",
    prompt_type: str = "",
    data_sig: str = "",
    seed: int | str = "",
    train_time_s: float | str = "",
    eval_time_s: float | str = "",
) -> Dict[str, str]: 
    """Create a new row for the experiment CSV.""" 
    row = {k: "" for k in COLUMNS} 
    row.update({ 
        "timestamp": datetime.now(timezone.utc).isoformat(),  
        "run_id": run_id,  
        "approach": approach,  
        "model": model,  
        "provider": provider,  
        "prompt_type": prompt_type,  
        "features_kind": features_kind,  
        "data_sig": data_sig,  
        "seed": seed,  
        "train_time_s": _scalar(train_time_s),
        "eval_time_s": _scalar(eval_time_s),
    })  
    return row  

def upsert_row(csv_path: Path, run_id: str, updates: Dict[str, object]) -> None:
    """Update the row with run_id; if missing, create a new row."""
    ensure_csv(csv_path)
    rows = []
    found = False
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row = {k: row.get(k, "") for k in COLUMNS}
            if row.get("run_id") == run_id:
                for k, v in updates.items():
                    if k in COLUMNS:
                        row[k] = str(_scalar(v))
                found = True
            rows.append(row)

    if not found:
        base = {k: "" for k in COLUMNS}
        base["timestamp"] = datetime.now(timezone.utc).isoformat()
        base["run_id"] = run_id
        for k, v in updates.items():
            if k in COLUMNS:
                base[k] = str(_scalar(v))
        rows.append(base)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

def append_row(csv_path: Path, row: Dict[str, object]) -> None:
    """Append a brand-new row that already matches COLUMNS."""
    ensure_csv(csv_path)
    safe = {k: str(_scalar(row.get(k, ""))) for k in COLUMNS}
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writerow(safe)
