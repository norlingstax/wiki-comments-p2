from __future__ import annotations
import json
from typing import Optional


# Utilities for parsing outputs
def parse_label_score_json(output: str) -> tuple[Optional[int], Optional[float]]:
    """
    Expecting: {"label": "toxic"|"non-toxic", "score": float in [0,1]}
    Returns (label_int, score_float) where label_int is 1 for toxic, 0 for non-toxic.
    """
    try:
        obj = json.loads(output.strip())
        lab = str(obj.get("label", "")).strip().lower()
        scr = obj.get("score", None)
        lbl = 1 if lab == "toxic" else 0 if lab == "non-toxic" else None
        if isinstance(scr, (int, float)):
            scr = max(0.0, min(1.0, float(scr)))
        else:
            scr = None
        return lbl, scr
    except Exception:
        return None, None

def label_to_prob(lbl: Optional[int], fallback: float = 0.5) -> float:
    """Map label int to probability; None returns fallback."""
    if lbl is None:
        return fallback
    return 1.0 if lbl == 1 else 0.0
