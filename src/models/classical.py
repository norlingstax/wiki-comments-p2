from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import compute_binary_metrics, save_json
from src.core.utils import Timer, init_logger, ensure_dir

logger = init_logger("classical")

# Config 
@dataclass
class ClassicalConfig:
    name: str = "logreg"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = None
    n_jobs: int = -1

    # io
    save_dir: str = "models/embeddings"
    artifacts_name: str = "logreg"

    # provenance (optional)
    data_signature: Optional[str] = None
    features_signature: Optional[str] = None
    seed: int = 42

# Builder 
def _build_estimator(cfg: ClassicalConfig) -> Pipeline:
    if cfg.name != "logreg":
        raise ValueError(f"Unknown model name: {cfg.name}")
    base = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        n_jobs=cfg.n_jobs,
        solver="lbfgs",
    )
    # Standardise inputs
    model = Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", base)])
    return model

# Training
def train_from_arrays(
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: ClassicalConfig,
    run_name: Optional[str] = None,
) -> Dict:
    """
    Train on precomputed arrays and save the model + meta.
    No evaluation here; do it in eval/evaluate.py.
    arrays: dict with keys: "train", "val", "test" -> (X, y)
    """
    X_train, y_train = arrays["train"]

    model = _build_estimator(cfg)
    timings: Dict[str, float] = {}

    with Timer() as t_fit:
        model.fit(X_train, y_train)
    timings["train_time_s"] = t_fit.elapsed or 0.0

    def _proba(m, X):
        if not len(X): return np.array([], dtype=np.float32)
        if hasattr(m, "predict_proba"): return m.predict_proba(X)[:, 1]
        mrg = m.decision_function(X).astype(np.float64)
        mrg = np.clip(mrg, -50, 50)
        return (1 / (1 + np.exp(-mrg))).astype(np.float32)

    X_val, y_val = arrays.get("val", (np.empty((0, X_train.shape[1])), np.array([], dtype=int)))

    y_train_prob = _proba(model, X_train)
    train_metrics = compute_binary_metrics(y_true=y_train, y_prob=y_train_prob, threshold=0.5)

    if len(X_val):
        y_val_prob = _proba(model, X_val)
        val_metrics = compute_binary_metrics(y_true=y_val, y_prob=y_val_prob, threshold=0.5)
    else:
        y_val_prob = np.array([], dtype=np.float32)
        val_metrics = {}
    
    save_root = ensure_dir(Path(cfg.save_dir))
    tag = run_name or cfg.artifacts_name
    run_dir = save_root / tag
    ensure_dir(run_dir)

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    
    save_json(train_metrics, run_dir / "metrics.train.json")
    if val_metrics:
        save_json(val_metrics, run_dir / "metrics.val.json")

    meta = {
        "config": asdict(cfg),
        "timings": timings,
        "shapes": {
            k: [int(v[0].shape[0]), int(v[0].shape[1])] for k, v in arrays.items() if len(v) == 2 and hasattr(v[0], "shape")
        },
        "approach": "embeddings+classical",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved model -> %s", model_path)

    return {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "timings": timings,
        "meta_path": str(run_dir / "run_meta.json"),
        "metrics_train": train_metrics,
        "metrics_val": val_metrics,
    }
