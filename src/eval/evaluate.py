from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix
)

from src.eval.metrics import compute_binary_metrics, save_json
from src.core.utils import init_logger, ensure_dir
from src.core.data_io import read_manifest
from src.features.embed_sentence import build_or_load_sentence_embeddings
from src.features.embed_wordvec import build_or_load_wordvec_features
from src.models.model_utils import build_or_load_tokenised
from src.eval.experiments import upsert_row

logger = init_logger("evaluate")


def _finalise_binary_eval(
    *,
    y_true,
    y_prob,
    threshold: float,
    run_dir: Path | None,
    experiments_csv: Path,
    run_id: str,
    extra_payload: dict | None = None,
):
    """Compute metrics + curves, write metrics JSON(s), and backfill experiments."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    payload = {
        "metrics": metrics,
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec.tolist(),
        "recall_curve": rec.tolist(),
    }
    if extra_payload:
        payload.update(extra_payload)

    if run_dir is not None:
        ensure_dir(run_dir)
        save_json(payload, Path(run_dir) / "metrics.test.json")

    upsert_row(
        Path(experiments_csv),
        run_id,
        {
            "roc_auc_test": metrics.get("roc_auc", ""),
            "pr_auc_test":  metrics.get("pr_auc", ""),
        },
    )
    
    metrics_dir = ensure_dir(Path("outputs") / "metrics")
    metrics_path = metrics_dir / f"{run_id}.test.json"
    save_json(payload, metrics_path)
    
    return metrics


def _make_prompt_provider(
    model_id: str,
):
    """Construct the Ollama prompts provider and return (provider, subdir, extra)."""
    from src.prompts.provider_ollama import OllamaProvider
    provider = OllamaProvider(model_id=model_id)
    subdir = f"ollama-{Path(str(model_id)).name.replace('/', '_')}"
    extra = {"provider": "ollama", "model_id": model_id}
    return provider, subdir, extra


# Embeddings
def _predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    """Predict P(y=1) for sklearn models, handling decision_function fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    m = model.decision_function(X).astype(np.float64)
    m = np.clip(m, -50, 50)
    return (1 / (1 + np.exp(-m))).astype(np.float32)


def evaluate_embeddings_run(
    run_dir: str | Path,
    manifests_dir: str | Path,
    processed_dir: str | Path,
    experiments_csv: str | Path,
    threshold: float = 0.5,
    run_id: Optional[str] = None,
    features_kind: str = "sentence",                # "sentence" | "wordvec"
    encoder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",  # for sentence
    vectors_path: str | Path | None = None,         # for wordvec
) -> Dict:
    """
    Loads the saved sklearn model in `run_dir` and evaluates on test split.
    """
    run_dir = Path(run_dir)
    model_path = run_dir / "model.joblib"
    model = joblib.load(model_path)

    features_kind = (features_kind or "sentence").lower()
    if features_kind == "sentence":
        arrays = build_or_load_sentence_embeddings(
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            model_id=encoder_model_id,
            normalize=True,
            cache=True,
        )
    elif features_kind == "wordvec":
        if not vectors_path:
            raise ValueError("features_kind='wordvec' requires vectors_path.")
        arrays = build_or_load_wordvec_features(
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            vectors_path=vectors_path,
            cache=True,
        )
    else:
        raise ValueError(f"Unknown features_kind: {features_kind}")

    X_test, y_test = arrays["test"]

    # Start eval timer before prediction to capture full inference time
    t_eval0 = time.perf_counter()
    y_prob = _predict_proba_sklearn(model, X_test)

    art_dir = run_dir
    rid = run_id or art_dir.name
    metrics = _finalise_binary_eval(  
        y_true=y_test,  
        y_prob=y_prob,  
        threshold=threshold,  
        run_dir=art_dir,  
        experiments_csv=Path(experiments_csv),  
        run_id=rid,  
        extra_payload={"approach": "embeddings"},  
    )  
    upsert_row(Path(experiments_csv), rid, {"eval_time_s": float(time.perf_counter() - t_eval0)})
    logger.info("Embeddings test ROC-AUC: %.4f", metrics["roc_auc"])
    return metrics


# Transformers
def evaluate_transformer_run(
    run_dir: str | Path,
    manifests_dir: str | Path,
    processed_dir: str | Path,
    experiments_csv: str | Path,
    threshold: float = 0.5,
    run_id: Optional[str] = None,
    max_length: int = 256,
) -> Dict:
    """
    Loads HF model from `run_dir` and evaluates on test split.
    """
    run_dir = Path(run_dir)
    model = AutoModelForSequenceClassification.from_pretrained(run_dir)
    tok   = AutoTokenizer.from_pretrained(run_dir, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = build_or_load_tokenised(
        manifests_dir=manifests_dir,
        processed_dir=processed_dir,
        model_id=tok.name_or_path,
        max_length=max_length,
    )

    y_true: list[int] = []
    y_prob: list[float] = []
    # Start eval timer before model inference loop
    t_eval0 = time.perf_counter()

    test_ds = ds["test"] 
    bs = 64 
    for i in range(0, len(test_ds), bs): 
        batch = test_ds[i : i + bs]  
    
        enc = {}
        if "input_ids" in batch:
            enc["input_ids"] = torch.as_tensor(batch["input_ids"], device=device)
        if "attention_mask" in batch:
            enc["attention_mask"] = torch.as_tensor(batch["attention_mask"], device=device)
        if "token_type_ids" in batch: 
            enc["token_type_ids"] = torch.as_tensor(batch["token_type_ids"], device=device)
    
        with torch.no_grad():
            logits = model(**enc).logits 
            prob1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    
        y_true.extend(int(x) for x in batch["label"])
        y_prob.extend(prob1.tolist())

    art_dir = run_dir
    rid = run_id or art_dir.name
    metrics = _finalise_binary_eval( 
        y_true=y_true, 
        y_prob=y_prob, 
        threshold=threshold, 
        run_dir=art_dir, 
        experiments_csv=Path(experiments_csv), 
        run_id=rid, 
        extra_payload={"approach": "transformer", "run_dir": str(art_dir)}, 
    ) 
    # Ensure GPU kernels complete before stopping timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    upsert_row(Path(experiments_csv), rid, {"eval_time_s": float(time.perf_counter() - t_eval0)})
    logger.info("Transformer test ROC-AUC: %.4f", metrics["roc_auc"])
    return metrics


# Prompts 
def _load_texts_and_labels(manifest_path: Path) -> Tuple[list[str], list[int]]:
    """Read JSONL manifest and return parallel lists of texts and int labels."""
    rows = read_manifest(manifest_path)
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    return texts, labels


def evaluate_prompts_run(   
    manifests_dir: str | Path,  
    experiments_csv: str | Path,  
    model_id: str = "phi3:mini",  
    run_id: str | None = None,  
    threshold: float = 0.5,  
    prompt_type: str | None = None,  
    prompt_templates: dict | None = None,  
    limit: int | None = None,
) -> Dict:
    """Run prompts provider on test set (optionally stratified-limited) and score."""
    
    test_path = Path(manifests_dir) / "test.jsonl"
    texts, y_true = _load_texts_and_labels(test_path)
    # Optional stratified limit
    if limit is not None and isinstance(limit, int) and 0 < limit < len(texts):
        idx0 = [i for i, y in enumerate(y_true) if y == 0]
        idx1 = [i for i, y in enumerate(y_true) if y == 1]
        n0 = len(idx0); n1 = len(idx1); total = n0 + n1
        if total > 0:
            p0 = n0 / total
            k0 = int(round(limit * p0))
            k1 = max(0, limit - k0)
            rng = np.random.default_rng(1337)
            sel0 = rng.choice(idx0, size=min(k0, n0), replace=False).tolist()
            sel1 = rng.choice(idx1, size=min(k1, n1), replace=False).tolist()
            sel = sorted(sel0 + sel1)
            texts = [texts[i] for i in sel]
            y_true = [y_true[i] for i in sel]
        else:
            texts = texts[:limit]
            y_true = y_true[:limit]

    t0 = time.perf_counter()
    provider, subdir, extra = _make_prompt_provider(  
        model_id,   
    )   
    # Take template for Ollama from YAML
    template = None
    if prompt_templates and prompt_type and isinstance(prompt_templates, dict):
        template = prompt_templates.get(prompt_type)
    try:
        y_prob = provider.predict_batch(texts, template=template)
    except TypeError:
        y_prob = provider.predict_batch(texts)
    elapsed_s = float(time.perf_counter() - t0)

    art_dir = None
    rid = run_id or subdir  

    payload_extra = dict(extra)
    if prompt_type:
        payload_extra["prompt_type"] = prompt_type

    metrics = _finalise_binary_eval(   
        y_true=y_true,   
        y_prob=y_prob,   
        threshold=threshold,   
        run_dir=art_dir,   
        experiments_csv=Path(experiments_csv),   
        run_id=rid,   
        extra_payload=payload_extra,   
    )   
    upsert_row(Path(experiments_csv), rid, {"eval_time_s": elapsed_s})
    logger.info("Prompts [%s] test ROC-AUC: %.4f", subdir, metrics["roc_auc"])  
    return metrics  


def evaluate(   
    approach: str,  
    *,  
    run_dir: Optional[str | Path] = None,  
    manifests_dir: str | Path,  
    processed_dir: str | Path,  
    experiments_csv: str | Path,  
    threshold: float = 0.5,  
    run_id: Optional[str] = None,  
    features_kind: str = "sentence",        # "sentence" | "wordvec"  
    encoder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2", # for sentence  
    vectors_path: str | Path | None = None, # for wordvec  
    provider_model_id: str = "phi3:mini",   
    provider_prompt_type: Optional[str] = None,        # prompt template key  
    provider_prompt_templates: Optional[dict] = None,  # name->template  
    max_length: int = 256,   
    limit: Optional[int] = None,
) -> Dict:   
    """
    Unified dispatcher:
      - approach="embeddings": needs run_dir pointing to saved sklearn model folder.
      - approach="transformer": needs run_dir pointing to HF-saved model folder.
      - approach="prompts": re-runs provider on test (Ollama)
    """
    if approach == "embeddings":
        assert run_dir is not None, "embeddings: run_dir (saved sklearn model dir) is required"
        return evaluate_embeddings_run(
            run_dir=run_dir,
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            experiments_csv=experiments_csv,
            threshold=threshold,
            run_id=run_id,
            features_kind=features_kind,
            encoder_model_id=encoder_model_id,
            vectors_path=vectors_path,
        )

    elif approach == "transformer":
        assert run_dir is not None, "transformer: run_dir (saved HF model dir) is required"
        return evaluate_transformer_run(
            run_dir=run_dir,
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            experiments_csv=experiments_csv,
            threshold=threshold,
            run_id=run_id,
            max_length=max_length,
        )

    elif approach == "prompts":  
        return evaluate_prompts_run(   
            manifests_dir=manifests_dir,   
            experiments_csv=experiments_csv,   
            model_id=provider_model_id,   
            run_id=run_id,   
            threshold=threshold,   
            prompt_type=provider_prompt_type,   
            prompt_templates=provider_prompt_templates,   
            limit=limit,
        )   

    else:
        raise ValueError(f"Unknown approach: {approach}")
