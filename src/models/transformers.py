from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from src.core.utils import Timer, init_logger, ensure_dir, set_seed
from src.models.model_utils import build_training_args, build_or_load_tokenised, system_info
from src.eval.metrics import compute_binary_metrics, save_json

logger = init_logger("transformers")

# Config 
@dataclass
class TransformerConfig:
    model_id: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 2
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    eval_steps: Optional[int] = 200
    seed: int = 42

    # io
    save_dir: str = "models/transformer"
    artifacts_name: str = "transformer_run"

    # provenance
    data_signature: Optional[str] = None
    tokenized_signature: Optional[str] = None

# Training
def finetune(
    manifests_dir: str | Path,
    processed_dir: str | Path,
    cfg: TransformerConfig,
) -> Dict:
    """
    Fine-tune and save model + training summary.
    """
    use_cuda = torch.cuda.is_available()
    logger.info("Device: %s | CUDA: %s",
                "cuda" if use_cuda else "cpu",
                torch.version.cuda if use_cuda else "n/a")
    
    set_seed(cfg.seed)

    # Tokenised dataset
    ds = build_or_load_tokenised(
        manifests_dir=manifests_dir,
        processed_dir=processed_dir,
        model_id=cfg.model_id,
        max_length=cfg.max_length,
    )

    # Model and tokeniser
    num_labels = 2
    config = AutoConfig.from_pretrained(
        cfg.model_id,
        num_labels=num_labels,
        id2label={0: "non-toxic", 1: "toxic"},
        label2id={"non-toxic": 0, "toxic": 1},
        problem_type="single_label_classification",
    )
    # ignore_mismatched_sizes=True to load checkpoints with a different head size
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    # Trainer
    out_dir = Path(cfg.save_dir) / cfg.artifacts_name
    tb = build_training_args(
        output_dir=out_dir,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_steps=cfg.eval_steps,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=tb.args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        compute_metrics=None,
        callbacks=tb.callbacks,
    )

    timings: Dict[str, float] = {}
    with Timer() as t_train:
        trainer.train()
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
    timings["train_time_s"] = t_train.elapsed or 0.0

    # evaluate TRAIN and VAL
    def _softmax_prob1(logits: np.ndarray) -> np.ndarray:
        # logits: (N, 2)
        x = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(x)
        return (e[:, 1] / e.sum(axis=1)).astype(np.float32)

    # TRAIN
    pred_train = trainer.predict(ds["train"])  # returns (predictions, label_ids, metrics)
    train_prob = _softmax_prob1(np.array(pred_train.predictions))
    train_labels = np.array(pred_train.label_ids, dtype=int)
    train_metrics = compute_binary_metrics(y_true=train_labels, y_prob=train_prob, threshold=0.5)
    save_json(train_metrics, out_dir / "metrics.train.json")

    # VAL
    pred_val = trainer.predict(ds["val"])
    val_prob = _softmax_prob1(np.array(pred_val.predictions))
    val_labels = np.array(pred_val.label_ids, dtype=int)
    val_metrics = compute_binary_metrics(y_true=val_labels, y_prob=val_prob, threshold=0.5)
    save_json(val_metrics, out_dir / "metrics.val.json")

    summary = {
        "config": asdict(cfg),
        "system": system_info(),
        "timings": timings,
        "approach": "transformer",
        "run_dir": str(out_dir),
    }
    ensure_dir(out_dir)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved model + summary -> %s", out_dir)

    return {
        "run_dir": str(out_dir),
        "summary_path": str(summary_path),
        "timings": timings,
        "metrics_train": train_metrics,
        "metrics_val": val_metrics,
    }
