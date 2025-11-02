from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer
from datasets import Dataset, DatasetDict

from src.core.data_io import read_manifest
from src.core.cache import cache_dir, save_json
from src.core.data_sig import manifest_signature
from src.core.utils import ensure_dir

# Training arguments / callbacks
@dataclass
class TrainerBuild:
    args: TrainingArguments
    callbacks: list

def build_training_args(
    output_dir: str | Path,
    lr: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 2,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    eval_steps: Optional[int] = None, 
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
) -> TrainerBuild:
    output_dir = str(ensure_dir(output_dir))
    eval_strategy = "steps" if eval_steps else "epoch"

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy=eval_strategy,
        logging_strategy=eval_strategy,
        save_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=True if eval_strategy == "steps" else False, 
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        report_to=[],
        save_safetensors=False,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)]
    return TrainerBuild(args=args, callbacks=callbacks)

# Tokenised dataset cache
def build_or_load_tokenised(
    manifests_dir: str | Path,
    processed_dir: str | Path,
    model_id: str,
    max_length: int = 256,
) -> DatasetDict:
    """
    Turn jsonl manifests (train/val/test) into a tokenized HF DatasetDict and cache to disk.
    Cache key includes: model_id, max_length, and combined manifest signature.
    """
    manif_sig = manifest_signature(manifests_dir)["combo"]
    params = dict(model=model_id, maxlen=max_length, data=manif_sig, ver=1)
    cdir = cache_dir(processed_dir, "tokenized", params)
    meta_path = Path(cdir) / "meta.json"

    if Path(cdir).exists() and meta_path.exists():
        return DatasetDict.load_from_disk(str(cdir))

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def load_split(name: str):
        items = read_manifest(Path(manifests_dir) / f"{name}.jsonl")
        texts  = [it["text"] for it in items]
        labels = [int(it["label"]) for it in items]
        return Dataset.from_dict({"text": texts, "label": labels})

    raw = DatasetDict({
        "train": load_split("train"),
        "val":   load_split("val"),
        "test":  load_split("test"),
    })

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["label"] = batch["label"]
        return enc

    ds = raw.map(tok, batched=True, remove_columns=["text"])
    try:
        ds = ds.with_format("torch")
    except Exception:
        pass

    ensure_dir(cdir)
    ds.save_to_disk(str(cdir))
    sizes = {k: len(v) for k, v in ds.items()}
    save_json({"params": params, "sizes": sizes}, meta_path)
    return ds


# System info
def system_info() -> Dict[str, str]:
    """System info for provenance."""
    info = {"device": "cpu", "gpu": "none"}
    try:
        import torch
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if info["device"] == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
