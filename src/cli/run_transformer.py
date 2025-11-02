from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import torch


from src.core.data_io import make_or_load_splits
from src.core.config import load_config
from src.core.utils import set_seed, init_logger, ensure_dir
from src.core.data_sig import manifest_signature
from src.eval.evaluate import evaluate
from src.models.transformers import TransformerConfig, finetune
from src.eval.experiments import new_run_row, append_row
from src.eval.plots import plot_confusion_from_json, plot_roc_from_json, plot_pr_from_json

logger = init_logger("run_transformer")


def main():
    """Run transformer pipeline (train -> eval -> interpret)"""
    ap = argparse.ArgumentParser(description="Transformer pipeline (train -> eval -> interpret)")
    ap.add_argument("--cfg", type=str, default="configs/transformer.yaml")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg.seed)
    print(f"CUDA available: {torch.cuda.is_available()}")    
      
    # Paths
    data_raw        = cfg.get("paths", "data_raw")
    manifests_dir   = cfg.get("paths", "manifests_dir")
    processed_dir   = cfg.get("paths", "processed_dir")
    outputs_dir     = cfg.get("paths", "outputs_dir", default="outputs")
    experiments_csv = Path(outputs_dir) / "experiments.csv"
    ensure_dir(outputs_dir)

    make_or_load_splits(
        raw_csv=data_raw,
        manif_dir=manifests_dir,
        test_size=float(cfg.get("data", "test_size", default=0.15)),
        val_size=float(cfg.get("data", "val_size", default=0.1)),
        stratify=bool(cfg.get("data", "stratify", default=True)),
    )
    
    data_sig = manifest_signature(manifests_dir)["combo"]
    run_id = args.run_name or datetime.now(timezone.utc).strftime("trans_%Y%m%d-%H%M%S")

    # Train
    tcfg = TransformerConfig(
        model_id=cfg.get("params", "model_id", default="distilbert-base-uncased"),
        max_length=int(cfg.get("params", "max_length", default=256)),
        batch_size=int(cfg.get("params", "batch_size", default=16)),
        lr=float(cfg.get("params", "lr", default=2e-5)),
        weight_decay=float(cfg.get("params", "weight_decay", default=0.01)),
        warmup_ratio=float(cfg.get("params", "warmup_ratio", default=0.1)),
        epochs=int(cfg.get("params", "epochs", default=2)),
        gradient_accumulation_steps=int(cfg.get("params", "gradient_accumulation_steps", default=1)),
        fp16=bool(cfg.get("params", "fp16", default=True)),
        eval_steps=int(cfg.get("params", "eval_steps", default=200)),
        seed=cfg.seed,
        save_dir=cfg.get("params", "save_dir", default="models/transformer"),
        artifacts_name=run_id,
        data_signature=data_sig,
    )

    logger.info("Finetuning %s | data_sig=%s", tcfg.model_id, data_sig)
    train_out = finetune(manifests_dir=manifests_dir, processed_dir=processed_dir, cfg=tcfg)
    run_dir = Path(train_out["run_dir"])

    # Create CSV stub
    row = new_run_row(
        run_id=run_id,
        approach="transformer",
        model=tcfg.model_id,
        data_sig=data_sig,
        seed=cfg.seed,
        train_time_s=train_out["timings"].get("train_time_s", ""),
        eval_time_s="",
    )
    # add train/val into the same row before writing
    row["roc_auc_train"] = train_out["metrics_train"].get("roc_auc", "")
    row["pr_auc_train"]  = train_out["metrics_train"].get("pr_auc", "")
    row["roc_auc_val"]   = train_out["metrics_val"].get("roc_auc", "")
    row["pr_auc_val"]    = train_out["metrics_val"].get("pr_auc", "")
    
    append_row(experiments_csv, row)
    logger.info("Wrote stub row to %s", experiments_csv)

    # Evaluate on TEST
    logger.info("Evaluating on test split…")
    _ = evaluate(
        "transformer",
        run_dir=run_dir,
        manifests_dir=manifests_dir,
        processed_dir=processed_dir,
        experiments_csv=experiments_csv,
        threshold=float(cfg.get("evaluation", "predict_threshold", default=0.5)),
        run_id=run_id,
        max_length=int(cfg.get("transformer", "max_length", default=256)),
    )

    # Plots
    fig_dir = ensure_dir(Path(outputs_dir) / "figures" / run_id)
    metrics_json = Path(outputs_dir) / "metrics" / f"{run_id}.test.json"
    plot_confusion_from_json(metrics_json, out_path=fig_dir / "cm.png")
    plot_roc_from_json(metrics_json, out_path=fig_dir / "roc.png")
    plot_pr_from_json(metrics_json, out_path=fig_dir / "pr.png")

    print(f"[OK] {run_id} | finetuned and evaluated, plots saved")


if __name__ == "__main__":
    main()
