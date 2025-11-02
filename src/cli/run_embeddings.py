from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.core.data_io import make_or_load_splits
from src.core.config import load_config
from src.core.utils import set_seed, init_logger, ensure_dir
from src.core.data_sig import manifest_signature
from src.eval.evaluate import evaluate
from src.models.classical import ClassicalConfig, train_from_arrays
from src.eval.experiments import append_row, new_run_row
from src.eval.plots import plot_confusion_from_json, plot_roc_from_json, plot_pr_from_json

logger = init_logger("run_embeddings")


def _load_features(cfg):
    """Build or load (X,y) arrays for train/val/test based on cfg.features.kind."""
    kind = cfg.get("features", "kind", default="sentence")
    processed_dir = cfg.get("paths", "processed_dir")
    manifests_dir = cfg.get("paths", "manifests_dir")

    if kind == "sentence":
        from src.features.embed_sentence import build_or_load_sentence_embeddings
        model_id   = cfg.get("features", "sentence_model", default="sentence-transformers/all-MiniLM-L6-v2")
        max_len    = int(cfg.get("features", "max_length", default=256))
        batch_size = int(cfg.get("features", "batch_size", default=64))
        lowercase  = bool(cfg.get("features", "lowercase", default=True))
        normalize  = bool(cfg.get("features", "normalize", default=True))
        arrays = build_or_load_sentence_embeddings(
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            model_id=model_id, max_length=max_len, batch_size=batch_size,
            normalize=normalize, lowercase=lowercase, cache=True,
        )
        feat_sig = f"sentence|{model_id}|maxlen={max_len}|norm={normalize}|lower={lowercase}"

    elif kind == "wordvec":
        from src.features.embed_wordvec import build_or_load_wordvec_features
        vectors_path = cfg.get("features", "vectors_path")
        if not vectors_path:
            raise ValueError("features.kind='wordvec' requires features.vectors_path in the config.")
        lowercase = bool(cfg.get("features", "lowercase", default=True))
        arrays = build_or_load_wordvec_features(
            manifests_dir=manifests_dir,
            processed_dir=processed_dir,
            vectors_path=vectors_path,
            lowercase=lowercase,
        )
        feat_sig = f"wordvec|{Path(vectors_path).name}|lower={lowercase}"

    else:
        raise ValueError(f"Unknown features.kind: {kind}")

    return arrays, feat_sig


def main():
    ap = argparse.ArgumentParser(description="Embeddings + classical pipeline (train -> eval -> interpret)")
    ap.add_argument("--cfg", type=str, default="configs/embeddings.yaml")
    ap.add_argument("--run-name", type=str, default=None, help="Optional run id prefix")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg.seed)

    # Paths
    data_raw        = cfg.get("paths", "data_raw")
    manifests_dir   = cfg.get("paths", "manifests_dir")
    processed_dir   = cfg.get("paths", "processed_dir")
    outputs_dir     = cfg.get("paths", "outputs_dir")
    feat_kind = cfg.get("features", "kind", default="sentence").lower()
    experiments_csv = Path(outputs_dir) / "experiments.csv"
    ensure_dir(outputs_dir)

    make_or_load_splits(
        raw_csv=data_raw,
        manif_dir=manifests_dir,
        test_size=float(cfg.get("data", "test_size", default=0.15)),
        val_size=float(cfg.get("data", "val_size", default=0.1)),
        stratify=bool(cfg.get("data", "stratify", default=True)),
    )

    # Signatures
    data_sig = manifest_signature(manifests_dir)["combo"]

    # Features
    arrays, feat_sig = _load_features(cfg)

    # Train
    if feat_kind == "sentence":
        run_id = args.run_name or datetime.now(timezone.utc).strftime("emb_sen_%Y%m%d-%H%M%S")
    else:
        run_id = args.run_name or datetime.now(timezone.utc).strftime("emb_wv_%Y%m%d-%H%M%S")
    model_cfg = ClassicalConfig(
        name=cfg.get("model", "name", default="logreg"),
        C=float(cfg.get("model", "C", default=1.0)),
        max_iter=int(cfg.get("model", "max_iter", default=200)),
        class_weight=cfg.get("model", "class_weight", default=None),
        n_jobs=int(cfg.get("train", "n_jobs", default=-1)),
        save_dir=cfg.get("train", "save_dir", default="models/embeddings"),
        artifacts_name=run_id,
        data_signature=data_sig,
        features_signature=feat_sig,
        seed=cfg.seed,
    )

    logger.info("Training %s on features: %s | data_sig=%s", model_cfg.name, feat_sig, data_sig)
    train_out = train_from_arrays(arrays=arrays, cfg=model_cfg, run_name=run_id)
    run_dir = Path(train_out["run_dir"])

    row = new_run_row(
        run_id=run_id,
        approach="embeddings",
        model=model_cfg.name,
        features_kind=feat_kind,
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
    
    eval_kwargs = {
        "features_kind": feat_kind,
    }
    
    if feat_kind == "sentence":
        eval_kwargs.update({
            "encoder_model_id": cfg.get("features", "sentence_model", default="sentence-transformers/all-MiniLM-L6-v2"),
        })
    elif feat_kind == "wordvec":
        eval_kwargs.update({
            "vectors_path": cfg.get("features", "vectors_path"),
        })
    else:
        raise ValueError(f"Unknown features.kind: {feat_kind}")
    
    # Evaluate on TEST
    logger.info("Evaluating on test split…")
    _ = evaluate(
        "embeddings",
        run_dir=run_dir,
        manifests_dir=manifests_dir,
        processed_dir=processed_dir,
        experiments_csv=experiments_csv,
        **eval_kwargs,
    )
    
    # Plots
    fig_dir = ensure_dir(Path(outputs_dir) / "figures" / run_id)
    metrics_json = Path(outputs_dir) / "metrics" / f"{run_id}.test.json"
    plot_confusion_from_json(metrics_json, out_path=fig_dir / "cm.png")
    plot_roc_from_json(metrics_json, out_path=fig_dir / "roc.png")
    plot_pr_from_json(metrics_json, out_path=fig_dir / "pr.png")

    print(f"[OK] {run_id} | trained and evaluated, plots saved")


if __name__ == "__main__":
    main()
