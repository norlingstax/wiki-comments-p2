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
from src.eval.experiments import append_row, new_run_row
from src.eval.plots import plot_confusion_from_json, plot_roc_from_json, plot_pr_from_json

logger = init_logger("run_prompts")


def main():
    """Run prompts evaluation with Ollama using config + CLI args."""
    ap = argparse.ArgumentParser(description="Prompts pipeline (eval -> interpret)")
    ap.add_argument("--cfg", type=str, default="configs/prompts.yaml")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--prompt-type", type=str, default=None, help="Prompt type: zero_shot|few_shot_4|role")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of test comments (stratified)")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg.seed)
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Paths
    data_raw        = cfg.get("paths", "data_raw")
    manifests_dir   = cfg.get("paths", "manifests_dir")
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

    # Provider selection (prompts use Ollama)
    provider_name    = cfg.get("provider", "name", default="ollama")
    provider_model   = cfg.get("provider", "model_id", default="phi3:mini")
    provider_bs      = int(cfg.get("provider", "batch_size", default=16))
    provider_max_len = cfg.get("provider", "max_length", default=None)
    if provider_max_len is not None:
        try:
            provider_max_len = int(provider_max_len)
        except Exception:
            provider_max_len = None
    provider_max_new = cfg.get("provider", "max_new_tokens", default=None)
    if provider_max_new is not None:
        try:
            provider_max_new = int(provider_max_new)
        except Exception:
            provider_max_new = None
    prompt_type_arg  = args.prompt_type
    # Load prompt templates from YAML into a dict[name->template]
    tpl_variants = cfg.get("templates", "variants", default=None)
    prompt_templates = None
    if tpl_variants:
        try:
            prompt_templates = {item["name"]: item["template"] for item in tpl_variants}
        except Exception:
            prompt_templates = None

    run_id = args.run_name or datetime.now(timezone.utc).strftime("prompt_%Y%m%d-%H%M%S")

    # Create CSV stub row
    row = new_run_row( 
        run_id=run_id, 
        approach="prompts", 
        model=provider_model, 
        provider=provider_name, 
        prompt_type=(prompt_type_arg or ""), 
        data_sig=data_sig, 
        seed=cfg.seed, 
    ) 
    append_row(experiments_csv, row)

    # Evaluate on TEST
    logger.info("Evaluating prompts on test split (provider: %s, model: %s)…", provider_name, provider_model)
    _ = evaluate(
        "prompts",
        manifests_dir=manifests_dir,
        processed_dir=cfg.get("paths", "processed_dir", default="data/processed"),
        experiments_csv=experiments_csv,
        threshold=float(cfg.get("evaluation", "predict_threshold", default=0.5)),
        run_id=run_id,
        provider_model_id=provider_model,
        provider_prompt_type=prompt_type_arg,
        provider_prompt_templates=prompt_templates,
        max_length=provider_max_len,
        limit=args.limit,
    )

    # Plots 
    fig_dir = ensure_dir(Path(outputs_dir) / "figures" / run_id)
    metrics_json = Path(outputs_dir) / "metrics" / f"{run_id}.test.json"
    plot_confusion_from_json(metrics_json, out_path=fig_dir / "cm.png")
    plot_roc_from_json(metrics_json, out_path=fig_dir / "roc.png")
    plot_pr_from_json(metrics_json, out_path=fig_dir / "pr.png")

    print(f"[OK] {run_id} | prompts evaluated, plots saved")


if __name__ == "__main__":
    main()
