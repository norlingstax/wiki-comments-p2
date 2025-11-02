#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# keep window open on success or error 
trap 'code=$?; echo; read -r -p "Exit code $code"; exit $code' EXIT

# Defaults
CSV="data/raw/wiki_comments.csv"
OUT="data/manifests"
TEST_SIZE="0.15"
VAL_SIZE="0.1"
STRATIFY="true"
SEED="1337"

usage() {
  cat <<EOF
Create (or load) dataset splits as JSONL manifests.

Usage:
  scripts/build_manifests.sh [--csv PATH] [--out DIR] [--test-size FLOAT] [--val-size FLOAT]
                             [--no-stratify] [--seed INT]

Defaults:
  --csv        $CSV
  --out        $OUT
  --test-size  $TEST_SIZE
  --val-size   $VAL_SIZE
  --seed       $SEED
  (stratified by default; use --no-stratify to disable)
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)        CSV="$2"; shift 2;;
    --out)        OUT="$2"; shift 2;;
    --test-size)  TEST_SIZE="$2"; shift 2;;
    --val-size)   VAL_SIZE="$2"; shift 2;;
    --no-stratify) STRATIFY="false"; shift 1;;
    --seed)       SEED="$2"; shift 2;;
    -h|--help)    usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Pick venv python
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
else
  echo "[build] No venv found. Create it first with: scripts/setup.sh"
  exit 1
fi

echo "[build] CSV: $CSV"
echo "[build] OUT: $OUT"
echo "[build] test_size: $TEST_SIZE  val_size: $VAL_SIZE  stratify: $STRATIFY  seed: $SEED"

# Run inline Python that calls make_or_load_splits
"$PY" - <<PYCODE
from pathlib import Path
import sys
from src.core.data_io import make_or_load_splits

csv = Path(r"${CSV}")
out = Path(r"${OUT}")
test_size = float("${TEST_SIZE}")
val_size  = float("${VAL_SIZE}")
seed      = int("${SEED}")
stratify  = "${STRATIFY}".lower() == "true"

paths = make_or_load_splits(
    raw_csv=csv,
    manif_dir=out,
    test_size=test_size,
    val_size=val_size,
    stratify=stratify,
    random_state=seed,
)

print("[build] Manifests ready:")
print("  train:", paths.train_path)
print("  val:  ", paths.val_path)
print("  test: ", paths.test_path)
PYCODE

echo "[build] done."
