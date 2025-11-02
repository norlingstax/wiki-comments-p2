#!/usr/bin/env bash
set -euo pipefail

# repo root
cd "$(dirname "$0")/.."

# keep window open on success or error 
trap 'code=$?; echo; read -r -p "Exit code $code"; exit $code' EXIT

# options
CFG="configs/transformer.yaml"
RUN_NAME=""

usage() {
  cat <<EOF
Usage: scripts/run_transformer.sh [--cfg PATH] [--run-name STRING]

Defaults:
  --cfg       $CFG
Notes:
  - Params (model kind, paths, etc.) come from the YAML config.
  - Use --run-name to tag the run_id (otherwise a timestamp is used).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg)
      CFG="$2"; shift 2;;
    --run-name)
      RUN_NAME="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# pick venv python
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
else
  echo "[run] No venv found. Create it first with: scripts/setup.sh"
  exit 1
fi

# run
echo "[run] using config: $CFG"
if [[ -n "$RUN_NAME" ]]; then
  "$PY" -m src.cli.run_transformer --cfg "$CFG" --run-name "$RUN_NAME"
else
  "$PY" -m src.cli.run_transformer --cfg "$CFG"
fi
