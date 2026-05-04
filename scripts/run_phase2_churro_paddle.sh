#!/usr/bin/env bash
# Run Phase 2 CHURRO-3B baseline using the dedicated venv (PaddleOCR-VL not in default Phase 2 list).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CHURRO_PADDLE_VENV:-$ROOT/.venv-churro-paddle}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing venv at $VENV — run: bash scripts/setup_churro_paddle_venv.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Prefer venv churro-ocr on PATH
export PATH="$VENV/bin:$PATH"
# Ensure Hugging Face runtime is registered (official Churro quick start; ~2s when already done).
export CHURRO_TORCH_BACKEND="${CHURRO_TORCH_BACKEND:-cu124}"
churro-ocr install hf --torch-backend "$CHURRO_TORCH_BACKEND"

exec python "$ROOT/scripts/run_phase2_baselines.py" --config "$ROOT/configs/phase2/with_churro.yaml" "$@"
