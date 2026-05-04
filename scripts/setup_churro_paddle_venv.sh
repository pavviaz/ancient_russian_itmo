#!/usr/bin/env bash
# Create Python 3.12 venv for CHURRO-3B + PaddleOCR-VL (not supported on Python 3.14).
# Requires: wget or curl, ~4 GB disk for torch + models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${CHURRO_PADDLE_VENV:-$ROOT/.venv-churro-paddle}"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv to ~/.local/bin …"
  if command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
fi

echo "Installing managed CPython 3.12 (uv) …"
uv python install 3.12

echo "Creating venv at $VENV …"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

echo "Installing PyTorch (CUDA 12.4 wheels) …"
uv pip install --python "$PY" "torch>=2.5,<2.7" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

echo "Installing transformers + project …"
uv pip install --python "$PY" "transformers>=4.50" accelerate sentencepiece einops \
  pillow pyyaml omegaconf tqdm orjson jiwer httpx beautifulsoup4 lxml matplotlib pytesseract
uv pip install --python "$PY" -e "$ROOT"

echo "Installing CHURRO (PyPI) + Hugging Face runtime (official quick start) …"
uv pip install --python "$PY" 'churro-ocr[hf]'
# Required per https://github.com/stanford-oval/Churro — installs/pins HF stack via uv in this venv.
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
churro-ocr install hf --torch-backend cu124

echo "Installing Paddle (CPU wheels from PyPI) + OCR extras …"
uv pip install --python "$PY" paddlepaddle paddleocr
uv pip install --python "$PY" "paddlex[ocr]==3.5.1"

# Optional: GPU Paddle (China mirror). Uncomment if PyPI CPU build is too slow and you have CUDA 12.x:
# uv pip install --python "$PY" paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "Optional: EasyOCR (single-venv full Phase 2) …"
uv pip install --python "$PY" easyocr || true

uv pip freeze --python "$PY" > "$ROOT/envs/freeze_churro_paddle_venv.txt"
echo "Done. Activate: source \"$VENV/bin/activate\""
echo "Run CHURRO baseline: bash \"$ROOT/scripts/run_phase2_churro_paddle.sh\""
