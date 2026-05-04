#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHON=".venv-churro-paddle/bin/python"
ACCELERATE=".venv-churro-paddle/bin/accelerate"
TRAIN_SCRIPT="scripts/external/train_text_to_image_lora_sdxl.py"

RUN_DIR="runs/phase3_sdxl_lora_birchbark_v1_20260503"
mkdir -p "$RUN_DIR"

"$PYTHON" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
git rev-parse HEAD > "$RUN_DIR/git_sha.txt" 2>/dev/null || true
nvidia-smi > "$RUN_DIR/nvidia_smi_start.txt" || true
cp configs/phase3/sdxl_lora_birchbark.yaml "$RUN_DIR/resolved_config.yaml"

"$ACCELERATE" launch \
  --mixed_precision bf16 \
  "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --variant fp16 \
  --train_data_dir "data/synthetic/sdxl_lora_gramoty_train" \
  --image_column image \
  --caption_column text \
  --resolution 1024 \
  --center_crop \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --max_train_steps 2 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 4 \
  --dataloader_num_workers 2 \
  --use_8bit_adam \
  --allow_tf32 \
  --mixed_precision bf16 \
  --rank 16 \
  --seed 1337 \
  --report_to tensorboard \
  --validation_prompt "<birchbark> Old Russian inscription scratched into birch bark, grey brown fibrous wooden surface, shallow dark incised Cyrillic letters, medieval Novgorod archaeological document photo" \
  --num_validation_images 2 \
  --validation_epochs 1 \
  --output_dir "$RUN_DIR"
