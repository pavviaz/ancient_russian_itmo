#!/usr/bin/env bash
# 1) Run CHURRO-3B fair-postproc on test_birch (~ 2.5-3 h on a single A100).
# 2) Then launch the r=64 LoRA cell (~ 5 h).
set -uo pipefail
cd "$(dirname "$0")/.."

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >&2; }

log "1) running CHURRO-3B (GPU 0)..."
mkdir -p reports/eval
CUDA_VISIBLE_DEVICES=0 \
    OUT_PRED=reports/eval/test_predictions_churro.jsonl \
    OUT_SUMMARY=reports/eval/test_summary_churro.json \
    bash scripts/run_churro_test_birch.sh > /tmp/churro_test.log 2>&1
RET=$?
log "   CHURRO finished (exit=$RET); tail:"
tail -15 /tmp/churro_test.log >&2

log "2) launching r=64 cell on GPU 0..."
PYTHON=.venv-qwen-edit-multi/bin/python
SPLITS_DIR=data/splits/phase4_v3
RUNS_DIR=runs/phase4_v6
out="${RUNS_DIR}/mixed_80_20_seed1337_r64"
mkdir -p "${out}"
CUDA_VISIBLE_DEVICES=0 nohup ${PYTHON} -u scripts/train_qwen_vl_lora.py \
    --model Qwen/Qwen3.5-2B \
    --train-jsonl "${SPLITS_DIR}/mixed_80_20_train.jsonl" \
    --val-jsonl   "${SPLITS_DIR}/val.jsonl" \
    --output-dir  "${out}" \
    --epochs 5 \
    --per-device-batch-size 4 --grad-accum 4 \
    --learning-rate 1e-4 --warmup-ratio 0.05 --max-grad-norm 0.5 \
    --eval-steps 100 --save-steps 100 --logging-steps 20 \
    --early-stopping-patience 5 --save-total-limit 2 \
    --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \
    --seed 1337 --num-workers 2 \
    > "${out}/train.log" 2>&1 &
RPID=$!
echo "${RPID}" > "${out}/.pid"
log "   r=64 launched, pid=${RPID}, log=${out}/train.log"
log "chain done; r=64 continues independently"
