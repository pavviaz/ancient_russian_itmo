#!/usr/bin/env bash
# launch_phase4_v6.sh - multi-seed variance + LoRA-rank ablation
#
# All cells run mixed_80_20 5-ep with the v3 broad-target LoRA (r=32 unless noted).
# Wave 0 (3 cells, GPUs 1,2,3, ~5 h each):
#   GPU 1: seed=2026 r=32  (variance)
#   GPU 2: seed=4242 r=32  (variance)
#   GPU 3: seed=1337 r=16  (LoRA-rank A4)
# Wave 1 (1 cell, GPU 0, ~5 h, kicks in once test-beam4 finishes):
#   GPU 0: seed=1337 r=64  (LoRA-rank A4)
#
# After wave 0+1 we have 3 seeds for r=32 (existing 1337 + 2026 + 4242)
# and 3 ranks for seed=1337 (16, 32, 64).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv-qwen-edit-multi/bin/python}
SPLITS_DIR=${SPLITS_DIR:-data/splits/phase4_v3}
RUNS_DIR=${RUNS_DIR:-runs/phase4_v6}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
mkdir -p "${RUNS_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >&2; }

LAUNCHED_PID=
launch_cell () {
    # usage: launch_cell GPU TAG SEED LORA_R [extra...]
    local gpu="$1" tag="$2" seed="$3" lora_r="$4"
    shift 4
    local out="${RUNS_DIR}/${tag}"
    mkdir -p "${out}"
    local logfile="${out}/train.log"

    local cmd=(
        ${PYTHON} -u scripts/train_qwen_vl_lora.py
        --model "${MODEL}"
        --train-jsonl "${SPLITS_DIR}/mixed_80_20_train.jsonl"
        --val-jsonl   "${SPLITS_DIR}/val.jsonl"
        --output-dir  "${out}"
        --epochs 5
        --per-device-batch-size 4
        --grad-accum 4
        --learning-rate 1e-4
        --warmup-ratio 0.05
        --max-grad-norm 0.5
        --eval-steps 100
        --save-steps 100
        --logging-steps 20
        --early-stopping-patience 5
        --save-total-limit 2
        --lora-r "${lora_r}"
        --lora-alpha "$(( lora_r * 2 ))"
        --lora-dropout 0.05
        --seed "${seed}"
        --num-workers 2
        "$@"
    )
    log "[gpu ${gpu}] launch ${tag}  out=${out}  seed=${seed}  r=${lora_r}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        nohup "${cmd[@]}" > "${logfile}" 2>&1 &
    LAUNCHED_PID=$!
    echo "${LAUNCHED_PID}" > "${out}/.pid"
    log "    pid=${LAUNCHED_PID}  log=${logfile}"
}

# -------- Wave 0 on GPUs 1,2,3 --------
log "=== Wave 0: launching 3 cells on GPUs 1,2,3 ==="
launch_cell 1 "mixed_80_20_seed2026_r32" 2026 32
PID1=$LAUNCHED_PID
launch_cell 2 "mixed_80_20_seed4242_r32" 4242 32
PID2=$LAUNCHED_PID
launch_cell 3 "mixed_80_20_seed1337_r16" 1337 16
PID3=$LAUNCHED_PID
log "wave 0 PIDs: 1=${PID1} 2=${PID2} 3=${PID3}"

# -------- Wait for GPU 0 to be free (test-beam4 finishes ~10min) --------
log "waiting for GPU 0 to be free (test-beam4 inference)..."
while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' 'NR==1 {print $2}' | grep -qE "[0-9]{4,}"; do
    sleep 30
done
log "GPU 0 is free"

# -------- Wave 1 on GPU 0 --------
log "=== Wave 1: launching r=64 cell on GPU 0 ==="
launch_cell 0 "mixed_80_20_seed1337_r64" 1337 64
PID0=$LAUNCHED_PID
log "wave 1 pid: 0=${PID0}"

# -------- Wait for everything --------
for p in "${PID1}" "${PID2}" "${PID3}" "${PID0}"; do
    log "waiting for pid=${p}..."
    wait "${p}" 2>/dev/null || true
    log "  pid=${p} done"
done

log "=== all v6 cells finished ==="
