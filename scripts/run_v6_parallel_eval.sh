#!/usr/bin/env bash
# Parallel re-evaluation of the 4 phase4_v6 cells across the 4 A100 GPUs.
# Each cell gets one GPU; on each GPU we evaluate the best checkpoint on
# val_birch (n=117) and test_birch (n=252) with beam=4.
#
# After all four finish we run scripts/aggregate_phase4_v6.py with --skip-test
# (it'll just collect the already-written summaries from reports/eval/).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p reports/eval

VENV="${VENV:-.venv-qwen-edit-multi}"
RUNS_DIR="runs/phase4_v6"
NUM_BEAMS=4
MAX_NEW_TOKENS=160

# Pair each cell with a GPU
declare -A CELL_GPU=(
  [mixed_80_20_seed1337_r16]=0
  [mixed_80_20_seed1337_r64]=1
  [mixed_80_20_seed2026_r32]=2
  [mixed_80_20_seed4242_r32]=3
)

# Find the checkpoint with the lowest eval_gen_cer
best_ckpt() {
  local run="$1"
  "${VENV}/bin/python" - <<PY
import json, sys
from pathlib import Path
run = Path("${run}")
best_cer = float("inf")
best_path = None
for ck in sorted(run.glob("checkpoint-*")):
    ts_file = ck / "trainer_state.json"
    if not ts_file.exists():
        continue
    ts = json.loads(ts_file.read_text())
    for entry in ts.get("log_history", []):
        cer = entry.get("eval_gen_cer")
        if cer is None:
            continue
        if cer < best_cer:
            best_cer = cer
            step = entry.get("step")
            cand = run / f"checkpoint-{step}"
            if cand.exists():
                best_path = cand
    bm = ts.get("best_model_checkpoint")
    if best_path is None and bm and Path(bm).exists():
        best_path = Path(bm)
        best_cer = float(ts.get("best_metric", best_cer))
print(f"{best_path}\t{best_cer:.4f}")
PY
}

PIDS=()
for cell in "${!CELL_GPU[@]}"; do
  gpu="${CELL_GPU[$cell]}"
  run="${RUNS_DIR}/${cell}"
  read -r ckpt cer < <(best_ckpt "$run")
  echo "[$cell] best ckpt: $ckpt (in-loop CER=$cer)  GPU $gpu"

  log="${run}/eval_v6_beam${NUM_BEAMS}.log"
  (
    set -e
    export CUDA_VISIBLE_DEVICES="$gpu"

    "${VENV}/bin/python" -u scripts/eval_qwen_vl_lora.py \
      --base-model Qwen/Qwen3.5-2B \
      --adapter   "$ckpt" \
      --jsonl     data/splits/phase4_v3/val.jsonl \
      --image-root . \
      --out-pred  "reports/eval/v6_${cell}_val_beam${NUM_BEAMS}.jsonl" \
      --out-summary "reports/eval/v6_${cell}_val_beam${NUM_BEAMS}.json" \
      --device cuda:0 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --num-beams "$NUM_BEAMS"

    "${VENV}/bin/python" -u scripts/eval_qwen_vl_lora.py \
      --base-model Qwen/Qwen3.5-2B \
      --adapter   "$ckpt" \
      --jsonl     data/interim/birchbark_test.jsonl \
      --image-root . \
      --out-pred  "reports/eval/v6_${cell}_test_beam${NUM_BEAMS}.jsonl" \
      --out-summary "reports/eval/v6_${cell}_test_beam${NUM_BEAMS}.json" \
      --device cuda:0 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --num-beams "$NUM_BEAMS"
  ) >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "  -> launched pid=$pid log=$log"
done

echo "Waiting for: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "All v6 evals finished."
