#!/usr/bin/env bash
# phase4_v4_status.sh — snapshot of v4 grid: chain log + per-cell summary
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS_DIR=${RUNS_DIR:-runs/phase4_v5}

echo "=== launcher log ==="
[[ -f "${RUNS_DIR}/_launcher.log" ]] && tail -20 "${RUNS_DIR}/_launcher.log" || echo "(no launcher log)"

echo
echo "=== gpu utilisation ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "=== running training pids (one parent per cell) ==="
# group by --output-dir, keep only the smallest pid per cell (the parent)
ps -ef | grep -E 'train_qwen_vl_lora' | grep -v grep \
    | awk '{
        pid=$2; ppid=$3;
        out="?";
        for(i=1;i<=NF;i++) if($i=="--output-dir"){out=$(i+1); break}
        print pid, ppid, out
    }' | sort -k3,3 -k1,1n \
       | awk '{ if($3 != prev){ printf "  pid=%-8s ppid=%-8s cell=%s\n", $1, $2, $3; prev=$3 } }' || true

echo
echo "=== per-cell summary ==="
.venv-qwen-edit-multi/bin/python << 'EOF'
import json, glob, os, subprocess
RUNS_DIR = os.environ.get("RUNS_DIR", "runs/phase4_v4")
rows = []
for d in sorted(glob.glob(f"{RUNS_DIR}/*/")):
    cell = d.rstrip("/").split("/")[-1]
    if cell.startswith("_"):
        continue
    pid = ""
    pid_file = os.path.join(d, ".pid")
    if os.path.exists(pid_file):
        pid = open(pid_file).read().strip()
    running = "DONE"
    if pid and os.path.exists(f"/proc/{pid}"):
        running = "RUNNING"
    states = sorted(glob.glob(f"{d}/checkpoint-*/trainer_state.json"))
    best_cer = best_step = "-"
    last_cer = last_nls = "-"
    last_step = "-"
    if states:
        try:
            state = json.load(open(states[-1]))
            best_cer = f"{state.get('best_metric'):.3f}" if state.get("best_metric") else "-"
            best_step = state.get("best_global_step", "-")
            last_step = state.get("global_step", "-")
            finals = [e for e in state.get("log_history", []) if "eval_gen_cer" in e]
            if finals:
                last_cer = f"{finals[-1]['eval_gen_cer']:.3f}"
                last_nls = f"{finals[-1]['eval_gen_nls']:.3f}"
        except Exception:
            pass
    rows.append((cell, running, str(last_step), str(best_step), best_cer, last_cer, last_nls))

print(f"{'cell':<45s} {'state':<8s} {'step':>6s} {'best':>6s} {'best_cer':>9s} {'last_cer':>9s} {'last_nls':>9s}")
for r in rows:
    print(f"{r[0]:<45s} {r[1]:<8s} {r[2]:>6s} {r[3]:>6s} {r[4]:>9s} {r[5]:>9s} {r[6]:>9s}")
EOF
