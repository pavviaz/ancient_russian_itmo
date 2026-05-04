# Birchbark OCR (research pipeline)

Experimental programme for Novgorod birchbark text recognition, following `agent_protocol.md`.

## Layout

- `src/birchbark_ocr/` — library code (data, eval, train, synth)
- `scripts/` — CLI entry points
- `configs/` — Hydra configs
- `data/raw/` — downloads (gitignored images); manifests committed where small
- `data/splits/` — frozen train/val/test document IDs + hashes
- `reports/` — phase reports

## Phase 2 — CHURRO-3B (Python 3.12 venv)

CHURRO uses **Python ≤3.12** and a **torch+cu124** stack in **`.venv-churro-paddle`**. Do **not** use Python 3.14 for this path. **PaddleOCR-VL** is not part of the reported Phase 2 baseline here (CPU Paddle was ~130 s/image); optional config: `configs/phase2/with_churro_paddle.yaml`.

```bash
bash scripts/setup_churro_paddle_venv.sh    # once: uv, CPython 3.12, venv, deps, `churro-ocr install hf`, freeze
source .venv-churro-paddle/bin/activate
# If you skipped the setup script: run once per venv — https://github.com/stanford-oval/Churro
#   churro-ocr install hf --torch-backend cu124
bash scripts/run_phase2_churro_paddle.sh     # CHURRO only → runs/phase2/raw_predictions_churro.jsonl
# Smoke: bash scripts/run_phase2_churro_paddle.sh --limit 5
```

After **`runs/phase2/metrics.json`** (core models) and **`runs/phase2/metrics_churro.json`** exist, run **`python scripts/merge_phase2_metrics.py`** for **`runs/phase2/metrics_phase2_all.json`** and **`reports/figs/baseline_phase2_all.png`**. Full write-up: **`reports/report_phase2.md`**.

## Phase 1 quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/gramoty_scrape.py index --output-dir data/raw/gramoty --delay-seconds 2.0
python scripts/gramoty_scrape.py scrape --output-dir data/raw/gramoty --delay-seconds 2.0 --limit 0
python scripts/make_birchbark_splits.py --index data/raw/gramoty/document_index.jsonl --out-dir data/splits --seed 1337
```

Respect gramoty.ru load: default throttle is 1 request per 2 seconds. `robots.txt` was not served at crawl time (404); throttle anyway.

## Hardware note

Confirm GPU with `nvidia-smi` before training phases; this environment may be CPU-only.
