# Phase 2 — Baseline inference (final report)

**Date:** 2026-05-03  
**Protocol:** `agent_protocol.md` §3  
**Test split:** `data/interim/birchbark_test.jsonl`, **n = 252** line crops (diplomatic gold + primary gramoty thumb).  
**Hardware (core + CHURRO runs):** NVIDIA RTX 4090 Laptop (16 GB); CHURRO uses PyTorch CUDA (`churro_ocr` HF backend, `max_new_tokens=128`).

## Metrics implementation

Implemented in `src/birchbark_ocr/eval/metrics.py`:


| Function                  | Description                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `normalize()`             | NFC, strip, collapse whitespace (§3.3).                                                                                      |
| `strip_square_brackets()` | Alternate eval variant for bracket stripping.                                                                                |
| `cer(pred, gold)`         | Character error rate via `jiwer.cer` on normalised strings; can exceed 1.0 when predictions are much longer than references. |
| `nls(pred, gold)`         | 1 - \mathrm{Lev}(p,g) / \max(                                                                                                |
| `exact(pred, gold)`       | Exact match after `normalize`.                                                                                               |
| `per_char_confusion(...)` | Needleman–Wunsch alignment scaffolding + filtered confusion counts.                                                          |


## Configuration

Hydra’s `@hydra.main` is not used here (Python 3.14 argparse edge case in an earlier environment). The runner is `**scripts/run_phase2_baselines.py`** with `**configs/phase2/*.yaml`**.

## Models in the Phase 2 baseline set


| Model                                                 | Role                                                                                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tesseract** (`rus`)                                 | Classical OCR lower bound.                                                                                                                     |
| **EasyOCR** (`ru`, `en`)                              | Off-the-shelf scene text reader.                                                                                                               |
| **TrOCR** `cyrillic-trocr/trocr-handwritten-cyrillic` | Handwriting-specialised Cyrillic encoder–decoder.                                                                                              |
| **Qwen3.5-0.8B** / **Qwen3.5-2B**                     | VLMs with palaeographer prompt (§3.2); images downscaled to `qwen_max_image_side: 896`.                                                        |
| **CHURRO-3B** `stanford-oval/churro-3B`               | Document-style XML line transcription via `churro_ocr` HF backend (`max_new_tokens=128`); run inside `**.venv-churro-paddle`** (CPython 3.12). |


**Excluded from the baseline:** **PaddleOCR-VL** — in this workspace the stack uses **CPU** `paddlepaddle` from PyPI; measured **~130 s per line image** (often saturating generation limits), so it is **not** reported as a comparable baseline. Optional experiment config remains in `configs/phase2/with_churro_paddle.yaml` if you install GPU Paddle and accept the cost.

## Aggregate metrics (mean over 252 test documents)

**Note on CER > 1:** VLMs and CHURRO can emit long boilerplate or XML; raw CER is unbounded. Bracket-stripped CER can still be high when the model repeats structure outside `[...]` conventions.


| Model      | n   | CER (raw) | CER (brackets stripped) | NLS       | Exact |
| ---------- | --- | --------- | ----------------------- | --------- | ----- |
| tesseract  | 252 | 1.420     | 1.604                   | 0.030     | 0.000 |
| easyocr    | 252 | 1.787     | 1.932                   | 0.039     | 0.000 |
| trocr      | 252 | 1.427     | 1.547                   | **0.076** | 0.000 |
| qwen35_08b | 252 | 5.118     | 1.004                   | 0.015     | 0.000 |
| qwen35_2b  | 252 | 6.590     | 1.088                   | 0.031     | 0.000 |
| churro_cli | 252 | 6.967     | 8.329                   | 0.013     | 0.000 |


Sources: `**runs/phase2/metrics.json`** (first five rows) and `**runs/phase2/metrics_churro.json`** (CHURRO). Combined JSON and figure: `**runs/phase2/metrics_phase2_all.json**`, `**reports/figs/baseline_phase2_all.png**` (from `scripts/merge_phase2_metrics.py`).

**Best mean NLS on this snapshot:** **TrOCR** (0.076). Absolute scores remain weak on birchbark line crops without domain adaptation; the table ranks systems on the **same** thumbs and gold.

### Slavonic auxiliary slice (CHURRO-DS)

**Not run** in this snapshot (separate JSONL + invocation).

## Commands

```bash
cd /home/pavviaz/Documents/ancient_russian_itmo

# Core five models (any Python with torch/transformers + tesseract + easyocr):
PYTHONPATH=src python scripts/run_phase2_baselines.py --config configs/phase2/default.yaml

# CHURRO only (3.12 venv; ~13 min GPU for 252 lines on this machine):
bash scripts/setup_churro_paddle_venv.sh   # once
bash scripts/run_phase2_churro_paddle.sh   # uses configs/phase2/with_churro.yaml
# Smoke: bash scripts/run_phase2_churro_paddle.sh --limit 10

# After both exist, merged table + figure:
PYTHONPATH=src python scripts/merge_phase2_metrics.py
```

CHURRO-only log from the full run: `logs/phase2_churro_only_252.log` (if created via `tee`).

## Deliverables


| Artifact                                  | Path                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------------------- |
| Core predictions + gzip                   | `runs/phase2/raw_predictions.jsonl`, `runs/phase2/raw_predictions.jsonl.gz`                 |
| Core metrics                              | `runs/phase2/metrics.json`                                                                  |
| Core NLS bars                             | `reports/figs/baseline_bars.png`                                                            |
| CHURRO predictions + gzip                 | `runs/phase2/raw_predictions_churro.jsonl`, `runs/phase2/raw_predictions_churro.jsonl.gz`   |
| CHURRO metrics + bars                     | `runs/phase2/metrics_churro.json`, `reports/figs/baseline_churro.png`                       |
| **Merged six-model metrics + bars**       | `**runs/phase2/metrics_phase2_all.json`**, `**reports/figs/baseline_phase2_all.png`**       |
| Optional Paddle experiment (not baseline) | `configs/phase2/with_churro_paddle.yaml`, `runs/phase2/metrics_churro_paddle.json` (if run) |
| Env / upstream notes                      | `BLOCKED.md`, `envs/freeze_churro_paddle_venv.txt`                                          |


## Environment snapshot

- **Core Phase 2:** see your local freeze if recorded (e.g. `envs/freeze_phase2_*.txt`).  
- **CHURRO venv:** `envs/freeze_churro_paddle_venv.txt`; requires `churro-ocr install hf` once per venv (see [Churro](https://github.com/stanford-oval/Churro)).  
- **Qwen3.5:** recent `transformers` (e.g. 5.7.x) with `Qwen3_5ForConditionalGeneration`.

## Discussion (protocol §3.4)

**Is CHURRO-3B already good enough on birchbark?** On this held-out set, **CHURRO does not beat TrOCR on mean NLS** (0.013 vs 0.076 — higher NLS is better). CHURRO’s **raw CER is very high** here, largely because the model returns **structured XML / long hypotheses** on single-line crops while gold is **continuous diplomatic text**; postprocessing keeps only `<Line>` fragments but length and template mismatch still dominate CER.

**PaddleOCR-VL** was dropped from the headline baseline after timing measurements (~130 s/image on CPU Paddle in this repo), not after a full 252-score comparison.

Next steps for stronger external alignment: fine-tune or prompt/rerank toward **plain** diplomatic output; or evaluate CHURRO on **full page** inputs closer to its training regime.