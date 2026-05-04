# BLOCKED — CHURRO-3B local inference (Phase 2)

## Resolution (2026-05-03) — **Python 3.12 venv**

CHURRO and PaddleOCR-VL are **not viable on Python 3.14** (no `paddlepaddle` wheels; CHURRO without `install hf` could hang). A dedicated **CPython 3.12** environment fixes **CHURRO** for full-test scoring (`configs/phase2/with_churro.yaml`, `churro_ocr` HF backend, `max_new_tokens=128`). **PaddleOCR-VL** remains installable in the same venv for experiments, but is **excluded from the reported Phase 2 baseline** after ~**130 s/image** on CPU Paddle here (see `reports/report_phase2.md`).

**Setup (automated):**

```bash
cd /home/pavviaz/Documents/ancient_russian_itmo
bash scripts/setup_churro_paddle_venv.sh
bash scripts/run_phase2_churro_paddle.sh          # CHURRO-only full test (with_churro.yaml); optional: --limit 10
```

- Venv path: `**.venv-churro-paddle/**` (override with `CHURRO_PADDLE_VENV`).
- Freeze: `**envs/freeze_churro_paddle_venv.txt**`.
- `run_phase2_baselines.py` now parses CHURRO `**<Line>**` XML into plain text for CER; set `**CHURRO_OCR_BIN**` if `churro-ocr` is not first on `PATH`.

**Upstream canonical flow** ([Churro README](https://github.com/stanford-oval/Churro), [Getting Started](https://stanford-oval.github.io/Churro/getting-started.html)):

```bash
uv tool install churro-ocr          # or: uv pip install 'churro-ocr[hf]' in a venv
churro-ocr install hf             # required once per environment; add e.g. --torch-backend cu124
churro-ocr transcribe --image scan.png --backend hf --model stanford-oval/churro-3B
```

Skipping `**install hf**` leaves the HF runtime half-configured and was a major cause of hangs / empty output on Python 3.14. The setup script now runs `**churro-ocr install hf --torch-backend cu124**` after installing the package.

**Paddle GPU (optional):** PyPI **CPU** `paddlepaddle` is installed by default (works on RTX for VL inference via Paddle’s stack). For faster GPU-native Paddle wheels, uncomment the mirror line in `scripts/setup_churro_paddle_venv.sh` (requires connectivity to `paddlepaddle.org.cn`).

---

**Date:** 2026-05-03  
**Tool:** `churro-ocr` (Python API + CLI), model `stanford-oval/churro-3B`, HF backend.

## Symptom

After model weights load successfully (`824/824` shards), **inference does not return within practical wall-clock** (no transcription output). Observed:

1. **Python API:** `OCRClient(backend).ocr(DocumentPage(...))` — hung past ~20 minutes on first image after download.
2. **CLI:** `churro-ocr transcribe --image … --backend hf --model stanford-oval/churro-3B` — `**timeout 300`** exited 124 after weight load with no transcribed text printed.

## Environment (repro)

- Linux, NVIDIA RTX 4090 Laptop GPU, CUDA available (`torch.cuda.is_available()` True).
- Python **3.14**, PyTorch **2.11.0+cu130**, `transformers` **5.7.0**, `torchvision` **0.26.0** (upgraded from mistaken **0.2.0** wheel off `cu124` index — that broke `qwen_vl_utils`).
- `churro-ocr` **0.3.0** installed per `pip install 'churro-ocr[hf]'`.

## First failure (fixed)

`qwen_vl_utils` import failed: `ImportError: cannot import name 'io' from 'torchvision'` — caused by **torchvision 0.2.0**. Resolved with:

```bash
pip install --user --break-system-packages 'torchvision>=0.20'
```

## Second failure (current blocker)

Hangs after weight load during OCR (exact stack frame not surfaced before timeout).

## Reproduction commands

```bash
cd /home/pavviaz/Documents/ancient_russian_itmo
pip install --user --break-system-packages 'churro-ocr[hf]' 'torchvision>=0.20'
# Official Churro docs also require (once per environment):
#   churro-ocr install hf [--torch-backend cu124|cu126|auto]
churro-ocr install hf --torch-backend cu124
timeout 300 churro-ocr transcribe \
  --image data/raw/gramoty/documents/novgorod__1/images/photo_novgorod_0001_1.jpg_thumb-large.jpeg \
  --backend hf \
  --model stanford-oval/churro-3B
```

## Suggested next steps (human)

- Try **Python 3.11** venv and stack versions from CHURRO README / `agent_protocol.md` §1.2 (pinned `transformers`, torch+cuda combo).
- Try **served** CHURRO via `openai-compatible` / vLLM if local HF path stays stuck.
- Set `HF_TOKEN` for reliable Hub downloads.

Phase 2 proceeds with **TrOCR Cyrillic** (and other models that run) until CHURRO is unblocked.

---

## Autonomous pass — 2026-05-03


| Item             | Result                                                                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CHURRO-3B**    | `churro-ocr transcribe` still **times out** at **120 s**/image on birchbark thumbs (empty prediction). Prefer **Python 3.11** venv, longer timeout, or served CHURRO (see §Suggested next steps). |
| **Tesseract**    | **Resolved on user machine** — apt-installed `tesseract` + `rus`; Phase 2 core pass ran (`tesseract_lang: rus`). Optional: `rus_old` from tessdata_best per protocol §3.1.                        |
| **EasyOCR**      | **Resolved** — user-installed; full-test pass ran on GPU.                                                                                                                                         |
| **PaddleOCR-VL** | **Still blocked on Python 3.14** — no `paddlepaddle` wheel; use Python ≤3.12 + GPU paddle stack.                                                                                                  |


