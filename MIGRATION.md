# Migration to a 4xA100 machine

This document is the single source of truth for restoring the project on a fresh
host. The repo only carries source code, configs, manifests, splits and small
JSONL labels. Everything else (raw images, derived crops, model weights, virtual
envs) is regenerated from those small files, or rsynced from this machine.

## 0. What's tracked vs. what is not

Tracked in git (small, deterministic, required to re-derive everything):

```
src/                              library code
scripts/                          CLI entry points (scrape / build / train)
configs/                          experiment configs
pyproject.toml, env.yml           dependency manifests
README.md, MIGRATION.md           docs
data/splits/birchbark_*.txt       frozen train/val/test doc IDs
data/raw/gramoty/manifest.jsonl   gramoty index + per-document text
data/raw/gramoty/document_index.jsonl
data/raw/suprasliensis/manifest.jsonl
# Note: per-document data/raw/gramoty/documents/<doc>/meta.json files are NOT
# tracked. They are byte-equivalent to rows of manifest.jsonl and are
# re-emitted by scripts/gramoty_scrape.py.
data/interim/suprasliensis_crops/crops.jsonl  per-crop labels (repo-relative paths; no images)
data/interim/birchbark_*.jsonl                doc-level split sheets
data/processed/unified_ocr/unified_*.jsonl    final OCR labels (repo-relative paths only)
```

Not tracked (regenerable; transferred via rsync OR re-derived via scripts):

```
data/raw/gramoty/documents/**/images/*           ~3.6 GB  scraped photos
data/raw/suprasliensis/{images,pages}/*          ~  650 MB scraped facsimiles
data/raw/ostromir/ostromir.txt                   ~  1 MB   single download
data/interim/suprasliensis_crops/images/*        ~  540 MB derived crops
data/interim/ostromir_synth_10line/*             ~  1.4 GB synth renders
data/processed/unified_ocr/images/*              ~  1.9 GB bark-tinted crops
data/synthetic/sdxl_lora_gramoty_train/*         ~  214 MB SDXL LoRA dataset
reports/figs/*                                    generated visual/debug artefacts
runs/                                            checkpoints + logs
.venv-*/                                         python envs
```

Total transferable working set on this machine: **~8 GB** raw+derived images
(without venvs, without runs).

## 1. Bootstrap on the new host

```bash
# Clone (assumes you've pushed; otherwise rsync the working tree, see step 3.B)
git clone <remote-url> ancient_russian_itmo
cd ancient_russian_itmo

# Base venv (Phase-1 utilities, scrapers, dataset builder)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# CHURRO / PaddleOCR baseline venv (large; uses uv + cu124)
bash scripts/setup_churro_paddle_venv.sh
```

If the setup script fails on the new host (different CUDA / driver), edit
`scripts/setup_churro_paddle_venv.sh` to match the local CUDA version (the
A100 box should accept the default `--torch-backend cu124`).

## 2. Restore the data — pick ONE option

### Option A — rsync the regenerated artefacts (fastest, deterministic)

Run from THIS machine (`/home/pavviaz/Documents/ancient_russian_itmo`):

```bash
TGT_HOST=a100box        # ssh alias for the 4xA100 machine
TGT_DIR=/path/on/a100/ancient_russian_itmo

rsync -avh --info=progress2 \
  --exclude '.venv*' --exclude 'runs' --exclude 'wandb' \
  --exclude '__pycache__' --exclude '.git' --exclude 'reports/figs' \
  ./ $TGT_HOST:$TGT_DIR/
```

This copies code + manifests + every regenerable image. ~8 GB.

If you want to split the transfer, use git for the small state and rsync only
for large data. Do **not** use broad `*.txt` rsync globs; generated per-sample
gold files under `data/interim/` are intentionally ignored.

```bash
# 1) Tiny: code + tracked manifests only (instant)
git push
ssh $TGT_HOST "git clone <remote-url> $TGT_DIR"

# 2) Big: only what you actually need on the GPU box
rsync -avh --info=progress2 data/raw/                $TGT_HOST:$TGT_DIR/data/raw/
rsync -avh --info=progress2 data/interim/            $TGT_HOST:$TGT_DIR/data/interim/
rsync -avh --info=progress2 data/processed/unified_ocr/ $TGT_HOST:$TGT_DIR/data/processed/unified_ocr/
```

### Option B — re-derive from scratch via scripts (no machine-to-machine copy)

Run on the NEW host after `Bootstrap` above. Order matters.

```bash
source .venv/bin/activate

# 1) Gramoty: index + scrape (THROTTLED, takes hours; respect the host)
python scripts/gramoty_scrape.py index   --output-dir data/raw/gramoty --delay-seconds 2.0
python scripts/gramoty_scrape.py scrape  --output-dir data/raw/gramoty --delay-seconds 2.0 --limit 0

# 2) Frozen splits (deterministic from the index)
python scripts/make_birchbark_splits.py --index data/raw/gramoty/document_index.jsonl \
  --out-dir data/splits --seed 1337
python scripts/build_interim_birchbark_jsonl.py

# 3) Codex Suprasliensis: facsimiles + per-line gold
python scripts/scrape_suprasliensis.py --output-dir data/raw/suprasliensis --delay-seconds 1.0
python scripts/crop_suprasliensis_crops.py \
  --manifest-dir data/raw/suprasliensis \
  --out-images-dir data/interim/suprasliensis_crops/images \
  --out-jsonl data/interim/suprasliensis_crops/crops.jsonl

# 4) Ostromir text (single small file)
mkdir -p data/raw/ostromir
curl -L -o data/raw/ostromir/ostromir.txt http://www.ponomar.net/files/ostromir.txt

# 5) Unified OCR dataset (gramoty photos + bark-tinted suprasliensis crops)
python scripts/build_unified_ocr_dataset.py --out-dir data/processed/unified_ocr
```

Sanity check after either option:

```bash
wc -l data/processed/unified_ocr/unified_*.jsonl
# expected:  4699 train  /  272 val  /  472 test
ls data/processed/unified_ocr/images/ | wc -l   # 2850 bark-tinted crops
```

## 3. Other restore knobs

### 3.A SDXL LoRA training set (optional; only if redoing Phase 3 LoRA)

```bash
python scripts/prepare_sdxl_lora_dataset.py \
  --train-jsonl data/interim/birchbark_train.jsonl \
  --raw-root data/raw/gramoty \
  --output-dir data/synthetic/sdxl_lora_gramoty_train
```

### 3.B Bring over the in-flight repo state directly (no remote needed)

If you have not pushed the latest commits:

```bash
rsync -avh --info=progress2 --exclude '.venv*' --exclude 'runs' --exclude 'wandb' \
  --exclude '__pycache__' \
  ./ $TGT_HOST:$TGT_DIR/
```

This brings the entire working tree including `.git`. It also brings ignored
small generated files if they exist; this is acceptable for a one-off handoff,
but the canonical portable state is the git-tracked subset described above.

### 3.C Model weights / runs (skip unless explicitly needed)

`runs/` is gitignored on purpose. If you want to keep a baseline checkpoint:

```bash
rsync -avh --info=progress2 runs/<exp_id>/checkpoint-best/ \
  $TGT_HOST:$TGT_DIR/runs/<exp_id>/checkpoint-best/
```

## 4. After migration — quick smoke test

On the A100 box:

```bash
nvidia-smi                                       # confirm 4xA100
source .venv-churro-paddle/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Re-run a tiny baseline to confirm everything is wired
bash scripts/run_phase2_churro_paddle.sh --limit 5
```

## 5. Going forward (Qwen-Image-2512 plan)

Phase 4 idea (not implemented yet):

1. Use **Qwen-Image-2512** to clear the original carved text from gramoty
   photos (text-removal mode), producing blank-bark backgrounds with
   pixel-true bark fibre.
2. Imprint exact ground-truth glyphs with a non-VLM rasteriser (PIL outline +
   incised effect from `scripts/generate_real_bark_overlay_samples.py`).
3. Pair the new image with exact text → leak-free synthetic OCR data.

Required on the A100 box:

```bash
pip install -U "diffusers>=0.31" "transformers>=4.45" accelerate safetensors
# Qwen-Image-2512 weights will be pulled lazily by diffusers; ~15 GB on disk.
```

A scratch driver script will live at `scripts/qwen_image_clear_bark.py`
(to be added in Phase 4).
