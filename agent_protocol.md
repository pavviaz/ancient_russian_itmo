# AI Agent Protocol — Birchbark Manuscript OCR with Qwen3.5-2B and Diffusion-Generated Synthetic Data

**Version:** 2.0 (drafted 2026-05-03)
**Target deliverable:** comprehensive experimental reports (NOT a paper draft) sufficient to support a VAK-tier journal submission written by humans afterwards.
**Compute budget assumption:** 1 × NVIDIA RTX 4090 Laptop (16 GB VRAM), local laptop / workstation. Original plan assumed 4 × A100; this revision adapts everything to single-consumer-GPU constraints.
**Language of all artifacts:** English. Russian / Old Russian / Old Church Slavonic text appears only inside data and inside transcriptions.

---

## 0. Mission Statement and Guard-rails

You are an autonomous research engineer. Your job is to execute a multi-phase experimental program on **birchbark manuscript text recognition** — the Old Russian / Old East Slavic letters scratched into birch bark in Novgorod and other northern Rus' centres between the XI and XV centuries. The target benchmark is the held-out portion of the Novgorod Birchbark Letters corpus (gramoty.ru, ~1100 documents).

The pipeline is: a small vision-language model (Qwen3.5-2B as the primary workhorse, Qwen3.5-0.8B as the lightweight backup) is fine-tuned on a mixture of (a) all available historical Cyrillic data (Digital Peter, Old Church Slavonic codices, modern Russian handwriting baselines) and (b) diffusion-generated synthetic line images, specifically conditioned to look like birchbark inscriptions (FLUX.1-dev or SD 3.5 + LoRA fine-tuned on real birchbark photographs). Evaluation is on a held-out subset of birchbark grammots that the model has never seen.

### Why this scope
- Birchbark is an under-served niche even in palaeographic OCR — public benchmarks for it do not exist beyond gramoty.ru itself.
- The corpus is genuinely low-resource (~1100 documents, of which many are fragments) but has high-quality scholarly transcriptions.
- A clear focused story ("specialise a generalist VLM into a tiny historical niche via synthesis-aware fine-tuning") sells well in a VAK paper and is realistic for a 1×4090 Laptop compute budget.
- Birchbark presents specific palaeographic challenges (no spaces between words, scratched-rather-than-inked surface, fragmented preservation) that make a baseline-vs-fine-tuned comparison genuinely informative.

### Non-goals
- **Do not write the paper.** Stop at structured Markdown/CSV/PNG reports. The human authors will write the manuscript.
- **Do not optimise for SOTA.** A clean, complete, ablated experimental record matters more than peak metrics. If a strong baseline already beats the fine-tuned model, report that honestly.
- **Do not chase generality.** The headline metric is birchbark CER / NLS. Other historical Cyrillic test sets are reported only as supporting generalisation evidence.
- **Do not invent data.** If a dataset is unreachable, say so in the report and continue with whatever is available. Never fabricate ground truth.

### Hard rules
1. Every phase MUST end with a `report_phaseN.md` containing tables, command lines used, hyperparameters, environment hash (commit SHA + `pip freeze`), runtime, and a short narrative.
2. Every metric MUST be computed on a held-out test set that was never seen during training, hyperparameter selection, or model selection.
3. All training runs MUST log to `wandb` (or local TensorBoard if wandb is unreachable) and persist checkpoints to disk with explicit naming `runs/<phase>_<exp_id>_<date>`.
4. Random seeds: fix to 1337 in every script; report when seed varies.
5. If a step requires more than 24 hours of wall-clock time, write an interim status report before continuing. On a single 4090 Laptop this WILL happen for diffusion fine-tuning and the main VLM run; expect it.
6. If a tool errors or behaves unexpectedly twice in a row, stop and write a `BLOCKED.md` describing the failure with reproduction steps. Do not silently work around problems.
7. **OOM is the default failure mode on a 16 GB card.** Whenever it occurs, the first response is to (a) lower batch size to 1, (b) enable gradient checkpointing if not enabled, (c) drop to QLoRA 4-bit if not already, (d) reduce `image_max_pixels` — in that order. Document the chosen workaround in the phase report.

---

## 1. Hardware and Environment

### 1.1 Hardware assumptions
- 1 × RTX 4090 Laptop (16 GB VRAM) accessible via CUDA. Confirm with `nvidia-smi` at start of each phase.
- ≥ 256 GB local SSD for datasets, synthetic images, and checkpoints. NVMe strongly preferred — diffusion-generated PNGs and FLUX checkpoints are large. Confirm with `df -h`.
- ≥ 32 GB system RAM (64 GB strongly recommended; FLUX inference and dataloaders both eat host memory aggressively).
- Internet access for HuggingFace, GitHub, arXiv. If isolated, fall back to mirrors (ModelScope for Qwen, GitHub mirror).

### 1.1a Compute reality check

This is a single-consumer-GPU project, not an HPC job. Plan accordingly:

| Workload | Realistic on 4090 Laptop 16 GB? | Notes |
|---|---|---|
| Qwen3.5-0.8B full fine-tune (bf16) | Yes | ~ 8 GB weights + optimiser. Batch 4 with grad ckpt. |
| Qwen3.5-2B full fine-tune (bf16) | Borderline | ~ 16 GB weights + opt. Use LoRA or gradient ckpt + bs=1. |
| Qwen3.5-2B LoRA (bf16) | Yes | Comfortable. Batch 2-4 with grad ckpt. |
| Qwen3.5-2B QLoRA (4-bit) | Yes | Fits with batch 4-8, 8 GB headroom for vision tokens. |
| FLUX.1-dev inference (bf16, full) | No | Needs ~ 16 GB just for weights. Use 4-bit / nf4 quant + CPU offload. ~ 20-40 s per image. |
| FLUX.1-dev LoRA training | Borderline | Possible with 4-bit base + bs=1, 8-16 hours per LoRA. |
| SD 3.5 Medium / SDXL inference | Yes | ~ 8-12 GB. Batch 1-2. |
| SDXL LoRA training | Yes | Comfortable. |
| Synthetic generation 60K images | No (too slow) | Reduced to 10-20K — see Phase 3. |

The reasonable target shape of the experimental campaign:
- ~10-20K synthetic images (not 60K).
- Primary base: **Qwen3.5-2B** with QLoRA. Backup: Qwen3.5-0.8B (full or LoRA).
- One main fine-tune run + 4-5 short ablations (not 10).
- Diffusion generator: prefer SDXL+LoRA; FLUX optional and only via 4-bit nf4.

### 1.2 Software stack
Create `env.yml` and `pyproject.toml` reproducibly:

```bash
# Create environment
mamba create -n birchbark python=3.11 -y
mamba activate birchbark

# Core ML stack
pip install --upgrade pip
pip install "torch>=2.5,<2.7" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.50.0" accelerate peft bitsandbytes
pip install unsloth==2026.04 unsloth_zoo  # for memory-efficient Qwen3.5 fine-tuning on consumer GPUs
pip install flash-attn --no-build-isolation
pip install trl datasets sentencepiece einops

# Diffusion stack — prioritise memory-efficient inference
pip install "diffusers>=0.32" "controlnet_aux>=0.0.10"
pip install "huggingface_hub>=0.27" safetensors
pip install optimum quanto  # for 4-bit / nf4 quant of FLUX

# Metrics and analysis
pip install jiwer rapidfuzz Levenshtein pandas scikit-learn matplotlib seaborn
pip install python-Levenshtein editdistance evaluate

# OCR baselines
pip install easyocr paddleocr paddlepaddle-gpu
pip install pytesseract  # plus apt install tesseract-ocr-rus

# Tracking
pip install wandb tensorboard

# Utilities
pip install hydra-core omegaconf rich tqdm orjson beautifulsoup4 lxml httpx
```

Persist the exact resolved versions:
```bash
pip freeze > envs/freeze_$(date +%Y%m%d).txt
```

### 1.3 Repository layout
```
birchbark-ocr/
├── README.md
├── env.yml
├── envs/
├── configs/                  # Hydra configs for each experiment
├── data/
│   ├── raw/                  # downloaded datasets, never modified
│   ├── interim/              # cleaned line images + JSONL
│   ├── synthetic/            # diffusion-generated images
│   └── splits/               # train.jsonl / val.jsonl / test.jsonl
├── src/
│   ├── data/                 # loaders, scrapers, splitters
│   ├── synth/                # diffusion fine-tune + generation
│   ├── train/                # Qwen3.5 fine-tuning
│   ├── eval/                 # metrics, decoders
│   └── analysis/             # confusion matrices, per-period plots
├── scripts/                  # one-off CLI entry points
├── runs/                     # wandb-mirrored checkpoints + logs
├── reports/                  # phase reports, final summary
└── notebooks/                # exploratory only — keep them out of the source of truth
```

Initialise as a git repo on day one. Tag the repo at the end of every phase with `phase<N>-final`.

---

## 2. Phase 1 — Data Acquisition and Curation

### 2.1 Data hierarchy

The data plan reflects the project's narrow focus on birchbark grammots. There are three concentric tiers of data:

1. **Primary corpus (gramoty.ru — Novgorod Birchbark Letters).** This is the headline benchmark and the only dataset that decides the paper. ~1100 documents, XI–XV c., scratched into birch bark, each with a scholarly transcription from the IRYa RAN / NovGU / MSU project. Treat it as gold.
2. **Auxiliary historical Cyrillic (training-only).** Non-birchbark sources used to give the model broad palaeographic context: Old Church Slavonic codices (Codex Suprasliensis), Stanford CHURRO-DS Slavonic subset, Transkribus public Cyrillic models' GT, and Digital Peter (XVIII c. Russian cursive). These never contribute to the test set.
3. **Modern Cyrillic and synthetic (training-only).** Modern Russian handwriting datasets and diffusion-generated synthetic line images. Pure capacity-fillers — they teach the model letter shapes and writing surfaces, not the target distribution.

### 2.2 Birchbark scrape (gramoty.ru) — handled first and most carefully

The Novgorod corpus at gramoty.ru is the single most relevant in-domain material. There is no machine-friendly export. Procedure:

1. Crawl the index `http://gramoty.ru/birchbark/document/list/` respecting `robots.txt`. Throttle to 1 request / 2 s.
2. For each document page extract: photo of the artefact, "Прорись" (line drawing) image, transcription with diacritics, dating range, place of finding.
3. Store raw HTML + images in `data/raw/gramoty/`. Write the transcription parser as a separate cleaning step in `data/interim/gramoty/`.
4. Manually inspect 30 random samples after parsing. gramoty.ru transcriptions use a custom convention for unclear letters and reconstructed text (square brackets, dots under letters, vertical bars for line breaks). Decide on a normalisation policy:
   - `[reconstructed]` → keep as `[reconstructed]` (this preserves uncertainty signal); evaluate model on both versions (with brackets / brackets stripped) and report.
   - `letter͡` (dotted under) → keep the dotted variant; some are crucial palaeographic signals.
   - `|` (line break inside one document) → split into separate line images via the prorisi line drawings.
   - `(...)` (editorial completion) → strip in normalised version, keep in raw version.
5. Birchbark inscriptions have **no spaces** between words and are often written **continuously**. This breaks WER computation. For this corpus compute CER only and a `lemma-level F1` after a separate tokenisation step.
6. **Test set freezing.** Split by physical document (not by line), stratified across centuries (XI, XII, XIII, XIV, XV) and across find sites (Novgorod / Staraya Russa / Smolensk / Pskov / Torzhok / others):
   - 70% train (~770 docs)
   - 10% val (~110 docs) — for hyperparameter selection only
   - 20% test (~220 docs) — never touched until the final run

   Save IDs to `data/splits/birchbark_{train,val,test}_ids.txt` and never modify the files. SHA256-hash them at creation time and verify the hash before any evaluation run.

7. From each document, extract individual lines using the prorisi (line drawings) when available, plus the photographs. Some grammots are 1-2 line fragments — use the whole image as a single sample. Target output: ~1500 train line images, ~250 val line images, ~500 test line images (numbers approximate; actual counts depend on legibility filtering).

### 2.3 Auxiliary data (training-only)

Acquire in this priority order; if any source is offline, log the failure and continue.

| Source | URL | Period / Script | Approx. usable lines |
|---|---|---|---|
| Codex Suprasliensis digital edition | http://csup.ilit.bas.bg/ | XI c. Old Church Slavonic uncial | several hundred pages |
| Stanford CHURRO-DS (filter to Slavonic / Old East Slavic) | https://huggingface.co/datasets/stanford-oval/churro-dataset | XI–XVII c. mixed | several thousand |
| Transkribus public model GT for Old Cyrillic | https://github.com/quinnanya/transkribus-models (pointers) | XI–XVI c. uncial / semi-uncial | small |
| Manuscripts.ru ("Манускрипт", IRYa RAN) | http://manuscripts.ru/ | XI–XVII c. | varies, mostly read-only |
| Digital Peter (Sber AIRI) | https://github.com/sberbank-ai/digital_peter_aij2020 | XVIII c. Russian cursive | 9694 |
| Cyrillic-trocr training set | https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic | mixed Slavonic+Russian+Ukrainian | ~6800 |
| HKR (HSE) | https://github.com/abdoelsayed2016/HKR_Dataset | modern Russian school | small support |
| HTR-United catalogue (filter Cyrillic) | https://htr-united.github.io/catalog.html | aggregator | discovery only |

For each downloaded dataset, write an ETL script that converts to a unified JSONL:

```jsonl
{"image_path": "data/interim/codex_suprasliensis/000123.png",
 "text": "и҆̀ рече гд҃ь ко а҆вра́му",
 "source": "codex_suprasliensis",
 "period": "XI",
 "script": "uncial",
 "surface": "parchment",
 "length_chars": 22,
 "split": "train_aux"}
```

The `split` field for everything in this tier is `train_aux` — never `val` or `test`.

### 2.4 Splits — final summary

| Split | Source | Use |
|---|---|---|
| `train_birch` | gramoty.ru, 70% by document | primary fine-tuning data |
| `val_birch` | gramoty.ru, 10% by document | model selection |
| `test_birch` | gramoty.ru, 20% by document | **headline number** |
| `train_aux` | Codex Suprasliensis + CHURRO Slavonic + Digital Peter + others | broaden the prior; mix into training |
| `train_synth` | diffusion-generated (Phase 3) | bulk capacity for letter shapes / surfaces |
| `eval_aux` | small held-out slices of CHURRO Slavonic, Digital Peter | optional generalisation evidence |

Run a leakage check: for each test image, confirm its filename and its hashed gold text are not present in any train shard. Document the result in `data/splits/leakage_audit.md`.

### 2.5 Phase 1 deliverables
- `reports/report_phase1.md` with a table of datasets, per-split sample counts, per-century histogram of `train_birch` and `test_birch`, character distribution comparison between birchbark and auxiliary corpora.
- `data/interim/*` with all cleaned line images and JSONL index.
- `data/splits/{birchbark_train,birchbark_val,birchbark_test}_ids.txt` with SHA256 hashes.
- `data/splits/leakage_audit.md`.

---

## 3. Phase 2 — Baseline Inference

Run a focused set of off-the-shelf models on the **birchbark test set** (primary) and on a small slice of `eval_aux` (CHURRO Slavonic, ~ 200 lines) for context. Compute CER and Normalized Levenshtein Similarity (NLS = `1 − Lev / max(len_pred, len_gold)`) — the same metric as the CHURRO paper, so cross-paper comparison is possible.

Skip exhaustive baselines from the original plan; on a single 4090 Laptop each model run takes hours. Pick the most informative ones.

### 3.1 Baseline matrix (priority order)

| # | Model | HF / repo | Why included |
|---|---|---|---|
| 1 | **CHURRO-3B** | `stanford-oval/churro-3B` | **Primary external baseline.** Strongest open-weight historical-OCR VLM (Sep 2025). The headline number to beat or to come close to. |
| 2 | **Qwen3.5-2B (zero-shot)** | `Qwen/Qwen3.5-2B` | Our base model, zero-shot. Establishes the starting point of the fine-tune curve. |
| 3 | **Qwen3.5-0.8B (zero-shot)** | `Qwen/Qwen3.5-0.8B` | Backup base model. Zero-shot reference. |
| 4 | TrOCR-handwritten-cyrillic | `cyrillic-trocr/trocr-handwritten-cyrillic` | Specialised Slavonic+Russian+Ukrainian fine-tune. Older but a strong line-level baseline. |
| 5 | PaddleOCR-VL-0.9B | `PaddlePaddle/PaddleOCR-VL` | Recent (Nov 2025), explicitly supports Cyrillic + historical documents. Sanity check that we are not embarrassed by an off-the-shelf model. |
| 6 | EasyOCR (Cyrillic) | `JaidedAI/EasyOCR` | Classical CRNN baseline; expected to be very weak on birchbark. |
| 7 | Tesseract 5 + `rus_old` | `tesseract-ocr/tessdata` | Lower-bound sanity check. |

Skip on consumer 4090 Laptop (originally listed, deprioritised here):
- Qwen2.5-VL-7B / Qwen3-VL-8B — superseded by Qwen3.5 generation; keep only if there is spare compute.
- InternVL3.5-8B / Llama-3.2-11B-Vision — would require quantisation just to run; not worth the time for a single comparison line.
- Donut, HRTR — niche; only run if the headline gap to CHURRO leaves doubt about the approach.

If commercial API budget is available, also evaluate Gemini 2.5 Pro and GPT-5 Vision on the birchbark test set (200 random images). Skip if no budget.

### 3.2 Prompts for VLMs

Use the CHURRO prompt verbatim where applicable (it is in the repo at `stanford-oval/Churro/prompts/`). For Qwen3.5 models:

```
You are an expert palaeographer reading Old Russian birchbark inscriptions
from medieval Novgorod (XI–XV century). The text is scratched into birch bark,
written continuously without spaces between words, and uses Old Cyrillic
letterforms including ѣ, ѧ, ѫ, ѳ, ѵ, ѡ, ѥ, ѩ, ѭ, ѯ, ѱ, ҂.

Transcribe the line in the image diplomatically — preserve original letterforms,
diacritics (titlas), superscript letters, and the original lack of word spacing.
Do not modernise. Do not add punctuation. Use square brackets [...] only for
characters that are visible but ambiguous; never invent missing text.

Output only the transcribed line, nothing else.
```

For each VLM, set `max_new_tokens = 256`, `temperature = 0`, `do_sample = False`. Disable thinking mode for Qwen3.5 unless explicitly testing it (Qwen3.5-2B gets a separate "thinking-mode" zero-shot run as a small ablation, since the model card shows non-trivial CER differences).

### 3.3 Evaluation

Implement metrics in `src/eval/metrics.py`:

- `cer(pred, gold)`: Levenshtein on characters / `len(gold)`. Use `jiwer.cer`. Compute on two normalised variants (with brackets / brackets stripped).
- `nls(pred, gold)`: `1 − Lev(pred, gold) / max(len(pred), len(gold))` — CHURRO-comparable.
- `exact(pred, gold)`: `pred.strip() == gold.strip()`.
- `per_char_confusion`: confusion matrix for the 30 most frequent characters in birchbark + the 12 historically interesting ones (ѣ ѧ ѫ ѳ ѵ ѡ ѥ ѩ ѭ ѯ ѱ ҂). Build via Needleman-Wunsch alignment of pred / gold.

Apply a single `normalize()` function (NFC unicode, strip leading/trailing space, collapse repeated whitespace) consistently to all baselines so the comparison is fair.

### 3.4 Phase 2 deliverables
- `reports/report_phase2.md` with a baseline table on birchbark test (CER, NLS, exact-match) and on a small Slavonic slice.
- `reports/figs/baseline_bars.png` — bar chart of NLS by model.
- `runs/phase2/raw_predictions.jsonl.gz` — every (image_id, model, prediction) tuple.
- A short discussion paragraph: "is CHURRO-3B already good enough on birchbark, or is there a clear gap to close?" — this informs whether the fine-tuning effort in Phase 4 is justified.

---

## 4. Phase 3 — Synthetic Data Generation Pipeline

Goal: ~10-20K line-level synthetic images that look like birchbark inscriptions, with paired ground truth. The original 60K target is unrealistic on a 4090 Laptop; 15K is the sweet spot — enough to matter for fine-tuning, few enough to generate in 1-2 days of compute.

### 4.1 Text source for prompts

Build a "text bank" of ~ 200K lines of authentic Slavonic / Old Russian / pre-modern Russian text. Sources:
- Gramoty.ru transcriptions (text-only export, NOT image — text reuse is fine; image reuse from test set is the hard line).
- Codex Suprasliensis text-only edition.
- Ostromir Gospel, Izbornik Sviatoslav, standard liturgical fragments ("Otche nash", Psalms).
- Manuscripta corpora available on manuscripts.ru.

Tokenise into "lines" of 5-25 words. Save as `data/synthetic/text_bank.txt`. Deduplicate.

**Critical:** before using any text line for synthesis, hash it and check it against the gold text of `test_birch`. Drop any synthetic line whose gold text duplicates (Levenshtein < 5 chars) any test-set gold. Document the dedup count in the phase report.

### 4.2 Diffusion generators — single-GPU choices

Tried in this order; pick the one that gives best downstream OCR utility per hour of compute.

| Generator | VRAM | Speed | Use |
|---|---|---|---|
| **SDXL + LoRA** (primary) | ~ 10-12 GB bf16 | ~ 4-6 s/img on 4090 Laptop | Main workhorse. Cheap to fine-tune (8h LoRA), fast to generate. |
| **SD 3.5 Medium** (alternative) | ~ 8 GB | ~ 3-5 s/img | Better small-text rendering than SDXL; try as a sanity check on a 1K subset. |
| **FLUX.1-dev + nf4 quant + offload** (optional) | ~ 12-16 GB nf4 | ~ 30-40 s/img | Highest visual quality but very slow. Generate a small subset (1-2K) only, primarily for Phase 5 ablation A4 ("does FLUX outperform SDXL as generator?"). |
| **Bezier-curve engine** (procedural baseline) | CPU only | < 1 s/img | From dbrainio/CyrillicHandwritingPOC. Ground truth is exact (no OCR-bootstrap noise). Use as procedural-vs-diffusion ablation. |

Skip FLUX.1-Kontext on 4090 Laptop — it adds another 8 GB on top of base FLUX, making it impractical.

### 4.3 LoRA fine-tuning of SDXL for birchbark style

Use `bghira/SimpleTuner` or `kohya-ss/sd-scripts`.

Hyperparameters (start here):
- LoRA rank: 16
- Learning rate: 1e-4 (UNet), 5e-5 (text encoders)
- Optimizer: `adamw_bf16`
- Resolution: 1024×1024 with line-image crops centered, padded with bark/parchment background
- Steps: 2000 (SDXL learns visual style fast)
- Trigger token: `<birchbark>` — append to every caption
- Captions: human-written, e.g. "<birchbark> handwritten Old Russian inscription scratched into birch bark, dark brown ink-like incisions on cream-coloured fibrous bark surface, 13th century Novgorod style"
- Batch size: 1 with grad accumulation 4
- Save checkpoints every 500 steps
- Required VRAM: ~ 18-20 GB; use 8-bit Adam if tight

Training corpus for the LoRA: 200-400 birchbark photographs from the **train split only** (never test). Caption every image manually or with a strong VLM (Qwen3.5-2B itself can write the descriptions).

If a higher-quality LoRA is needed and time allows, repeat with FLUX.1-dev: nf4-quantised base, rank 16, 3000 steps, ~ 16 hours on 4090 Laptop. Optional.

### 4.4 Generation pass

For each (text-bank line, style preset), generate one image:

```python
prompt = f"<birchbark> {style_preset}, line of Cyrillic text reading: {line_text}"
# style_preset cycles through:
# - "weathered birchbark with dark scratches, 11th century Novgorod"
# - "well-preserved birchbark, fine sharp incisions, 13th century"
# - "fragmented birchbark with cracks and stains, 14th century"
# - "darkened birchbark with faded letters, late 12th century"
```

**Critical pitfall — diffusion models cannot spell.** The generated image will contain *Cyrillic-looking glyphs* that resemble the prompt text but are not letter-perfect. DO NOT use the prompt text as the OCR ground truth. Two options:

**Option A — noisy-bootstrap (recommended primary).** Generate the image, then run a strong OCR (CHURRO-3B + your previous-iteration fine-tuned model) to read the synthetic image. Use the OCR reading as the ground truth. Reject samples where two readings disagree on more than 30% of characters. Same trick as in CHURRO and the Manchu Qwen2.5-VL paper. Document the noise rate in the report.

**Option B — glyph-conditional (cleaner ground truth).** Render the text first with a Cyrillic font (Old Standard, Ponomar Unicode), then condition the diffusion model on this rendered glyph layout via ControlNet (canny / lineart). This way the ground truth is exactly the rendered text. As in arXiv:2305.19543 (glyph-conditional DDPM). Lower visual realism than Option A but no label noise.

Run both pipelines on smaller scales (5K each) in Phase 3 and compare downstream OCR utility in Phase 5 ablation A4.

### 4.5 Augmentation chain

After generation, apply a stochastic augmentation pipeline:
- Elastic distortion (alpha=20-60, sigma=4-8)
- Faded-letter simulation (multiplicative gamma 0.5-1.4)
- Random cracks (thin curved strokes superimposed)
- Random stains (Gaussian dark blobs, low opacity)
- Background bark-texture overlay from a 30-image library scraped from public-domain birchbark photos (NOT from gramoty.ru test set)
- Geometric: ±3° rotation, ±5% scale

Save augmented + clean variants. Include both in the synthetic train set.

### 4.6 Phase 3 deliverables
- 10-20K synthetic line images in `data/synthetic/v1/`, JSONL with text + style + generator metadata
- `reports/report_phase3.md` with FID between synthetic and real birchbark, sample grids, per-generator stats, noise rate from Option A bootstrapping
- `reports/figs/synth_examples.png` — 4×8 grid showing each generator + style preset

---

## 5. Phase 4 — Main Training: Qwen3.5-2B Fine-tuning

### 5.1 Why Qwen3.5-2B as the workhorse

Qwen3.5 (Feb 2026, dense + MoE family) is the latest unified VL line from the Qwen team. The benchmark numbers from the official Qwen3.5-2B model card are striking:

- OCRBench: 84.5 (thinking) / 85.4 (non-thinking) — **higher** than Qwen3-VL-4B (80.8) and on par with much larger models.
- OmniDocBench1.5: 79.8 / 80.9 — **higher** than Qwen3-VL-4B (80.0).
- CC-OCR: 72.9 / 75.8 — strong in the small-text regime.

Architecturally Qwen3.5 uses early-fusion VL training, Gated DeltaNet + sparse attention, and tied LM-output embeddings. The early-fusion design is particularly useful for OCR — the visual tokens are aligned with text tokens earlier in the network, which seems to help fine-grained character recognition.

**Primary base model: Qwen3.5-2B.** Fits on a 4090 Laptop with QLoRA + grad ckpt; with full LoRA in bf16 batch 1-2 also works.
**Backup base model: Qwen3.5-0.8B** (architecture: 24 layers, 1024 hidden dim, 0.9B total params with vision encoder). Use for fast iteration, ablations that need many runs, and as a published alternative if 2B fine-tuning behaves poorly.

Do NOT use Qwen3.5-9B / 27B / 122B-A10B / 397B-A17B — these exist in the family but are out of scope for a 4090 Laptop.

### 5.2 Training framework

Use `unsloth` — it explicitly supports Qwen3.5 (per the Feb 2026 release notes) and gives ~ 2× speedup with 50% VRAM cut on consumer GPUs. If unsloth misbehaves with Qwen3.5 specifically, fall back to vanilla `transformers + peft + accelerate`. DeepSpeed is unnecessary on a single GPU.

Reference notebooks (adapt, do not blindly copy):
- https://docs.unsloth.ai/models/qwen3-vl-run-and-fine-tune (Qwen3-VL; close but not identical to Qwen3.5)
- https://kaitchup.substack.com/p/qwen3-vl-fine-tuning-on-your-computer (consumer-GPU practices)

### 5.3 Training data composition (main run)

```
70% synthetic (Phase 3 output)
20% train_aux (Codex Suprasliensis + CHURRO Slavonic + Digital Peter)
10% train_birch (gramoty.ru train split)
```

Why heavy on synthetic: the train_birch shard is small (~ 1500 line images) — it would overfit immediately if used pure. Synthetic gives capacity for letter shapes and surface textures; train_aux gives palaeographic context; train_birch teaches the actual target distribution. The 10% real-birchbark proportion is small but applied late in each epoch via curriculum (see 5.5).

### 5.4 Hyperparameters (Qwen3.5-2B + QLoRA, bf16)

```yaml
model: Qwen/Qwen3.5-2B
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []   # do NOT unfreeze the vision encoder unless an ablation tells us to
training:
  per_device_batch_size: 2
  grad_accumulation_steps: 16    # effective batch 32
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  num_train_epochs: 3
  max_seq_length: 1024            # birchbark lines are short; keep this tight to save VRAM
  optim: paged_adamw_8bit
  bf16: true
  gradient_checkpointing: true
  weight_decay: 0.01
data:
  image_max_pixels: 451584         # 768 * 28 * 28 — birchbark lines are short, do not over-tokenise
  image_min_pixels: 100352
  prompt: "<see birchbark palaeographer prompt above>"
eval:
  evaluation_strategy: steps
  eval_steps: 200
  save_steps: 200
  metric_for_best_model: cer_birchbark_val
  greater_is_better: false
  early_stopping_patience: 4
seed: 1337
```

Expected wall-clock: ~ 24-36 hours on a single 4090 Laptop for 3 epochs over ~ 20K samples. If too slow, drop epochs to 2 first, then drop to Qwen3.5-0.8B.

### 5.5 Critical pitfalls on consumer hardware

- **Qwen image token budget.** `image_max_pixels` directly controls vision tokens per image. Birchbark line crops are short and wide; keep aspect ratio ~ 1:5 to 1:10 (height ~ 70 px, width ~ 600-800 px after preprocessing). Print actual token count of the first batch — if vision tokens exceed 800 per image you are wasting VRAM.
- **OOM cascade.** First OOM on 4090 Laptop → drop batch to 1, raise grad_acc to 32. Second OOM → reduce `image_max_pixels` to 200704. Third OOM → fall back to Qwen3.5-0.8B (full LoRA, no QLoRA needed). Document each fallback in the phase report.
- **Tokenisation of rare Cyrillic codepoints.** ѣ, ѫ, ѡ, ҂ may map to multi-byte sequences in BPE. Verify by encoding/decoding a 100-character Old Cyrillic test string and confirming round-trip equality. If lossy, prepend a tokeniser-extension step.
- **Combining titlo `◌҃` (U+0483).** Test that the tokeniser preserves this combining mark in round-trip. If not, normalise to a base-letter + visible glyph form before training.
- **Synthetic-real distribution shift.** Report CER on `val_birch` and `val_synth` (held-out 5% of synthetic) **separately every epoch**. If `val_synth` drops to ~ 0 while `val_birch` plateaus, the model is overfitting to diffusion artefacts. Increase real-birchbark proportion or stop training.
- **Curriculum.** Try the 3-stage anneal: (i) synth + aux for 1 epoch, (ii) mixed for 1 epoch, (iii) real-birchbark heavy for 0.5 epoch. Compare against single-stage mixed in ablation A2.
- **Bf16 only.** Never fp16 with Qwen3.5; numerical instability is well-documented.
- **Flash-attention 2.** Required for memory efficiency. Verify with `print(model.config._attn_implementation)`.
- **Vision encoder freeze.** Default frozen. An ablation can try unfreezing, but on 4090 Laptop + small data it is more likely to hurt than help.

### 5.6 Three runs in Phase 4

Train three models (sequentially, not in parallel — one 4090 Laptop):
1. `synth_only` — train_synth + train_aux only, no real birchbark.
2. `real_only` — train_birch + train_aux only, no synthetic.
3. `mixed` (the headline run) — full 70/20/10 mix as in 5.3.

Each gives a checkpoint. Evaluate all three on `val_birch` + `test_birch` + `eval_aux`. The contrast `mixed` vs `real_only` quantifies the synthetic-data contribution, which is the central claim of the paper.

### 5.7 Phase 4 deliverables
- Three checkpoints (LoRA adapters only) in `runs/phase4/{synth_only,real_only,mixed}/`
- `reports/report_phase4.md` — training curves (loss, CER per val set), final test scores on birchbark + aux, comparison to Phase 2 baselines (delta NLS / CER per model).

---

## 6. Phase 5 — Ablations

Each ablation is a separate fine-tune run on the 4090 Laptop. Each takes ~ 12-30 hours. Realistic budget: 4-5 ablations. Do these in priority order; stop early if compute runs out.

### 6.1 Ablation matrix (priority-ordered)

| ID | Axis | Values | Cost (GPU-h) | Why it matters |
|---|---|---|---|---|
| A1 | synth/real ratio | mixed-default (ref) vs 0/100 (real_only, already done in P4) vs 100/0 (synth_only, already done in P4) vs 50/50 vs 90/10 | ~ 2 × 24h (the 50/50 and 90/10 runs) | Headline of the paper — what is the optimal mix? |
| A2 | Diffusion generator | SDXL+LoRA (ref) vs Bezier procedural | ~ 1 × 24h (one extra train run on Bezier-only synth) | Does diffusion realism matter, or is procedural synth enough? |
| A3 | Curriculum | single-stage mixed (ref) vs 3-stage anneal | ~ 1 × 30h | Common claim in low-resource OCR papers; cheap to test. |
| A4 | LoRA rank | 32 (ref) vs 16 vs 64 | ~ 2 × 24h | Standard sanity check. |
| A5 | Model size | Qwen3.5-2B (ref) vs Qwen3.5-0.8B | ~ 1 × 18h | Does the smaller backup model work? |
| A6 | Vision encoder | frozen (ref) vs trainable | ~ 1 × 30h | Optional. |
| A7 | Augmentation | full chain (ref) vs none vs geometric only | ~ 2 × 24h | Optional. |
| A8 | image_max_pixels | 451584 (ref) vs 902168 vs 200704 | ~ 2 × 24h | Optional. |
| A9 | Decoding | greedy (ref) vs beam=4 vs Qwen-thinking-mode | inference only | Free; run in any case. |
| A10 | Glyph-conditional vs noisy-bootstrap synth | Option A (ref) vs Option B (Phase 3.4) | ~ 1 × 24h | Run if Phase 3 produces both pipelines. |

**Minimum viable set if compute is tight:** A1 (the 50/50 and 90/10 runs) + A2 + A9. That gives one synth-ratio sweep, the procedural-vs-diffusion comparison, and the free decoding ablation. Together with Phase 4's three runs, this is ~ 6 trained models — enough for a credible ablation table.

If everything goes well, also do A4 (LoRA rank) and A5 (smaller model) for completeness.

Skip on consumer GPU: anything that needs MoE handling (was A10 in v1; not applicable here).

### 6.2 Phase 5 deliverables
- One CSV per ablation in `reports/tables/ablation_<id>.csv`
- `reports/report_phase5.md` with overlay plots showing the ablated curve vs reference
- A "winners" table identifying the best value for each axis, with the human-author note "if budget had allowed, we would also have explored X"

---

## 7. Phase 6 — Error Analysis

Focused on birchbark only — that is the headline benchmark.

### 7.1 Per-character confusion matrix

For the best `mixed` model on `test_birch`:
- Build a 60×60 character confusion matrix using a global alignment (Needleman-Wunsch) between predicted and gold strings.
- Plot as a heatmap with axes ordered by frequency.
- Highlight 12 historically interesting characters (ѣ, ѧ, ѫ, ѳ, ѵ, ѡ, ѥ, ѩ, ѭ, ѯ, ѱ, ҂) with bold borders.
- Save raw matrix as `reports/tables/char_confusion_birchbark.csv`.
- Save heatmap as `reports/figs/char_confusion_birchbark.png`.

### 7.2 Slice analysis

Bucket `test_birch` along three axes and report CER per bucket:

| Axis | Buckets |
|---|---|
| Century | XI, XII, XIII, XIV, XV |
| Find site | Novgorod, Staraya Russa, Smolensk, Pskov, Torzhok, other |
| Preservation state | well-preserved, fragmented, heavily damaged (manual labels on 50 samples; extrapolate via simple visual heuristic for the rest) |

Output a 2-D heatmap (century × preservation) of CER. Save as `reports/figs/slice_cer_birchbark.png`.

### 7.3 Failure modes

Take the 50 worst predictions (highest CER on `test_birch`) and:
- Manually classify each into one category: *segmentation error* (line crop wrong), *rare character missed*, *titlas misread*, *vynosnye bukvy missed*, *ligature confusion*, *bark damage / illegibility*, *no-space-segmentation error* (model inserted spurious spaces), *other*.
- Aggregate counts in `reports/tables/failure_modes.csv`.
- Save 12 representative examples as image grids with predicted vs gold overlay.

### 7.4 Phase 6 deliverables
- All of the above in `reports/report_phase6.md`.

---

## 8. Phase 7 — Final Synthesis Report

### 8.1 What to produce

`reports/FINAL.md` containing:
1. **Headline numbers** — one table: best CER and NLS on `test_birch`, with delta over (a) the best Phase-2 baseline, (b) CHURRO-3B specifically, and (c) Qwen3.5-2B zero-shot.
2. **Method summary** — 1 page Markdown, no figures, explaining what was trained and how.
3. **Key findings** — at most 5 bullets (e.g. "synthetic data from SDXL-LoRA gives +X.X NLS over no synthetic on birchbark", "LoRA rank 32 is optimal", "Qwen3.5-2B beats Qwen3.5-0.8B by Y.Y NLS").
4. **Reproducibility checklist** — git SHA, env hash, data versions, seed.
5. **Pointers** — file paths to every artifact (checkpoints, predictions, configs, logs).
6. **Open questions for the human authors** — list of ambiguities or methodological choices that warrant a paragraph in the final paper.

### 8.2 Hand-off package

Zip the following into `reports/handoff_<date>.zip`:
- All `report_phase*.md`
- All `reports/tables/*.csv`
- All `reports/figs/*.png`
- `reports/FINAL.md`
- `configs/` directory
- `data/splits/` directory (NOT raw images — too large)
- The best checkpoint as a HuggingFace-format folder (LoRA adapters only, ~ 100-200 MB depending on rank)

Stop here. Do NOT write a paper draft. Notify the human authors that hand-off is ready.

---

## 9. Cross-cutting pitfalls and gotchas

### 9.1 Dataset version control
HuggingFace datasets get re-uploaded silently. Pin every dataset to a specific revision SHA in your data download script. Record the SHA in the dataset's JSONL metadata.

### 9.2 Tokenizer drift between Qwen versions
Qwen3.5 tokeniser is *not* identical to Qwen3-VL or Qwen2.5-VL. Encoding the same Cyrillic string can yield different token sequences. When loading a checkpoint, *always* load the matching tokeniser from the same model card; never mix.

### 9.3 max_pixels and OOM on 4090 Laptop
Qwen vision models compute attention over all image patches. Doubling `max_pixels` quadruples VRAM use for that batch. On a 16 GB card OOM is the default failure; the OOM cascade in 5.5 is your standard playbook.

### 9.4 Birchbark transcription conventions
- Many lines have no spaces between words. Disable WER, use CER + word-segmentation-aware F1.
- Some grammots have multiple inscriptions on the same artefact, sometimes in different orientations. Crop carefully; for ambiguous orientation, exclude from train and test alike.
- Reconstructed text in `[brackets]` and dotted-under letters represent uncertainty. Decide whether to keep, strip, or normalise these BEFORE training. Record the decision in `data/interim/birchbark_normalisation_policy.md` and never change it mid-experiment.
- Some grammots are dated to a range (e.g. "XII–XIII century"). For stratified splits, use the median; for slice analysis, use the earliest endpoint.

### 9.5 Numeric / titlo characters
The combining titlo `◌҃` (U+0483) is a combining mark. Test on the empty string + titlo to confirm tokeniser round-trip. If lossy, normalise titles into a base-letter + visible glyph form (e.g., `<TITLO>`) before training and apply the same normalisation at evaluation.

### 9.6 Diffusion models cannot spell — repeat for emphasis
This is the single most common failure mode of synthetic-data pipelines for OCR. If you skip the noisy-bootstrap or glyph-conditional mitigation in Phase 3 and use prompt text as ground truth, the model will be trained on garbage and will appear to learn but will fail on real birchbark. Do not skip Phase 3.4.

### 9.7 Evaluation contamination
Diffusion-generated images sometimes happen to render text that exactly matches some test-set text by chance (the text bank overlaps with public-domain manuscripts that the test set is also drawn from). Run a textual de-duplication: for each synthetic line, hash the gold text and check no test-set gold has the same hash or Levenshtein < 5. Drop matches from the synthetic train set.

### 9.8 Reproducibility
Wrap every training script with Hydra. Save the resolved config as YAML next to the checkpoint. Re-running with the same config + same data should reproduce within ±0.5 CER.

### 9.9 Reporting honesty
If the fine-tuned Qwen3.5-2B does *not* beat CHURRO-3B on birchbark, say so plainly. CHURRO is a domain-specialised 3B-parameter model and may be hard to surpass on a 4090 Laptop with 1500 real birchbark training images. The story can still be: "narrow-domain specialisation of a 2B generalist VLM via diffusion synthesis brings it within X% of a domain-specialised SOTA, ablations isolate the contribution of each component, and we provide the first published benchmark on the gramoty.ru held-out split". A negative-or-mixed result honestly reported is a valid VAK paper.

### 9.10 Compute discipline
Maintain a running log of GPU-hours consumed in `logs/compute.csv` with columns `phase, run_id, hours, purpose`. The 4090 Laptop budget is finite; monitor.

### 9.11 Laptop-specific issues
- **Thermal throttling.** Sustained training on a laptop 4090 Laptop (mobile or eGPU) can hit thermal limits. Monitor `nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv -l 60` during long runs. If clock drops > 15% from base, pause and improve cooling before continuing.
- **Power.** A laptop on battery throttles GPU power. Always plug in. Always disable sleep / hibernation during training.
- **Disk speed.** Synthetic image generation produces tens of GB of PNGs. NVMe ≥ 3 GB/s read recommended; SATA SSD will bottleneck the dataloader.
- **Driver versions.** CUDA 12.4 + cuDNN 9 + nightly PyTorch are required for Qwen3.5 + flash-attention 2 to play together. Pin them.

---

## 10. Reading list to skim before starting

These eleven papers / repos are the absolute minimum context. Skim them, summarise key claims into `reports/related_work.md`, and reference them by short label in subsequent reports.

1. **CHURRO** — Semnani et al., 2025, https://arxiv.org/abs/2509.19768. The strongest VLM for historical OCR, sets the metric (NLS), provides the dataset, and is the primary external baseline.
2. **Digital Peter** — Potanin et al., 2021, https://arxiv.org/pdf/2103.09354. Russian XVIII century HTR, the canonical Russian benchmark; auxiliary training data.
3. **Rabus, "Recognizing handwritten text in Slavic manuscripts"** — Universität Freiburg, https://www.academia.edu/38835297/. The reference for Old Cyrillic / Church Slavonic HTR via Transkribus; sets the prior baseline on Slavonic uncial.
4. **CyrillicHandwritingPOC** — dbrainio, 2023, https://arxiv.org/abs/2311.15896. Synthetic Cyrillic handwriting via Bezier curves + post-OCR T5 correction. Source of the procedural-baseline generator.
5. **Glyph-conditional DDPM for OCR** — Ding et al., 2023, https://arxiv.org/abs/2305.19543. The technique behind Phase 3 Option B (glyph-conditioned generation).
6. **Manchu Qwen2.5-VL OCR** — arXiv 2507.06761, 2025. Direct methodological template — low-resource VLM-OCR via diffusion-generated synthetic data; almost identical recipe to ours.
7. **Qwen3-VL Technical Report** — arXiv 2511.21631. Architecture details that mostly carry over to Qwen3.5.
8. **Qwen3.5 blog post** — https://qwen.ai/blog?id=qwen3.5 + the model card https://huggingface.co/Qwen/Qwen3.5-2B. Differences from Qwen3-VL, OCR benchmark numbers, recommended sampling parameters.
9. **PaddleOCR-VL** — https://huggingface.co/PaddlePaddle/PaddleOCR-VL. Strong recent open-weights baseline (Cyrillic + historical).
10. **Yandex archives OCR (Habr write-up)** — https://habr.com/ru/companies/yandex/articles/712510/. Industrial perspective on Russian XVIII–XIX c. handwriting OCR; source of synthetic-data tricks.
11. **gramoty.ru project pages** — http://gramoty.ru/birchbark/, plus the academic descriptions at IRYa RAN and NovGU. Necessary to understand the transcription convention, the [bracket] policy, and the dating system.

---

## 11. Execution order

Execute the phases strictly sequentially: Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Do not attempt to run multiple training runs in parallel — there is one 4090 Laptop. Each phase ends with a written report; only after the report is committed do you move to the next phase. Long-running steps (Phase 3 synth generation, Phase 4 training, each Phase 5 ablation) will exceed 24 hours of wall-clock; write interim status notes when that happens.

Within Phase 5, run ablations sequentially in priority order: A1 (50/50) → A1 (90/10) → A2 → A9 → A3 → A4 → A5 → A6 → A7 → A8 → A10. The first four are highest-value; if compute runs out, stop after A2 and document the omission. A9 (decoding) is inference-only and free; always do it.

---

**End of protocol.** When you finish Phase 7, write a single message to the human authors: `Hand-off ready at reports/handoff_<date>.zip. Highlights: <three bullet points of headline findings>.` Then stop.
