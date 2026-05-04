# Phase 3 — Synthetic data: **three-source plan** (gramoty + Ostromir + Codex Suprasliensis)

**Date:** 2026-05-03  
**Status:** Research + **Suprasliensis ETL/crop scripts** in-repo; Ostromir wood-incision examples generated; first SDXL+LoRA style model trained.  
**Protocol:** `agent_protocol.md` (text bank, test leakage, no fabricated GT).

## 1. Goal

Build training data for birchbark-style generation / conditioning using **only**:

1. **Gramoty.ru** — in-domain birchbark photos + gold (existing Phase 1 pipeline; train text reusable with test-dedup).  
2. **Ostromir Gospel** — large authentic early Cyrillic / OCS **plain text** (no pixel-aligned line corpus).  
3. **Codex Suprasliensis** — [obdurodon digital edition](https://suprasliensis.obdurodon.org/): facsimile JPEG per folio + **per-line** diplomatic Slavonic in HTML (`span.os`).

## 2. Ostromir — **what we do with it**

Ostromir is **not** used as “download aligned line crops” (there is no standard line-image corpus like Suprasliensis facsimiles + edition list).

**Use:** treat a diplomatic plain-text file (e.g. [ponomar.net/files/ostromir.txt](http://www.ponomar.net/files/ostromir.txt)) as a **text bank** of line- or clause-sized shards. **Synthesize images** by rendering those strings with:

- Church Slavonic–friendly **serif / uncial** fonts (titlo combining marks, yus, yat, abbreviation glyphs).  
- **Wood / birchbark-like backgrounds** (grey-brown grain, scratches, cracks, uneven illumination).  
- **Incised text effect** (dark groove + light/shadow offsets), with varied per-line x positions rather than a typeset left-aligned block.

**Ground truth:** the exact Unicode string fed to the renderer (protocol-aligned “procedural exact” path, Option B–adjacent). Run the same **test_birch** hash / Levenshtein gate on shards as for any other text bank.

**Generated sample set:** `data/interim/ostromir_synth_10line/` contains **1,845** PNGs and **1,845** matching `.gold.txt` files, rendered as non-overlapping 10-line chunks from **18,451** cleaned Ostromir plaintext lines. Command:

```bash
PYTHONPATH=src python3 scripts/render_ostromir_samples.py \
  --all-chunks \
  --lines-per-image 10 \
  --width 1500 \
  --font-size 34 \
  --line-gap 10 \
  --margin 48 \
  --workers 16 \
  --seed 101 \
  --output-dir data/interim/ostromir_synth_10line
```

## 3. Codex Suprasliensis — **scrape + crops + GT**

### 3.1 Scrape (`scripts/scrape_suprasliensis.py`)

- Discover folios from `index.html` (`pages/supr*.html`).  
- For each folio: save HTML under `data/raw/suprasliensis/pages/`, download `images/<folio>.jpg`, append one JSONL record to `manifest.jsonl` with `lines[]` from `span.os` only (Greek `span.gk` ignored).  
- Default `--delay-seconds 0.75` between requests; use `--limit` for smoke tests.

### 3.2 Crops (`scripts/crop_suprasliensis_crops.py`)

- Read `manifest.jsonl`; for each folio draw **3–5 consecutive** edition lines (`--min-lines` / `--max-lines`).  
- **Geometry:** the edition gives one logical line per `<li>` but does **not** ship facsimile pixel boxes. The script detects high-pass dark-stroke row peaks, selects exactly `n_lines` centers, derives line boundaries from adjacent centers, then crops the requested span plus padding (`--equal-split` is only a debug fallback).  
- **Ground truth:** join the corresponding `lines[i]` with a **space** between lines; if a line ends with `-` (word wrap), drop the hyphen and concatenate the next line **without** an extra space (`merge_lines_for_gt`).  
- **Caveat:** if a band misaligns the ink, the image can disagree slightly with the edition line; crops are for **approximate** real-manuscript supervision, not guaranteed pixel-perfect line boxes.

**Example:**

```bash
python3 scripts/scrape_suprasliensis.py --output-dir data/raw/suprasliensis
python3 scripts/crop_suprasliensis_crops.py \
  --manifest-dir data/raw/suprasliensis \
  --out-images-dir data/interim/suprasliensis_crops/images \
  --out-jsonl data/interim/suprasliensis_crops/crops.jsonl
```

## 4. Gramoty (unchanged role)

Train-split JSONL images + transcriptions for in-domain supervision; **never** test-document text or test images in generator training. All synthetic text shards must pass the protocol **test-dedup** rules.

## 5. Dedup and next implementation steps

1. Normalise (NFC) per `birchbark_normalisation_policy.md`.  
2. Drop any candidate line / crop GT that matches **test_birch** (exact hash or Levenshtein < 5 to any test line).  
3. Build SDXL LoRA style dataset from **train-split gramoty photos only** (no val/test photos).  
4. Train first diffusion model: **SDXL + LoRA**, rank 16, trigger token `<birchbark>`, 2000 steps, local TensorBoard logging.

## 6. SDXL + LoRA v1 result

- Dataset: `data/synthetic/sdxl_lora_gramoty_train/` — **400** train-split gramoty photos padded to 1024 px with a constant `<birchbark>` style caption.
- Config: `configs/phase3/sdxl_lora_birchbark.yaml`.
- Launcher: `scripts/run_sdxl_lora_birchbark.sh`.
- Run dir: `runs/phase3_sdxl_lora_birchbark_v1_retry_20260503/`.
- Training: SDXL base, UNet LoRA rank 16, batch 1, gradient accumulation 4, bf16, 8-bit Adam, **2000** optimisation steps.
- Runtime: **1:43:01** for the successful retry (about 3.08 s/step).
- Checkpoints retained: `checkpoint-1100` … `checkpoint-2000`; final adapter: `pytorch_lora_weights.safetensors` (~46 MB).
- Smoke samples: `reports/figs/sdxl_lora_samples/sdxl_lora_sample_00.png` and `sdxl_lora_sample_01.png`.
- Note: the first full run stopped externally around step 399 with no traceback and no checkpoint because checkpointing was initially every 500 steps. The retry uses `checkpointing_steps=100`.

## 7. Glyph-conditioned prototype

- Script: `scripts/generate_sdxl_glyph_conditioned_samples.py`.
- Method: render exact Ostromir strings into a high-contrast glyph control image, then run `StableDiffusionXLControlNetPipeline` with `diffusers/controlnet-canny-sdxl-1.0` plus the trained birchbark LoRA.
- Smoke output: `reports/figs/sdxl_glyph_conditioned_smoke/`.
- Files: 2 generated PNGs, 2 glyph-control PNGs, 2 `.gold.txt` files.
- Initial observation: the generated images follow the broad line layout and look authentic, but Canny conditioning still allows glyph-level drift and can introduce unwanted museum-display framing. For OCR-grade exact GT, the next variant should use stronger glyph-preserving conditioning (lineart/softedge or a custom ControlNet/adapter over rendered glyph maps), tighter crop prompts/negatives, and optionally IP-Adapter only as a low-strength style reference from train-split gramoty photos.

### 7.1 Mini-ablation after prompt tightening

- Sweep script: `scripts/run_glyph_conditioning_sweep.py`.
- Prompt changes: close-up bark surface filling the whole frame; explicit negatives for museum case, glass, shelf, border, wide shot, printed ink, clean font.
- Canny sweep: `reports/figs/glyph_conditioning_sweep/canny_partial_contact_sheet.png`.
  - `scale=0.55`: strong birchbark texture but invents dense pseudo-glyphs.
  - `scale=0.80`: best Canny trade-off so far; line layout is plausible but glyph identities are still not exact.
  - `scale=1.05`: follows broad rows but glyphs become faint / washed or dissolve into bark texture.
- MistoLine sweep: `reports/figs/glyph_conditioning_sweep_mistoline/glyph_conditioning_sweep_contact_sheet.png`.
  - `scale=0.65`: visually best among tested settings; less museum framing, good bark style.
  - `scale=0.90`: too faint/washed; still not exact at the character level.
- SoftEdge attempt: `SargeZT/controlnet-sd-xl-1.0-softedge-dexined` failed to load in Diffusers because expected `diffusion_pytorch_model*.safetensors` weights were not present.

**Current recommendation:** do **not** scale this pipeline as OCR-labelled data yet. These outputs are good style references but still label-noisy at the glyph level. The next technical step should be a glyph-preserving image-to-image / ControlNet pipeline where the rendered glyph map is kept visible as a structural substrate (or a custom ControlNet trained on rendered glyph maps), with SDXL LoRA adding bark texture around it. IP-Adapter remains useful only as low-strength style conditioning from train-split gramoty photos, not as the main glyph constraint.

### 7.2 Exact-glyph bark overlay prototype

- Script: `scripts/generate_exact_glyph_bark_samples.py`.
- Method: generate **blank** 1024 px birchbark / manuscript-fragment backgrounds with SDXL + the trained LoRA, then deterministically engrave the exact rendered glyph mask onto the generated surface after diffusion.
- Why this variant matters: the visible text is no longer invented by SDXL/ControlNet, so OCR ground truth is exact by construction. SDXL only contributes background/style.
- Outputs:
  - First filled-mask attempt: `reports/figs/exact_glyph_bark_samples/contact_sheet.png` — label-safe but too printed / over-cracked.
  - Thin scratch attempt: `reports/figs/exact_glyph_bark_samples_thin/contact_sheet.png` — better text integration but prompt drifted toward natural birch tree bark.
  - Artifact-style prompt: `reports/figs/exact_glyph_bark_samples_artifact/contact_sheet.png` — best current prototype; flatter manuscript-fragment backgrounds and exact glyph masks.
- Current assessment: this is the first scalable OCR-labelled synthetic route that does **not** corrupt glyph identities. It should be treated as the near-term data-generation path, with further tuning on (1) glyph stroke style / font choice, (2) generated blank-background realism, and (3) optional low-strength IP-Adapter style references only after the exact mask remains visible.

### 7.3 Real-bark overlay pipeline (recommended)

After SDXL-generated blank-bark backgrounds still looked synthetic and let text overflow the bark, we replaced the diffusion-only background path with a **real-photo pipeline** that uses authentic gramoty bark surfaces:

- Script: `scripts/generate_real_bark_overlay_samples.py`.
- Steps: (1) load an original train-split gramoty photo via `source_path` from `data/synthetic/sdxl_lora_gramoty_train/metadata.jsonl`; (2) extract the bark foreground with HSV warm-hue prior + GrabCut + largest connected component; (3) crop to the bark bbox and resize keeping aspect ratio (snapped to multiples of 64); (4) word-wrap text into the **largest inscribed axis-aligned rectangle** of the eroded bark mask (so text can never spill onto the studio background); (5) detect existing scratched strokes via morphological **blackhat** and run `cv2.inpaint` only inside a dilated halo around the planned text area, leaving the rest of the bark texture intact; (6) engrave the exact glyph mask using shadow + highlight + dark fill, intersected with the bark mask.
- Output: `<stem>.png` final, `.background.png` (after local stroke removal), `.bark_mask.png`, `.glyphs.png`, `.strokes.png` (debug), `.gold.txt` exact GT, plus `metadata.jsonl` with image path, source photo, layout/seed, line bboxes, engrave parameters.
- Sample sweep results:
  - `reports/figs/real_bark_overlay_smoke/` — first pass without inpaint, shows constrained text masking already works (text stays inside bark).
  - `reports/figs/real_bark_overlay_v3/` — global blackhat inpaint over-smoothed fibrous bark (over-inclusive stroke mask).
  - `reports/figs/real_bark_overlay_v4..v5/` — switched to **localized** inpaint only under the planned text area; bark texture preserved everywhere else, but residual original strokes still visible under the text and engraved text looked printed.
  - `reports/figs/real_bark_overlay_v6..v7/` — added clean-patch sampling from the same image with LAB colour matching and cosine feathering; eliminated original strokes but introduced a faint rectangular seam when the source patch came from a brighter bark area.
  - `reports/figs/real_bark_overlay_v8/` — switched glyph rendering from solid filled letters to **outline-only** (Pillow `stroke_width=1`, `fill=0`, `stroke_fill=255`), so text now reads as a thin scratched ring rather than a printed glyph body.
  - `reports/figs/real_bark_overlay_v10/contact_sheet.png` — current best. Under-text region is cleaned by **median + bilateral filter on the destination itself** (no foreign patch imported), then enhanced with locally-derived high-frequency detail and a tiny grain. There is no rectangular seam, the colour matches the surrounding bark on every sample, and original inscriptions under the new text are erased.
- We also tried SDXL inpaint with our LoRA (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) but at any usable strength it drifted off-distribution (LoRA was trained against base SDXL UNet), so the deterministic OpenCV path is the chosen one.

**Status:** this real-bark overlay route is the recommended generator going forward: backgrounds are unimpeachably realistic (they are real gramoty), GT is exact by construction, text never escapes the bark, original strokes under the new text are erased without a visible seam, and engraved glyphs are rendered as scratched outlines. Future tuning is cosmetic only — font choice (uncial / Old Cyrillic instead of Liberation Serif), optional perspective warp, and richer photo selection that prefers gramoty with sparser original inscriptions.

## 8. References

- Codex Suprasliensis site: [suprasliensis.obdurodon.org](https://suprasliensis.obdurodon.org/)  
- Ostromir plain text (example): [ponomar.net/files/ostromir.txt](http://www.ponomar.net/files/ostromir.txt)
