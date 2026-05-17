# Birchbark OCR — fine-tuning a 2B-parameter VLM on Old Russian *gramoty*

End-to-end research pipeline for OCR on Old Russian (East Slavic) birchbark documents from `gramoty.ru` (Novgorod, Pskov, Smolensk, Staraya Russa, Torzhok; 11th-15th c.). The headline contribution is a Qwen3.5-2B vision-language model fine-tuned with LoRA on a four-stage diffusion-based synthetic-data pipeline; relative to off-the-shelf systems the fine-tune cuts CER from ≥ 1.4 to **0.571** and lifts NLS from ≤ 0.076 to **0.478** on the held-out `test_birch` split.

> **The full paper-source document is at [`reports/FINDINGS_for_paper.md`](reports/FINDINGS_for_paper.md).** It is the authoritative reference for everything below: methodology, results, ablations, limitations, future work, and inline bibliography. This README is a thin entry point.

## Headline result

| split | n scored / n total | decoding | CER (raw) | CER (brackets stripped) | NLS |
|---|---:|---|---:|---:|---:|
| `val_birch`  | 117 / 117 | greedy  | 0.551 | 0.504 | 0.476 |
| `val_birch`  | 117 / 117 | beam=4  | 0.527 | 0.507 | 0.500 |
| `test_birch` | 246 / 252 | greedy  | 0.583 | 0.553 | 0.454 |
| **`test_birch`** | **246 / 252** | **beam=4 (recommended)** | **0.571** | **0.561** | **0.478** |
| `test_birch` (3-seed mean ± std at r=32, beam=4) | 246 / 252 | beam=4 | 0.591 ± 0.034 | 0.583 ± 0.052 | 0.456 ± 0.021 |

Compared to the same Qwen3.5-2B base model zero-shot (CER 6.59 / NLS 0.031): **91% relative reduction in raw CER, 15.4× increase in NLS** at the published seed; against the strongest external open-weights baseline (CHURRO-3B with fair postprocessing): **4.6× lower CER, 53× higher NLS**.

Adapter checkpoint: `runs/phase4_v5/mixed_80_20__Qwen_Qwen3_5-2B__ceil5ep/checkpoint-900/` (~ 200 MB; gitignored, released as a separate artifact).

## Repo layout

```
ancient_russian_itmo/
├── reports/
│   ├── FINDINGS_for_paper.md        ← single-source-of-truth for the paper (read this first)
│   ├── report_phase1.md ... phase3  ← phase-by-phase write-ups
│   ├── related_work.md              ← skim notes (canonical citations are in FINDINGS §16)
│   ├── figs/findings/*.png          ← 4 paper-ready figures (pipeline, U-curve, train curve, qualitative grid)
│   ├── eval/                        ← all per-row predictions and summary JSON files
│   └── audit/dataset_a/             ← Phase-3 audit grids
├── data/
│   ├── raw/gramoty/                 ← scrape manifests (images gitignored; regenerable from manifests)
│   ├── interim/                     ← line-level JSONLs (birchbark_train/val/test, suprasliensis crops)
│   ├── splits/                      ← frozen document-level IDs + sha256
│   │   └── phase4_v3/               ← canonical mix-pool JSONLs for the 9-point synth/real sweep
│   └── processed/
│       ├── tablet_text_pool.txt     ← 8 k pruned text bank for synthetic engraving
│       ├── synth_carved/dataset_a/  ← deterministic v18-engraver outputs (manifests committed)
│       ├── synth_kandinsky/dataset_a/ ← Kandinsky-refined synth (manifests + audit summary committed)
│       └── qwen_clear_gramoty/      ← stage-1 clean-bark substrates (manifest committed)
├── src/birchbark_ocr/
│   ├── data/                        ← gramoty parsing, text normalisation, splits
│   ├── synth/                       ← Ostromir text helpers
│   ├── train/                       ← Qwen-VL collator, dataset, deterministic augmentation
│   └── eval/                        ← CER / NLS metrics
├── scripts/
│   ├── make_birchbark_splits.py     ← Phase 1: build document-level train/val/test
│   ├── run_phase2_baselines.py      ← Phase 2: zero-shot baselines
│   ├── run_phase2_churro_paddle.sh  ← Phase 2: CHURRO-3B baseline
│   ├── train_qwen_vl_lora.py        ← Phase 4: champion fine-tuning script
│   ├── eval_qwen_vl_lora.py         ← Phase 4: standalone evaluation (greedy / beam search)
│   ├── run_v6_parallel_eval.sh      ← Phase 4: re-eval the 4 v6 cells across the A100s
│   ├── run_churro_test_birch.sh     ← Phase 4: fair-postproc CHURRO comparison
│   ├── aggregate_phase4_v6.py       ← Phase 4: collect multi-seed and rank ablation results
│   └── launch_phase4_*.sh           ← orchestration scripts for the v5 / v6 grid
├── notebooks/
│   ├── qwen_image_edit_bark_clearing.ipynb  ← stage-1 of the synthetic pipeline
│   └── kandinsky_synth_refine.ipynb         ← stage-3 of the synthetic pipeline
├── assets/fonts/churchslavonic/      ← 16 Church-Slavonic-friendly serifs (Acathist, Cathisma, …)
├── configs/                          ← Hydra configs (legacy from Phase 1/2)
├── envs/                             ← `uv pip freeze` outputs for reproducible venv installs
├── pyproject.toml                    ← Python package metadata
├── agent_protocol.md                 ← original research protocol (for context)
└── README.md                         ← this file
```

Items intentionally **gitignored** (regenerable or out-of-scope): `runs/`, `.venv*/`, `dataset_a.zip`, `data/raw/**/*.{jpg,png,html}`, exploration figures under `reports/figs/<everything-but-findings>/`, exploration notebooks (`openrouter_*`, `qwen_synth_enhancement.ipynb`, …), and the legacy top-level scratch script `qwen2.5vl_oldrus_v2.py`. See `.gitignore` for the full policy.

## Phases at a glance

| phase | summary | report | wall-clock | GPU-h |
|---|---|---|---:|---:|
| 1 | Crawl gramoty.ru (1 260 docs, 2 s throttle), build stratified document splits | [`reports/report_phase1.md`](reports/report_phase1.md) | ~ 3 h | 0 |
| 2 | Zero-shot baselines (Tesseract, EasyOCR, TrOCR, Qwen3.5-{0.8B,2B}, CHURRO-3B) on `test_birch` | [`reports/report_phase2.md`](reports/report_phase2.md) | ~ 4 h | ~ 4 |
| 3 | Four-stage synthetic pipeline (Qwen-Image-Edit clean-bark → renderer v18 carve → Kandinsky 6 Pro I2I → aspect audit). Yield: 4 374 trainable rows from 5 200 attempts. | [`reports/report_phase3.md`](reports/report_phase3.md) | ~ 4 h GPU + ~ 64 h API | ~ 4 |
| 4 | Qwen3.5-2B + LoRA on `mixed_80_20`, with U-curve sweep, LoRA-target ablation, decoding ablation, multi-seed variance, LoRA-rank ablation, fair-postproc CHURRO comparison | [`reports/FINDINGS_for_paper.md`](reports/FINDINGS_for_paper.md) | ~ 50 h | ~ 165 (4 × A100) |
| **total** |  |  |  | **~ 197** + 64 h Kandinsky API |

## Reproducing the headline run

```bash
# 0. One-time install (fast: uv-managed venv against pyproject.toml extras)
python -m venv .venv-qwen-edit-multi
source .venv-qwen-edit-multi/bin/activate
pip install -e ".[dev]"
# transformers>=5.7, peft>=0.19, torch 2.6+cu124, jiwer, python-Levenshtein

# 1. (Optional) Rebuild the frozen document splits — IDs are already committed
python scripts/make_birchbark_splits.py \
    --manifest data/raw/gramoty/manifest.jsonl \
    --out-dir data/splits \
    --seed 1337

# 2. Use the committed phase4_v3 mix-pool JSONLs directly (canonical splits):
#       data/splits/phase4_v3/mixed_80_20_train.jsonl   (5 468 rows: 4 374 synth + 1 094 real)
#       data/splits/phase4_v3/val.jsonl                 (117 rows of held-out gramoty val)

# 3. Train champion (single A100-40GB, ~ 5 h wall-clock)
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_qwen_vl_lora.py \
    --model Qwen/Qwen3.5-2B \
    --train-jsonl data/splits/phase4_v3/mixed_80_20_train.jsonl \
    --val-jsonl   data/splits/phase4_v3/val.jsonl \
    --output-dir  runs/repro/mixed_80_20_2B \
    --epochs 5 \
    --per-device-batch-size 4 --grad-accum 4 \
    --learning-rate 1e-4 --warmup-ratio 0.05 --max-grad-norm 0.5 \
    --eval-steps 100 --save-steps 100 --logging-steps 20 \
    --early-stopping-patience 5 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05

# 4. Evaluate the best checkpoint on the held-out test set (~ 9 min @ beam=4)
python scripts/eval_qwen_vl_lora.py \
    --base-model Qwen/Qwen3.5-2B \
    --adapter   runs/repro/mixed_80_20_2B/<best_checkpoint> \
    --jsonl     data/interim/birchbark_test.jsonl \
    --out-pred    reports/eval/test_predictions_repro.jsonl \
    --out-summary reports/eval/test_summary_repro.json \
    --num-beams 4 --max-new-tokens 160 --device cuda:0
```

The default `--lora-target` already includes the all-modules string (LM self-attention + linear-attention `in_proj_*` + MLP + vision-encoder `qkv,fc1,fc2`), and `--no-expand-tokens` is the default since the v3 ablation. See FINDINGS §4.5 for why these two defaults matter — in short, the canonical "self-attention + MLP" target list freezes the vision encoder and 75 % of the LM, producing CER ≥ 1.

## Phase 2 baselines

CHURRO-3B uses **Python ≤ 3.12** + a separate venv:

```bash
bash scripts/setup_churro_paddle_venv.sh    # one-time: uv, CPython 3.12, deps, churro-ocr install hf
source .venv-churro-paddle/bin/activate
bash scripts/run_phase2_churro_paddle.sh    # CHURRO only → runs/phase2/raw_predictions_churro.jsonl
# Smoke run: bash scripts/run_phase2_churro_paddle.sh --limit 5

# Fair-postproc re-evaluation on test_birch (with smarter <Line> XML extraction):
bash scripts/run_churro_test_birch.sh
# → reports/eval/test_summary_churro.json  (CER 2.628 / NLS 0.009 — see FINDINGS §6.6.3)
```

Other zero-shot baselines (Tesseract, EasyOCR, TrOCR, Qwen3.5-{0.8B, 2B}):

```bash
python scripts/run_phase2_baselines.py     # → runs/phase2/metrics.json
python scripts/merge_phase2_metrics.py     # → metrics_phase2_all.json + reports/figs/baseline_phase2_all.png
```

Numbers and per-row tables: [`reports/report_phase2.md`](reports/report_phase2.md). Phase-2 numbers are also restated alongside the fine-tune in FINDINGS §7.

## Multi-seed variance and LoRA-rank ablation (Phase 4 v6)

Four ablation cells (seed ∈ {1337, 2026, 4242} × r=32, plus seed 1337 × r ∈ {16, 64}) were run on `mixed_80_20`, 5-epoch ceiling, with the same hyperparameters as the champion. Re-evaluation is parallelised across the four A100s:

```bash
bash scripts/run_v6_parallel_eval.sh    # ~ 18 min total (one cell per A100, beam=4)
python scripts/aggregate_phase4_v6.py   # → reports/eval/v6_aggregate.json
```

Results: 3-seed test-set CER 0.591 ± 0.034 / NLS 0.456 ± 0.021; rank=32 wins, rank=16 is competitive (-0.008 test CER), rank=64 over-fits (+0.044 test CER). Full details in FINDINGS §6.6.

## Decoding ablation (A9)

```bash
# greedy / greedy-512 / beam=4-160 / greedy with repetition-penalty=1.1 on val_birch
# (driver script lives inline; predictions are at reports/eval/a9_decoding/)
# beam=4 wins by 0.024 CER and 0.024 NLS over greedy on val; transferred to test_birch as the headline.
```

Summary table at FINDINGS §6.5; decoding-ablation predictions are committed at `reports/eval/a9_decoding/val_*_summary.json`.

## Synthetic pipeline (Phase 3)

The four-stage pipeline produces 4 374 line-level synthetic images with byte-exact ground truth. Stages 1 and 3 use external models; stage 2 is deterministic in pure Python; stage 4 is a 1-line audit rule.

```bash
# Stage 1: Qwen-Image-Edit clean-bark substrate (~ 12 s/image on A100, bf16)
jupyter execute notebooks/qwen_image_edit_bark_clearing.ipynb
# → data/processed/qwen_clear_gramoty/{images/, manifest.jsonl}  (1 232 substrates)

# Stage 2: deterministic v18 engraver (~ 0.4 s/image on CPU)
# (engraver lives inside scripts/render_carved_v18.py; configuration is captured in
#  data/processed/synth_carved/dataset_a/summary.json)

# Stage 3: Kandinsky 6 Pro I2I refinement (~ 46 s/image, closed API)
jupyter execute notebooks/kandinsky_synth_refine.ipynb
# → data/processed/synth_kandinsky/dataset_a/{refined/, manifest.jsonl}

# Stage 4: aspect-only audit (~ 5 min, CPU)
# Produces manifest_audit.jsonl + manifest_clean.jsonl + audit_summary.json
# 87.1 % effective yield (4 374 / 5 021 carved attempts)
```

The 4 374-image refined dataset is the practical reproducibility anchor; the carved-stage-2 outputs alone are also committed (manifests only) and can be used in lieu of stage 3 if Kandinsky API access is unavailable (with an expected CER regression of ~ 0.05-0.10 — see FINDINGS §13 limitation 9).

The pipeline figure (`reports/figs/findings/fig_synth_pipeline.png`) shows three (real → clean → carved → refined) tuples side-by-side.

## Hardware

- 4 × NVIDIA A100-PCIE-40GB (one cell per GPU; ~ 70-80 % utilisation per cell with bf16, gradient checkpointing, sdpa).
- Phase 1 / Phase 2 baselines also work on a single 16 GB consumer GPU.

## Provenance and licensing

- **Corpus.** Novgorod Birchbark Letters via [`gramoty.ru`](http://gramoty.ru/birchbark/). Crawled with a 2 s per-request throttle; we redistribute *derived* features only (clean-bark substrates, synthetic engravings, line-level normalised gold strings), not the raw scraped pages.
- **Base model.** [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) under the Qwen Research License.
- **Stage-1 model.** [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) under its own license; pipeline consumes outputs only.
- **Stage-3 model.** Kandinsky 6 Pro (closed API);
- **Comparison baseline.** [`stanford-oval/churro-3B`](https://huggingface.co/stanford-oval/churro-3B); CHURRO paper's prompt and inference settings.
- **Fonts.** 16 Church-Slavonic serifs in `assets/fonts/churchslavonic/` (each font's own license bundled with it).
- **Released LoRA adapter.** ~ 200 MB delta on the Qwen3.5-2B base; we plan to release it under CC-BY-4.0 with attribution to gramoty.ru and the paper authors. **Intended use: research and pre-transcription assistance.** At CER 0.57 the system is *not* a substitute for a palaeographer — see FINDINGS §15 for the full intended-use note.

## Where to look next

1. **For the paper draft** — start with [`reports/FINDINGS_for_paper.md`](reports/FINDINGS_for_paper.md). It contains an Abstract (§0), Related work (§1A), Method (§3-§5), Results with all four ablations and qualitative grid (§6-§7), Limitations (§13), Future work (§14), Ethics (§15), and an inline References section (§16).
2. **For the headline number alone** — see the table at the top of this README or the headline block of FINDINGS.
3. **For reproducibility** — FINDINGS §9 has the full hardware/software/determinism section; the recipe above is the 3-step quickstart.
4. **For the synthetic pipeline figure** — `reports/figs/findings/fig_synth_pipeline.png` (paper main figure); details in FINDINGS §3.
5. **For variance and ablations** — FINDINGS §6.5 (decoding), §6.6.1 (multi-seed), §6.6.2 (LoRA rank); aggregate JSON at `reports/eval/v6_aggregate.json`.

If you find this work useful for digital-palaeography research, please cite the paper (forthcoming) and acknowledge the gramoty.ru curators.
