# Phase 1 — Data acquisition and curation (report)

**Date:** 2026-05-03  
**Protocol:** `agent_protocol.md` v2.0  
**Repository root:** `/home/pavviaz/Documents/ancient_russian_itmo`

## Environment and hardware check

| Check | Result |
|-------|--------|
| `nvidia-smi` | **Available** — NVIDIA GeForce RTX 4090 Laptop, 16376 MiB, driver 595.58, CUDA 13.2 |
| Disk (`df -h` on workspace volume) | `/dev/nvme1n1p2` **846 GB free** of 929 GB (5% used) |
| `robots.txt` | `http(s)://gramoty.ru/robots.txt` returned **404** at crawl time; crawler uses **2 s** minimum delay between requests (§2.2). |

### Git commit SHA

**N/A** — `git` is not installed in this environment (`git: command not found`). Install git locally and record `git rev-parse HEAD` for reproducibility.

### Resolved Python packages

| Snapshot | Scope |
|----------|--------|
| `envs/freeze_20260503.txt` | Core scraping stack (httpx, pandas, hydra, …) from early Phase 1 |
| `envs/freeze_phase2_20260503.txt` | User-site additions for baseline stack (`torch`, `transformers`, `jiwer`, `churro-ocr`, `matplotlib`, …) — **see Phase 2 report for caveats (Python 3.14)** |

### Commands (Phase 1)

```bash
cd /home/pavviaz/Documents/ancient_russian_itmo
PYTHONPATH=src pip install --user --break-system-packages -e .   # PEP 668 environment

python3 scripts/gramoty_scrape.py index --output-dir data/raw/gramoty --delay-seconds 2.0

# Full corpus (long-running; resume skips existing page.html unless --refetch)
PYTHONPATH=src python3 scripts/gramoty_scrape.py scrape \
  --output-dir data/raw/gramoty --delay-seconds 2.0 --limit 0

# If scrape is interrupted: rebuild manifest from per-doc meta.json, then interim JSONL
PYTHONPATH=src python3 scripts/rebuild_manifest_from_disk.py
PYTHONPATH=src python3 scripts/build_interim_birchbark_jsonl.py

# Splits were originally frozen from list metadata (1260 docs). Do NOT regenerate from a partial manifest.
PYTHONPATH=src python3 scripts/make_birchbark_splits.py \
  --manifest data/raw/gramoty/manifest.jsonl --out-dir data/splits --seed 1337   # only after full scrape + review

PYTHONPATH=src python3 scripts/leakage_audit.py
```

**Hyperparameters:** split seed `1337`; throttle `delay_seconds=2.0`; stratification = century × site bucket (`birchbark_splits.py`).

### Runtime (scraping)

- **Index:** ~2 s wall-clock.  
- **Full scrape (completed):** `python3 scripts/gramoty_scrape.py scrape … --limit 0` finished **1260 / 1260** documents. Observed wall-clock **~3 h 1 min** end-to-end on this machine (~**8.6 s/doc** average including tqdm overhead, HTML, images, and **2 s** throttle). Final line: `Manifest: data/raw/gramoty/manifest.jsonl (1260 docs)`.  
- **`manifest.jsonl`:** **1260** records (matches `document_index.jsonl`).

---

## Datasets

| Source | Role | Status |
|--------|------|--------|
| gramoty.ru birchbark list + documents | Primary corpus | **Complete:** **1260** indexed + **1260** scraped (`manifest.jsonl`), per-document `page.html`, `meta.json`, and downloaded thumbs under `data/raw/gramoty/documents/`. |
| Codex Suprasliensis, CHURRO-DS, Digital Peter, … | train_aux | **Not ingested** (future Phase 1 extension per protocol). |

### Per-split document counts (frozen IDs — list metadata)

Stratified by century × coarse site bucket, seed **1337** (unchanged):

| Split | Documents |
|-------|-----------|
| train | 884 |
| val | 124 |
| test | 252 |
| **Total** | **1260** |

Files: `data/splits/birchbark_{train,val,test}_ids.txt` + `.sha256`, `data/splits/split_summary.json`.

**Policy:** Splits stay tied to the **full** indexed corpus. After **complete** scrape, optionally re-run `make_birchbark_splits.py --manifest ...` if document-page metadata systematically overrides list fields; then **re-hash** split sidecars.

**Post-scrape verification (2026-05-03):** `manifest.jsonl` was rebuilt from **1260** `meta.json` files. A stratified split recomputed from that manifest (seed **1337**) matched the frozen `birchbark_{train,val,test}_ids.txt` sets **exactly** (884 / 124 / 252). Split files and **`data/splits/*.sha256`** were **not** changed.

### Interim JSONL (document-level)

Built by `scripts/build_interim_birchbark_jsonl.py` from `manifest.jsonl` + frozen split IDs:

| File | Purpose |
|------|---------|
| `data/interim/birchbark_train.jsonl` | Train shard rows (present in manifest) |
| `data/interim/birchbark_val.jsonl` | Val shard rows |
| `data/interim/birchbark_test.jsonl` | Test shard rows |

**Counts after full scrape:** **884** train / **124** val / **252** test rows — aligned with frozen split ID files.

Each row includes: `doc_id`, `split`, `text` (diplomatic transcription), `image_paths`, `primary_image` (photo thumb preferred), optional `url`, `date_raw`, `city`.

### Century / site histograms

Unchanged from list-derived statistics (see previous report revision): XI–XV century counts and Novgorod-heavy site buckets remain valid for the **index**.

### Character distribution (birchbark vs auxiliary)

**Still pending** full diplomatic export + auxiliary JSONL (§2.3).

---

## Parsing and normalisation

- Transcription (diplomatic): `div.text-area.original-text` on document pages (`gramoty.py`).  
- Images: `img` with `/thumbs/photo_*` and `/thumbs/drawing_*` → `documents/<safe_id>/images/`.  
- Policy: `data/interim/birchbark_normalisation_policy.md`.

---

## Leakage audit

See **`data/splits/leakage_audit.md`** (regenerated on full interim JSONL).

- **Document ID overlap:** none.  
- **Image basename overlap:** none.  
- **Normalised gold-text hash overlap:** **2** test documents (`novgorod/327`, `novgorod/680`) share a SHA256 with some train document — likely **duplicate diplomatic strings** across different artefacts; **human review** recommended before treating eval as contamination-free. Script exits non-zero (**REVIEW**).

---

## Narrative

Infrastructure now covers **resume-friendly manifest reconstruction**, **document-level interim JSONL** for Phase 2 wiring, and a **real leakage script** (`scripts/leakage_audit.py`). The **full gramoty scrape** (**1260** documents) **completed** successfully with final **`manifest.jsonl`**. Interim shards match frozen splits; leakage automation flagged **two** duplicate normalised-text hashes on the **test** split for manual review. Next blocking work for “Phase 1 complete” beyond downloads is **line-image segmentation** and auxiliary ETL per protocol.

---

## Open items before tagging `phase1-final`

1. ~~Full **`scrape`** + **`manifest.jsonl` (1260)~~ — **done**.  
2. Optionally regenerate splits from **`manifest.jsonl`** if document-page metadata materially differs from list rows; **re-hash** split sidecars if IDs change.  
3. ~~Rebuild **`data/interim/birchbark_*.jsonl`**~~ — **done** (884 / 124 / 252).  
4. Resolve **leakage REVIEW** for duplicate normalised gold text (`novgorod/327`, `novgorod/680`) — drop, merge, or document as benign.  
5. Line-level crops + extended character histograms (§2.2–2.3).  
6. Install **git**, commit, and `git tag phase1-final`.
