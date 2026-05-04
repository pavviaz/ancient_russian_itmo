# Leakage audit (document-level)

**Train JSONL:** `data/interim/birchbark_train.jsonl` (884 rows)
**Test JSONL:** `data/interim/birchbark_test.jsonl` (252 rows)

## 1. Document ID overlap (must be empty)

- Overlapping doc_ids: **0**
- None.

## 2. Image filename overlap

Train shard basenames (all `image_paths`) are indexed; test rows are scanned for basename collisions (unexpected if filenames are unique per artefact photo).

- Collisions: **0**

## 3. Normalised gold-text hash overlap

SHA256 of `normalize(transcription_diplomatic)` for train vs test; duplicates can indicate shared template text across different documents (rare).

- Test rows whose hash appears in train: **2**

```json
[
  {
    "doc_id": "novgorod/327",
    "sha256_norm_text": "3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112"
  },
  {
    "doc_id": "novgorod/680",
    "sha256_norm_text": "3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112"
  }
]
```

## Verdict

- **REVIEW** — see counts above; investigate before trusting eval.
