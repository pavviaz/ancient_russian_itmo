#!/usr/bin/env python3
"""Document-level leakage checks between birchbark train and test JSONL shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import orjson

from birchbark_ocr.eval.metrics import normalize


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows


def basename_set(paths: list[str | None]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        if not p:
            continue
        out.add(Path(p.replace("\\", "/")).name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/interim/birchbark_train.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("data/interim/birchbark_test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/splits/leakage_audit.md"))
    args = ap.parse_args()

    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)

    train_doc_ids = {r["doc_id"] for r in train_rows}
    test_doc_ids = {r["doc_id"] for r in test_rows}
    doc_overlap = sorted(train_doc_ids & test_doc_ids)

    train_names: set[str] = set()
    train_text_hashes: set[str] = set()
    for r in train_rows:
        train_names |= basename_set(r.get("image_paths") or [])
        t = normalize(str(r.get("text") or ""))
        if t:
            train_text_hashes.add(hashlib.sha256(t.encode("utf-8")).hexdigest())

    filename_hits: list[dict] = []
    hash_hits: list[dict] = []
    for r in test_rows:
        did = r["doc_id"]
        for ip in r.get("image_paths") or []:
            bn = Path(ip.replace("\\", "/")).name
            if bn in train_names:
                filename_hits.append({"doc_id": did, "basename": bn})
        t = normalize(str(r.get("text") or ""))
        if t:
            hx = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if hx in train_text_hashes:
                hash_hits.append({"doc_id": did, "sha256_norm_text": hx})

    lines: list[str] = [
        "# Leakage audit (document-level)",
        "",
        f"**Train JSONL:** `{args.train}` ({len(train_rows)} rows)",
        f"**Test JSONL:** `{args.test}` ({len(test_rows)} rows)",
        "",
        "## 1. Document ID overlap (must be empty)",
        "",
        f"- Overlapping doc_ids: **{len(doc_overlap)}**",
    ]
    if doc_overlap:
        lines.append(f"- IDs: `{doc_overlap[:20]}`" + (" …" if len(doc_overlap) > 20 else ""))
    else:
        lines.append("- None.")
    lines += [
        "",
        "## 2. Image filename overlap",
        "",
        "Train shard basenames (all `image_paths`) are indexed; test rows are scanned for basename collisions "
        "(unexpected if filenames are unique per artefact photo).",
        "",
        f"- Collisions: **{len(filename_hits)}**",
    ]
    if filename_hits[:10]:
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(filename_hits[:10], indent=2, ensure_ascii=False))
        lines.append("```")

    lines += [
        "",
        "## 3. Normalised gold-text hash overlap",
        "",
        "SHA256 of `normalize(transcription_diplomatic)` for train vs test; duplicates can indicate shared "
        "template text across different documents (rare).",
        "",
        f"- Test rows whose hash appears in train: **{len(hash_hits)}**",
    ]
    if hash_hits[:10]:
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(hash_hits[:10], indent=2, ensure_ascii=False))
        lines.append("```")

    lines += ["", "## Verdict", ""]
    ok = not doc_overlap and not filename_hits and not hash_hits
    lines.append(
        "- **PASS** — no doc overlap, filename collisions, or shared normalised-text hashes."
        if ok
        else "- **REVIEW** — see counts above; investigate before trusting eval."
    )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
