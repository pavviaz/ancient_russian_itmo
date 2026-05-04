#!/usr/bin/env python3
"""Build document-level train/val/test JSONL from manifest + frozen split id lists."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson


def load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def preferred_primary_image(images: list[str]) -> str | None:
    """Prefer photograph thumbs over line drawings for OCR baselines."""
    if not images:
        return None
    photos = [p for p in images if "photo_" in p.replace("\\", "/")]
    draws = [p for p in images if "drawing_" in p.replace("\\", "/")]
    cand = photos or draws or images
    cand_sorted = sorted(cand)
    return cand_sorted[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/raw/gramoty/manifest.jsonl"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    args = ap.parse_args()

    train_ids = load_ids(args.splits_dir / "birchbark_train_ids.txt")
    val_ids = load_ids(args.splits_dir / "birchbark_val_ids.txt")
    test_ids = load_ids(args.splits_dir / "birchbark_test_ids.txt")

    def split_for(did: str) -> str | None:
        if did in train_ids:
            return "train"
        if did in val_ids:
            return "val"
        if did in test_ids:
            return "test"
        return None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for line in args.manifest.read_bytes().splitlines():
        if not line.strip():
            continue
        m = orjson.loads(line)
        did = m["doc_id"]
        sp = split_for(did)
        if sp is None:
            continue
        images = m.get("images") or []
        primary = preferred_primary_image(images)
        row = {
            "doc_id": did,
            "split": sp,
            "text": m.get("transcription_diplomatic") or "",
            "text_spaced": m.get("transcription_spaced"),
            "image_paths": images,
            "primary_image": primary,
            "url": m.get("url"),
            "date_raw": m.get("date_raw"),
            "city": m.get("city"),
        }
        buckets[sp].append(row)

    for name in ("train", "val", "test"):
        path = args.out_dir / f"birchbark_{name}.jsonl"
        with path.open("wb") as f:
            for row in sorted(buckets[name], key=lambda r: r["doc_id"]):
                f.write(orjson.dumps(row))
                f.write(b"\n")
        print(f"Wrote {path} ({len(buckets[name])} docs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
