#!/usr/bin/env python3
"""Prepare train-split gramoty photos for SDXL LoRA style fine-tuning."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import orjson
from PIL import Image, ImageOps

CAPTION = (
    "<birchbark> Old Russian inscription scratched into birch bark, "
    "grey brown fibrous wooden surface, shallow dark incised Cyrillic letters, "
    "medieval Novgorod birchbark gramota, archaeological document photo"
)


def load_train_photo_records(jsonl_path: Path, raw_root: Path) -> list[dict]:
    rows: list[dict] = []
    for line in jsonl_path.read_bytes().splitlines():
        if not line.strip():
            continue
        row = orjson.loads(line)
        if row.get("split") != "train":
            continue
        candidates = [
            p
            for p in row.get("image_paths", [])
            if "/photo_" in p or Path(p).name.startswith("photo_")
        ]
        primary = row.get("primary_image")
        if primary and ("/photo_" in primary or Path(primary).name.startswith("photo_")):
            candidates.insert(0, primary)
        for rel in candidates:
            path = raw_root / rel
            if path.exists():
                rows.append(
                    {
                        "doc_id": row["doc_id"],
                        "image_path": path,
                        "date_raw": row.get("date_raw"),
                        "city": row.get("city"),
                    }
                )
                break
    return rows


def fit_square(im: Image.Image, size: int, pad_rgb: tuple[int, int, int]) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), pad_rgb)
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, default=Path("data/interim/birchbark_train.jsonl"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/gramoty"))
    p.add_argument("--output-dir", type=Path, default=Path("data/synthetic/sdxl_lora_gramoty_train"))
    p.add_argument("--max-images", type=int, default=400)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = load_train_photo_records(args.train_jsonl, args.raw_root)
    rng.shuffle(rows)
    rows = rows[: args.max_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] = []
    pad = (132, 124, 105)

    for i, row in enumerate(rows):
        out_name = f"gramoty_train_{i:04d}.png"
        with Image.open(row["image_path"]) as im:
            fit_square(im, args.resolution, pad).save(args.output_dir / out_name)
        metadata.append(
            {
                "file_name": out_name,
                "text": CAPTION,
                "doc_id": row["doc_id"],
                "source_path": str(row["image_path"]),
                "date_raw": row.get("date_raw"),
                "city": row.get("city"),
            }
        )

    meta_path = args.output_dir / "metadata.jsonl"
    meta_path.write_bytes(b"\n".join(orjson.dumps(r) for r in metadata) + b"\n")
    print(f"Wrote {len(metadata)} images + captions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
