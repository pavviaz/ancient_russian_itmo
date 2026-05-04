#!/usr/bin/env python3
"""Build a unified OCR dataset from gramoty photos + Suprasliensis crops.

Outputs one JSONL per split with normalised rows:

    {"image": "<repo-relative path>", "text": "<gold text>", "source": "gramoty|suprasliensis",
     "doc_id": "...", "split": "train|val|test"}

Suprasliensis crops are *bark-tinted* on the fly (duotone colour shift +
optional bark texture multiply) so they look closer to a birchbark surface.
Originals are kept untouched.

Notes:
  - Gramoty splits come from ``data/splits/birchbark_{train,val,test}_ids.txt``.
  - Suprasliensis is added entirely to the train split (it's parchment, used
    for character-level pretraining; we never validate on parchment).
  - Augmentations should run at training time (Albumentations); this script
    only normalises images and labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LACUNA_RE = re.compile(r"[…⁞]+|\[[^\]]+\]|\([^)]+\)")


def portable_path(path: Path, root: Path | None = None) -> str:
    """Return a repo-relative path when possible, otherwise the original path."""
    root = (root or Path.cwd()).resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def normalise_text(text: str) -> str:
    """Strip restoration markup and collapse whitespace.

    Keeps Cyrillic characters (including supplements) and standard punctuation.
    Removes editorial brackets, lacunae markers, em dashes used as separators,
    and runs of whitespace.
    """
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[\(\)\[\]\{\}]", "", t)
    t = t.replace("…", " ").replace("⁞", " ")
    t = re.sub(r"-{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_split_ids(splits_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        p = splits_dir / f"birchbark_{name}_ids.txt"
        out[name] = {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return out


def gramoty_rows(
    manifest: Path,
    splits: dict[str, set[str]],
    *,
    only_photos: bool = True,
    skip_drawings: bool = True,
    min_chars: int = 12,
):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        did = m["doc_id"]
        split = next((s for s, ids in splits.items() if did in ids), None)
        if split is None:
            continue
        text = normalise_text(m.get("transcription_diplomatic") or "")
        if len(text) < min_chars:
            continue
        for img_rel in m.get("images") or []:
            name = Path(img_rel).name
            if skip_drawings and name.startswith("drawing_"):
                continue
            if only_photos and not name.startswith("photo_"):
                continue
            yield {
                "image_relpath": img_rel,
                "text": text,
                "doc_id": did,
                "split": split,
                "source": "gramoty",
            }


def suprasliensis_rows(jsonl: Path, *, min_chars: int = 12):
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        text = normalise_text(m.get("gold") or "")
        if len(text) < min_chars:
            continue
        yield {
            "image_path": m["image_path"],
            "original_image": m.get("source_image"),
            "text": text,
            "doc_id": m.get("folio_id", ""),
            "split": "train",
            "source": "suprasliensis",
        }


# ---------------------------------------------------------------------------
# Suprasliensis -> bark colour shift
# ---------------------------------------------------------------------------

# Duotone palette: dark ink colour and light bark colour. These were eyeballed
# from real gramoty.ru photos: warm dark brown and warm tan/biscuit.
DARK_RGB = np.array([28, 18, 10], dtype=np.float32)
LIGHT_RGB = np.array([198, 156, 106], dtype=np.float32)


def bark_tint_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Map luminance to a warm duotone bark palette.

    Output(x) = LIGHT * t(x) + DARK * (1 - t(x)),
    where t(x) is contrast-stretched luminance in [0, 1].
    """
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = rgb.astype(np.float32)
    lo, hi = np.percentile(gray, (5, 95))
    if hi - lo < 1:
        hi = lo + 1
    t = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    out = LIGHT_RGB[None, None, :] * t[..., None] + DARK_RGB[None, None, :] * (1 - t[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_bark_grain(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add subtle warm grain so the duotone surface does not look paper-flat."""
    h, w = rgb.shape[:2]
    fine = rng.normal(0, 5, size=(h, w)).astype(np.float32)
    coarse = rng.normal(0, 10, size=(h // 8 + 1, w // 8 + 1)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    grain = (fine + coarse)[..., None]
    out = np.clip(rgb.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return out


def multiply_bark_texture(rgb: np.ndarray, bark_rgb: np.ndarray) -> np.ndarray:
    """Soft-light blend with a real bark texture sample for fibre realism."""
    h, w = rgb.shape[:2]
    bg = cv2.resize(bark_rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # Use only the high-frequency part of the bark sample (fibre detail), not its
    # low-frequency colour, so the duotone palette stays in charge.
    blur = cv2.GaussianBlur(bg, (0, 0), sigmaX=12, sigmaY=12)
    detail = bg - blur
    out = rgb.astype(np.float32) + detail * 110.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_bark_shift(
    pil_img: Image.Image,
    *,
    bark_pool: list[Path] | None,
    rng: np.random.Generator,
) -> Image.Image:
    rgb = np.array(pil_img.convert("RGB"))
    out = bark_tint_grayscale(rgb)
    out = add_bark_grain(out, rng)
    if bark_pool:
        bp = bark_pool[int(rng.integers(0, len(bark_pool)))]
        try:
            bark = np.array(Image.open(bp).convert("RGB"))
            # Crop a random patch the size of the input region.
            bh, bw = bark.shape[:2]
            ph, pw = out.shape[:2]
            if bh > ph and bw > pw:
                y0 = int(rng.integers(0, bh - ph))
                x0 = int(rng.integers(0, bw - pw))
                bark = bark[y0 : y0 + ph, x0 : x0 + pw]
            out = multiply_bark_texture(out, bark)
        except Exception:
            pass
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/raw/gramoty/manifest.jsonl"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument(
        "--gramoty-root",
        type=Path,
        default=Path("data/raw/gramoty"),
        help="Root for image_relpath resolution.",
    )
    ap.add_argument(
        "--suprasliensis",
        type=Path,
        default=Path("data/interim/suprasliensis_crops/crops.jsonl"),
    )
    ap.add_argument(
        "--bark-pool-dir",
        type=Path,
        default=Path("data/raw/gramoty/documents"),
        help=(
            "Directory under which photo_*.jp(e)g files are sampled for bark texture detail. "
            "Searched recursively."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/unified_ocr"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--max-suprasliensis",
        type=int,
        default=0,
        help="If > 0, sample at most this many parchment crops (debug).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = load_split_ids(args.splits_dir)

    out_img_dir = args.out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    bark_pool: list[Path] = []
    if args.bark_pool_dir and args.bark_pool_dir.exists():
        bark_pool = sorted(args.bark_pool_dir.rglob("photo_*.jpeg")) + sorted(
            args.bark_pool_dir.rglob("photo_*.jpg")
        )
    print(f"bark texture pool: {len(bark_pool)} photos")

    bucket_files = {
        s: (args.out_dir / f"unified_{s}.jsonl").open("w", encoding="utf-8")
        for s in ("train", "val", "test")
    }
    counts = {s: {"gramoty": 0, "suprasliensis": 0} for s in ("train", "val", "test")}
    skipped = 0

    for row in gramoty_rows(args.manifest, splits):
        img_rel = row["image_relpath"]
        candidates = [
            args.gramoty_root / img_rel.lstrip("/"),
            Path(img_rel),
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            skipped += 1
            continue
        sp = row["split"]
        bucket_files[sp].write(
            json.dumps(
                {
                    "image": portable_path(src),
                    "text": row["text"],
                    "source": "gramoty",
                    "doc_id": row["doc_id"],
                    "split": sp,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        counts[sp]["gramoty"] += 1

    sup = list(suprasliensis_rows(args.suprasliensis))
    if args.max_suprasliensis > 0:
        random.Random(args.seed).shuffle(sup)
        sup = sup[: args.max_suprasliensis]
    print(f"suprasliensis crops to bark-tint: {len(sup)}")
    for i, row in enumerate(sup):
        src = Path(row["image_path"])
        if not src.exists():
            skipped += 1
            continue
        try:
            img = Image.open(src)
        except Exception:
            skipped += 1
            continue
        tinted = apply_bark_shift(img, bark_pool=bark_pool, rng=rng)
        out_name = f"sup_{src.stem}.png"
        out_path = out_img_dir / out_name
        tinted.save(out_path)
        bucket_files["train"].write(
            json.dumps(
                {
                    "image": portable_path(out_path),
                    "text": row["text"],
                    "source": "suprasliensis",
                    "doc_id": row["doc_id"],
                    "split": "train",
                    "original_image": row.get("original_image") or portable_path(src),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        counts["train"]["suprasliensis"] += 1
        if (i + 1) % 200 == 0:
            print(f"  bark-tinted {i + 1}/{len(sup)}")

    for f in bucket_files.values():
        f.close()
    print("Done. Skipped:", skipped)
    print("Counts:")
    for sp, by_src in counts.items():
        total = sum(by_src.values())
        print(f"  {sp:5s}: total={total}  " + " ".join(f"{k}={v}" for k, v in by_src.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
