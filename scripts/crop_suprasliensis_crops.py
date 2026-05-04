#!/usr/bin/env python3
"""Build random multi-line facsimile crops from Suprasliensis manifest + JPEGs.

The HTML gives **one diplomatic line per** ``<ol><li>`` (``span.os``), which is reliable
**logical** line segmentation and transcription gold. It does **not** ship **pixel**
``(x, y)`` boxes on the facsimile JPEG, so vertical bands are still inferred on the image.

Each sample uses 3–5 consecutive ``<li>`` lines. Ground truth joins those strings with a
**space** between lines; if a line ends with ``-`` (word split across lines), the hyphen is
dropped and the next line is concatenated **without** an extra space.

Line *geometry* uses ink-trimmed margins and a smoothed row-ink profile (``--equal-split`` for
uniform vertical bands).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import orjson
from PIL import Image, ImageFilter


def portable_path(path: Path, root: Path | None = None) -> str:
    """Return a repo-relative path when possible, otherwise the original path."""
    root = (root or Path.cwd()).resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        rows.append(orjson.loads(line))
    return [r for r in rows if "error" not in r and r.get("n_lines", 0) > 0 and r.get("lines")]


def merge_lines_for_gt(lines: list[str]) -> str:
    """Join consecutive diplomatic lines (space between lines; hyphenated wraps glued)."""
    merged = ""
    for t in lines:
        t = t.strip()
        if not t:
            continue
        if not merged:
            merged = t
            continue
        if merged.endswith("-"):
            merged = merged[:-1] + t
        else:
            merged += " " + t
    return merged


def ink_bbox(
    gray: np.ndarray,
    row_peak_frac: float = 0.06,
    col_peak_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Outer box of pixels in rows/columns whose mean darkness exceeds a peak-relative threshold."""
    inv = 255 - gray.astype(np.float32)
    row_s = inv.mean(axis=1)
    col_s = inv.mean(axis=0)
    rthr = float(np.max(row_s) * row_peak_frac + 1e-6)
    cthr = float(np.max(col_s) * col_peak_frac + 1e-6)
    ys = np.where(row_s >= rthr)[0]
    xs = np.where(col_s >= cthr)[0]
    if ys.size == 0 or xs.size == 0:
        return 0, gray.shape[0], 0, gray.shape[1]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    return y0, y1, x0, x1


def smooth_1d(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(ap, ker, mode="valid")


def local_minima(y: np.ndarray) -> np.ndarray:
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1


def highpass_dark_profile(gray: np.ndarray, margin_frac: float = 0.08) -> np.ndarray:
    """Row score for dark strokes after subtracting a blurred local background."""
    h, w = gray.shape
    x0 = int(w * margin_frac)
    x1 = max(x0 + 1, int(w * (1 - margin_frac)))
    im = Image.fromarray(gray.astype(np.uint8))
    bg = np.array(im.filter(ImageFilter.GaussianBlur(radius=25))).astype(np.float32)
    residual = bg - gray.astype(np.float32)
    central = residual[:, x0:x1]
    threshold = 50.0
    return (central > threshold).sum(axis=1).astype(np.float32)


def line_centers_from_dark_peaks(gray: np.ndarray, n_lines: int) -> list[int] | None:
    """Detect one y-center per diplomatic line using high-pass dark-stroke row peaks."""
    h = gray.shape[0]
    score = highpass_dark_profile(gray)
    if not np.any(score):
        return None

    k = max(5, (h // 450) | 1)
    sm = smooth_1d(score, k)
    min_y = int(h * 0.05)
    max_y = int(h * 0.96)
    min_dist = max(18, h // max(1, n_lines * 3))
    min_score = max(20.0, float(np.quantile(sm[min_y:max_y], 0.70)))

    peaks: list[tuple[int, float]] = []
    for y in range(max(min_y + 5, 5), min(max_y - 5, h - 5)):
        if sm[y] >= min_score and sm[y] == np.max(sm[y - 5 : y + 6]):
            peaks.append((y, float(sm[y])))
    if len(peaks) < n_lines:
        return None

    selected: list[tuple[int, float]] = []
    for y, s in sorted(peaks, key=lambda p: p[1], reverse=True):
        if all(abs(y - yy) >= min_dist for yy, _ in selected):
            selected.append((y, s))
        if len(selected) >= n_lines:
            break
    if len(selected) != n_lines:
        return None
    return [y for y, _ in sorted(selected)]


def y_band_edges(gray: np.ndarray, n_lines: int, equal_split: bool) -> list[int]:
    """Return len n_lines+1 increasing y coordinates (pixel) splitting content into n_lines bands."""
    centers = None if equal_split else line_centers_from_dark_peaks(gray, n_lines)
    if centers:
        gaps = [b - a for a, b in zip(centers, centers[1:])]
        median_gap = int(np.median(gaps)) if gaps else max(1, gray.shape[0] // n_lines)
        edges = [max(0, int(round(centers[0] - 0.55 * median_gap)))]
        edges.extend(int(round((a + b) / 2)) for a, b in zip(centers, centers[1:]))
        edges.append(min(gray.shape[0], int(round(centers[-1] + 0.55 * median_gap))))
        return edges

    y0, y1, _, _ = ink_bbox(gray)
    h = max(1, y1 - y0)
    equal_edges = [y0 + int(round(h * i / n_lines)) for i in range(n_lines + 1)]
    if equal_split or n_lines <= 1:
        return equal_edges

    sub = gray[y0:y1]
    inv = (255 - sub.astype(np.float32)) / 255.0
    prof = inv.mean(axis=1)
    k = max(5, h // 120)
    sm = smooth_1d(prof, k)
    mins = local_minima(sm)
    win = max(3, h // max(8, 2 * n_lines))
    cuts: list[int] = []
    for j in range(1, n_lines):
        t = int(j * h / n_lines)
        lo, hi = max(1, t - win), min(h - 2, t + win)
        idx = int(np.argmin(sm[lo:hi]) + lo)
        near = mins[(mins >= lo) & (mins <= hi)]
        if near.size:
            idx = int(near[np.argmin(np.abs(near - t))])
        cuts.append(idx)
    cuts.sort()
    min_sep = max(3, h // (3 * n_lines))
    merged: list[int] = []
    for c in cuts:
        if not merged or c - merged[-1] >= min_sep:
            merged.append(c)
        else:
            merged[-1] = (merged[-1] + c) // 2
    if len(merged) != n_lines - 1:
        return equal_edges
    edges = [y0] + [y0 + c for c in merged] + [y1]
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    edges[-1] = y1
    return edges


def x_band_for_rows(gray: np.ndarray, y0: int, y1: int, pad_ratio: float) -> tuple[int, int]:
    row0, row1, x0, x1 = ink_bbox(gray[y0:y1, :])
    _ = row0, row1
    w = gray.shape[1]
    pad = int((x1 - x0) * pad_ratio + 4)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    return x0, x1


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "x"


def cmd_crops(args: argparse.Namespace) -> int:
    root = Path(args.manifest_dir)
    man_path = root / "manifest.jsonl"
    if not man_path.exists():
        print(f"Missing {man_path}; run scripts/scrape_suprasliensis.py first", file=sys.stderr)
        return 1

    records = load_manifest(man_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    out_img = Path(args.out_images_dir)
    out_meta = Path(args.out_jsonl)
    out_img.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written: list[dict] = []

    for rec in tqdm_records(records, desc="folios"):
        folio = rec["folio_id"]
        rel_img = rec["image_path"]
        img_path = root / rel_img
        if not img_path.exists():
            continue
        lines: list[str] = rec["lines"]
        n = len(lines)
        if n < args.min_lines:
            continue

        with Image.open(img_path) as im:
            gray = np.array(im.convert("L"))
        edges = y_band_edges(gray, n, equal_split=args.equal_split)

        for _ in range(args.crops_per_folio):
            span = rng.randint(args.min_lines, min(args.max_lines, n))
            start = rng.randint(0, n - span)
            y_a = edges[start]
            y_b = edges[start + span]
            pad_y = max(2, int((y_b - y_a) * args.pad_y_ratio))
            y0 = max(0, y_a - pad_y)
            y1 = min(gray.shape[0], y_b + pad_y)
            x0, x1 = x_band_for_rows(gray, y0, y1, args.pad_x_ratio)
            crop = Image.fromarray(gray[y0:y1, x0:x1])
            gt = merge_lines_for_gt(lines[start : start + span])

            stem = f"{safe_slug(folio)}_{start:03d}_{span}l_{rng.getrandbits(32):08x}"
            out_path = out_img / f"{stem}.png"
            crop.save(out_path)

            written.append(
                {
                    "folio_id": folio,
                    "line_start": start,
                    "line_end": start + span,
                    "n_lines": span,
                    "gold": gt,
                    "image_path": portable_path(out_path),
                    "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "source_image": portable_path(img_path),
                }
            )

    out_meta.write_bytes(b"\n".join(orjson.dumps(r) for r in written) + (b"\n" if written else b""))
    print(f"Wrote {len(written)} crops to {out_img}; metadata {out_meta}")
    return 0


def tqdm_records(records: list[dict], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(records, desc=desc)
    except Exception:  # noqa: BLE001
        return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/raw/suprasliensis"),
        help="Directory containing manifest.jsonl and images/",
    )
    p.add_argument(
        "--out-images-dir",
        type=Path,
        default=Path("data/interim/suprasliensis_crops/images"),
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/interim/suprasliensis_crops/crops.jsonl"),
    )
    p.add_argument("--min-lines", type=int, default=3)
    p.add_argument("--max-lines", type=int, default=5)
    p.add_argument("--crops-per-folio", type=int, default=4)
    p.add_argument("--pad-y-ratio", type=float, default=0.03)
    p.add_argument("--pad-x-ratio", type=float, default=0.04)
    p.add_argument("--equal-split", action="store_true", help="Uniform vertical bands (debug baseline)")
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N manifest rows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return cmd_crops(args)


if __name__ == "__main__":
    raise SystemExit(main())
