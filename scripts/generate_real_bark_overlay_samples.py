#!/usr/bin/env python3
"""Robust label-safe synthetic samples.

Pipeline:
  1. Load a real train-split gramoty photo, guaranteeing authentic bark texture.
  2. Extract the bark foreground mask (HSV warm-hue prior + GrabCut + largest CC).
  3. Build a forbidden-symbol mask from the gramoty facsimile drawing when
     available, plus photo-local dark stroke detection.
  4. Crop to the bark bbox keeping aspect ratio.
  5. Render only inside a clean rectangle of bark that does not intersect the
     forbidden-symbol mask. This avoids the impossible task of globally erasing
     dense 3D carvings.
  6. Smooth only the chosen text rectangle, then engrave exact glyph outlines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from birchbark_ocr.synth.ostromir_render import (
    _default_font_path,
    _wrap_text,
    iter_ostromir_content_lines,
)

OSTROMIR_URL = "http://www.ponomar.net/files/ostromir.txt"


def load_text(url: str, cache: Path | None) -> str:
    if cache and cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        r = client.get(url)
        r.raise_for_status()
        text = r.text
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return text


def bark_mask(rgb: np.ndarray) -> np.ndarray:
    """Return uint8 0/255 mask of the bark surface in an original gramoty photo.

    Uses GrabCut seeded by a high-saturation warm-hue prior, then keeps the largest
    connected component. Robust to grey studio backgrounds, rulers, captions.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    warm = ((H >= 5) & (H <= 32) & (S > 55) & (V > 60)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), 2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        warm = np.where(labels == keep, 1, 0).astype(np.uint8)
    if warm.sum() < 0.005 * warm.size:
        return np.zeros((h, w), dtype=np.uint8)
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[warm == 1] = cv2.GC_PR_FGD
    eroded = cv2.erode(warm, np.ones((25, 25), np.uint8), 1)
    gc_mask[eroded == 1] = cv2.GC_FGD
    border = np.zeros_like(warm)
    border[:8] = 1; border[-8:] = 1; border[:, :8] = 1; border[:, -8:] = 1
    gc_mask[border == 1] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (warm * 255).astype(np.uint8)
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), 2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        out = np.where(labels == keep, 255, 0).astype(np.uint8)
    return out


def find_drawing_for_photo(photo_path: Path) -> Path | None:
    """Look for a `drawing_*.gif/png` file next to a `photo_*.jpg` of the same gramota."""
    if not photo_path.parent.exists():
        return None
    candidates: list[Path] = []
    for pat in ("drawing_*.gif", "drawing_*.png", "drawing_*.jpg", "drawing_*.jpeg"):
        candidates.extend(sorted(photo_path.parent.glob(pat)))
    return candidates[0] if candidates else None


def stroke_mask_from_drawing(
    drawing_path: Path, photo_rgb: np.ndarray, photo_bark: np.ndarray
) -> np.ndarray | None:
    """Return a forbidden-symbol mask in photo coordinates from the facsimile drawing.

    Pipeline:
      1. Threshold the drawing to a binary ink map.
      2. Largest external contour = bark silhouette (drops ruler + caption blobs).
      3. Mask drawing strokes to the silhouette interior (so only ink inside bark survives).
      4. Affine-warp from the drawing's silhouette bbox to the photo's bark bbox.
      5. Restrict to the photo's bark mask and slightly dilate so placement avoids strokes.

    Returns None if the drawing cannot be parsed.
    """
    try:
        d = np.array(Image.open(drawing_path).convert("L"))
    except Exception:
        return None
    _, ink = cv2.threshold(d, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    silhouette = max(contours, key=cv2.contourArea)
    if cv2.contourArea(silhouette) < 0.05 * d.size:
        return None
    sil_mask = np.zeros_like(ink)
    cv2.drawContours(sil_mask, [silhouette], -1, 255, thickness=cv2.FILLED)
    sil_inside = cv2.erode(sil_mask, np.ones((9, 9), np.uint8), 1)
    strokes_in = cv2.bitwise_and(ink, sil_inside)
    dx, dy, dw, dh = cv2.boundingRect(silhouette)
    ys, xs = np.where(photo_bark > 0)
    if len(xs) == 0 or dw < 16 or dh < 16:
        return None
    px0, py0, px1, py1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pw, ph = px1 - px0, py1 - py0
    src_pts = np.float32([[dx, dy], [dx + dw, dy], [dx, dy + dh]])
    dst_pts = np.float32([[px0, py0], [px0 + pw, py0], [px0, py0 + ph]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(
        strokes_in, M, (photo_rgb.shape[1], photo_rgb.shape[0]), flags=cv2.INTER_NEAREST
    )
    warped = cv2.bitwise_and(warped, photo_bark)
    # This mask is used for placement exclusion, not global erasing. A moderate
    # halo is enough: it keeps new text away from old symbols without destroying
    # bark texture.
    long_side = max(pw, ph)
    radius = max(3, int(round(long_side * 0.004)))
    k = radius | 1
    warped = cv2.dilate(warped, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    if warped.sum() < 200:
        return None
    return warped


def text_stroke_mask(rgb: np.ndarray, bark: np.ndarray) -> np.ndarray:
    """Stroke density map (uint8) of any dark scratched marks inside the bark area."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if bark.sum() == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((11, 11), np.uint8))
    _, strokes = cv2.threshold(bh, 18, 255, cv2.THRESH_BINARY)
    strokes = cv2.bitwise_and(strokes, strokes, mask=bark)
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    return strokes


def find_clean_patch(
    stroke: np.ndarray, bark: np.ndarray, h: int, w: int, exclude_box: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    """Sliding-window search for a (h, w) rectangle inside bark, low stroke density,
    preferring patches near the planned text rectangle (so colour/lighting match).
    """
    H, W = stroke.shape
    if h <= 0 or w <= 0 or h > H or w > W:
        return None
    bark_int = cv2.integral((bark > 0).astype(np.uint8))
    stroke_int = cv2.integral((stroke > 0).astype(np.uint8))
    ex0, ey0, ex1, ey1 = exclude_box
    cx, cy = (ex0 + ex1) / 2.0, (ey0 + ey1) / 2.0
    best = None
    step = max(6, min(h, w) // 10)
    target_bark = h * w * 0.95
    diag = float((H * H + W * W) ** 0.5)
    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            if not (x + w <= ex0 - 4 or x >= ex1 + 4 or y + h <= ey0 - 4 or y >= ey1 + 4):
                continue
            inside = (
                bark_int[y + h, x + w] - bark_int[y, x + w] - bark_int[y + h, x] + bark_int[y, x]
            )
            if inside < target_bark:
                continue
            density = (
                stroke_int[y + h, x + w] - stroke_int[y, x + w] - stroke_int[y + h, x] + stroke_int[y, x]
            )
            pcx, pcy = x + w / 2.0, y + h / 2.0
            dist = ((pcx - cx) ** 2 + (pcy - cy) ** 2) ** 0.5 / diag
            score = density / (h * w) + 0.18 * dist
            if best is None or score < best[0]:
                best = (score, y, x)
    if best is None:
        return None
    return best[1], best[2]


def composite_clean_patch(
    bg: np.ndarray, bark: np.ndarray, stroke: np.ndarray, text_rect: tuple[int, int, int, int]
) -> np.ndarray:
    """Erase original strokes inside ``text_rect`` *in place*: median blur (kills
    narrow scratches but keeps local colour and lighting), then re-inject
    high-frequency bark texture from a colour-matched nearby clean patch so the
    region does not become a smooth blob. No hard seam ever appears.
    """
    x0, y0, x1, y1 = text_rect
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return bg
    dst = bg[y0:y1, x0:x1].copy()
    # Median wipes scratches; bilateral cleans residual halos while keeping bark color.
    smooth = cv2.medianBlur(dst, 11)
    smooth = cv2.medianBlur(smooth, 7)
    smooth = cv2.bilateralFilter(smooth, 9, 35, 35)
    # Re-inject local high-frequency bark noise so the area is not flat.
    # We borrow only fine-scale luminance variance from the destination itself
    # (post-median, so original strokes are gone), avoiding any foreign patch.
    base = cv2.GaussianBlur(smooth, (0, 0), sigmaX=3.0, sigmaY=3.0)
    detail = smooth.astype(np.int16) - base.astype(np.int16)
    smooth = np.clip(base.astype(np.int16) + (detail * 1.2).astype(np.int16), 0, 255).astype(np.uint8)
    rng_noise = np.random.RandomState(int(x0 * 7 + y0 * 13 + h * 17 + w * 19) & 0x7FFFFFFF)
    grain = (rng_noise.rand(*smooth.shape[:2]).astype(np.float32) - 0.5) * 6.0
    smooth = np.clip(smooth.astype(np.float32) + grain[..., None], 0, 255).astype(np.uint8)
    feather = max(18, int(min(h, w) * 0.25))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fy = np.minimum(yy, h - 1 - yy)
    fx = np.minimum(xx, w - 1 - xx)
    alpha = np.minimum(np.minimum(fy, fx) / feather, 1.0)
    alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
    alpha3 = alpha[..., None]
    blended = dst.astype(np.float32) * (1 - alpha3) + smooth.astype(np.float32) * alpha3
    bg[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def fit_to_canvas(img_rgb: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to bark bbox + small pad, scale so longest side == max_side, snap to 64.

    Returns (rgb, bark_mask) of the same shape; aspect ratio of the bark is preserved
    so the output looks like a real elongated gramota strip rather than a square scene.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return cv2.resize(img_rgb, (max_side, max_side)), cv2.resize(mask, (max_side, max_side))
    pad = 28
    x0, x1 = max(0, xs.min() - pad), min(img_rgb.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img_rgb.shape[0], ys.max() + pad)
    crop = img_rgb[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = max_side / max(h, w)
    new_w = max(64, int(round(w * scale / 64)) * 64)
    new_h = max(64, int(round(h * scale / 64)) * 64)
    rgb = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(cmask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb, out_mask


def fit_mask_to_canvas(extra_mask: np.ndarray, bark_mask_src: np.ndarray, max_side: int) -> np.ndarray:
    """Apply the same bark-bbox crop/resize as ``fit_to_canvas`` to an extra mask."""
    ys, xs = np.where(bark_mask_src > 0)
    if len(xs) == 0:
        return cv2.resize(extra_mask, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
    pad = 28
    x0, x1 = max(0, xs.min() - pad), min(extra_mask.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(extra_mask.shape[0], ys.max() + pad)
    crop = extra_mask[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = max_side / max(h, w)
    new_w = max(64, int(round(w * scale / 64)) * 64)
    new_h = max(64, int(round(h * scale / 64)) * 64)
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def find_low_symbol_rect(
    bark: np.ndarray,
    forbidden: np.ndarray,
    *,
    font_size: int,
    line_gap: int,
    lines_per_image: int,
    max_symbol_density: float,
) -> tuple[int, int, int, int] | None:
    """Find a large bark rectangle with the lowest old-symbol density.

    We do not require zero old marks: real birchbark is scratched everywhere.
    The goal is to avoid dense original letters under/around the new text.
    """
    H, W = bark.shape
    bark01 = (bark > 0).astype(np.uint8)
    forbid01 = (forbidden > 0).astype(np.uint8)
    bark_int = cv2.integral(bark01)
    forbid_int = cv2.integral(forbid01)

    needed_h = int((font_size + line_gap) * min(lines_per_image, 3) + font_size)
    widths = [int(W * f) for f in (0.68, 0.55, 0.42, 0.32)]
    heights = [int(max(needed_h, H * f)) for f in (0.45, 0.35, 0.27, 0.20)]
    best: tuple[float, int, int, int, int] | None = None

    for h in heights:
        for w in widths:
            if h > H or w > W or h < font_size * 2 or w < font_size * 5:
                continue
            step = max(10, min(h, w) // 8)
            for y in range(0, H - h + 1, step):
                for x in range(0, W - w + 1, step):
                    bark_area = (
                        bark_int[y + h, x + w]
                        - bark_int[y, x + w]
                        - bark_int[y + h, x]
                        + bark_int[y, x]
                    )
                    coverage = bark_area / float(h * w)
                    if coverage < 0.82:
                        continue
                    forbid_area = (
                        forbid_int[y + h, x + w]
                        - forbid_int[y, x + w]
                        - forbid_int[y + h, x]
                        + forbid_int[y, x]
                    )
                    density = forbid_area / max(float(bark_area), 1.0)
                    if density > max_symbol_density:
                        continue
                    # Prefer cleaner windows, then larger windows.
                    score = density + 0.04 * (1.0 - coverage) - 0.02 * ((h * w) / (H * W))
                    if best is None or score < best[0]:
                        best = (score, x, y, x + w, y + h)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def render_glyph_mask_in(
    bark_mask_pil: Image.Image,
    lines: list[str],
    *,
    font_size: int,
    line_gap: int,
    rng: random.Random,
    stroke_width: int,
) -> tuple[Image.Image, str, list[tuple[int, int, int, int]], tuple[int, int, int, int] | None]:
    """Render glyph mask constrained inside an inner clean rectangle.

    Returns (filled_mask, gold_text, per_line_bboxes, text_bounding_rect). The
    text bounding rect is the union of all rendered line bboxes, expanded by a
    small margin; it is used downstream for clean-bark patch composition.
    """
    bark_np = np.array(bark_mask_pil) > 127
    H, W = bark_np.shape
    eroded = cv2.erode(bark_np.astype(np.uint8) * 255, np.ones((max(3, font_size // 2),) * 2, np.uint8), iterations=1)
    rect = largest_axis_rect(eroded > 0)
    if rect is None:
        return Image.new("L", (W, H), 0), "", [], None
    x0, y0, x1, y1 = rect
    font = ImageFont.truetype(str(_default_font_path()), font_size)
    max_w = x1 - x0
    wrapped: list[str] = []
    rendered_lines: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(line, font, max_w))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    y = y0 + rng.randint(0, max(1, (y1 - y0) // 6))
    drift = rng.randint(-20, 20)
    bboxes: list[tuple[int, int, int, int]] = []
    for text in wrapped:
        bb = font.getbbox(text)
        line_h = bb[3] - bb[1]
        tw = bb[2] - bb[0]
        if y + line_h > y1:
            break
        max_x = x1 - tw
        drift += rng.randint(-25, 25)
        x = min(max_x, max(x0, x0 + drift + rng.randint(0, max(1, max_x - x0))))
        # Outline-only render: fill=0 (invisible on black bg), stroke renders thin 1-2 px ring.
        draw.text(
            (x, y), text, font=font, fill=0, stroke_width=1, stroke_fill=255
        )
        bboxes.append((x, y, x + tw, y + line_h))
        rendered_lines.append(text)
        y += line_h + line_gap + rng.randint(-3, 4)
    if stroke_width > 1:
        for _ in range(stroke_width - 1):
            mask = mask.filter(ImageFilter.MaxFilter(3))
    if not bboxes:
        return mask, "", [], None
    pad = max(8, font_size // 4)
    bx0 = max(0, min(b[0] for b in bboxes) - pad)
    by0 = max(0, min(b[1] for b in bboxes) - pad)
    bx1 = min(W, max(b[2] for b in bboxes) + pad)
    by1 = min(H, max(b[3] for b in bboxes) + pad)
    return mask, "\n".join(rendered_lines), bboxes, (bx0, by0, bx1, by1)


def largest_axis_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest all-True axis-aligned rectangle in a boolean mask (max histogram)."""
    if mask.sum() == 0:
        return None
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        row = mask[y]
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] > cur:
                idx, hh = stack.pop()
                area = hh * (x - idx)
                if area > best[0]:
                    best = (area, idx, y - hh + 1, x - 1, y)
                start = idx
            stack.append((start, int(cur)))
    if best[0] == 0:
        return None
    return best[1], best[2], best[3], best[4]


def engrave(
    bg_rgb: Image.Image,
    filled_mask: Image.Image,
    bark_mask_pil: Image.Image,
    *,
    strength: float,
    rng: random.Random,
) -> Image.Image:
    """Draw text as thin scratched outline strokes (not solid printed letters).

    Outline = filled - eroded(filled). The outline is drawn as a dark groove
    with a faint lower-right highlight, jittered slightly for an organic feel.
    """
    bark_np = np.array(bark_mask_pil)
    outline = np.array(filled_mask)
    outline = np.minimum(outline, bark_np).astype(np.uint8)
    # Random fade so strokes look hand-scratched, not vectorial.
    noise = (np.random.RandomState(rng.randint(0, 2**31)).rand(*outline.shape) * 110 + 145).astype(np.uint8)
    outline = np.minimum(outline, noise).astype(np.uint8)
    # Tiny geometric jitter (1px shift in random direction) per row band, for wobble.
    jitter = outline.copy()
    band = max(6, outline.shape[0] // 80)
    for y0 in range(0, outline.shape[0], band):
        dx = int(round((np.random.rand() - 0.5) * 2))
        if dx == 0:
            continue
        seg = outline[y0 : y0 + band]
        shifted = np.roll(seg, dx, axis=1)
        jitter[y0 : y0 + band] = shifted
    outline = np.maximum(outline, jitter).astype(np.uint8)
    groove_pil = Image.fromarray(outline).filter(ImageFilter.GaussianBlur(radius=0.35))
    bg = bg_rgb.convert("RGBA")
    dark = Image.new("RGBA", bg.size, (18, 13, 9, 0))
    dark.putalpha(groove_pil.point(lambda v: int(min(255, v * 0.95 * strength))))
    highlight_pil = Image.fromarray(outline).filter(ImageFilter.GaussianBlur(radius=0.5))
    highlight = Image.new("RGBA", bg.size, (235, 220, 188, 0))
    highlight.putalpha(highlight_pil.point(lambda v: int(v * 0.25 * strength)))
    bg.alpha_composite(highlight, (1, 1))
    bg.alpha_composite(dark)
    return bg.convert("RGB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photos-dir", type=Path, default=Path("data/synthetic/sdxl_lora_gramoty_train"))
    p.add_argument(
        "--photos-metadata",
        type=Path,
        default=Path("data/synthetic/sdxl_lora_gramoty_train/metadata.jsonl"),
        help="If given, load original photos from each row's source_path (richer than padded squares).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("reports/figs/real_bark_overlay_samples"))
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--lines-per-image", type=int, default=5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--font-size", type=int, default=42)
    p.add_argument("--line-gap", type=int, default=14)
    p.add_argument("--stroke-width", type=int, default=1)
    p.add_argument("--engrave-strength", type=float, default=1.4)
    p.add_argument(
        "--max-symbol-density",
        type=float,
        default=0.10,
        help="Reject candidate windows whose forbidden-symbol mask covers more than this fraction.",
    )
    p.add_argument("--max-attempts", type=int, default=120)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--cache", type=Path, default=Path("data/raw/ostromir/ostromir.txt"))
    p.add_argument(
        "--keep-full-bark",
        action="store_true",
        help="Keep the whole bark plaque instead of cropping to the selected clean window.",
    )
    p.add_argument(
        "--no-inpaint",
        action="store_true",
        help="Skip cv2 stroke inpaint (debug: keeps original text on bark)",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_text(OSTROMIR_URL, args.cache)
    pool = iter_ostromir_content_lines(raw)
    rng = random.Random(args.seed)

    photos: list[Path] = []
    if args.photos_metadata and args.photos_metadata.exists():
        for line in args.photos_metadata.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sp = row.get("source_path")
            if sp and Path(sp).exists():
                photos.append(Path(sp))
    if not photos:
        photos = sorted(args.photos_dir.glob("*.png"))
    if not photos:
        print("No source photos available", file=sys.stderr)
        return 1

    records: list[dict] = []
    attempts = 0
    i = 0
    while i < args.n_samples and attempts < args.max_attempts:
        attempts += 1
        photo_path = photos[rng.randrange(len(photos))]
        rgb = np.array(Image.open(photo_path).convert("RGB"))
        bark = bark_mask(rgb)
        if bark.sum() < 0.02 * bark.size:
            print(f"  skip {photo_path.name}: bark mask too small", file=sys.stderr)
            continue
        # Use drawings/detected strokes as placement exclusion. We do not try to
        # globally erase old carvings, because that makes muddy bark.
        drawing_path = find_drawing_for_photo(photo_path)
        drawing_forbidden = None
        if drawing_path is not None and not args.no_inpaint:
            drawing_forbidden = stroke_mask_from_drawing(drawing_path, rgb, bark)
        crop_rgb, crop_mask = fit_to_canvas(rgb, bark, args.size)
        crop_forbidden = text_stroke_mask(crop_rgb, crop_mask)
        if drawing_forbidden is not None:
            crop_forbidden = cv2.bitwise_or(
                crop_forbidden,
                fit_mask_to_canvas(drawing_forbidden, bark, args.size),
            )
        # Expand old-symbol zones before ranking candidate windows. This is a
        # placement prior, not an erase mask.
        forbidden_halo = max(11, args.font_size // 3)
        if forbidden_halo % 2 == 0:
            forbidden_halo += 1
        crop_forbidden = cv2.dilate(
            crop_forbidden,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (forbidden_halo, forbidden_halo)),
            1,
        )
        clean_rect = find_low_symbol_rect(
            crop_mask,
            crop_forbidden,
            font_size=args.font_size,
            line_gap=args.line_gap,
            lines_per_image=args.lines_per_image,
            max_symbol_density=args.max_symbol_density,
        )
        if clean_rect is None:
            print(f"  skip {photo_path.name}: no clean rect for text", file=sys.stderr)
            continue
        rx0, ry0, rx1, ry1 = clean_rect
        if not args.keep_full_bark:
            pad = max(24, args.font_size)
            cx0, cy0 = max(0, rx0 - pad), max(0, ry0 - pad)
            cx1, cy1 = min(crop_rgb.shape[1], rx1 + pad), min(crop_rgb.shape[0], ry1 + pad)
            crop_rgb = crop_rgb[cy0:cy1, cx0:cx1]
            crop_mask = crop_mask[cy0:cy1, cx0:cx1]
            crop_forbidden = crop_forbidden[cy0:cy1, cx0:cx1]
            clean_rect = (rx0 - cx0, ry0 - cy0, rx1 - cx0, ry1 - cy0)
            rx0, ry0, rx1, ry1 = clean_rect
        clean_placement = np.zeros_like(crop_mask)
        clean_placement[ry0:ry1, rx0:rx1] = 255
        clean_placement = cv2.bitwise_and(clean_placement, crop_mask)
        bg_pil = Image.fromarray(crop_rgb)
        bark_pil = Image.fromarray(crop_mask)
        placement_pil = Image.fromarray(clean_placement)

        start = rng.randint(0, len(pool) - args.lines_per_image)
        lines = pool[start : start + args.lines_per_image]
        layout_seed = rng.randint(0, 2**30)
        layout_rng = random.Random(layout_seed)
        glyphs, gold, bboxes, text_rect = render_glyph_mask_in(
            placement_pil,
            lines,
            font_size=args.font_size,
            line_gap=args.line_gap,
            rng=layout_rng,
            stroke_width=args.stroke_width,
        )
        if not gold.strip() or text_rect is None:
            print(f"  skip {photo_path.name}: no clean rect for text", file=sys.stderr)
            continue

        if not args.no_inpaint:
            stroke = text_stroke_mask(crop_rgb, crop_mask)
            cleaned = composite_clean_patch(crop_rgb.copy(), crop_mask, stroke, text_rect)
            bg_pil = Image.fromarray(cleaned)
            Image.fromarray(stroke).save(args.output_dir / f"real_bark_overlay_{i:02d}.strokes.png")

        final = engrave(bg_pil, glyphs, bark_pil, strength=args.engrave_strength, rng=layout_rng)

        stem = f"real_bark_overlay_{i:02d}"
        bg_pil.save(args.output_dir / f"{stem}.background.png")
        bark_pil.save(args.output_dir / f"{stem}.bark_mask.png")
        Image.fromarray(crop_forbidden).save(args.output_dir / f"{stem}.forbidden.png")
        placement_pil.save(args.output_dir / f"{stem}.placement_mask.png")
        glyphs.convert("RGB").save(args.output_dir / f"{stem}.glyphs.png")
        final.save(args.output_dir / f"{stem}.png")
        (args.output_dir / f"{stem}.gold.txt").write_text(gold + "\n", encoding="utf-8")
        records.append(
            {
                "image": str(args.output_dir / f"{stem}.png"),
                "background": str(args.output_dir / f"{stem}.background.png"),
                "bark_mask": str(args.output_dir / f"{stem}.bark_mask.png"),
                "forbidden_mask": str(args.output_dir / f"{stem}.forbidden.png"),
                "placement_mask": str(args.output_dir / f"{stem}.placement_mask.png"),
                "glyphs": str(args.output_dir / f"{stem}.glyphs.png"),
                "gold": str(args.output_dir / f"{stem}.gold.txt"),
                "text": gold,
                "source_photo": str(photo_path),
                "source_lines_start": start,
                "source_lines_end": start + args.lines_per_image,
                "seed": args.seed + i,
                "layout_seed": layout_seed,
                "line_bboxes": bboxes,
                "clean_rect": clean_rect,
                "engrave_strength": args.engrave_strength,
                "stroke_width": args.stroke_width,
                "inpainted": not args.no_inpaint,
                "used_drawing_mask": drawing_forbidden is not None,
                "drawing_path": str(drawing_path) if drawing_path else None,
            }
        )
        print(args.output_dir / f"{stem}.png")
        i += 1

    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
