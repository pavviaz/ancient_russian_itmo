"""Render Ostromir diplomatic text lines as birchbark-like incised synthetic images."""

from __future__ import annotations

import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Edition noise in ponomar.net plaintext export
_TAG_RE = re.compile(r"<telia>|<kopie>|<left\s+marginal>", re.IGNORECASE)
_LEAD_RE = re.compile(r"^[\d.\s]+")  # strip "2.1  1" style prefixes


def strip_edition_markup(line: str) -> str:
    s = _TAG_RE.sub("", line)
    s = _LEAD_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_ostromir_content_lines(raw: str) -> list[str]:
    """Return cleaned non-empty lines suitable for rendering (longer fragments preferred)."""
    out: list[str] = []
    for raw_line in raw.splitlines():
        s = strip_edition_markup(raw_line)
        if len(s) < 12:
            continue
        # Heuristic: mostly Cyrillic / historic Cyrillic letters (not folio headers)
        def _is_cyr_related(ch: str) -> bool:
            o = ord(ch)
            if ch.isalpha() and (
                "\u0400" <= ch <= "\u04ff"
                or "\u0500" <= ch <= "\u052f"
                or "\u2de0" <= ch <= "\u2dff"
                or "\ua640" <= ch <= "\ua69f"
                or 0x1C80 <= o <= 0x1C88
            ):
                return True
            return ch in "ѢѣѤѥ҃҇ⷩⷢꙋꙊꙑꙐ"

        letters = [c for c in s if c.isalpha()]
        if not letters:
            continue
        if sum(_is_cyr_related(c) for c in letters) < max(4, len(letters) // 3):
            continue
        out.append(s)
    return out


def _default_font_path() -> Path:
    for p in (
        Path("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("No suitable TTF (Noto Serif / Liberation / DejaVu Serif)")


def _wood_base(w: int, h: int, rng: random.Random) -> Image.Image:
    """Grey-brown birchbark / wood surface with horizontal grain and scratches."""
    base = Image.new("RGB", (w, h))
    px = base.load()
    base_gray = rng.randint(105, 135)
    base_col = (
        base_gray + rng.randint(4, 14),
        base_gray + rng.randint(-2, 8),
        base_gray - rng.randint(8, 18),
    )
    for y in range(h):
        band = rng.randint(-5, 5) if y % rng.randint(7, 19) == 0 else 0
        grain = int(10 * rng.random())
        for x in range(w):
            slow = int(12 * (x / max(w - 1, 1)) + 8 * (y / max(h - 1, 1)))
            n = rng.randint(-10, 10) + band + grain - slow
            px[x, y] = tuple(max(0, min(255, c + n)) for c in base_col)

    im = base.filter(ImageFilter.GaussianBlur(radius=0.7))
    draw = ImageDraw.Draw(im, "RGBA")
    # Horizontal wood fibres and cracks
    for _ in range(max(28, h // 5)):
        y = rng.randrange(0, h)
        x0 = rng.randrange(0, max(1, w // 3))
        x1 = rng.randrange(max(x0 + 10, w // 2), w)
        col = rng.choice([(45, 38, 32, 45), (190, 178, 155, 28), (25, 22, 20, 65)])
        draw.line([(x0, y), (x1, y + rng.randint(-2, 2))], fill=col, width=rng.choice([1, 1, 2]))
    for _ in range(rng.randint(4, 9)):
        y = rng.randrange(15, max(16, h - 15))
        x0 = rng.randrange(0, max(1, w // 2))
        pts = [(x0, y)]
        for _j in range(rng.randint(3, 7)):
            pts.append((min(w - 1, pts[-1][0] + rng.randint(35, 160)), pts[-1][1] + rng.randint(-8, 8)))
        draw.line(pts, fill=(24, 20, 18, rng.randint(45, 90)), width=rng.choice([1, 2, 3]))
    return im


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if font.getbbox(text)[2] - font.getbbox(text)[0] <= max_width:
        return [text]
    words = text.split()
    if len(words) <= 1:
        # Character-wise wrap for unspaced fragments
        lines: list[str] = []
        cur = ""
        for ch in text:
            trial = cur + ch
            bbox = font.getbbox(trial)
            if bbox[2] - bbox[0] <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines if lines else [text]

    lines = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        bbox = font.getbbox(trial)
        if bbox[2] - bbox[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines if lines else [text]


def _engrave_text(
    im: Image.Image,
    text_lines: list[str],
    font: ImageFont.FreeTypeFont,
    line_heights: list[int],
    *,
    margin: int,
    line_gap: int,
    rng: random.Random,
) -> None:
    """Draw text as shallow incisions: highlight above-left, shadow below-right, dark groove."""
    w, h = im.size
    mask = Image.new("L", im.size, 0)
    md = ImageDraw.Draw(mask)
    y = margin
    drift = rng.randint(-35, 35)
    for i, t in enumerate(text_lines):
        tw = font.getbbox(t)[2] - font.getbbox(t)[0]
        max_x = max(margin, w - margin - tw)
        # Line starts vary like real scratched documents rather than a clean typeset block.
        if i == 0:
            x = rng.randint(margin, max_x)
        else:
            drift += rng.randint(-45, 50)
            x = min(max_x, max(margin, margin + drift + rng.randint(0, max(1, max_x - margin))))
        md.text((x, y), t, font=font, fill=rng.randint(185, 235))
        y += line_heights[i] + line_gap + rng.randint(-3, 4)

    groove = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
    shadow = Image.new("RGBA", im.size, (22, 18, 15, 0))
    shadow.putalpha(groove.point(lambda v: int(v * 0.50)))
    highlight = Image.new("RGBA", im.size, (205, 195, 170, 0))
    highlight.putalpha(groove.point(lambda v: int(v * 0.28)))
    fill = Image.new("RGBA", im.size, (35, 30, 25, 0))
    fill.putalpha(groove.point(lambda v: int(v * 0.38)))

    im.alpha_composite(highlight, (-2, -2))
    im.alpha_composite(shadow, (2, 2))
    im.alpha_composite(fill)


def render_ostromir_sample(
    lines: list[str],
    *,
    width: int = 1400,
    margin: int = 56,
    font_size: int = 38,
    line_gap: int = 14,
    rng: random.Random | None = None,
    font_path: Path | None = None,
) -> tuple[Image.Image, str]:
    """Return (RGB image, exact gold string used: lines joined by newline)."""
    rng = rng or random.Random()
    font_p = font_path or _default_font_path()
    font = ImageFont.truetype(str(font_p), font_size)
    gold = "\n".join(lines)
    max_text_w = width - 2 * margin
    wrapped: list[str] = []
    for ln in lines:
        wrapped.extend(_wrap_text(ln, font, max_text_w))
    line_heights = []
    for t in wrapped:
        bb = font.getbbox(t)
        line_heights.append(bb[3] - bb[1])
    text_h = sum(line_heights) + line_gap * (len(wrapped) - 1 if wrapped else 0)
    height = max(320, margin * 2 + text_h + 40)

    im = _wood_base(width, height, rng).convert("RGBA")
    _engrave_text(
        im,
        wrapped,
        font,
        line_heights,
        margin=margin,
        line_gap=line_gap,
        rng=rng,
    )

    if rng.random() < 0.85:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 0.28)))
    if rng.random() < 0.35:
        overlay = Image.new("RGB", im.size, (255, 255, 255))
        od = ImageDraw.Draw(overlay)
        for x in range(0, width, 6):
            f = 0.94 + 0.06 * (x / max(width - 1, 1))
            od.line([(x, 0), (x, height)], fill=(int(255 * f),) * 3, width=5)
        im = Image.blend(im.convert("RGB"), overlay, 0.06).convert("RGBA")

    return im.convert("RGB"), gold
