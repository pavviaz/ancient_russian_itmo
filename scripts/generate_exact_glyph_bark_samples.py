#!/usr/bin/env python3
"""Generate label-safe synthetic samples: SDXL+LoRA bark background + exact glyph engraving."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import httpx
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from birchbark_ocr.synth.ostromir_render import (
    _default_font_path,
    _wrap_text,
    iter_ostromir_content_lines,
)

OSTROMIR_URL = "http://www.ponomar.net/files/ostromir.txt"

BACKGROUND_PROMPT = (
    "<birchbark> macro close-up of a flat aged medieval birchbark manuscript fragment, "
    "grey brown wooden tablet surface fills the whole image, smooth fibrous horizontal grain, "
    "subtle stains and shallow scratches, archaeological archive photograph, no writing"
)
BACKGROUND_NEGATIVE = (
    "letters, text, inscription, glyphs, alphabet, symbols, black ink, printed writing, "
    "museum display, glass case, shelf, border, frame, label, plaque, wide shot, object on table"
    ", deep cracks, alligator cracked paint, tree trunk, natural birch tree bark, black lenticels, "
    "white birch bark pattern, bark wall, forest"
)


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


def render_exact_mask(
    lines: list[str],
    *,
    size: int,
    font_size: int,
    margin: int,
    line_gap: int,
    stroke_width: int,
    rng: random.Random,
) -> tuple[Image.Image, str]:
    """Return exact text mask and GT. Mask layout varies like scratched birchbark lines."""
    font = ImageFont.truetype(str(_default_font_path()), font_size)
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(line, font, size - 2 * margin))
    heights = [font.getbbox(t)[3] - font.getbbox(t)[1] for t in wrapped]
    total_h = sum(heights) + max(0, len(heights) - 1) * line_gap
    y = max(margin, (size - total_h) // 2 + rng.randint(-35, 35))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    drift = rng.randint(-100, 80)
    for i, text in enumerate(wrapped):
        tw = font.getbbox(text)[2] - font.getbbox(text)[0]
        max_x = max(margin, size - margin - tw)
        drift += rng.randint(-70, 75)
        x = min(max_x, max(margin, margin + drift + rng.randint(0, max(1, max_x - margin))))
        draw.text((x, y), text, font=font, fill=235)
        y += heights[i] + line_gap + rng.randint(-4, 5)
    if stroke_width > 1:
        for _ in range(stroke_width - 1):
            mask = mask.filter(ImageFilter.MaxFilter(3))
    return mask, "\n".join(lines)


def engrave_exact_glyphs(background: Image.Image, mask: Image.Image, *, strength: float) -> Image.Image:
    """Composite exact glyph mask as incised strokes over a generated bark background."""
    bg = background.convert("RGBA")
    groove = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
    # Carving uses offsets rather than opaque printed letters.
    shadow = Image.new("RGBA", bg.size, (18, 14, 10, 0))
    shadow.putalpha(groove.point(lambda v: int(v * 0.42 * strength)))
    highlight = Image.new("RGBA", bg.size, (230, 218, 190, 0))
    highlight.putalpha(groove.point(lambda v: int(v * 0.22 * strength)))
    dark_fill = Image.new("RGBA", bg.size, (26, 21, 17, 0))
    dark_fill.putalpha(groove.point(lambda v: int(v * 0.10 * strength)))
    bg.alpha_composite(highlight, (-2, -2))
    bg.alpha_composite(shadow, (2, 2))
    bg.alpha_composite(dark_fill)
    # Add a faint displaced groove line to make strokes scratched rather than typographic.
    fine = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.25))
    edge = Image.new("RGBA", bg.size, (8, 6, 4, 0))
    edge.putalpha(fine.point(lambda v: int(v * 0.42 * strength)))
    bg.alpha_composite(edge)
    return bg.convert("RGB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("reports/figs/exact_glyph_bark_samples"))
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--lines-per-image", type=int, default=5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--font-size", type=int, default=44)
    p.add_argument("--margin", type=int, default=76)
    p.add_argument("--line-gap", type=int, default=18)
    p.add_argument("--stroke-width", type=int, default=1)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--engrave-strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--cache", type=Path, default=Path("data/raw/ostromir/ostromir.txt"))
    p.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--lora-dir", type=Path, default=Path("runs/phase3_sdxl_lora_birchbark_v1_retry_20260503"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for SDXL background generation.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_text(OSTROMIR_URL, args.cache)
    pool = iter_ostromir_content_lines(raw)
    rng = random.Random(args.seed)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(str(args.lora_dir))
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    records: list[dict[str, object]] = []
    for i in range(args.n_samples):
        start = rng.randint(0, len(pool) - args.lines_per_image)
        lines = pool[start : start + args.lines_per_image]
        layout_seed = rng.randint(0, 2**30)
        layout_rng = random.Random(layout_seed)
        mask, gold = render_exact_mask(
            lines,
            size=args.size,
            font_size=args.font_size,
            margin=args.margin,
            line_gap=args.line_gap,
            stroke_width=args.stroke_width,
            rng=layout_rng,
        )
        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        background = pipe(
            BACKGROUND_PROMPT,
            negative_prompt=BACKGROUND_NEGATIVE,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        final = engrave_exact_glyphs(background, mask, strength=args.engrave_strength)
        stem = f"exact_glyph_bark_{i:02d}"
        background.save(args.output_dir / f"{stem}.background.png")
        mask.convert("RGB").save(args.output_dir / f"{stem}.mask.png")
        final.save(args.output_dir / f"{stem}.png")
        (args.output_dir / f"{stem}.gold.txt").write_text(gold + "\n", encoding="utf-8")
        records.append(
            {
                "image": str(args.output_dir / f"{stem}.png"),
                "background": str(args.output_dir / f"{stem}.background.png"),
                "mask": str(args.output_dir / f"{stem}.mask.png"),
                "gold": str(args.output_dir / f"{stem}.gold.txt"),
                "text": gold,
                "source": "ostromir",
                "source_start_line": start,
                "source_end_line": start + args.lines_per_image,
                "seed": args.seed + i,
                "layout_seed": layout_seed,
                "lines_per_image": args.lines_per_image,
                "background_prompt": BACKGROUND_PROMPT,
                "background_negative_prompt": BACKGROUND_NEGATIVE,
                "engrave_strength": args.engrave_strength,
                "stroke_width": args.stroke_width,
            }
        )
        print(args.output_dir / f"{stem}.png")
    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
