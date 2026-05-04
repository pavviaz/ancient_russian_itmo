#!/usr/bin/env python3
"""Generate SDXL+LoRA samples conditioned on exact rendered Old Cyrillic glyph layouts."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from birchbark_ocr.synth.ostromir_render import iter_ostromir_content_lines

OSTROMIR_URL = "http://www.ponomar.net/files/ostromir.txt"
PROMPT = (
    "<birchbark> Old Russian inscription scratched into birch bark, grey brown fibrous "
    "wooden surface, shallow dark incised Cyrillic letters, medieval Novgorod birchbark "
    "gramota, archaeological document photo"
)
NEGATIVE_PROMPT = (
    "modern paper, printed ink, latin text, clean typography, typed document, colorful, "
    "watermark, signature, extra symbols"
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


def default_font_path() -> Path:
    for p in (
        Path("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ):
        if p.exists():
            return p
    raise FileNotFoundError("No suitable serif font found")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if font.getbbox(text)[2] - font.getbbox(text)[0] <= max_width:
        return [text]
    words = text.split()
    out: list[str] = []
    cur: list[str] = []
    for word in words:
        trial = " ".join(cur + [word])
        if font.getbbox(trial)[2] - font.getbbox(trial)[0] <= max_width:
            cur.append(word)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = [word]
    if cur:
        out.append(" ".join(cur))
    return out


def render_condition(
    lines: list[str],
    *,
    size: int,
    font_size: int,
    margin: int,
    line_gap: int,
    rng: random.Random,
) -> Image.Image:
    """White glyph strokes on black background, suitable for canny/line ControlNet."""
    font = ImageFont.truetype(str(default_font_path()), font_size)
    im = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(im)
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_text(line, font, size - 2 * margin))
    heights = [font.getbbox(t)[3] - font.getbbox(t)[1] for t in wrapped]
    total_h = sum(heights) + max(0, len(heights) - 1) * line_gap
    y = max(margin, (size - total_h) // 2 + rng.randint(-40, 40))
    drift = rng.randint(-160, 120)
    for i, text in enumerate(wrapped):
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        max_x = max(margin, size - margin - tw)
        drift += rng.randint(-80, 90)
        x = min(max_x, max(margin, margin + drift + rng.randint(0, max(1, max_x - margin))))
        draw.text((x, y), text, font=font, fill=255)
        y += heights[i] + line_gap + rng.randint(-4, 5)
    # Slightly thicken strokes so ControlNet sees continuous line structure.
    return im.filter(ImageFilter.MaxFilter(3)).convert("RGB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("reports/figs/sdxl_glyph_conditioned"))
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--lines-per-image", type=int, default=5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--font-size", type=int, default=46)
    p.add_argument("--margin", type=int, default=80)
    p.add_argument("--line-gap", type=int, default=18)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=6.5)
    p.add_argument("--controlnet-scale", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cache", type=Path, default=Path("data/raw/ostromir/ostromir.txt"))
    p.add_argument("--url", default=OSTROMIR_URL)
    p.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--controlnet-model", default="diffusers/controlnet-canny-sdxl-1.0")
    p.add_argument(
        "--lora-dir",
        type=Path,
        default=Path("runs/phase3_sdxl_lora_birchbark_v1_retry_20260503"),
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this script.", file=sys.stderr)
        return 1

    raw = load_text(args.url, args.cache)
    pool = iter_ostromir_content_lines(raw)
    rng = random.Random(args.seed)
    starts = [rng.randint(0, len(pool) - args.lines_per_image) for _ in range(args.n_samples)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(str(args.lora_dir))
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    for i, start in enumerate(starts):
        lines = pool[start : start + args.lines_per_image]
        gold = "\n".join(lines)
        cond = render_condition(
            lines,
            size=args.size,
            font_size=args.font_size,
            margin=args.margin,
            line_gap=args.line_gap,
            rng=random.Random(rng.randint(0, 2**30)),
        )
        cond_path = args.output_dir / f"glyph_condition_{i:02d}.png"
        out_path = args.output_dir / f"sdxl_glyph_conditioned_{i:02d}.png"
        txt_path = args.output_dir / f"sdxl_glyph_conditioned_{i:02d}.gold.txt"
        cond.save(cond_path)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        image = pipe(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=cond,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=gen,
        ).images[0]
        image.save(out_path)
        txt_path.write_text(gold + "\n", encoding="utf-8")
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
