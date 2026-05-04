#!/usr/bin/env python3
"""Mini-ablation for SDXL+LoRA glyph conditioning settings."""

from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw, ImageFont

from generate_sdxl_glyph_conditioned_samples import (
    NEGATIVE_PROMPT,
    OSTROMIR_URL,
    load_text,
    render_condition,
)
from birchbark_ocr.synth.ostromir_render import iter_ostromir_content_lines

TIGHT_PROMPT = (
    "<birchbark> macro close-up crop of an Old Russian birchbark inscription, "
    "the bark surface fills the whole frame, shallow dark scratched Cyrillic letters carved "
    "into grey brown fibrous wood, medieval Novgorod gramota fragment, archival photograph, "
    "no border, no glass, no museum shelf"
)
TIGHT_NEGATIVE = (
    NEGATIVE_PROMPT
    + ", museum display, glass case, shelf, frame, mat, wall, room, label, plaque, border, "
    "wide shot, object floating on background, printed black ink, clean font"
)

CONTROLNETS = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "mistoline": "TheMistoAI/MistoLine",
    "softedge": "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
}


def slug(s: str) -> str:
    return s.replace("/", "__").replace(".", "_").replace("-", "_")


def load_controlnet(model_id: str) -> ControlNetModel:
    try:
        return ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    except Exception:
        return ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )


def make_contact_sheet(
    output_dir: Path,
    condition_paths: list[Path],
    rows: list[tuple[str, float, int, Path]],
    *,
    thumb: int = 256,
) -> None:
    font = ImageFont.load_default()
    labels_h = 26
    cols = 1 + len({(name, scale) for name, scale, _idx, _path in rows})
    sample_ids = sorted({idx for _name, _scale, idx, _path in rows})
    cell_w, cell_h = thumb, thumb + labels_h
    sheet = Image.new("RGB", (cols * cell_w, len(sample_ids) * cell_h), (235, 235, 230))
    draw = ImageDraw.Draw(sheet)
    col_keys = sorted({(name, scale) for name, scale, _idx, _path in rows})
    for r, sample_idx in enumerate(sample_ids):
        cond = Image.open(condition_paths[sample_idx]).convert("RGB")
        cond.thumbnail((thumb, thumb))
        sheet.paste(cond, (0, r * cell_h + labels_h))
        draw.text((4, r * cell_h + 5), f"condition {sample_idx}", fill=(0, 0, 0), font=font)
        for c, (name, scale) in enumerate(col_keys, start=1):
            match = [p for n, s, i, p in rows if n == name and s == scale and i == sample_idx]
            if not match:
                continue
            im = Image.open(match[0]).convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (c * cell_w, r * cell_h + labels_h))
            draw.text((c * cell_w + 4, r * cell_h + 5), f"{name} {scale}", fill=(0, 0, 0), font=font)
    sheet.save(output_dir / "glyph_conditioning_sweep_contact_sheet.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("reports/figs/glyph_conditioning_sweep"))
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--lines-per-image", type=int, default=5)
    p.add_argument("--scales", nargs="+", type=float, default=[0.55, 0.8, 1.05])
    p.add_argument("--models", nargs="+", default=["canny", "mistoline", "softedge"])
    p.add_argument("--steps", type=int, default=22)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=20260503)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--font-size", type=int, default=46)
    p.add_argument("--control-guidance-end", type=float, default=0.9)
    p.add_argument("--cache", type=Path, default=Path("data/raw/ostromir/ostromir.txt"))
    p.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument(
        "--lora-dir",
        type=Path,
        default=Path("runs/phase3_sdxl_lora_birchbark_v1_retry_20260503"),
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_text(OSTROMIR_URL, args.cache)
    pool = iter_ostromir_content_lines(raw)
    rng = random.Random(args.seed)
    starts = [rng.randint(0, len(pool) - args.lines_per_image) for _ in range(args.n_samples)]

    conditions: list[Image.Image] = []
    condition_paths: list[Path] = []
    for i, start in enumerate(starts):
        lines = pool[start : start + args.lines_per_image]
        cond = render_condition(
            lines,
            size=args.size,
            font_size=args.font_size,
            margin=80,
            line_gap=18,
            rng=random.Random(rng.randint(0, 2**30)),
        )
        conditions.append(cond)
        path = args.output_dir / f"condition_{i:02d}.png"
        cond.save(path)
        condition_paths.append(path)
        (args.output_dir / f"sample_{i:02d}.gold.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows: list[tuple[str, float, int, Path]] = []
    failures: list[str] = []
    for name in args.models:
        model_id = CONTROLNETS.get(name, name)
        try:
            controlnet = load_controlnet(model_id)
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
            for scale in args.scales:
                for i, cond in enumerate(conditions):
                    gen = torch.Generator(device="cuda").manual_seed(args.seed + i + int(scale * 1000))
                    image = pipe(
                        TIGHT_PROMPT,
                        negative_prompt=TIGHT_NEGATIVE,
                        image=cond,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        controlnet_conditioning_scale=scale,
                        control_guidance_end=args.control_guidance_end,
                        generator=gen,
                    ).images[0]
                    path = args.output_dir / f"{slug(name)}_scale_{scale:.2f}_sample_{i:02d}.png"
                    image.save(path)
                    rows.append((name, scale, i, path))
                    print(path)
            del pipe, controlnet
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name} ({model_id}): {type(e).__name__}: {e}")
            print(f"FAILED {name}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    if rows:
        make_contact_sheet(args.output_dir, condition_paths, rows)
    if failures:
        (args.output_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
