#!/usr/bin/env python3
"""Standalone inference + evaluation for a Qwen3.5-VL + LoRA checkpoint.

Reconstructed minimally to evaluate the v5 champion checkpoint on
data/interim/birchbark_test.jsonl after the training-side scripts were
lost in a branch switch.

Usage:
    .venv-qwen-edit-multi/bin/python scripts/eval_qwen_vl_lora.py \\
        --base-model Qwen/Qwen3.5-2B \\
        --adapter   runs/phase4_v5/mixed_80_20__Qwen_Qwen3_5-2B__ceil5ep/checkpoint-900 \\
        --jsonl     data/interim/birchbark_test.jsonl \\
        --image-root . \\
        --out-pred  reports/eval/test_predictions.jsonl \\
        --out-summary reports/eval/test_summary.json \\
        --max-rows 252
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
# Load text_norm directly by file path to avoid pulling birchbark_ocr.data.__init__,
# which imports gramoty -> bs4 (not in this venv).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_text_norm",
    REPO_ROOT / "src" / "birchbark_ocr" / "data" / "text_norm.py",
)
_tn = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tn)
normalize_text = _tn.normalize_text
is_meaningful_text = _tn.is_meaningful_text

# eval/metrics has no heavy deps
sys.path.insert(0, str(REPO_ROOT / "src"))
from birchbark_ocr.eval.metrics import (  # noqa: E402
    cer as cer_fn,
    nls as nls_fn,
    strip_square_brackets,
)


SYSTEM_PROMPT = (
    "You are an expert palaeographer specialising in medieval East Slavic "
    "birchbark documents (gramoty) from Novgorod, Moscow, and other Old Russian "
    "centres, 11th–15th centuries. The image is a photograph or drawing of an "
    "inscription incised into birchbark with a stylus. Transcribe every visible "
    "character of the inscription verbatim, line by line, in the diplomatic "
    "continuous form used at gramoty.ru (no editorial expansions, no "
    "modernisation, no commentary). Preserve the original Old Cyrillic letter "
    "forms (ѣ, ѫ, ѡ, ѥ, ѩ, ѭ, titlo, etc.). Where letters are damaged or "
    "illegible, output a single '-' character. Output the transcription only, "
    "with no extra prose."
)


def _resize_for_qwen(img: Image.Image, max_pixels: int = 451584,
                     min_pixels: int = 100352) -> Image.Image:
    """Resize so total pixels are within [min, max] and dimensions multiples of 28."""
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    pix = w * h
    if pix > max_pixels:
        scale = (max_pixels / pix) ** 0.5
        w, h = int(w * scale), int(h * scale)
    elif pix < min_pixels:
        scale = (min_pixels / pix) ** 0.5
        w, h = int(w * scale), int(h * scale)
    # Round to multiples of 28 (Qwen patch size)
    w = max(28, (w // 28) * 28)
    h = max(28, (h // 28) * 28)
    return img.resize((w, h), Image.BICUBIC)


def _build_messages(prompt: str, image: Image.Image) -> list:
    """Match the chat-template structure used in training."""
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Transcribe."},
            ],
        },
    ]


def _resolve_image_path(row: dict, image_root: Path) -> Path | None:
    """Try a few image-path conventions used across the dataset variants."""
    candidates = []
    for key in ("image_path", "primary_image", "image_paths"):
        v = row.get(key)
        if isinstance(v, str) and v:
            candidates.append(v)
        elif isinstance(v, list) and v:
            candidates.append(v[0])
    for c in candidates:
        p = Path(c)
        if p.is_absolute() and p.exists():
            return p
        # Relative paths like "documents/.../images/photo_xxx.jpg" are stored
        # under data/raw/gramoty/. Try a few likely prefixes.
        for prefix in (Path(""), Path("data/raw/gramoty"), Path("data/interim")):
            cand = (image_root / prefix / c).resolve()
            if cand.exists():
                return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", required=True,
                    help="Path to a checkpoint directory containing adapter_config.json")
    ap.add_argument("--jsonl", required=True,
                    help="Path to JSONL with rows containing image_path/primary_image and text")
    ap.add_argument("--image-root", default=str(REPO_ROOT))
    ap.add_argument("--out-pred", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = all rows")
    ap.add_argument("--text-field", default="text",
                    help="JSONL field that contains the gold text "
                         "(use 'text' for already-normalised; this script "
                         "always re-applies normalize_text before scoring)")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=1,
                    help="1 = greedy / sampling; >1 = beam search")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--length-penalty", type=float, default=1.0,
                    help="Only used when num-beams > 1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    print(f"[+] loading base {args.base_model} ...", flush=True)
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Model
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Model
    from peft import PeftModel

    proc = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = _Model.from_pretrained(
        args.base_model,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    print(f"[+] loading adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model = model.to(args.device).eval()
    print(f"[+] model on {args.device} dtype={args.dtype}", flush=True)

    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines()]
    if args.max_rows:
        rows = rows[: args.max_rows]
    image_root = Path(args.image_root).resolve()
    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out_pred, "w", encoding="utf-8")

    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    eos_id = proc.tokenizer.eos_token_id

    n_total = 0
    n_skipped_no_img = 0
    n_skipped_meaningless = 0
    cer_raws: list[float] = []
    cer_strips: list[float] = []
    nls_raws: list[float] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        gold_raw = row.get(args.text_field, "")
        gold_norm = normalize_text(gold_raw)
        if not is_meaningful_text(gold_norm, min_visible_chars=5):
            n_skipped_meaningless += 1
            continue

        img_path = _resolve_image_path(row, image_root)
        if img_path is None:
            n_skipped_no_img += 1
            continue

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"[skip-img] {img_path}: {e}", flush=True)
            n_skipped_no_img += 1
            continue
        img = _resize_for_qwen(img)

        msgs = _build_messages(SYSTEM_PROMPT, img)
        text = proc.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(args.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if args.num_beams > 1:
            gen_kwargs["length_penalty"] = args.length_penalty
            gen_kwargs["early_stopping"] = True
        with torch.no_grad():
            gen = model.generate(**inputs, **gen_kwargs)
        # Drop the prompt tokens from the generated ids
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = gen[0, prompt_len:].tolist()
        pred = proc.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        c_raw = cer_fn(pred, gold_norm)
        c_strip = cer_fn(strip_square_brackets(pred), strip_square_brackets(gold_norm))
        n_raw = nls_fn(pred, gold_norm)
        cer_raws.append(c_raw)
        cer_strips.append(c_strip)
        nls_raws.append(n_raw)
        n_total += 1

        out = {
            "row_index": i,
            "doc_id": row.get("doc_id"),
            "image_path": str(img_path),
            "gold_raw": gold_raw,
            "gold_norm": gold_norm,
            "pred": pred,
            "cer_raw": c_raw,
            "cer_brackets_stripped": c_strip,
            "nls": n_raw,
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            mean_cer = sum(cer_raws) / max(1, len(cer_raws))
            mean_nls = sum(nls_raws) / max(1, len(nls_raws))
            elapsed = time.time() - t0
            print(f"[{i+1:4d}/{len(rows)}]  scored={n_total}  "
                  f"meanCER={mean_cer:.4f}  meanNLS={mean_nls:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    fout.close()

    summary = {
        "adapter": args.adapter,
        "base_model": args.base_model,
        "jsonl": args.jsonl,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "n_rows_input": len(rows),
        "n_rows_scored": n_total,
        "n_skipped_no_img": n_skipped_no_img,
        "n_skipped_meaningless": n_skipped_meaningless,
        "mean_cer_raw": sum(cer_raws) / max(1, len(cer_raws)),
        "mean_cer_brackets_stripped": sum(cer_strips) / max(1, len(cer_strips)),
        "mean_nls": sum(nls_raws) / max(1, len(nls_raws)),
        "elapsed_sec": time.time() - t0,
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
