#!/usr/bin/env python3
"""Phase 2 OCR baselines on birchbark JSONL (YAML config + argparse overrides)."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import orjson
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from birchbark_ocr.eval.metrics import cer, exact, nls, normalize, strip_square_brackets

log = logging.getLogger(__name__)

# agent_protocol.md §3.2 — Qwen3.5 palaeographer prompt
BIRCHBARK_VLM_PROMPT = """You are an expert palaeographer reading Old Russian birchbark inscriptions
from medieval Novgorod (XI–XV century). The text is scratched into birch bark,
written continuously without spaces between words, and uses Old Cyrillic
letterforms including ѣ, ѧ, ѫ, ѳ, ѵ, ѡ, ѥ, ѩ, ѭ, ѯ, ѱ, ҂.

Transcribe the line in the image diplomatically — preserve original letterforms,
diacritics (titlas), superscript letters, and the original lack of word spacing.
Do not modernise. Do not add punctuation. Use square brackets [...] only for
characters that are visible but ambiguous; never invent missing text.

Output only the transcribed line, nothing else."""

_easyocr_readers: dict[tuple[tuple[str, ...], bool], Any] = {}
_trocr_bundle: dict[str, tuple[Any, Any, Any]] = {}
_qwen_bundles: dict[str, tuple[Any, Any, str]] = {}
_churro_clients: dict[tuple[str, int], Any] = {}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows


def resolve_image(repo_root: Path, primary: str | None, paths: list[str]) -> Path | None:
    gramoty_root = repo_root / "data/raw/gramoty"
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    candidates.extend(paths or [])
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_absolute() and p.exists():
            return p
        for base in (repo_root, gramoty_root):
            trial = base / p
            if trial.exists():
                return trial
    return None


def predict_tesseract(img_path: Path, lang: str) -> str:
    import pytesseract

    im = Image.open(img_path).convert("RGB")
    return pytesseract.image_to_string(im, lang=lang) or ""


def predict_easyocr(img_path: Path, langs: list[str], gpu: bool) -> str:
    import easyocr

    key = (tuple(langs), gpu)
    if key not in _easyocr_readers:
        _easyocr_readers[key] = easyocr.Reader(langs, gpu=gpu, verbose=False)
    reader = _easyocr_readers[key]
    lines = reader.readtext(str(img_path), detail=0, paragraph=True)
    if isinstance(lines, list):
        return "\n".join(str(x) for x in lines if x)
    return str(lines)


def _predict_churro_subprocess_fallback(img_path: Path, model: str, timeout_s: int) -> str:
    """Legacy `churro-ocr transcribe` path (no max_new_tokens cap; upstream default ~25k)."""
    import os
    import shutil

    bin_name = os.environ.get("CHURRO_OCR_BIN", "churro-ocr")
    cmd = [bin_name, "transcribe", "--image", str(img_path), "--backend", "hf", "--model", model]
    if not shutil.which(bin_name):
        log.warning("churro-ocr not on PATH (%s); activate .venv-churro-paddle or set CHURRO_OCR_BIN", bin_name)
        return ""
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("churro-cli timeout (%ss) for %s", timeout_s, img_path)
        return ""
    if r.returncode != 0:
        log.warning("churro-cli failed (%s): %s", r.returncode, (r.stderr or "")[:500])
        return ""
    return _churro_output_to_plaintext((r.stdout or "").strip())


def predict_churro_cli(img_path: Path, model: str, timeout_s: int, max_new_tokens: int = 128) -> str:
    """CHURRO on Hugging Face: use `churro_ocr` with a hard generation cap.

    The published CLI does not expose generation limits; the HF backend defaults to
    ``max_new_tokens`` ≈ 25k (`DEFAULT_OCR_MAX_TOKENS`), which can spiral on bad images.
    """
    import concurrent.futures

    try:
        from churro_ocr.ocr import OCRClient
        from churro_ocr.providers.builder import build_ocr_backend
        from churro_ocr.providers.specs import HuggingFaceOptions, OCRBackendSpec
    except ImportError:
        log.warning(
            "churro_ocr not importable; falling back to churro-ocr CLI without max_new_tokens cap "
            "(install churro-ocr[hf] in this environment)."
        )
        return _predict_churro_subprocess_fallback(img_path, model, timeout_s)

    key = (model, int(max_new_tokens))
    if key not in _churro_clients:
        log.info("Loading CHURRO HF backend %s (max_new_tokens=%s)", model, max_new_tokens)
        backend = build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=model,
                options=HuggingFaceOptions(
                    model_kwargs={"device_map": "auto", "torch_dtype": "auto"},
                    generation_kwargs={"max_new_tokens": int(max_new_tokens), "do_sample": False},
                ),
            )
        )
        _churro_clients[key] = OCRClient(backend)

    client = _churro_clients[key]

    def _run() -> str:
        page = client.ocr_image(image_path=img_path)
        return _churro_output_to_plaintext((page.text or "").strip())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning("churro HF timeout (%ss) for %s", timeout_s, img_path)
            return ""


def _churro_output_to_plaintext(s: str) -> str:
    """CHURRO CLI prints HistoricalDocument XML; extract <Line> text for CER."""
    if not s or "<HistoricalDocument" not in s:
        return s
    import re

    lines = re.findall(r"<Line>([^<]*)</Line>", s, flags=re.IGNORECASE)
    if lines:
        return "".join(lines)
    return re.sub(r"<[^>]+>", "", s)


def predict_trocr(img_path: Path, model_name: str, device_s: str) -> str:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = torch.device(device_s if torch.cuda.is_available() and device_s == "cuda" else "cpu")
    if model_name not in _trocr_bundle:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        model.eval()
        _trocr_bundle[model_name] = (processor, model, device)
    processor, model, device = _trocr_bundle[model_name]
    im = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=im, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def _tensor_inputs_to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def predict_qwen35(
    img_path: Path, hf_model_id: str, device_s: str, max_image_side: int = 896
) -> str:
    """Qwen3.5 unified VL (HF `Qwen3_5ForConditionalGeneration`) — agent_protocol §3.2."""
    import gc

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    use_cuda = torch.cuda.is_available() and device_s == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > max_image_side:
        s = max_image_side / m
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)

    if hf_model_id not in _qwen_bundles:
        log.info("Loading Qwen3.5 %s …", hf_model_id)
        processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=dtype,
            device_map="cuda" if use_cuda else None,
            trust_remote_code=True,
        )
        if not use_cuda:
            model = model.to(device)
        model.eval()
        _qwen_bundles[hf_model_id] = (processor, model, str(device))

    processor, model, _ = _qwen_bundles[hf_model_id]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": BIRCHBARK_VLM_PROMPT},
            ],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _tensor_inputs_to_device(inputs, device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        trimmed = out_ids[:, in_len:]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    except Exception as e:
        log.warning("Qwen3.5 inference failed (%s): %s", hf_model_id, e)
        return ""
    finally:
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()


def unload_trocr_models() -> None:
    import gc

    import torch

    _trocr_bundle.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def unload_qwen_models() -> None:
    import gc

    import torch

    for _mid, (proc, model, _) in list(_qwen_bundles.items()):
        del proc, model
    _qwen_bundles.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


_paddle_vl_ocr: Any = None


def _paddle_vl_collect_texts(obj: Any) -> list[str]:
    """Best-effort text extraction from PaddleOCRVL / PaddleX result dicts."""
    acc: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if lk in ("markdown", "text", "ocr_text", "content", "transcription") and isinstance(v, str):
                acc.append(v)
            elif lk == "rec_texts" and isinstance(v, (list, tuple)):
                acc.extend(str(x) for x in v if x)
            elif lk == "rec_text" and isinstance(v, str):
                acc.append(v)
            else:
                acc.extend(_paddle_vl_collect_texts(v))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            acc.extend(_paddle_vl_collect_texts(x))
    return acc


def predict_paddle_vl(
    img_path: Path,
    lang: str,
    *,
    max_new_tokens: int = 512,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    repetition_penalty: float | None = None,
    temperature: float | None = None,
    timeout_s: int = 300,
) -> str:
    """PaddleOCR-VL when paddlepaddle+paddleocr are installed (often not on Python 3.14).

    PaddleX defaults to very large ``max_new_tokens`` (e.g. 8192), which on CPU can look
    like a hang. We pass a smaller cap and optional wall-clock timeout per image.
    """
    import concurrent.futures

    global _paddle_vl_ocr
    try:
        from paddleocr import PaddleOCRVL  # type: ignore[import-not-found]
    except Exception as e:
        log.warning("PaddleOCR-VL unavailable: %s", e)
        return ""
    try:
        if _paddle_vl_ocr is None:
            # Line crops: skip heavy layout model when possible (faster cold start).
            _paddle_vl_ocr = PaddleOCRVL(use_layout_detection=False)

        pred_kw: dict[str, Any] = {
            "use_layout_detection": False,
            "max_new_tokens": int(max_new_tokens),
        }
        if max_pixels is not None:
            pred_kw["max_pixels"] = int(max_pixels)
        if min_pixels is not None:
            pred_kw["min_pixels"] = int(min_pixels)
        if repetition_penalty is not None:
            pred_kw["repetition_penalty"] = float(repetition_penalty)
        if temperature is not None:
            pred_kw["temperature"] = float(temperature)

        def _run() -> Any:
            return _paddle_vl_ocr.predict(str(img_path), **pred_kw)

        if timeout_s and timeout_s > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run)
                try:
                    out = fut.result(timeout=float(timeout_s))
                except concurrent.futures.TimeoutError:
                    log.warning(
                        "PaddleOCR-VL timeout (%ss) for %s — raise paddle_timeout_seconds or "
                        "paddle_vl_max_new_tokens if hits are too frequent.",
                        timeout_s,
                        img_path,
                    )
                    return ""
        else:
            out = _run()

        if isinstance(out, list) and out:
            texts = []
            for page in out:
                texts.extend(_paddle_vl_collect_texts(page))
            if texts:
                return "\n".join(texts).strip()
            return ""
        if isinstance(out, str):
            return out.strip()
        return ""
    except Exception as e:
        log.warning("PaddleOCR-VL predict failed: %s", e)
        return ""


def dispatch_prediction(model: str, img_path: Path, cfg: DictConfig) -> str:
    m = model.lower().strip()
    if m == "tesseract":
        import shutil

        if not shutil.which("tesseract"):
            log.warning("tesseract binary not found; skipping OCR")
            return ""
        return predict_tesseract(img_path, str(cfg.tesseract_lang))
    if m == "easyocr":
        langs = [str(x) for x in cfg.easyocr_langs]
        gpu = str(cfg.device) == "cuda"
        try:
            return predict_easyocr(img_path, langs, gpu=gpu)
        except Exception as e:
            log.warning("EasyOCR unavailable (%s); install deps or skip.", e)
            return ""
    if m == "churro_cli":
        to = int(cfg.get("churro_timeout_seconds", 180))
        cap = int(cfg.get("churro_max_new_tokens", 128))
        return predict_churro_cli(img_path, str(cfg.churro_model), to, max_new_tokens=cap)
    if m == "trocr":
        return predict_trocr(img_path, str(cfg.trocr_model), str(cfg.device))
    if m == "qwen35_2b":
        side = int(cfg.get("qwen_max_image_side", 896))
        return predict_qwen35(img_path, str(cfg.qwen35_2b_model), str(cfg.device), max_image_side=side)
    if m == "qwen35_08b":
        side = int(cfg.get("qwen_max_image_side", 896))
        return predict_qwen35(img_path, str(cfg.qwen35_08b_model), str(cfg.device), max_image_side=side)
    if m == "paddle_vl":
        max_px = cfg.get("paddle_vl_max_pixels")
        min_px = cfg.get("paddle_vl_min_pixels")
        rep = cfg.get("paddle_vl_repetition_penalty")
        temp = cfg.get("paddle_vl_temperature")
        return predict_paddle_vl(
            img_path,
            str(cfg.get("paddle_lang", "latin")),
            max_new_tokens=int(cfg.get("paddle_vl_max_new_tokens", 512)),
            max_pixels=int(max_px) if max_px not in (None, "") else None,
            min_pixels=int(min_px) if min_px not in (None, "") else None,
            repetition_penalty=float(rep) if rep not in (None, "") else None,
            temperature=float(temp) if temp not in (None, "") else None,
            timeout_s=int(cfg.get("paddle_timeout_seconds", 300)),
        )
    if m == "qwen_vl":
        # Legacy alias → Qwen3.5-2B
        side = int(cfg.get("qwen_max_image_side", 896))
        return predict_qwen35(
            img_path, str(cfg.get("qwen35_2b_model", "Qwen/Qwen3.5-2B")), str(cfg.device), max_image_side=side
        )
    raise ValueError(f"Unknown model backend: {model}")


def summarise(preds: list[str], golds: list[str]) -> dict[str, Any]:
    n = len(preds)
    cers_raw = [cer(p, g) for p, g in zip(preds, golds, strict=True)]
    cers_strip = [cer(strip_square_brackets(p), strip_square_brackets(g)) for p, g in zip(preds, golds, strict=True)]
    nlss = [nls(p, g) for p, g in zip(preds, golds, strict=True)]
    ex = [exact(p, g) for p, g in zip(preds, golds, strict=True)]
    return {
        "n": n,
        "cer_mean_raw": sum(cers_raw) / n if n else float("nan"),
        "cer_mean_brackets_stripped": sum(cers_strip) / n if n else float("nan"),
        "nls_mean": sum(nlss) / n if n else float("nan"),
        "exact_match_rate": sum(ex) / n if n else float("nan"),
    }


def plot_bars(metrics_by_model: dict[str, dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(metrics_by_model.keys())
    nls_vals = [metrics_by_model[m]["nls_mean"] for m in names]
    plt.figure(figsize=(max(6, len(names) * 1.2), 4))
    plt.bar(names, nls_vals, color="steelblue")
    plt.ylabel("Mean NLS (↑ better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Phase 2 birchbark baselines")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/phase2/default.yaml",
        help="YAML config (defaults mirror Hydra layout under configs/phase2/)",
    )
    ap.add_argument("--repo-root", type=Path, default=None, help="Project root (default: parent of configs/)")
    ap.add_argument("--limit", type=int, default=None, help="Override cfg.limit (0 = all)")
    args = ap.parse_args()

    repo_root = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    cfg = OmegaConf.load(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    test_path = repo_root / str(cfg.test_jsonl)
    if not test_path.exists():
        log.error("Missing test JSONL at %s — build interim after scrape.", test_path)
        sys.exit(2)

    rows = load_rows(test_path)
    limit = int(cfg.limit or 0)
    if limit > 0:
        rows = rows[:limit]

    preds_path = repo_root / str(cfg.predictions_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    models = [str(m) for m in cfg.models]
    all_records: list[dict[str, Any]] = []

    for model in models:
        log.info("Running model=%s on %d docs", model, len(rows))
        for row in tqdm(rows, desc=model):
            img = resolve_image(repo_root, row.get("primary_image"), row.get("image_paths") or [])
            gold = row.get("text") or ""
            doc_id = row.get("doc_id")
            if img is None:
                pred = ""
                log.debug("skip missing image doc_id=%s", doc_id)
            else:
                try:
                    pred = dispatch_prediction(model, img, cfg)
                except Exception as e:
                    log.warning("predict failed doc_id=%s model=%s err=%s", doc_id, model, e)
                    pred = ""
            all_records.append(
                {
                    "doc_id": doc_id,
                    "model": model,
                    "prediction": pred,
                    "gold": gold,
                    "primary_image": str(img) if img else None,
                }
            )
        if str(model).lower() == "trocr":
            unload_trocr_models()
        if str(model).lower().startswith("qwen35"):
            unload_qwen_models()

    with preds_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    gz_path = preds_path.with_suffix(preds_path.suffix + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
        for rec in all_records:
            gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote predictions %s and %s", preds_path, gz_path)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        preds = [r["prediction"] for r in all_records if r["model"] == model]
        golds = [r["gold"] for r in all_records if r["model"] == model]
        metrics_by_model[model] = summarise(preds, golds)

    metrics_path = repo_root / str(cfg.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")
    log.info("Wrote metrics %s", metrics_path)

    fig_path = repo_root / str(cfg.fig_path)
    try:
        plot_bars(metrics_by_model, fig_path)
        log.info("Wrote figure %s", fig_path)
    except Exception as e:
        log.warning("Could not plot bars: %s", e)

    cfg_dump = repo_root / "runs/phase2/resolved_config.yaml"
    cfg_dump.parent.mkdir(parents=True, exist_ok=True)
    cfg_dump.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")


if __name__ == "__main__":
    main()
