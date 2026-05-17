#!/usr/bin/env bash
# Re-run CHURRO-3B on data/interim/birchbark_test.jsonl using the .venv-churro-paddle
# environment, applying the same gold-text normalization (text_norm.normalize_text)
# we use for our Qwen3.5-2B + LoRA evaluation. Provides the "fair-comparison"
# CHURRO number requested in §A5 of FINDINGS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${CHURRO_PADDLE_VENV:-$ROOT/.venv-churro-paddle}"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[!] Missing venv at $VENV — bootstrap with scripts/setup_churro_paddle_venv.sh" >&2
    exit 1
fi

export PATH="$HOME/.local/bin:$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CHURRO_TORCH_BACKEND="${CHURRO_TORCH_BACKEND:-cu124}"
# `churro-ocr install hf` registers the HuggingFace backend; no-op if registered.
"$VENV/bin/churro-ocr" install hf --torch-backend "$CHURRO_TORCH_BACKEND" >/dev/null 2>&1 || true

OUT_PRED="${OUT_PRED:-reports/eval/test_predictions_churro.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-reports/eval/test_summary_churro.json}"
mkdir -p "$(dirname "$OUT_PRED")"

exec "$VENV/bin/python" - <<'PY'
"""CHURRO-3B inference + fair postprocessing for the test_birch comparison."""
from __future__ import annotations

import json
import os
import sys
import time
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[0] if False else Path.cwd()

# Load text_norm by file path (dependency-free)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_text_norm", REPO / "src" / "birchbark_ocr" / "data" / "text_norm.py")
_tn = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_tn)
normalize_text = _tn.normalize_text
is_meaningful_text = _tn.is_meaningful_text

# Lightweight CER/NLS via jiwer + Levenshtein (already in this venv)
import jiwer
from Levenshtein import distance as lev

def cer_fn(pred: str, gold: str) -> float:
    if not gold:
        return 0.0 if not pred else 1.0
    return jiwer.cer(gold, pred)

def nls_fn(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    L = max(len(pred), len(gold))
    if L == 0:
        return 1.0
    return 1.0 - lev(pred, gold) / L

def strip_brackets(s: str) -> str:
    return re.sub(r"[\[\]]", "", s)


def churro_xml_to_plain(s: str) -> str:
    """Smarter than scripts/run_phase2_baselines.py::_churro_output_to_plaintext.

    CHURRO-3B emits a full <HistoricalDocument><Metadata>…</Metadata><Body>…
    structure where text lines are nested:
        <Line>foo<Word>bar</Word></Line>
    The original phase-2 regex ``<Line>([^<]*)</Line>`` matches only Lines
    *without* inner tags; on real CHURRO output that almost always fails, and
    the function falls back to stripping all tags — which then includes
    metadata strings ("Hebrew", "rtl", "Parchment manuscript, …") that
    massively inflate CER.

    This version:
      1. Tries `<Line>(.+?)</Line>` (DOTALL, non-greedy) and inner-strips tags.
      2. Falls back to `<TextLine>`, `<TextRegion>`.
      3. Falls back to extracting only the body (after </Metadata>) and
         stripping tags — never returns the metadata block.
      4. If nothing structural is found, returns the input unchanged.
    """
    if not s or "<HistoricalDocument" not in s:
        return s
    for tag in ("Line", "TextLine", "TextRegion"):
        pat = re.compile(rf"<{tag}\b[^>]*>(.+?)</{tag}>", re.DOTALL | re.IGNORECASE)
        bodies = pat.findall(s)
        if bodies:
            return " ".join(re.sub(r"<[^>]+>", "", b).strip() for b in bodies if b.strip())
    # Body-only fallback: drop everything before </Metadata>, strip tags from the rest.
    after_meta = re.split(r"</Metadata>", s, maxsplit=1, flags=re.IGNORECASE)
    body = after_meta[1] if len(after_meta) > 1 else s
    return re.sub(r"<[^>]+>", " ", body).strip()


def main():
    out_pred = Path(os.environ.get("OUT_PRED", "reports/eval/test_predictions_churro.jsonl"))
    out_summary = Path(os.environ.get("OUT_SUMMARY", "reports/eval/test_summary_churro.json"))
    jsonl_path = Path(os.environ.get("TEST_JSONL", "data/interim/birchbark_test.jsonl"))
    image_root = Path(os.environ.get("IMAGE_ROOT", str(REPO))).resolve()
    max_rows = int(os.environ.get("MAX_ROWS", "0"))
    model_id = os.environ.get("CHURRO_MODEL", "stanford-oval/churro-3B")

    print(f"[+] loading CHURRO ({model_id}) ...", flush=True)
    from churro_ocr.ocr import OCRClient
    from churro_ocr.providers.builder import build_ocr_backend
    from churro_ocr.providers.specs import HuggingFaceOptions, OCRBackendSpec

    # Match scripts/run_phase2_baselines.py::predict_churro_cli for fair comparison
    # with the existing Phase 2 numbers — same gen kwargs, same model loader.
    backend = build_ocr_backend(OCRBackendSpec(
        provider="hf",
        model=model_id,
        options=HuggingFaceOptions(
            model_kwargs={"device_map": "auto", "torch_dtype": "auto"},
            generation_kwargs={"max_new_tokens": 128, "do_sample": False},
        ),
    ))
    client = OCRClient(backend)

    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    if max_rows:
        rows = rows[:max_rows]

    out_pred.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_pred, "w", encoding="utf-8")

    cers_raw, cers_strip, nlss = [], [], []
    n_total = 0; n_skip_img = 0; n_skip_txt = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        gold_raw = row.get("text", "")
        gold = normalize_text(gold_raw)
        if not is_meaningful_text(gold, min_visible_chars=5):
            n_skip_txt += 1; continue

        # resolve image
        ip = None
        for key in ("image_path", "primary_image", "image_paths"):
            v = row.get(key)
            if isinstance(v, list) and v: v = v[0]
            if not v: continue
            cand = Path(v)
            if cand.is_absolute() and cand.exists():
                ip = cand; break
            for prefix in (Path(""), Path("data/raw/gramoty"), Path("data/interim")):
                p2 = (image_root / prefix / v).resolve()
                if p2.exists():
                    ip = p2; break
            if ip: break
        if ip is None:
            n_skip_img += 1; continue

        try:
            page = client.ocr_image(image_path=ip)
            raw_text = (page.text or "").strip()
        except Exception as e:
            print(f"[!] {ip}: {e}", flush=True)
            raw_text = ""

        pred = churro_xml_to_plain(raw_text)
        # The same normalization we apply to gold (NFC + editorial-strip + parens)
        # applied to CHURRO output too — this is the "fair-postproc" comparison.
        pred_norm = normalize_text(pred)

        c_raw = cer_fn(pred_norm, gold)
        c_strip = cer_fn(strip_brackets(pred_norm), strip_brackets(gold))
        n = nls_fn(pred_norm, gold)
        cers_raw.append(c_raw); cers_strip.append(c_strip); nlss.append(n)
        n_total += 1

        rec = {"row_index": i, "doc_id": row.get("doc_id"),
               "image_path": str(ip),
               "gold_norm": gold,
               "churro_raw_xml": raw_text,
               "churro_xml_to_plain": pred,
               "churro_post_normalized": pred_norm,
               "cer_raw": c_raw, "cer_brackets_stripped": c_strip, "nls": n}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if (i+1) % 5 == 0 or i == len(rows)-1:
            mc = sum(cers_raw)/max(1,len(cers_raw))
            mn = sum(nlss)/max(1,len(nlss))
            print(f"[{i+1:4d}/{len(rows)}] scored={n_total}  meanCER={mc:.4f}  meanNLS={mn:.4f}  el={time.time()-t0:.0f}s", flush=True)

    f.close()
    summary = {
        "model": model_id,
        "jsonl": str(jsonl_path),
        "n_input": len(rows),
        "n_scored": n_total,
        "n_skip_no_img": n_skip_img,
        "n_skip_meaningless": n_skip_txt,
        "mean_cer_raw_with_normalization": sum(cers_raw)/max(1,len(cers_raw)),
        "mean_cer_brackets_stripped": sum(cers_strip)/max(1,len(cers_strip)),
        "mean_nls": sum(nlss)/max(1,len(nlss)),
        "elapsed_sec": time.time() - t0,
        "postprocessing": {
            "stage1_xml_strip": "_churro_output_to_plaintext (extract <Line>...</Line>)",
            "stage2_text_norm": "src/birchbark_ocr/data/text_norm.py::normalize_text",
            "stage3_brackets":  "regex strip [ and ] for the bracket-stripped CER column",
            "rationale": "Identical normalization applied to gold AND prediction",
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
PY
