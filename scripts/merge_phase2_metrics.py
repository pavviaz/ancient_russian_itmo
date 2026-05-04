#!/usr/bin/env python3
"""Merge core Phase 2 metrics + CHURRO into one JSON and one NLS bar chart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Preferred display order for the combined figure / table
MODEL_ORDER = [
    "tesseract",
    "easyocr",
    "trocr",
    "qwen35_08b",
    "qwen35_2b",
    "churro_cli",
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    core_path = root / "runs/phase2/metrics.json"
    churro_path = root / "runs/phase2/metrics_churro.json"
    out_json = root / "runs/phase2/metrics_phase2_all.json"
    out_fig = root / "reports/figs/baseline_phase2_all.png"

    if not core_path.is_file():
        print(f"Missing {core_path}", file=sys.stderr)
        sys.exit(2)
    if not churro_path.is_file():
        print(f"Missing {churro_path} — run: bash scripts/run_phase2_churro_paddle.sh", file=sys.stderr)
        sys.exit(2)

    core = json.loads(core_path.read_text(encoding="utf-8"))
    churro = json.loads(churro_path.read_text(encoding="utf-8"))
    if "churro_cli" not in churro:
        print(f"{churro_path} has no churro_cli key", file=sys.stderr)
        sys.exit(2)

    merged: dict[str, dict] = {}
    for name in MODEL_ORDER:
        if name in core:
            merged[name] = core[name]
        elif name in churro:
            merged[name] = churro[name]
    for k, v in core.items():
        merged.setdefault(k, v)
    for k, v in churro.items():
        merged.setdefault(k, v)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skip figure (no matplotlib): {e}", file=sys.stderr)
        return

    names = [m for m in MODEL_ORDER if m in merged]
    nls_vals = [merged[m]["nls_mean"] for m in names]
    plt.figure(figsize=(max(6, len(names) * 1.2), 4))
    plt.bar(names, nls_vals, color="steelblue")
    plt.ylabel("Mean NLS (↑ better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
