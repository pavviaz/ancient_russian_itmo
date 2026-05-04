#!/usr/bin/env python3
"""Rebuild manifest.jsonl from per-document meta.json files (resume after interrupt)."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gramoty-dir", type=Path, default=Path("data/raw/gramoty"))
    args = ap.parse_args()
    docs = args.gramoty_dir / "documents"
    metas: list[Path] = sorted(docs.glob("*/meta.json"))
    rows: list[dict] = []
    for p in metas:
        rows.append(orjson.loads(p.read_bytes()))
    rows.sort(key=lambda r: r.get("doc_id", ""))
    out = args.gramoty_dir / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    print(f"Wrote {out} ({len(rows)} docs) from {len(metas)} meta.json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
