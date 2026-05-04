#!/usr/bin/env python3
"""Build frozen train/val/test document ID lists + SHA256 sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import orjson

from birchbark_ocr.data.birchbark_splits import stratified_split_doc_ids


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in manifest_path.read_bytes().splitlines():
        if not line.strip():
            continue
        m = orjson.loads(line)
        rows.append(
            {
                "doc_id": m["doc_id"],
                "date_raw": m.get("date_raw") or m.get("list_date_raw") or "",
                "city": m.get("city") or m.get("list_city") or "",
            }
        )
    return rows


def load_index_rows(index_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in index_path.read_bytes().splitlines():
        if not line.strip():
            continue
        r = orjson.loads(line)
        rows.append(
            {
                "doc_id": r["doc_id"],
                "date_raw": r.get("date_raw") or "",
                "city": r.get("city") or "",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.jsonl after scrape")
    ap.add_argument("--index", type=Path, default=None, help="document_index.jsonl (list only)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.manifest:
        rows = load_manifest_rows(args.manifest)
    elif args.index:
        rows = load_index_rows(args.index)
    else:
        raise SystemExit("Provide --manifest or --index")

    train, val, test = stratified_split_doc_ids(rows, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_ids(name: str, ids: list[str]) -> None:
        p = args.out_dir / name
        body = "\n".join(sorted(ids)) + "\n"
        p.write_text(body, encoding="utf-8")
        hpath = p.with_suffix(p.suffix + ".sha256")
        hx = hashlib.sha256(body.encode("utf-8")).hexdigest()
        hpath.write_text(f"{hx}  {p.name}\n", encoding="utf-8")

    write_ids("birchbark_train_ids.txt", train)
    write_ids("birchbark_val_ids.txt", val)
    write_ids("birchbark_test_ids.txt", test)

    summary = {
        "seed": args.seed,
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "source": str(args.manifest or args.index),
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
