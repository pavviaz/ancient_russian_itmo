#!/usr/bin/env python3
"""Download gramoty.ru birchbark list + per-document HTML, images, and metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import orjson
from tqdm import tqdm

from birchbark_ocr.data.gramoty import (
    LIST_URL,
    GramotyClient,
    parse_document_page,
    parse_list_page,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def cmd_index(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_html = out / "list.html"
    client = GramotyClient(delay_seconds=args.delay_seconds)
    try:
        html = client.get_text(LIST_URL)
        raw_html.write_text(html, encoding="utf-8")
        entries = parse_list_page(html)
        rows = [
            {
                "doc_id": e.doc_id,
                "url": e.url,
                "title": e.title,
                "date_raw": e.date_raw,
                "city": e.city,
                "summary": e.summary,
            }
            for e in entries
        ]
        write_jsonl(out / "document_index.jsonl", rows)
        print(f"Wrote {len(rows)} entries to {out / 'document_index.jsonl'}")
    finally:
        client.close()
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    index_path = out / "document_index.jsonl"
    if not index_path.exists():
        print("Run `index` first or provide existing document_index.jsonl", file=sys.stderr)
        return 1
    rows = [
        orjson.loads(line)
        for line in index_path.read_bytes().splitlines()
        if line.strip()
    ]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = GramotyClient(delay_seconds=args.delay_seconds)
    docs_dir = out / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    try:
        for row in tqdm(rows, desc="documents"):
            did = row["doc_id"]
            url = row["url"]
            safe = did.replace("/", "__")
            ddir = docs_dir / safe
            ddir.mkdir(parents=True, exist_ok=True)
            html_path = ddir / "page.html"
            if not html_path.exists() or args.refetch:
                html = client.get_text(url)
                html_path.write_text(html, encoding="utf-8")
            else:
                html = html_path.read_text(encoding="utf-8")
            rec = parse_document_page(html, url, did)
            meta = {
                "doc_id": rec.doc_id,
                "url": rec.url,
                "transcription_diplomatic": rec.transcription_diplomatic,
                "transcription_spaced": rec.transcription_spaced,
                "metadata": rec.metadata,
                "image_paths_relative": rec.image_paths_relative,
                "list_date_raw": row.get("date_raw"),
                "list_city": row.get("city"),
            }
            # Prefer document table for city/date if present
            if "Город" in rec.metadata:
                meta["city"] = rec.metadata["Город"]
            else:
                meta["city"] = row.get("city") or ""
            if "Условная дата" in rec.metadata:
                meta["date_raw"] = rec.metadata["Условная дата"]
            else:
                meta["date_raw"] = row.get("date_raw") or ""

            img_dir = ddir / "images"
            img_dir.mkdir(exist_ok=True)
            saved_images: list[str] = []
            for rel in rec.image_paths_relative:
                name = rel.split("/")[-1]
                dest = img_dir / name
                if not dest.exists() or args.refetch:
                    data = client.download_bytes(rel)
                    dest.write_bytes(data)
                saved_images.append(str(dest.relative_to(out)))
            meta["images"] = saved_images
            (ddir / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
            manifest.append(meta)
    finally:
        client.close()

    write_jsonl(out / "manifest.jsonl", manifest)
    print(f"Manifest: {out / 'manifest.jsonl'} ({len(manifest)} docs)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="gramoty.ru birchbark scraper")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="Fetch and parse document list")
    pi.add_argument("--output-dir", type=str, default="data/raw/gramoty")
    pi.add_argument("--delay-seconds", type=float, default=2.0)
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("scrape", help="Fetch each document + images")
    ps.add_argument("--output-dir", type=str, default="data/raw/gramoty")
    ps.add_argument("--delay-seconds", type=float, default=2.0)
    ps.add_argument("--limit", type=int, default=0, help="0 = all")
    ps.add_argument("--refetch", action="store_true")
    ps.set_defaults(func=cmd_scrape)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
