#!/usr/bin/env python3
"""Download Codex Suprasliensis (obdurodon) folio HTML, facsimile JPEGs, and line manifests.

Respect the host: default delay between requests, optional --limit for smoke tests.
Each folio page uses ``<ol><li>`` for **one diplomatic Slavonic line** per item (``span.os``);
Greek parallels are ignored. That is **logical** line segmentation (gold text), not pixel boxes
on the JPEG.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import orjson
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://suprasliensis.obdurodon.org/"
INDEX_URL = urljoin(BASE, "index.html")
PAGE_RE = re.compile(r"pages/(supr[0-9a-z]+\.html)", re.I)


def discover_folio_pages(client: httpx.Client) -> list[str]:
    r = client.get(INDEX_URL)
    r.raise_for_status()
    ids = sorted(set(PAGE_RE.findall(r.text)))
    return ids


def parse_lines_from_html(html: str) -> tuple[list[str], str | None]:
    """Return (slavonic_lines, image_src_relative_or_none)."""
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []
    for li in soup.select("ol > li"):
        os_spans = li.select("span.os")
        if not os_spans:
            continue
        parts = [s.get_text(separator="", strip=True) for s in os_spans]
        text = "".join(parts).strip()
        if text:
            lines.append(text)
    img = soup.select_one("img#smallImage")
    rel = img.get("src") if img else None
    return lines, rel


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def cmd_scrape(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    pages_dir = out / "pages"
    images_dir = out / "images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": args.user_agent}
    delay = args.delay_seconds
    manifest: list[dict] = []

    with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
        folios = discover_folio_pages(client)
        if args.limit and args.limit > 0:
            folios = folios[: args.limit]

        for name in tqdm(folios, desc="folios"):
            page_url = urljoin(BASE, f"pages/{name}")
            folio_id = Path(name).stem  # supr001r
            html_path = pages_dir / name

            try:
                if not html_path.exists() or args.refetch:
                    r = client.get(page_url)
                    r.raise_for_status()
                    html_path.write_text(r.text, encoding="utf-8")
                    time.sleep(delay)

                html = html_path.read_text(encoding="utf-8")
                lines, img_rel = parse_lines_from_html(html)
                if not img_rel:
                    manifest.append(
                        {
                            "folio_id": folio_id,
                            "page_url": page_url,
                            "error": "no_facsimile_img",
                            "n_lines": len(lines),
                            "lines": lines,
                        }
                    )
                    continue

                img_url = urljoin(page_url, img_rel)
                img_name = Path(img_rel).name
                img_path = images_dir / img_name

                if not img_path.exists() or args.refetch:
                    ir = client.get(img_url)
                    ir.raise_for_status()
                    img_path.write_bytes(ir.content)
                    time.sleep(delay)

                manifest.append(
                    {
                        "folio_id": folio_id,
                        "page_url": page_url,
                        "image_url": img_url,
                        "image_path": str(img_path.relative_to(out)),
                        "html_path": str(html_path.relative_to(out)),
                        "n_lines": len(lines),
                        "lines": lines,
                    }
                )
            except Exception as e:  # noqa: BLE001 — record per-folio failures
                manifest.append(
                    {
                        "folio_id": folio_id,
                        "page_url": page_url,
                        "error": str(e),
                        "n_lines": 0,
                        "lines": [],
                    }
                )
                time.sleep(delay)

    man_path = out / "manifest.jsonl"
    write_jsonl(man_path, manifest)
    ok = sum(1 for m in manifest if "error" not in m)
    print(f"Wrote {man_path} ({ok}/{len(manifest)} folios with image+lines)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/suprasliensis"),
        help="Root output directory (pages/, images/, manifest.jsonl)",
    )
    p.add_argument("--delay-seconds", type=float, default=0.75, help="Pause between HTTP requests")
    p.add_argument("--limit", type=int, default=0, help="If >0, scrape only first N folios (debug)")
    p.add_argument("--refetch", action="store_true", help="Overwrite existing HTML/images")
    p.add_argument(
        "--user-agent",
        default="birchbark-ocr-research/0.1 (contact: local; polite crawl)",
        help="HTTP User-Agent",
    )
    args = p.parse_args()
    return cmd_scrape(args)


if __name__ == "__main__":
    raise SystemExit(main())
