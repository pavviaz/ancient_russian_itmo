#!/usr/bin/env python3
"""Generate sample wood-incised renders from Ostromir diplomatic plaintext."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import random
import sys
from pathlib import Path

import httpx

from birchbark_ocr.synth.ostromir_render import iter_ostromir_content_lines, render_ostromir_sample

OSTROMIR_URL = "http://www.ponomar.net/files/ostromir.txt"


def load_text(url: str, cache: Path | None) -> str:
    if cache and cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    with httpx.Client(follow_redirects=True, timeout=120.0) as c:
        r = c.get(url)
        r.raise_for_status()
        t = r.text
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(t, encoding="utf-8")
    return t


def render_one(job: tuple[int, int, list[str], str, int, int, int, int, int]) -> str:
    """Render one sample and write PNG + GT. Kept top-level for multiprocessing."""
    i, seed, chunk, output_dir, width, margin, font_size, line_gap, n_digits = job
    im, gold = render_ostromir_sample(
        chunk,
        width=width,
        margin=margin,
        font_size=font_size,
        line_gap=line_gap,
        rng=random.Random(seed),
    )
    out = Path(output_dir)
    stem = f"ostromir_sample_{i:0{n_digits}d}"
    png = out / f"{stem}.png"
    txt = out / f"{stem}.gold.txt"
    im.save(png)
    txt.write_text(gold + "\n", encoding="utf-8")
    return str(png)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/interim/ostromir_render_samples"))
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--lines-per-image", type=int, default=3, help="Consecutive edition lines stacked per PNG")
    p.add_argument("--all-chunks", action="store_true", help="Render non-overlapping chunks across the whole cleaned Ostromir text")
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--font-size", type=int, default=38)
    p.add_argument("--margin", type=int, default=56)
    p.add_argument("--line-gap", type=int, default=14)
    p.add_argument("--workers", type=int, default=1, help="Parallel render workers")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--cache", type=Path, default=Path("data/raw/ostromir/ostromir.txt"))
    p.add_argument("--url", default=OSTROMIR_URL)
    args = p.parse_args()

    try:
        raw = load_text(args.url, args.cache)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to fetch Ostromir text: {e}", file=sys.stderr)
        return 1

    pool = iter_ostromir_content_lines(raw)
    if len(pool) < args.lines_per_image + 5:
        print("Too few usable lines after cleaning.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.all_chunks:
        starts = list(range(0, len(pool) - args.lines_per_image + 1, args.lines_per_image))
    else:
        starts = [rng.randint(0, len(pool) - args.lines_per_image) for _ in range(args.n_samples)]

    n_digits = max(2, len(str(len(starts) - 1)))
    jobs = [
        (
            i,
            rng.randint(0, 2**30),
            pool[start : start + args.lines_per_image],
            str(out),
            args.width,
            args.margin,
            args.font_size,
            args.line_gap,
            n_digits,
        )
        for i, start in enumerate(starts)
    ]

    if args.workers <= 1:
        for job in jobs:
            print(render_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for png in ex.map(render_one, jobs, chunksize=max(1, len(jobs) // (args.workers * 8))):
                print(png)

    meta = out / "README.txt"
    meta.write_text(
        "Each PNG is a synthetic wood/birchbark incision render; matching .gold.txt is exact Unicode used.\n"
        "Source text: ponomar.net Ostromir file; edition tags stripped for rendering.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
