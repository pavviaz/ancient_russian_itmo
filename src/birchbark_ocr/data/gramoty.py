"""gramoty.ru list/document fetch and HTML parsing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://gramoty.ru"
LIST_URL = f"{BASE}/birchbark/document/list/"


@dataclass
class ListEntry:
    url: str
    doc_id: str  # e.g. novgorod/109 or novgorod/98/100
    title: str
    date_raw: str
    city: str
    summary: str


@dataclass
class DocumentRecord:
    doc_id: str
    url: str
    transcription_diplomatic: str
    transcription_spaced: str | None
    metadata: dict[str, str]
    image_paths_relative: list[str] = field(default_factory=list)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def parse_list_page(html: str) -> list[ListEntry]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[ListEntry] = []
    for table in soup.find_all("table"):
        thead = table.find("thead")
        if thead is None:
            continue
        header_spans = thead.select("th span")
        if not header_spans or header_spans[0].get_text(strip=True) != "Номер":
            continue
        tbody = table.find("tbody")
        if tbody is None:
            continue
        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 4:
                continue
            a = cells[0].find("a", href=True)
            if not a:
                continue
            href = a["href"]
            url = href if href.startswith("http") else urljoin(BASE, href)
            m = re.search(r"/birchbark/document/show/([^/]+/.+)/?", url)
            doc_id = m.group(1).rstrip("/") if m else url
            rows.append(
                ListEntry(
                    url=url,
                    doc_id=doc_id,
                    title=a.get_text(strip=True),
                    date_raw=cells[1].get_text(strip=True),
                    city=cells[2].get_text(strip=True),
                    summary=cells[3].get_text(strip=True),
                )
            )
    # De-dupe by URL preserving order
    seen: set[str] = set()
    out: list[ListEntry] = []
    for r in rows:
        if r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


def parse_document_page(html: str, url: str, doc_id: str) -> DocumentRecord:
    soup = BeautifulSoup(html, "lxml")
    areas = soup.select("div.text-area")
    diplomatic = ""
    spaced: str | None = None
    originals = [d for d in areas if "original-text" in d.get("class", [])]
    others = [d for d in areas if "original-text" not in d.get("class", [])]
    if originals:
        diplomatic = _norm_ws(originals[0].get_text("\n"))
    if len(areas) >= 2:
        spaced = _norm_ws(areas[1].get_text("\n"))
    elif others and not originals:
        diplomatic = _norm_ws(others[0].get_text("\n"))

    meta: dict[str, str] = {}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 2:
                continue
            k = tds[0].get_text(" ", strip=True)
            v = tds[1].get_text(" ", strip=True)
            if k and v:
                meta[k] = v

    imgs: list[str] = []
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "/thumbs/photo_" in src or "/thumbs/drawing_" in src:
            imgs.append(src)

    return DocumentRecord(
        doc_id=doc_id,
        url=url,
        transcription_diplomatic=diplomatic,
        transcription_spaced=spaced,
        metadata=meta,
        image_paths_relative=sorted(set(imgs)),
    )


class GramotyClient:
    def __init__(
        self,
        base: str = BASE,
        timeout: float = 60.0,
        delay_seconds: float = 2.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self.delay_seconds = delay_seconds
        self._client = httpx.Client(
            timeout=timeout,
            headers=headers
            or {
                "User-Agent": "birchbark-ocr-research/0.1 (+https://github.com/local; contact: academic)",
                "Accept-Language": "ru,en;q=0.9",
            },
            follow_redirects=True,
        )
        self._last_fetch = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GramotyClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.delay_seconds - (now - self._last_fetch)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch = time.monotonic()

    def get_text(self, url: str) -> str:
        self._throttle()
        r = self._client.get(url)
        r.raise_for_status()
        return r.text

    def download_bytes(self, path: str) -> bytes:
        url = path if path.startswith("http") else urljoin(self.base + "/", path.lstrip("/"))
        self._throttle()
        r = self._client.get(url)
        r.raise_for_status()
        return r.content
