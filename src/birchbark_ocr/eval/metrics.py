"""Character-level OCR metrics (CER, NLS, exact) and optional confusion scaffolding."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

import jiwer

# Historically interesting Old Cyrillic codepoints (protocol §3.3)
DEFAULT_HISTORIC_CHARS = tuple("ѣѧѫѳѵѡѥѩѭѯѱ҂")


def normalize(text: str) -> str:
    """NFC unicode, strip ends, collapse repeated whitespace (protocol §3.3)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text).strip()
    return " ".join(s.split())


def strip_square_brackets(text: str) -> str:
    """Remove [...] segments for alternate evaluation (protocol §2.2 / §3.3)."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return normalize("".join(out))


def levenshtein(a: str, b: str) -> int:
    """Classic O(nm) Levenshtein distance (no extra deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[len(b)]


def nls(pred: str, gold: str) -> float:
    """
    Normalized Levenshtein Similarity (CHURRO-style): 1 − Lev / max(len(pred), len(gold)).
    Empty vs empty → 1.0; empty vs non-empty → 0.0.
    """
    p, g = normalize(pred), normalize(gold)
    denom = max(len(p), len(g))
    if denom == 0:
        return 1.0
    return 1.0 - levenshtein(p, g) / denom


def cer(pred: str, gold: str) -> float:
    """Character error rate via jiwer (Levenshtein / len(reference))."""
    p, g = normalize(pred), normalize(gold)
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return float(jiwer.cer(g, p))


def exact(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def _nw_align_chars(a: str, b: str, match: int = 1, mismatch: int = -1, gap: int = -1) -> list[tuple[str | None, str | None]]:
    """Global alignment (Needleman–Wunsch) returning aligned character pairs."""
    na, nb = len(a), len(b)
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    ptr = [[0] * (nb + 1) for _ in range(na + 1)]  # 0 diag, 1 up, 2 left
    for i in range(1, na + 1):
        dp[i][0] = dp[i - 1][0] + gap
        ptr[i][0] = 1
    for j in range(1, nb + 1):
        dp[0][j] = dp[0][j - 1] + gap
        ptr[0][j] = 2
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sc_diag = dp[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            sc_up = dp[i - 1][j] + gap
            sc_left = dp[i][j - 1] + gap
            best = max(sc_diag, sc_up, sc_left)
            dp[i][j] = best
            if best == sc_diag:
                ptr[i][j] = 0
            elif best == sc_up:
                ptr[i][j] = 1
            else:
                ptr[i][j] = 2
    pairs: list[tuple[str | None, str | None]] = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i][j] == 0:
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ptr[i][j] == 1):
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def per_char_confusion(
    pred: str,
    gold: str,
    *,
    top_k_frequent: int = 30,
    extra_chars: Iterable[str] = DEFAULT_HISTORIC_CHARS,
) -> dict[str, Any]:
    """
    Confusion counts from NW alignment of normalized strings.
    Returns raw pair counts plus a filtered square subset for frequent + historic chars.
    """
    p, g = normalize(pred), normalize(gold)
    pairs = _nw_align_chars(g, p)  # gold → rows, pred → cols convention
    full_counts: Counter[tuple[str, str]] = Counter()
    for gc, pc in pairs:
        gg = gc if gc is not None else "<gap>"
        pp = pc if pc is not None else "<gap>"
        full_counts[(gg, pp)] += 1

    freq = Counter(g)
    for ch in list(freq.keys()):
        if ch == " ":
            del freq[ch]
    top_chars = {c for c, _ in freq.most_common(top_k_frequent)}
    top_chars.update(extra_chars)
    filtered: Counter[tuple[str, str]] = Counter()
    for (gg, pp), v in full_counts.items():
        if gg in top_chars or pp in top_chars:
            filtered[(gg, pp)] += v

    return {
        "full_counts": full_counts,
        "filtered_counts": filtered,
        "top_chars": sorted(top_chars),
    }


def aggregate_confusion(per_doc_results: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    """Merge filtered confusion dicts from per_char_confusion outputs."""
    out: Counter[tuple[str, str]] = Counter()
    for r in per_doc_results:
        fc = r.get("filtered_counts")
        if isinstance(fc, Counter):
            out.update(fc)
    return out


def macro_avg(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def safe_mean_cer_nls(preds: list[str], golds: list[str]) -> tuple[float, float]:
    cers = [cer(p, g) for p, g in zip(preds, golds, strict=True)]
    nlss = [nls(p, g) for p, g in zip(preds, golds, strict=True)]
    return macro_avg(cers), macro_avg(nlss)
