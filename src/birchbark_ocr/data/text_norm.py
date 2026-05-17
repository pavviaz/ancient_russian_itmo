"""Text normalisation for birchbark OCR training & evaluation.

Reconstructed from the v3 / v5 trainer behaviour after the original module was
lost in a branch switch. Validated to match data/splits/phase4_v3/val.jsonl
(text_raw -> text) with ~99% exact agreement (116/117 rows).

Rules:
  1. NFC unicode normalise.
  2. Strip a small allow-list of *editorial* Russian phrases that gramoty.ru
     editors prepend to their transcriptions ("оборот:", "Второй фрагмент:",
     "Строка N:", "Фрагмент N:", "Для остатков следующей строки..."). These
     are scholar-supplied commentary, not characters that the OCR model
     should reproduce.
  3. Strip ellipsis markers (… and ASCII ...).
  4. Strip parenthesised editorial expansions (...) — content between parens
     is what the editor SUPPLIED for missing letters, not what is visible.
     The square brackets [...] are kept (those are visible-but-ambiguous chars).
  5. Collapse repeated spaces; trim leading/trailing whitespace.
"""

from __future__ import annotations

import re
import unicodedata

_EDIT_PHRASES = [
    r"\bоборот\b\s*:?",
    r"\bпервый\s+фрагмент\b\s*:?",
    r"\bвторой\s+фрагмент\b\s*:?",
    r"\bтретий\s+фрагмент\b\s*:?",
    r"\bчетвёртый\s+фрагмент\b\s*:?",
    r"\bфрагмент\s*\d+\s*:?",
    r"\bстрока\s*\d+\s*:?",
    r"\bдля\s+остатков\s+следующей\s+строки[^…\n]*",
]
_EDITORIAL = re.compile("|".join(_EDIT_PHRASES), re.IGNORECASE)
_PARENS = re.compile(r"\([^()]*\)")
_ELLIPSIS = re.compile(r"…+|\.{3,}")
_LEAD = re.compile(r"^\s+")
_TRAIL = re.compile(r"\s+$")
_WS = re.compile(r" {2,}")


def normalize_text(s: str) -> str:
    """Return the diplomatic, editorial-stripped form used as OCR target."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _EDITORIAL.sub("", s)
    s = _ELLIPSIS.sub(" ", s)
    s = _PARENS.sub("", s)
    s = _WS.sub(" ", s)
    s = _LEAD.sub("", s)
    s = _TRAIL.sub("", s)
    s = _WS.sub(" ", s)
    return s


_VISIBLE_RE = re.compile(r"[^\s\-·]")


def is_meaningful_text(s: str, min_visible_chars: int = 5) -> bool:
    """A row is OCR-useful only if it has at least N visible (non-whitespace,
    non-dash, non-middle-dot) characters."""
    if not s:
        return False
    return len(_VISIBLE_RE.findall(s)) >= min_visible_chars
