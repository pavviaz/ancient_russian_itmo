"""Stratified document splits for birchbark (by century × site bucket)."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass


def century_from_conditional_date(date_raw: str) -> str:
    """Map '←1100‒1120→' or '1100‒1120' to Roman century label using median year."""
    nums = [int(x) for x in re.findall(r"\d{3,4}", date_raw)]
    if not nums:
        return "unknown"
    med = sum(nums) / len(nums)
    if med <= 1100:
        return "XI"
    if med <= 1200:
        return "XII"
    if med <= 1300:
        return "XIII"
    if med <= 1400:
        return "XIV"
    return "XV"


def site_bucket(city: str) -> str:
    c = city.strip().lower()
    mapping = [
        ("новгород", "Novgorod"),
        ("русса", "Staraya_Russa"),
        ("смоленск", "Smolensk"),
        ("псков", "Pskov"),
        ("торжок", "Torzhok"),
        ("тверь", "Tver"),
        ("рязань", "Staraya_Ryazan"),
        ("витебск", "Vitebsk"),
        ("вологда", "Vologda"),
        ("москва", "Moscow"),
        ("мстиславль", "Mstislavl"),
        ("переяславль", "Pereslavl"),
    ]
    for needle, bucket in mapping:
        if needle in c:
            return bucket
    return "other"


@dataclass
class SplitSpec:
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    def __post_init__(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(s, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"ratios must sum to 1, got {s}")


def _split_counts(n: int, spec: SplitSpec) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    nt = int(round(n * spec.train_ratio))
    nv = int(round(n * spec.val_ratio))
    ns = n - nt - nv
    if ns < 0:
        nt += ns
        ns = 0
    if nt < 1:
        nt = 1
        nv = min(nv, max(0, n - nt - ns))
        ns = n - nt - nv
    if nt + nv + ns > n:
        nt = n - nv - ns
    return nt, nv, ns


def stratified_split_doc_ids(
    rows: list[dict[str, str]],
    seed: int = 1337,
    spec: SplitSpec | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    rows: each dict needs keys doc_id, date_raw (from list), city (Город from doc metadata preferred).
    """
    spec = spec or SplitSpec()
    rng = random.Random(seed)
    strata: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        did = r["doc_id"]
        date_raw = r.get("date_raw") or ""
        city = r.get("city") or ""
        cent = century_from_conditional_date(date_raw)
        site = site_bucket(city)
        strata[f"{cent}|{site}"].append(did)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for _, ids in sorted(strata.items()):
        ids_u = sorted(set(ids))
        rng.shuffle(ids_u)
        n = len(ids_u)
        nt, nv, ns = _split_counts(n, spec)
        # Tiny strata: ensure disjoint sizes match n
        while nt + nv + ns > n:
            if nv > 0:
                nv -= 1
            elif ns > 0:
                ns -= 1
            else:
                nt -= 1
        while nt + nv + ns < n:
            nt += 1
        train.extend(ids_u[:nt])
        val.extend(ids_u[nt : nt + nv])
        test.extend(ids_u[nt + nv : nt + nv + ns])

    all_ids = {r["doc_id"] for r in rows}
    assigned = set(train + val + test)
    leftover = sorted(all_ids - assigned)
    rng.shuffle(leftover)
    train.extend(leftover)

    return train, val, test
