#!/usr/bin/env python3
"""Placeholder leakage check: extend once train_birch JSONL + test lines exist."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/splits/leakage_audit.md"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        """# Leakage audit (Phase 1 stub)

**Status:** Pending full line-level JSONL for `train_birch` and `test_birch`.

Per protocol §2.4, for each test image confirm:

1. Filename not present in any train shard.
2. SHA256 of gold text not present in train.

## Commands (when data exists)

```bash
python scripts/leakage_audit.py --train data/interim/birchbark_train.jsonl --test data/interim/birchbark_test.jsonl
```

## Result

- Run pending.

""",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
