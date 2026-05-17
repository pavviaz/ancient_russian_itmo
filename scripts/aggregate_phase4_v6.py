#!/usr/bin/env python3
"""Aggregate phase4_v6 multi-seed + LoRA-rank results.

Walks each ``runs/phase4_v6/*/`` directory:
  1. Reads ``checkpoint-<best>/trainer_state.json`` to find the best in-loop
     ``eval_gen_cer`` and the corresponding adapter path.
  2. Runs ``scripts/eval_qwen_vl_lora.py`` on val_birch + test_birch to get the
     full-117 / full-252 CER/NLS numbers.
  3. Writes ``reports/eval/v6_aggregate.json`` with mean ± std, and a markdown
     table for direct paste-in to FINDINGS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def best_checkpoint(run_dir: Path) -> tuple[Path, float] | None:
    """Find the checkpoint dir with the lowest eval_gen_cer in trainer_state."""
    ckpts = sorted(run_dir.glob("checkpoint-*"))
    best_path = None
    best_cer = float("inf")
    for ck in ckpts:
        ts_file = ck / "trainer_state.json"
        if not ts_file.exists():
            continue
        ts = json.loads(ts_file.read_text())
        # The "best" entry of the state at this checkpoint
        for entry in ts.get("log_history", []):
            cer = entry.get("eval_gen_cer")
            if cer is None:
                continue
            if cer < best_cer:
                best_cer = cer
                # Map back to the checkpoint path that step corresponds to
                best_path = ck if entry.get("step") == ts.get("global_step") else best_path
        # As a fallback, use trainer's own best_model_checkpoint
        if best_path is None and ts.get("best_model_checkpoint"):
            best_path = Path(ts["best_model_checkpoint"])
            best_cer = float(ts.get("best_metric", best_cer))
    if best_path is None or not best_path.exists():
        # Walk again: best_model_checkpoint pointer might be in any trainer_state
        for ck in ckpts:
            ts = json.loads((ck / "trainer_state.json").read_text())
            cand = ts.get("best_model_checkpoint")
            if cand and Path(cand).exists():
                return Path(cand), float(ts.get("best_metric", float("nan")))
        return None
    return best_path, best_cer


def run_eval(adapter: Path, jsonl: Path, out_pred: Path, out_summary: Path,
             *, num_beams: int = 1, max_new_tokens: int = 160,
             gpu: str = "0", venv: str = ".venv-qwen-edit-multi") -> dict:
    """Invoke scripts/eval_qwen_vl_lora.py and parse the summary back."""
    cmd = [
        f"{venv}/bin/python", "scripts/eval_qwen_vl_lora.py",
        "--base-model", "Qwen/Qwen3.5-2B",
        "--adapter", str(adapter),
        "--jsonl", str(jsonl),
        "--image-root", ".",
        "--out-pred", str(out_pred),
        "--out-summary", str(out_summary),
        "--device", "cuda:0",
        "--max-new-tokens", str(max_new_tokens),
        "--num-beams", str(num_beams),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    subprocess.run(cmd, check=True, env=env)
    return json.loads(out_summary.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/phase4_v6")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--num-beams", type=int, default=4,
                    help="A9 confirmed beam=4 wins on val.")
    ap.add_argument("--out", default="reports/eval/v6_aggregate.json")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    val_jsonl = Path("data/splits/phase4_v3/val.jsonl")
    test_jsonl = Path("data/interim/birchbark_test.jsonl")
    Path("reports/eval").mkdir(parents=True, exist_ok=True)

    cells = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        bc = best_checkpoint(run)
        if bc is None:
            print(f"[skip] no checkpoint in {run}")
            continue
        adapter, in_loop_cer = bc
        print(f"[{run.name}]  best ckpt = {adapter.name}  in-loop CER = {in_loop_cer:.4f}")

        out_val = Path(f"reports/eval/v6_{run.name}_val.jsonl")
        sum_val = Path(f"reports/eval/v6_{run.name}_val.json")
        val = run_eval(adapter, val_jsonl, out_val, sum_val,
                       num_beams=args.num_beams, gpu=args.gpu)

        test = None
        if not args.skip_test:
            out_test = Path(f"reports/eval/v6_{run.name}_test.jsonl")
            sum_test = Path(f"reports/eval/v6_{run.name}_test.json")
            test = run_eval(adapter, test_jsonl, out_test, sum_test,
                            num_beams=args.num_beams, gpu=args.gpu)

        cells.append({
            "run_name": run.name,
            "best_checkpoint": str(adapter),
            "in_loop_cer": in_loop_cer,
            "val_117": {"cer": val["mean_cer_raw"],
                        "cer_strip": val["mean_cer_brackets_stripped"],
                        "nls": val["mean_nls"], "n": val["n_rows_scored"]},
            "test_252": ({"cer": test["mean_cer_raw"],
                          "cer_strip": test["mean_cer_brackets_stripped"],
                          "nls": test["mean_nls"], "n": test["n_rows_scored"]}
                         if test else None),
        })

    # Aggregate the seed-multiplicity for r=32 cells
    r32 = [c for c in cells if "_r32" in c["run_name"]]
    if r32:
        cers = [c["val_117"]["cer"] for c in r32]
        nlss = [c["val_117"]["nls"] for c in r32]
        agg = {
            "n_seeds": len(r32),
            "val_cer_mean": statistics.mean(cers),
            "val_cer_std": statistics.stdev(cers) if len(cers) > 1 else 0.0,
            "val_nls_mean": statistics.mean(nlss),
            "val_nls_std": statistics.stdev(nlss) if len(nlss) > 1 else 0.0,
        }
    else:
        agg = None

    out = {
        "cells": cells,
        "r32_aggregate": agg,
        "decoding": {"num_beams": args.num_beams, "max_new_tokens": 160},
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print()
    print("=" * 76)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
