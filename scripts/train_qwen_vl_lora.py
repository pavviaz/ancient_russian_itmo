#!/usr/bin/env python3
"""Fine-tune Qwen3.5-VL on Phase-4 birchbark splits with LoRA.

Reconstructed after the previous source was lost in a branch switch. The
hyperparameters, LoRA target list, and prompt template match the existing
``mixed_80_20__Qwen_Qwen3_5-2B__ceil5ep/checkpoint-900`` champion exactly
(verified from its ``training_args.bin`` and ``adapter_config.json``).

Headline behaviour
------------------
* Backbone: ``Qwen/Qwen3.5-2B`` loaded as ``AutoModelForImageTextToText``,
  bf16, ``attn_implementation='sdpa'`` (FlashAttention-2 falls back).
* PEFT: LoRA r=32, alpha=64, dropout=0.05; targets cover **the vision encoder
  blocks**, **all 4 Mamba projections**, **standard self-attn (q,k,v,o)**, and
  the **MLP feed-forwards** (gate/up/down). This is the v3 fix that broke
  CER >= 1: training only ``q,k,v,o`` left the vision tower frozen and 75%
  of LM attention as Mamba layers untouched.
* Loss: standard CE on assistant tokens only -- prompt is masked to -100.
* Eval: standard cross-entropy + a generation-time CER/NLS callback that
  decodes a fixed, deterministic subset of validation rows every
  ``--eval-steps``. We select on ``eval_gen_cer`` (lower is better).

Usage
-----
::

    .venv-qwen-edit-multi/bin/python scripts/train_qwen_vl_lora.py \\
        --model Qwen/Qwen3.5-2B \\
        --train-jsonl data/splits/phase4_v3/mixed_80_20_train.jsonl \\
        --val-jsonl   data/splits/phase4_v3/val.jsonl \\
        --output-dir  runs/phase4_v6/mixed_80_20_seed2026 \\
        --epochs 5 --early-stopping-patience 5 --seed 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- birchbark_ocr.train.data imports augment_pil_image which only needs PIL/numpy.
# We avoid pulling in birchbark_ocr.data (which imports gramoty -> bs4).
from birchbark_ocr.train.data import (  # noqa: E402
    Phase4OCRDataset,
    build_collator,
    DEFAULT_PROMPT,
)

# eval/metrics is a small pure-python module
from birchbark_ocr.eval.metrics import (  # noqa: E402
    cer as cer_fn,
    nls as nls_fn,
)


# ---- LoRA target modules (matches checkpoint-900/adapter_config.json) ----
DEFAULT_LORA_TARGETS = [
    # ViT (vision encoder blocks)
    "linear_fc1", "linear_fc2",
    # ViT attention pooling
    "qkv", "out_proj",
    # LM standard self-attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # LM Mamba layers (linear-attention path)
    "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z",
    # LM MLP feed-forwards
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)

    # train hyperparams (defaults match v5 champion)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--lr-scheduler", default="cosine")
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--early-stopping-patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Defaults to --seed.")

    # LoRA
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="*", default=None,
                    help="Override LoRA target_modules. Default = v3 broad targets.")

    # gen-eval
    ap.add_argument("--gen-eval-rows", type=int, default=64,
                    help="How many val rows to generate on per eval step.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=160)
    ap.add_argument("--gen-no-repeat-ngram-size", type=int, default=4)

    # image budget
    ap.add_argument("--image-max-pixels", type=int, default=451584)
    ap.add_argument("--image-min-pixels", type=int, default=100352)

    # misc
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing",
                    dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    return ap.parse_args()


# -------- gen-eval callback --------
class GenEvalCallback:
    """Cheap, always-on generation-time CER/NLS evaluation.

    Runs ``model.generate()`` on a deterministic prefix of the val set every
    ``eval_steps`` and emits ``eval_gen_cer`` / ``eval_gen_nls`` into the
    Trainer log. This metric is what we steer model selection on (via
    ``metric_for_best_model='eval_gen_cer'``, ``greater_is_better=False``).
    """

    def __init__(self, processor, val_rows, *, prompt: str,
                 image_max_pixels: int, image_min_pixels: int,
                 max_new_tokens: int, no_repeat_ngram_size: int,
                 max_rows: int, normalize_text):
        self.processor = processor
        self.val_rows = val_rows[:max_rows]
        self.prompt = prompt
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self.max_new_tokens = max_new_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.normalize_text = normalize_text
        # Lazy collator — will be built once for prompt-only encoding
        self._collator = None

    def _ensure_collator(self):
        if self._collator is None:
            self._collator = build_collator(
                self.processor, train=False, prompt=self.prompt,
                image_max_pixels=self.image_max_pixels,
                image_min_pixels=self.image_min_pixels,
            )
        return self._collator

    @torch.no_grad()
    def __call__(self, model, device) -> dict[str, float]:
        col = self._ensure_collator()
        was_training = model.training
        model.eval()

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        cers, nlss, n = [], [], 0
        for i, row in enumerate(self.val_rows):
            try:
                enc = col.encode_prompt_only(row, augment_train=False)
            except Exception as e:  # noqa: BLE001
                print(f"[gen-eval] skip {row.sample_id}: {e}", flush=True)
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            try:
                gen = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[gen-eval] generate-failed {row.sample_id}: {e}", flush=True)
                continue
            prompt_len = enc["input_ids"].shape[1]
            gen_ids = gen[0, prompt_len:].tolist()
            pred = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gold = self.normalize_text(row.text)
            if i < 3:
                print(f"[gen-eval] gold={gold!r}", flush=True)
                print(f"[gen-eval] pred={pred!r}", flush=True)
            cers.append(cer_fn(pred, gold))
            nlss.append(nls_fn(pred, gold))
            n += 1

        if was_training:
            model.train()
        if not n:
            return {"eval_gen_cer": float("nan"), "eval_gen_nls": float("nan"),
                    "eval_gen_n": 0}
        return {
            "eval_gen_cer": sum(cers) / n,
            "eval_gen_nls": sum(nlss) / n,
            "eval_gen_n": n,
        }


def main() -> None:
    args = parse_args()
    args.data_seed = args.data_seed if args.data_seed is not None else args.seed
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save args for repro
    (out / "config.json").write_text(
        json.dumps({**vars(args), "lora_targets":
                    args.lora_targets or DEFAULT_LORA_TARGETS}, indent=2),
        encoding="utf-8")

    # text_norm — load by file path to avoid bs4 dependency
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_text_norm",
        REPO_ROOT / "src" / "birchbark_ocr" / "data" / "text_norm.py")
    _tn = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_tn)
    normalize_text = _tn.normalize_text

    # Device sanity
    n_dev = torch.cuda.device_count()
    print(f"[+] device count = {n_dev}", flush=True)
    print(f"[+] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}", flush=True)
    print(f"[+] base model = {args.model}", flush=True)

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Model
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Model

    print(f"[+] loading base model in {'bf16' if args.bf16 else 'fp32'}…", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = _Model.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
        )
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"[warn] {args.attn_impl} unavailable ({e}); falling back to sdpa", flush=True)
        model = _Model.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ---- LoRA ----
    from peft import LoraConfig, get_peft_model
    targets = args.lora_targets or DEFAULT_LORA_TARGETS
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Data ----
    train_ds = Phase4OCRDataset(args.train_jsonl, limit=args.limit_train)
    val_ds = Phase4OCRDataset(args.val_jsonl, limit=args.limit_val)
    print(f"[+] train rows = {len(train_ds)}, val rows = {len(val_ds)}", flush=True)

    train_collator = build_collator(
        proc, train=True, prompt=DEFAULT_PROMPT,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
    )
    eval_collator = build_collator(
        proc, train=False, prompt=DEFAULT_PROMPT,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
    )

    # Trainer needs a single collator. Use the train one for the dataloader,
    # but the gen-eval callback will side-step it for generation.
    # ---- Trainer ----
    from transformers import (
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from transformers.trainer_callback import TrainerCallback

    class _DeferredEarlyStop(EarlyStoppingCallback):
        """Variant of EarlyStoppingCallback that does NOT trigger on_evaluate.

        We need this because our gen-eval metric (``eval_gen_cer``) is added
        to the metrics dict *after* ``super().evaluate()`` returns. The base
        ``EarlyStoppingCallback.on_evaluate`` fires from inside that
        ``super().evaluate()`` and would always reset the patience counter
        because the metric is missing at that point. Our trainer override
        calls ``check_metric_and_stop`` manually after augmentation.
        """

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            return  # no-op; manual dispatch from QwenOCRTrainer.evaluate

        def check_metric_and_stop(self, args, state, control, metrics):
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check)
            if metric_value is None:
                return
            self.check_metric_value(args, state, control, metric_value)
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True

    gen_eval = GenEvalCallback(
        processor=proc,
        val_rows=val_ds.rows,
        prompt=DEFAULT_PROMPT,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        max_new_tokens=args.gen_max_new_tokens,
        no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
        max_rows=args.gen_eval_rows,
        normalize_text=normalize_text,
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_gen_cer",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.data_seed,
        report_to=[],  # no W&B etc.
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=(args.num_workers > 0),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    early_stop = _DeferredEarlyStop(early_stopping_patience=args.early_stopping_patience)

    class QwenOCRTrainer(Trainer):
        """Trainer that augments evaluate() with generation-time CER/NLS.

        Overriding evaluate() (rather than using a TrainerCallback) is the
        right hook: the metrics dict we return is what the parent class
        passes to ``_determine_best_metric``, ``log_metrics``, and the
        early-stopping check. Putting the new keys there is the only way
        to steer ``metric_for_best_model='eval_gen_cer'``.
        """

        def __init__(self, *a, gen_eval: GenEvalCallback,
                     early_stop: _DeferredEarlyStop, **kw):
            super().__init__(*a, **kw)
            self._gen_eval = gen_eval
            self._early_stop = early_stop

        def evaluate(self, eval_dataset=None, ignore_keys=None,
                     metric_key_prefix="eval"):
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            device = next(self.model.parameters()).device
            extra = self._gen_eval(self.model, device)
            renamed = {(f"{metric_key_prefix}_{k.removeprefix('eval_')}"
                        if k.startswith("eval_") else f"{metric_key_prefix}_{k}"): v
                       for k, v in extra.items()}
            metrics.update(renamed)
            self.log(renamed)
            self._early_stop.check_metric_and_stop(self.args, self.state,
                                                   self.control, metrics)
            return metrics

    trainer = QwenOCRTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=train_collator,
        gen_eval=gen_eval,
        early_stop=early_stop,
        callbacks=[early_stop],
    )

    print(f"[+] starting training (resume=False)", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"[+] training done in {time.time()-t0:.0f}s", flush=True)

    # Save the (best) LoRA adapter
    final_dir = Path(args.output_dir) / "lora_final"
    trainer.model.save_pretrained(str(final_dir))
    proc.save_pretrained(str(final_dir))
    print(f"[+] saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()
