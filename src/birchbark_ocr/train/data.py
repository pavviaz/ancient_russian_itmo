"""Phase-4 dataset and collator for Qwen2-VL / Qwen3-VL / Qwen3.5 fine-tuning.

A split JSONL (``data/splits/phase4/<mix>_train.jsonl`` or ``val.jsonl``)
is consumed row-by-row. Each row turns into a Qwen-VL chat conversation::

    [
      {"role": "user", "content": [
          {"type": "image", "image": <PIL.Image>},
          {"type": "text",  "text": <user prompt>},
      ]},
      {"role": "assistant", "content": [
          {"type": "text", "text": <gold transcription>},
      ]},
    ]

Critical correctness notes for Qwen processors (esp. Qwen3.5):

1. The chat template renders one ``<|image_pad|>`` per image; the *processor*
   then expands that placeholder into N tokens (one per visual patch). The
   number depends on the image's H, W. So you cannot derive the prompt
   boundary by tokenising the prompt text alone -- you have to run the same
   processor on the prompt-only conversation with the same images and read
   ``input_ids.shape[1]`` from the resulting encoding.

2. Qwen3.5's chat template injects ``<think>\n\n</think>\n\n`` between
   ``<|im_start|>assistant\n`` and the answer (the "no-thinking" placeholder).
   We pass ``enable_thinking=False`` for clarity, but the resulting prefix is
   identical between full and prompt-only renderings, so masking by
   ``prompt-only encoded length`` cleanly excludes that prefix.

3. At evaluation time we want generation to start *before* the answer, so
   the generation path uses the prompt-only encoding -- never the full one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from birchbark_ocr.train.augmentation import augment_pil_image


DEFAULT_PROMPT = (
    "You are an expert palaeographer specialising in medieval East Slavic "
    "birchbark documents (gramoty) from Novgorod, Moscow, and other Old Russian "
    "centres, 11th–15th centuries. The image is a photograph or drawing of an "
    "inscription incised into birchbark with a stylus. Transcribe every visible "
    "character of the inscription verbatim, line by line, in the diplomatic "
    "continuous form used at gramoty.ru (no editorial expansions, no "
    "modernisation, no commentary). Preserve the original Old Cyrillic letter "
    "forms (ѣ, ѫ, ѡ, ѥ, ѩ, ѭ, titlo, etc.). Where letters are damaged or "
    "illegible, output a single '-' character. Output the transcription only, "
    "with no extra prose."
)


@dataclass
class Phase4Row:
    sample_id: str
    image_path: Path
    text: str
    source: str  # "real_birch" | "synth"
    augment_seed: int | None
    augment_replica_index: int


class Phase4OCRDataset(Dataset[Phase4Row]):
    """Read-only view of a Phase-4 train/val JSONL.

    Returns lightweight :class:`Phase4Row` records; image decoding + augment
    happens lazily inside the collator so workers can pipeline I/O.
    """

    def __init__(self, jsonl_path: Path | str, *, limit: int | None = None) -> None:
        self.path = Path(jsonl_path)
        self.rows: list[Phase4Row] = []
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.rows.append(Phase4Row(
                    sample_id=str(rec["sample_id"]),
                    image_path=Path(rec["image_path"]),
                    text=str(rec.get("text", "")),
                    source=str(rec["source"]),
                    augment_seed=(int(rec["augment_seed"])
                                  if rec.get("augment_seed") is not None else None),
                    augment_replica_index=int(rec.get("augment_replica_index", 0)),
                ))
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Phase4Row:
        return self.rows[idx]


def _load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_for_qwen(img: Image.Image, *, image_max_pixels: int,
                     image_min_pixels: int, align_to: int) -> Image.Image:
    """Resize so total pixels fall in [min, max], aligning H/W to ``align_to``.

    Qwen processors do their own smart-resize, but doing the coarse-grained
    downscale here lets us trust the per-batch image-token budget (vital for
    multi-GPU throughput) and keeps the prompt-only encoding deterministic
    when paired with the same image.
    """
    w, h = img.size
    pix = w * h
    target_w, target_h = w, h
    if pix > image_max_pixels:
        scale = (image_max_pixels / pix) ** 0.5
        target_w = max(align_to, int(round(w * scale)))
        target_h = max(align_to, int(round(h * scale)))
    elif pix < image_min_pixels:
        scale = (image_min_pixels / pix) ** 0.5
        target_w = max(align_to, int(round(w * scale)))
        target_h = max(align_to, int(round(h * scale)))
    target_w = max(align_to, (target_w // align_to) * align_to)
    target_h = max(align_to, (target_h // align_to) * align_to)
    if (target_w, target_h) != (w, h):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


class QwenVLCollator:
    """Batch a list of :class:`Phase4Row` into Qwen-VL processor inputs.

    Train batches carry ``labels`` with the prompt portion masked to -100
    (loss is computed only on the assistant answer + ``<|im_end|>``).

    Eval batches (``train=False``) compute the same masked labels so
    ``Trainer.evaluate()`` can report a comparable cross-entropy. For
    generation, use :meth:`encode_prompt_only` instead -- it returns a
    prompt-only encoding suitable for ``model.generate(...)``.
    """

    def __init__(self, processor: Any, *, train: bool,
                 prompt: str = DEFAULT_PROMPT,
                 image_max_pixels: int = 451584,
                 image_min_pixels: int = 100352,
                 align_to: int = 28) -> None:
        self.processor = processor
        self.train = train
        self.prompt = prompt
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self.align_to = align_to

    # ---- public API ----
    def __call__(self, batch: list[Phase4Row]) -> dict[str, torch.Tensor]:
        images: list[Image.Image] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        for row in batch:
            img = self._prep_image(row)
            images.append(img)
            full_texts.append(self._render_full(row.text))
            prompt_texts.append(self._render_prompt())
        # Encode prompt-only (with images!) to recover token-level prompt boundary.
        # This is the **critical** Qwen3.5 trick: the chat template emits the
        # raw <|image_pad|> token which the processor expands to N visual tokens
        # depending on H,W -- so prompt length must be measured **with images**.
        prompt_lens = []
        for txt, img in zip(prompt_texts, images):
            enc_p = self._processor_call([txt], [img])
            prompt_lens.append(int(enc_p["input_ids"].shape[1]))
        enc = self._processor_call(full_texts, images)
        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        # Mask the prompt portion + pad tokens
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.shape[1])
            labels[i, :plen] = -100
        labels[enc["input_ids"] == pad_id] = -100
        enc["labels"] = labels
        return enc

    def encode_prompt_only(self, row: Phase4Row, *,
                           augment_train: bool | None = None) -> dict[str, torch.Tensor]:
        """Return a prompt-only encoding suitable for ``model.generate()``.

        Used by :class:`QwenOCRTrainer` to score CER/NLS at eval time. The
        gold text is *not* part of the encoding -- if it were, the model
        would just continue from after the answer and produce nonsense.
        """
        img = self._prep_image(row, augment_train=augment_train)
        prompt_text = self._render_prompt()
        return self._processor_call([prompt_text], [img])

    # ---- internals ----
    def _prep_image(self, row: Phase4Row, *,
                    augment_train: bool | None = None) -> Image.Image:
        do_aug = self.train if augment_train is None else augment_train
        img = _load_image(row.image_path)
        seed = row.augment_seed if do_aug else None
        img = augment_pil_image(img, seed=seed, source=row.source)
        img = _resize_for_qwen(img,
                               image_max_pixels=self.image_max_pixels,
                               image_min_pixels=self.image_min_pixels,
                               align_to=self.align_to)
        return img

    def _render_full(self, gold: str) -> str:
        msgs = [
            {"role": "system",
             "content": [{"type": "text", "text": self.prompt}]},
            {"role": "user",
             "content": [{"type": "image"},
                         {"type": "text", "text": "Transcribe."}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": gold}]},
        ]
        return self._apply_template(msgs, add_generation_prompt=False)

    def _render_prompt(self) -> str:
        msgs = [
            {"role": "system",
             "content": [{"type": "text", "text": self.prompt}]},
            {"role": "user",
             "content": [{"type": "image"},
                         {"type": "text", "text": "Transcribe."}]},
        ]
        return self._apply_template(msgs, add_generation_prompt=True)

    def _apply_template(self, msgs: list[dict[str, Any]], *,
                        add_generation_prompt: bool) -> str:
        # Try with enable_thinking=False (Qwen3.5+); fall back if the kwarg is
        # unknown (older processor versions).
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            return self.processor.apply_chat_template(
                msgs, **kwargs, enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(msgs, **kwargs)

    def _processor_call(self, text: list[str],
                        images: list[Image.Image]) -> dict[str, torch.Tensor]:
        try:
            return self.processor(
                text=text, images=images,
                return_tensors="pt", padding=True,
            )
        except TypeError:
            # Older processors used `padding="longest"`.
            return self.processor(
                text=text, images=images,
                return_tensors="pt", padding="longest",
            )


def build_collator(processor: Any, *, train: bool,
                   prompt: str = DEFAULT_PROMPT,
                   image_max_pixels: int = 451584,
                   image_min_pixels: int = 100352,
                   align_to: int = 28) -> QwenVLCollator:
    return QwenVLCollator(processor, train=train, prompt=prompt,
                          image_max_pixels=image_max_pixels,
                          image_min_pixels=image_min_pixels,
                          align_to=align_to)
