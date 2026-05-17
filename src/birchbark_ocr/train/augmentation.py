"""Deterministic image augmentation for real_birch training replicas.

Phase-4 JSONL rows carry an ``augment_seed`` field that uniquely identifies
the augmented variant of a real birchbark photo. The trainer asks for the
augmented image at dataloading time and *must* produce the same pixels for
the same seed -- otherwise the dataloader becomes a hidden source of noise
and runs become non-reproducible across restarts.

This module implements that contract: ``augment_pil_image(img, seed, source)``
is pure with respect to ``(img, seed, source)``. The aug pipeline is
deliberately gentle - birchbark photos are already a tough domain and we
don't want to push them off-distribution.

For ``source == "synth"`` or ``seed is None`` (the un-augmented "replica 0"
row of every real document), this returns the input unchanged.

NOTE (post-recovery): this file was reconstructed from the .pyc bytecode
left in ``src/birchbark_ocr/train/__pycache__`` after the source was lost in
a branch switch. Function signatures, parameter ranges, and per-step gating
probabilities match the decompile; the exact PIL/NumPy implementation of
``_scale_translate``, ``_photometric``, ``_gamma``, and ``_gaussian_noise``
follows the natural stdlib idiom.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # noqa: F401  (kept for parity)

Source = Literal["real_birch", "synth"]


def augment_pil_image(image: Image.Image, seed: int | None, source: Source) -> Image.Image:
    """Apply a deterministic augmentation chain.

    Parameters
    ----------
    image
        Source image. Will be converted to RGB internally.
    seed
        ``None`` means "no augmentation, return as-is". Any non-None integer
        is fed into a fresh ``np.random.default_rng`` so the same seed always
        yields the same pixels.
    source
        ``real_birch`` -> full geometric+photometric+wear chain.
        ``synth``      -> very light photometric only (synth is already varied).
    """
    if seed is None:
        return image if image.mode == "RGB" else image.convert("RGB")
    img = image if image.mode == "RGB" else image.convert("RGB")
    rng = np.random.default_rng(int(seed))
    if source == "real_birch":
        return _aug_real(img, rng)
    if source == "synth":
        return _aug_synth(img, rng)
    return img


def _aug_real(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = _rotate_small(img, rng, max_deg=4)
    img = _scale_translate(img, rng, scale=(0.92, 1.08), tx=0.04, ty=0.04)
    img = _photometric(img, rng,
                       brightness=(0.88, 1.12),
                       contrast=(0.88, 1.15),
                       saturation=(0.85, 1.10))
    img = _gamma(img, rng, gamma=(0.88, 1.12))
    if rng.random() < 0.55:
        img = _slight_blur(img, rng, max_radius=0.7)
    if rng.random() < 0.35:
        img = _gaussian_noise(img, rng, sigma=(2, 6))
    if rng.random() < 0.20:
        img = _jpeg_compress(img, rng, quality=(55, 85))
    return img


def _aug_synth(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = _photometric(img, rng,
                       brightness=(0.95, 1.05),
                       contrast=(0.95, 1.05),
                       saturation=(0.95, 1.05))
    if rng.random() < 0.20:
        img = _slight_blur(img, rng, max_radius=0.5)
    return img


def _rotate_small(img: Image.Image, rng: np.random.Generator, *, max_deg: float) -> Image.Image:
    angle = float(rng.uniform(-max_deg, max_deg))
    if abs(angle) < 0.05:
        return img
    return img.rotate(angle, resample=Image.Resampling.BICUBIC,
                      expand=False, fillcolor=(0, 0, 0))


def _scale_translate(img: Image.Image, rng: np.random.Generator, *,
                     scale: tuple[float, float], tx: float, ty: float) -> Image.Image:
    """Affine: small isotropic scale + small translation, identity-centred."""
    s = float(rng.uniform(scale[0], scale[1]))
    dx = float(rng.uniform(-tx, tx))
    dy = float(rng.uniform(-ty, ty))
    if abs(s - 1.0) < 1e-3 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return img
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    inv = 1.0 / s
    a, b, c = inv, 0.0, cx - inv * cx + dx * w
    d, e, f = 0.0, inv, cy - inv * cy + dy * h
    return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                         resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def _photometric(img: Image.Image, rng: np.random.Generator, *,
                 brightness: tuple[float, float],
                 contrast: tuple[float, float],
                 saturation: tuple[float, float]) -> Image.Image:
    b = float(rng.uniform(*brightness))
    c = float(rng.uniform(*contrast))
    s = float(rng.uniform(*saturation))
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


def _gamma(img: Image.Image, rng: np.random.Generator, *, gamma: tuple[float, float]) -> Image.Image:
    g = float(rng.uniform(*gamma))
    if abs(g - 1.0) < 1e-3:
        return img
    inv = 1.0 / g
    lut = [min(255, int(round(((i / 255.0) ** inv) * 255.0))) for i in range(256)]
    if img.mode == "RGB":
        return img.point(lut * 3)
    return img.point(lut)


def _slight_blur(img: Image.Image, rng: np.random.Generator, *, max_radius: float) -> Image.Image:
    r = float(rng.uniform(0.2, max_radius))
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _gaussian_noise(img: Image.Image, rng: np.random.Generator, *,
                    sigma: tuple[float, float]) -> Image.Image:
    s = float(rng.uniform(*sigma))
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, s, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def _jpeg_compress(img: Image.Image, rng: np.random.Generator, *,
                   quality: tuple[int, int]) -> Image.Image:
    """Fake JPEG ringing / blockiness for a fraction of replicas."""
    import io
    q = int(rng.integers(quality[0], quality[1] + 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
