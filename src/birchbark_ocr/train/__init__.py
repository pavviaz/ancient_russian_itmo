"""Phase 4 Qwen-VL fine-tuning: data, augmentation, training loop."""

from birchbark_ocr.train.augmentation import augment_pil_image
from birchbark_ocr.train.data import Phase4OCRDataset, build_collator

__all__ = ["augment_pil_image", "Phase4OCRDataset", "build_collator"]
