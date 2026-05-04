# Birchbark transcription normalisation policy

**Version:** 1.0 (frozen for experiments — do not change mid-campaign without a new version bump)

## Raw vs normalised

| Convention | Raw (stored as `text_raw`) | Normalised (`text_norm`) |
|------------|----------------------------|---------------------------|
| Reconstructed segments `[…]` | Kept | Kept for primary eval; optional eval with brackets stripped |
| Editorial completions `(…)` | Kept | **Stripped** in `text_norm` |
| Dotted-under / combining marks | Kept (Unicode as on gramoty.ru) | Same |
| Line breaks `\|` in edition | Split into physical lines during line-image ETL | N/A at document level |

## Evaluation

- **CER** on diplomatic continuous transcription (no WER as primary; word boundaries often absent).
- Report **lemma-level F1** only after a separate tokenisation step (future script).

## Source

Transcriptions are taken from gramoty.ru `div.text-area.original-text` (diplomatic continuous).
