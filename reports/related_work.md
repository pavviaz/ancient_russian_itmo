# Related work (skim notes)

Short labels for cross-referencing in later phase reports. Full bibliography belongs in the human-written paper.


| Label                  | Reference                                                                                                      | Note                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| CHURRO                 | Semnani et al., 2025, [https://arxiv.org/abs/2509.19768](https://arxiv.org/abs/2509.19768)                     | Historical OCR VLM; **NLS** metric; primary external baseline (`stanford-oval/churro-3B`). |
| Digital Peter          | Potanin et al., 2021, [https://arxiv.org/pdf/2103.09354](https://arxiv.org/pdf/2103.09354)                     | Russian XVIII c. HTR; auxiliary training data.                                             |
| Rabus                  | Freiburg Slavic HTR / Transkribus context                                                                      | Old Cyrillic manuscript OCR tradition.                                                     |
| CyrillicHandwritingPOC | dbrainio, arXiv:2311.15896                                                                                     | Procedural Bezier synthetic Cyrillic; procedural baseline generator.                       |
| Glyph-conditional DDPM | Ding et al., arXiv:2305.19543                                                                                  | Glyph layout conditioning for synthetic OCR training.                                      |
| Manchu Qwen2.5-VL OCR  | arXiv:2507.06761                                                                                               | Low-resource VLM + synthetic diffusion data recipe.                                        |
| Qwen3-VL TR            | arXiv:2511.21631                                                                                               | Architecture context for Qwen3.x VL line.                                                  |
| Qwen3.5                | [https://qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5) , HF `Qwen/Qwen3.5-2B`                      | Primary fine-tune backbone.                                                                |
| PaddleOCR-VL           | HF `PaddlePaddle/PaddleOCR-VL`                                                                                 | Optional heavy baseline; omitted from headline Phase 2 table (CPU timing in this repo).   |
| Yandex archives Habr   | [https://habr.com/ru/companies/yandex/articles/712510/](https://habr.com/ru/companies/yandex/articles/712510/) | Industrial Russian handwriting OCR perspective.                                            |
| gramoty.ru             | [http://gramoty.ru/birchbark/](http://gramoty.ru/birchbark/)                                                   | Corpus and editorial conventions.                                                          |
