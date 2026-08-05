# hp-corpus

EN–ZH parallel-corpus extraction pipeline for Chapter 1 of *Harry Potter and the Sorcerer's Stone* (J.K. Rowling). Renders PDF pages, runs OCR (or extracts the embedded text layer), cleans to plain UTF-8, segments into sentences with stable IDs, and aligns EN↔ZH with LaBSE + Vecalign.

## Copyright boundary

This repository contains **code, configuration, schemas, manifests, synthetic-fixture tests, and documentation only**. It must never contain the source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights. The `.gitignore` enforces this; verify with `git ls-files` after every pipeline run.

## Setup

Requires Python 3.11–3.12 (PaddlePaddle does not yet ship wheels for 3.13/3.14). Use [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.12
uv pip install -e ".[ocr,align,dev]"
```

Vecalign is cloned (not pip-installed):

```bash
git clone https://github.com/thompsonb/vecalign.git vendor/vecalign
```

Place the two source PDFs at `data/raw/hp1_en.pdf` and `data/raw/hp1_zh.pdf`. Configure bibliographic metadata, expected hashes, and page ranges in `config/hp1_{en,zh}.yaml`.

## Usage

```bash
hp-corpus validate
hp-corpus run --config config/hp1_en.yaml
hp-corpus run --config config/hp1_zh.yaml
hp-corpus align --en data/segmented/hp1_en_ch01.jsonl --zh data/segmented/hp1_zh_ch01.jsonl --output data/aligned/
```

All commands print metadata and counts only; no novel text reaches the terminal.
