# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`hp-corpus` is a research pipeline that turns two source PDFs of *Harry Potter and the Sorcerer's Stone* (Scholastic 1998 English edition + People's Literature Publishing House Su Nong Chinese translation, 2018 23rd printing) into a sentence-aligned EN–ZH parallel corpus. Currently scoped to **Chapter 1 only** (EN pages 13–29, ZH pages 7–18).

## Copyright boundary — read this first

The repo may **never** contain source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights. `.gitignore` enforces this; `tests/test_gitignore.py` guards it. Allowed in git: code, configs, schemas, manifests, synthetic-fixture tests, docs, aggregate stats. **Never** paste substantial excerpts from either novel into terminal output, tests, README, ADRs, commits, or reports.

If you need to demonstrate behavior, use synthetic text — every test fixture does.

## Source files (gitignored, must be present locally)

- `data/raw/hp1_en.pdf` — Scholastic 1998, 327 pp, has embedded text layer. SHA-256 `f1009a6d…beefa6`.
- `data/raw/hp1_zh.pdf` — People's Lit. 2018, 93 pp, image-only. SHA-256 `d2ab2782…7428d`.

If either is missing, ask the user. Renaming on import is intentional — configs reference the short `hp1_{en,zh}.pdf` names, not the original long filenames.

## Toolchain

Python `>=3.11,<3.13` (PaddlePaddle has no wheels for 3.13/3.14). PyMuPDF for both English text-layer extraction and Chinese page rendering. PaddleOCR PP-OCRv5 for Chinese OCR. LaBSE via `sentence-transformers` + an inline DP for alignment (vendored `vecalign` is optional — `src/hp_corpus/align.py` re-implements the DP).

`uv` is the package manager. Tsinghua/Aliyun PyPI mirrors are dramatically faster from China — prefix heavy installs with `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`.

## Pipeline

```
hp-corpus validate                              # SHA-256 + page-count + text-layer checks
hp-corpus render    --config config/hp1_*.yaml  # PDF → PNG (only the chapter range)
hp-corpus ocr       --config config/hp1_*.yaml  # PNG → OCRBlock JSONL (PaddleOCR) OR text-layer
hp-corpus clean     --config config/hp1_*.yaml  # OCRBlock JSONL → CleanSentence JSONL + .txt + _notes.jsonl
hp-corpus segment   --config config/hp1_*.yaml  # CleanSentence JSONL → Segment JSONL with stable IDs
hp-corpus run       --config config/hp1_*.yaml  # render → ocr → clean → segment, one config
hp-corpus align     --en … --zh … --output …    # cross-language alignment
```

All output goes under `data/{pages,ocr_raw,text_clean,segmented,aligned,embeddings}/` — all gitignored.

## Architecture notes

- **Module layout** (`src/hp_corpus/`): `render.py` (PDF I/O + validation) → `ocr.py` (PaddleOCR or PyMuPDF unified JSONL) → `clean.py` (header/footnote/page-number/paragraph heuristics) → `segment.py` (lang-aware sentence split with stable IDs) → `align.py` (LaBSE embeddings + inline DP). `schema.py` is the single source of truth for record types. `cli.py` wires subcommands.
- **IDs**: `{book}_{lang}_ch{NN}_p{NNNN}_s{NNN}`. Page number lives in a `source_pages` field on the Segment, not encoded in the ID — paragraph/sentence ordinals within a chapter are the canonical identity. Same input → same IDs.
- **Cleaning heuristics** are conservative: cross-page paragraph rejoin when no terminator; hyphen repair only when demonstrably a line-wrap (next char lowercase); footnote markers `①②③` split into `_notes.jsonl` rather than dropped; recurring `header_patterns` are exact-matched and the first occurrence is preserved as `chapter_title`.
- **Cleaning is pure-functional**: `clean_blocks(blocks, config) → CleanResult`. Raw OCR is never mutated; any cleaned artifact can be rebuilt from `data/ocr_raw/` + config.
- **Heavy deps are lazy**: PaddleOCR and sentence-transformers are imported inside the functions that need them, so unit tests run without them.

## Tests

`uv run pytest` runs the unit suite (synthetic fixtures only — no PDFs, no models, no LaBSE). 40 tests. Integration tests (marked `@pytest.mark.integration`) require the source PDFs and installed models.

## Working in this repo

- When touching cleaning rules, add or update a synthetic-fixture test in `tests/test_clean.py`. The rules are heuristic and easy to regress.
- When changing the schema in `src/hp_corpus/schema.py`, run `git ls-files` afterwards to confirm no tracked corpus file slipped in.
- Never bypass `.gitignore`. If a path needs to be tracked, ask the user first — the default is "nothing under `data/`, `models/`, `vendor/`, `artifacts/`, `tmp/` is tracked."
- Vecalign clone (`vendor/vecalign/`) is optional. The DP in `src/hp_corpus/align.py` does not depend on it.
