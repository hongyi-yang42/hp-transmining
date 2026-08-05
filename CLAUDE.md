# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This repository currently contains **no source code, build system, or tests**. It holds only the source data under `bookdata/`:

- `bookdata/J.K. Rowling - HP 1 - Harry Potter and the Sorcerer's Stone.pdf` — English source text (≈3.4 MB).
- `bookdata/中文_哈利波特与魔法石 KIC.pdf` — Chinese translation (≈56 MB, image-based / KIC scan).

There is no `package.json`, `pyproject.toml`, `requirements.txt`, `Makefile`, or git history. When the user asks for "build" / "test" / "run" commands, none exist — ask which stack they want before inventing one.

## Apparent purpose

The directory name (`HP_transmining`) and the paired English/Chinese PDFs indicate this is a **bilingual translation-mining project for Harry Potter and the Sorcerer's Stone**. Likely work includes PDF text extraction, sentence alignment across the two languages, and terminology/parallel-corpus analysis. **Confirm scope with the user** before assuming — no extraction scripts, alignment data, or downstream artifacts exist on disk yet.

## Working with the source PDFs

- The English PDF is text-based and can be extracted directly (e.g. `pdftotext`, `pdfplumber`, `PyMuPDF`).
- The Chinese PDF is a large KIC scan and is almost certainly **image-only** — `pdftotext` will return little or nothing. OCR (e.g. Tesseract with `chi_sim`, PaddleOCR, or a cloud OCR service) is required before any text mining. Verify this assumption before building a pipeline on top of it.
- The two books are different editions; chapter boundaries and page numbers will not line up. Alignment is by content, not by layout.
