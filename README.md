# hp-transmining

A research pipeline for **translation mining of definiteness across languages**, reproducing Bremmers et al. (2022) *"Translation Mining: Definiteness across Languages"* on *Harry Potter and the Sorcerer's Stone* (J.K. Rowling).

The pipeline turns three source PDFs — Scholastic 1998 (English), People's Lit. 2018 (Chinese, Su Nong), and Carlsen 2013 (German, Klaus Fritz) — into a sentence-aligned DE/EN/ZH parallel corpus, runs UD parsing, and extracts the paper's German prepositional-phrase dataset, mapped to EN and ZH translation equivalents.

> **Internal package name**: `hp-corpus` (the CLI entry point). The repo name `hp-transmining` reflects the research goal — Bremmers et al.'s translation-mining methodology applied to definiteness.

## Goal

Reproduce the paper's PP dataset from Chapters 1–3:

1. UD-parse all three languages → CoNLL-U
2. Run the paper's `conll-extractor` on the German output
3. Cross-validate hits against `FILTER_CONTRACTED_123` / `FILTER_PP` from `conll_extractor.prepositions.data`
4. Map each German PP to its EN/ZH translation via LaBSE-based alignment
5. MDS visualization of definiteness dimensions (deferred)

**Status** (Ch.1 validated):
- ✅ UD parsing complete for all three languages (DE 362 / EN 340 / ZH 351 sentences)
- ✅ Paper extractor adapted — `scripts/run_paper_extractor.py`
- ✅ Contracted PPs: 37/44 (84%) match the paper's filter; uncontracted: 12/39 (31%)
- ⏳ Ch.2 + Ch.3 extraction in progress
- ⏳ Cross-lingual PP mapping + MDS deferred until Ch.1–3 validates end-to-end

See `CLAUDE.md` for full methodology and current blockers.

## Copyright boundary — read this first

This repository contains **code, configuration, schemas, manifests, synthetic-fixture tests, and documentation only**. It must **never** contain source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights.

- `.gitignore` enforces this; `tests/test_gitignore.py` guards it.
- All commands print only metadata and counts — no novel text reaches the terminal, the test suite, the README, ADRs, commits, or reports.
- If you need to demonstrate behavior in a test or doc, use synthetic text (every existing fixture does).

## Pipeline

```
hp-corpus validate                              # SHA-256 + page-count + text-layer checks
hp-corpus run       --config config/hp1_*.yaml  # render → ocr → clean → segment, one shot
hp-corpus parse     --config config/hp1_*.yaml  # Segment JSONL → CoNLL-U (Stanza, needs models)
hp-corpus align     --src … --tgt … --out-name … # cross-language LaBSE + DP alignment
```

Module layout (`src/hp_corpus/`):

| Stage | Module | Output |
|---|---|---|
| PDF → PNG | `render.py` | `data/pages/` |
| PNG → text | `ocr.py` | `data/ocr_raw/` (PaddleOCR JSONL or PyMuPDF text layer) |
| Text → sentences | `clean.py` → `segment.py` | `data/text_clean/` → `data/segmented/` |
| Sentences → CoNLL-U | `parse.py` | `data/parsed/` |
| Cross-lingual | `align.py` | `data/aligned/` |

IDs are deterministic: `{book}_{lang}_ch{NN}_p{NNNN}_s{NNN}`. Page number lives in a `source_pages` field on the Segment, not encoded in the ID — paragraph/sentence ordinals within a chapter are the canonical identity. Same input → same IDs.

All outputs under `data/` are gitignored.

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

For Bremmers reproduction, also clone the paper's extractor:

```bash
git clone https://github.com/time-in-translation/conll-extractor.git vendor/conll-extractor
```

Place the three source PDFs at `data/raw/hp1_{en,zh,de}.pdf`. Configure bibliographic metadata, expected SHA-256 hashes, and chapter page ranges in `config/hp1_{en,zh,de}.yaml`.

### Stanza models (for `hp-corpus parse`)

HuggingFace is blocked from CN; use the ModelScope mirror:

```bash
uv run python -c "
from modelscope import snapshot_download
for lang in ('de', 'en', 'zh'):
    snapshot_download(f'stanfordnlp/stanza-{lang}' if lang != 'zh' else 'stanfordnlp/stanza-zh-hans', local_dir=f'~/stanza_resources/{lang}')
"
mv ~/stanza_resources/zh/* ~/stanza_resources/zh-hans/  # Stanza expects zh-hans/
uv run python scripts/patch_stanza_lemma_format.py     # one-time v3→v2 format fix
```

## Usage

```bash
# Full pipeline per language
hp-corpus run   --config config/hp1_en.yaml
hp-corpus run   --config config/hp1_zh.yaml
hp-corpus run   --config config/hp1_de.yaml

# UD parsing (requires Stanza models)
hp-corpus parse --config config/hp1_en.yaml
hp-corpus parse --config config/hp1_zh.yaml
hp-corpus parse --config config/hp1_de.yaml

# Cross-lingual alignment
hp-corpus align --src data/segmented/hp1_en_ch01.jsonl \
                --tgt data/segmented/hp1_zh_ch01.jsonl \
                --out-name hp1_en_zh_ch01.jsonl
```

For Bremmers PP extraction, see `scripts/run_paper_extractor.py`.

## Tests

```bash
uv run pytest                       # 49 unit tests, synthetic fixtures only
uv run pytest -m integration        # requires source PDFs + installed models
```

## Working in this repo

- When touching cleaning rules, add a synthetic-fixture test in `tests/test_clean.py`.
- When changing `src/hp_corpus/schema.py`, run `git ls-files` to confirm no tracked corpus file slipped in.
- Never bypass `.gitignore`. Default: nothing under `data/`, `models/`, `vendor/`, `artifacts/`, `tmp/` is tracked.
- The paper's filter lists in `vendor/conll-extractor/conll_extractor/prepositions/data` are the paper's exact annotation set — do not modify them. If a Ch.2–3 PP doesn't match, log it as an edition/parser error rather than patching the list.
