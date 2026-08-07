# hp-transmining

**Translation mining on Harry Potter.** Reproducing Bremmers et al. (2022) — *"Translation Mining: Definiteness across Languages"* — by rebuilding the parallel corpus and prepositional-phrase extraction pipeline from the three translations of *Harry Potter and the Sorcerer's Stone*.

Translation mining extracts grammatical categories (here: definiteness) from parallel translations rather than from manual annotation. The three editions — Scholastic 1998 (English), People's Lit. 2018 (Chinese, Su Nong), and Carlsen 2013 (German, Klaus Fritz) — supply a sentence-aligned DE/EN/ZH corpus on which to run the paper's PP extractor and study how definiteness surfaces across languages.

## What this project does

- **Builds a parallel corpus from PDFs.** Renders image-only PDFs, runs PaddleOCR (ZH/DE) and embedded-text-layer extraction (EN), applies conservative cleaning heuristics, segments into sentences with stable IDs, and aligns sentence pairs across languages via LaBSE embeddings + dynamic programming.
- **Parses with Universal Dependencies.** Stanza (tokenize + mwt + pos + lemma + depparse) → CoNLL-U for all three languages, preserving German contraction range lines (`5-6 im`) so the paper's extractor works unmodified.
- **Extracts prepositional phrases.** An adapted `conll-extractor` runs over the German CoNLL-U and every hit is cross-validated against `FILTER_CONTRACTED_123` / `FILTER_PP` from the paper's annotation set.

## Pipeline

```
hp-corpus validate                              # SHA-256 + page-count + text-layer checks
hp-corpus run       --config config/hp1_*.yaml  # render → ocr → clean → segment
hp-corpus parse     --config config/hp1_*.yaml  # → CoNLL-U (Stanza)
hp-corpus align     --src … --tgt … --out-name … # cross-language LaBSE + DP alignment
```

| Stage | Module | Output |
|---|---|---|
| PDF → PNG | `render.py` | `data/pages/` |
| PNG → text | `ocr.py` | `data/ocr_raw/` |
| Text → sentences | `clean.py` → `segment.py` | `data/text_clean/` → `data/segmented/` |
| Sentences → CoNLL-U | `parse.py` | `data/parsed/` |
| Cross-lingual | `align.py` | `data/aligned/` |

All outputs under `data/` are gitignored. Sentence IDs are deterministic: `{book}_{lang}_ch{NN}_p{NNNN}_s{NNN}` — same input always yields the same corpus.

## Setup

Requires Python 3.11–3.12 (PaddlePaddle has no wheels for 3.13/3.14) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv -p 3.12
uv pip install -e ".[ocr,align,dev]"
git clone https://github.com/thompsonb/vecalign.git vendor/vecalign
git clone https://github.com/time-in-translation/conll-extractor.git vendor/conll-extractor
```

Bring your own copies of the three PDFs and place them at `data/raw/hp1_{en,zh,de}.pdf` (gitignored). Configure bibliographic metadata, expected SHA-256 hashes, and chapter page ranges in `config/hp1_{en,zh,de}.yaml`.

For UD parsing, install Stanza models. HuggingFace is blocked from CN — use the ModelScope mirror:

```bash
uv run python -c "
from modelscope import snapshot_download
for lang in ('de', 'en', 'zh'):
    snapshot_download(f'stanfordnlp/stanza-{lang}' if lang != 'zh' else 'stanfordnlp/stanza-zh-hans', local_dir=f'~/stanza_resources/{lang}')
"
mv ~/stanza_resources/zh/* ~/stanza_resources/zh-hans/   # Stanza expects zh-hans/
uv run python scripts/patch_stanza_lemma_format.py       # one-time v3→v2 format fix
```

## Copyright boundary

This repository contains **code, configuration, schemas, manifests, synthetic-fixture tests, and documentation only**. It must never contain source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights.

`.gitignore` enforces this and `tests/test_gitignore.py` guards it. All commands print only metadata and counts — no novel text reaches the terminal, the test suite, the README, commits, or reports. Tests use synthetic fixtures exclusively.
