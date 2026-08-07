# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`hp-corpus` is a research pipeline whose **primary goal is reproducing Bremmers et al. (2022) "Translation Mining: Definiteness across Languages"** — extracting the paper's German PP dataset (contracted + uncontracted) from *Harry Potter and the Sorcerer's Stone*, then mapping each PP to its EN and ZH translation equivalents.

The repo turns three source PDFs (Scholastic 1998 English + People's Lit. 2018 Su Nong Chinese + Carlsen 2013 Klaus Fritz German) into sentence-aligned DE/EN/ZH parallel corpora, runs UD parsing, and extracts prepositional phrases per the paper's methodology.

**Working scope**: Chapters 1–3 for all three languages. Ch.1–3 PP extraction complete (110 / 132 contracted PPs match `FILTER_CONTRACTED_123` = 83%; combined Ch.1–3 ≈ paper's expected ~96–113 annotated set). Cross-lingual PP mapping (DE PP → EN/ZH equivalents) and MDS visualization are the remaining work.

**Out of scope until cross-lingual mapping validates**: Ch.4+ (paper's minimal-pair selection from later chapters), MDS visualization.

## Copyright boundary — read this first

The repo may **never** contain source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights. `.gitignore` enforces this; `tests/test_gitignore.py` guards it. Allowed in git: code, configs, schemas, manifests, synthetic-fixture tests, docs, aggregate stats. **Never** paste substantial excerpts from any of the three novels into terminal output, tests, README, ADRs, commits, or reports.

If you need to demonstrate behavior, use synthetic text — every test fixture does.

## Source files (gitignored, must be present locally)

- `data/raw/hp1_en.pdf` — Scholastic 1998, 327 pp, has embedded text layer. SHA-256 `f1009a6d…beefa6`.
- `data/raw/hp1_zh.pdf` — People's Lit. 2018, 93 pp, image-only. SHA-256 `d2ab2782…7428d`.
- `data/raw/hp1_de.pdf` — Carlsen 2013 (Klaus Fritz translation), 340 pp, image-only (IA Scribe text layer is too noisy). SHA-256 `eeefb4df…15aac`.

If any is missing, ask the user. Renaming on import is intentional — configs reference the short `hp1_{en,zh,de}.pdf` names, not the original long filenames.

## Toolchain

Python `>=3.11,<3.13` (PaddlePaddle has no wheels for 3.13/3.14). PyMuPDF for English text-layer extraction and ZH/DE page rendering. PaddleOCR PP-OCRv5 for Chinese and German OCR. LaBSE via `sentence-transformers` + an inline DP for alignment (vendored `vecalign` is optional — `src/hp_corpus/align.py` re-implements the DP). **Stanza 1.14** for UD parsing (tokenize+mwt+pos+lemma+depparse) → CoNLL-U.

`uv` is the package manager. Tsinghua/Aliyun PyPI mirrors are dramatically faster from China — prefix heavy installs with `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`.

**Stanza models**: HuggingFace is blocked from CN; use the ModelScope mirror (`modelscope.cn/stanfordnlp/stanza-{de,en,zh-hans}`). As of 2026-08, ModelScope's lemma files use `dicts_version=3` (JSON-in-gzip); Stanza 1.14 only supports v2 (pickle-in-gzip). Run `scripts/patch_stanza_lemma_format.py` once after download (idempotent). ZH models land under `~/stanza_resources/zh/` but Stanza expects `zh-hans/` — `mv ~/stanza_resources/zh/* ~/stanza_resources/zh-hans/` to fix.

## Pipeline

```
hp-corpus validate                              # SHA-256 + page-count + text-layer checks
hp-corpus render    --config config/hp1_*.yaml  # PDF → PNG (only the chapter range)
hp-corpus ocr       --config config/hp1_*.yaml  # PNG → OCRBlock JSONL (PaddleOCR) OR text-layer
hp-corpus clean     --config config/hp1_*.yaml  # OCRBlock JSONL → CleanSentence JSONL + .txt + _notes.jsonl
hp-corpus segment   --config config/hp1_*.yaml  # CleanSentence JSONL → Segment JSONL with stable IDs
hp-corpus run       --config config/hp1_*.yaml  # render → ocr → clean → segment, one config
hp-corpus parse     --config config/hp1_*.yaml  # Segment JSONL → CoNLL-U (Stanza, requires models)
hp-corpus align     --src … --tgt … --out-name … # cross-language alignment
```

All output goes under `data/{pages,ocr_raw,text_clean,segmented,parsed,aligned,embeddings,extracted}/` — all gitignored.

## Architecture notes

- **Module layout** (`src/hp_corpus/`): `render.py` (PDF I/O + validation) → `ocr.py` (PaddleOCR or PyMuPDF unified JSONL) → `clean.py` (header/footnote/page-number/paragraph heuristics) → `segment.py` (lang-aware sentence split with stable IDs) → `parse.py` (Stanza UD → CoNLL-U with MWT range lines) → `align.py` (LaBSE embeddings + inline DP). `schema.py` is the single source of truth for record types. `cli.py` wires subcommands.
- **IDs**: `{book}_{lang}_ch{NN}_p{NNNN}_s{NNN}`. Page number lives in a `source_pages` field on the Segment, not encoded in the ID — paragraph/sentence ordinals within a chapter are the canonical identity. Same input → same IDs.
- **Cleaning heuristics** are conservative: cross-page paragraph rejoin when no terminator; hyphen repair only when demonstrably a line-wrap (next char lowercase); footnote markers `①②③` split into `_notes.jsonl` rather than dropped; recurring `header_patterns` are exact-matched and the first occurrence is preserved as `chapter_title`. German follows English rules at `clean.py:346` (space-joined + de-hyphenated) — ZH remains unspaced.
- **Cleaning is pure-functional**: `clean_blocks(blocks, config) → CleanResult`. Raw OCR is never mutated; any cleaned artifact can be rebuilt from `data/ocr_raw/` + config.
- **Heavy deps are lazy**: PaddleOCR, sentence-transformers, and stanza are imported inside the functions that need them, so unit tests run without them.
- **MWT preservation**: Stanza's tokenizer emits range lines (e.g. `5-6 im`) for German contractions even without the MWT processor. `parse.py` writes these range lines faithfully; `scripts/normalize_conllu_mwt.py` collapses them into single tokens (POS/head from first component) for paper-extractor compatibility.

## Bremmers reproduction

**Goal**: Extract the paper's PP dataset (Ch.1–3 all contracted PPs + Ch.4+ contracted PPs that have uncontracted minimal-pair counterparts) and align each to EN/ZH translations.

**Method** (paper §2 + footnote 2):
1. UD-parse all three languages → CoNLL-U (`hp-corpus parse`)
2. Run `time-in-translation/conll-extractor` on DE CoNLL-U (vendored at `vendor/conll-extractor/`, gitignored)
3. Cross-validate hits against the paper's `FILTER_CONTRACTED_123` / `FILTER_PP` lists (in `conll_extractor.prepositions.data`)
4. Map each DE PP to its EN/ZH counterpart via the existing alignment
5. MDS visualization of definiteness dimensions

**Status**:
- ✅ Step 1 done for Ch.1–3 — DE 1005 sent / 16430 tok; EN 805 / 14263; ZH 854 / 15660
- ✅ Step 2 done for Ch.1–3 — `scripts/run_paper_extractor.py` adapts `process_single`
- ✅ Step 3 done for Ch.1–3 — contracted: 110/132 (83%) match `FILTER_CONTRACTED_123`; uncontracted: 27/107 (25%) match `FILTER_PP`. The 83% rate across all 3 chapters is consistent with Ch.1's 84%, confirming the extraction method generalizes.
- ✅ Alignments regenerated for Ch.1–3: DE↔EN, DE↔ZH, EN↔ZH (LaBSE+vecalign, ~96–98% of source sentences aligned per chapter)
- ⏳ Step 4 (DE PP → EN/ZH mapping) — next up
- ⏳ Step 5 (MDS) deferred

**Scripts** (`scripts/`):
- `run_paper_extractor.py` — paper-faithful extraction (CONTRACTED + PREPOSITIONS/DETERMINERS) + filter validation
- `normalize_conllu_mwt.py` — collapse MWT range lines for paper extractor
- `patch_stanza_lemma_format.py` — one-time Modelscope v3 → Stanza v2 lemma converter
- `extract_de_pps.py` — regex-based PP extractor, kept as orthogonal cross-validation (do not extend)
- `build_review_tsv.py` — generate manual-review TSV from alignment output

**Known error modes** seen in Ch.1–3 (22 contracted non-matches categorize as):
- Edition-specific PPs the paper's filter doesn't list (~11): am Tag, im Bad, ins Reptilienhaus, vom Augenblick, beim Gestank, zur Arbeit, im Erdgeschoß, im Donner, beim Gedanke — real German PPs the paper didn't annotate.
- Lemma errors (~6): Stanza lemmatized -en nouns as -e (Stolpern→Stolper, Brüllen→Brüll, Stillen→Stille, Inneren→Inneres).
- Parser errors (~5): adjective mistagged as NN (zum gut, am gut, am schlimm, im Vorbeigleite).
- The 80 uncontracted non-matches are mostly relative-pronoun `der` (~49) — the paper's extractor can't distinguish article `der` from relative `der`, so it over-extracts; this is a known methodological limitation, not a bug.

## Tests

`uv run pytest` runs the unit suite (synthetic fixtures only — no PDFs, no models, no LaBSE, no Stanza). 49 tests. Integration tests (marked `@pytest.mark.integration`) require the source PDFs and installed models.

## Working in this repo

- When touching cleaning rules, add or update a synthetic-fixture test in `tests/test_clean.py`. The rules are heuristic and easy to regress.
- When changing the schema in `src/hp_corpus/schema.py`, run `git ls-files` afterwards to confirm no tracked corpus file slipped in.
- Never bypass `.gitignore`. If a path needs to be tracked, ask the user first — the default is "nothing under `data/`, `models/`, `vendor/`, `artifacts/`, `tmp/` is tracked."
- Vecalign clone (`vendor/vecalign/`) is optional. The DP in `src/hp_corpus/align.py` does not depend on it.
- conll-extractor clone (`vendor/conll-extractor/`) is required for Bremmers reproduction. Filter lists in `conll_extractor.prepositions.data` are the paper's exact annotation set — do not modify them; if a Ch.2–3 PP doesn't match, log it as an edition/parser error rather than patching the list.
