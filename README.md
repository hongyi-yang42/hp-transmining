# hp-transmining

**Translation mining on Harry Potter.** Reproducing Bremmers et al. (2022) — *"Translation Mining: Definiteness across Languages"* — by rebuilding the parallel corpus and prepositional-phrase extraction pipeline from the three translations of *Harry Potter and the Sorcerer's Stone*.

Translation mining extracts grammatical categories (here: definiteness) from parallel translations rather than from manual annotation. The three sources — Scholastic 1998 (English), a born-digital ebook conversion of the People's Literature Publishing House Chinese translation (Su Nong; print-edition provenance unverified), and Carlsen 1998 first edition (German, Klaus Fritz) — are used to build a **machine-aligned** DE/EN/ZH corpus on which to run the paper's PP extractor.

**Scope:** the full novel (Ch.1–17). The pipeline produces a machine-built German PP candidate pool joined to EN/ZH aligned contexts (`dataset_scope = full_novel_machine_candidate_pool`, human-editable columns blank). The paper's final 96 data points are an **external aggregate benchmark** — the paper publishes no per-item list, and our analysis sample is defined by a separate selection procedure on this pool (see `docs/ANALYSIS_SAMPLE_DESIGN.md`), not by recovering those 96. The machine pool is not an analysis-ready corpus.

## What this project does

- **Builds a parallel corpus from PDFs.** Extracts embedded text layers (all three computational sources are born-digital PDFs; PaddleOCR remains wired in for image-only scans, and an earlier image-only ZH scan was retired after an audit showed OCR line loss), applies conservative cleaning heuristics, segments into sentences with stable IDs, and aligns sentence pairs across languages via `intfloat/multilingual-e5-base` embeddings + dynamic programming.
- **Parses with Universal Dependencies.** Stanza (tokenize + mwt + pos + lemma + depparse) → CoNLL-U for all three languages, preserving German contraction range lines (`5-6 im`) so the paper's extractor works unmodified.
- **Extracts prepositional phrases.** An adapted `conll-extractor` runs over the German CoNLL-U; each extracted form–lemma pair is checked against the authors' public filter resources (`FILTER_CONTRACTED_123` / `FILTER_PP`) for extraction QA. This check does **not** identify membership in the paper's final 96 trilingual contexts.

## Pipeline

```
hp-corpus validate                              # SHA-256 + page-count + text-layer checks
hp-corpus run       --config config/hp1_*.yaml  # render → ocr → clean → segment
hp-corpus parse     --config config/hp1_*.yaml  # → CoNLL-U (Stanza)
hp-corpus align     --src … --tgt … --out-name … # cross-language e5-base + DP alignment

# Extraction & cross-lingual mapping (post-alignment, see scripts/):
uv run python scripts/run_paper_extractor.py            # paper-faithful PP extraction on DE CoNLL-U
uv run python scripts/run_alignments_v2.py --chapters 4 … 17  # chapter-pair alignments + manifests (default 1 2 3)
uv run python scripts/build_full_novel_annotation.py    # full-novel machine candidate pool (human columns blank)
# → human review/annotation fills data/derived/step4/*.tsv per docs/STEP4_ANNOTATION.md
```

| Stage | Module | Output |
|---|---|---|
| PDF → PNG | `render.py` | `data/pages/` |
| PNG → text | `ocr.py` | `data/ocr_raw/` |
| Text → sentences | `clean.py` → `segment.py` | `data/text_clean/` → `data/segmented/` |
| Sentences → CoNLL-U | `parse.py` | `data/parsed/` |
| Cross-lingual | `align.py` | `data/aligned/` |

All outputs under `data/` are gitignored. Sentence IDs are deterministic: `{book}_{lang}_ch{NN}_p{NNNN}_s{NNN}`, where `p` is the chapter-scoped **paragraph ordinal** and `s` the sentence ordinal within its paragraph — page numbers live in a separate `source_pages` field, so IDs do not shift with pagination. Same input always yields the same corpus.

## Setup

Requires Python 3.11–3.12 (PaddlePaddle has no wheels for 3.13/3.14) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv -p 3.12
uv pip install -e ".[ocr,align,parse,dev]"
git clone https://github.com/time-in-translation/conll-extractor.git vendor/conll-extractor
cd vendor/conll-extractor && git checkout 4a8a220 && cd ../..   # pin to the validated revision (recorded in vendor/conll-extractor.commit)
```

`src/hp_corpus/align.py` ships its own dynamic-programming aligner — no `vecalign` checkout is needed.

Bring your own copies of the three PDFs and place them under `data/raw/` (gitignored). EN uses the short name `hp1_en.pdf`; the computational ZH source is a born-digital ebook PDF (filename referenced by `config/hp1_zh_ch*.yaml`); DE uses the long original filename `1998 Harry Potter und der Stein der Weisen -- 1998.pdf` (the configs reference it directly). Configure bibliographic metadata and chapter page ranges in `config/hp1_{en,zh,de}*.yaml`.

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

## Citation

If you use this code or build on the methodology, please cite the source paper:

```bibtex
@article{bremmers2022translation,
  title     = {Translation Mining: Definiteness across Languages (A Reply to Jenks 2018)},
  author    = {Bremmers, David and Liu, Jianan and van der Klis, Martijn and Le Bruyn, Bert},
  journal   = {Linguistic Inquiry},
  volume    = {53},
  number    = {4},
  pages     = {735--752},
  year      = {2022},
  publisher = {MIT Press},
  doi       = {10.1162/ling_a_00423}
}
```

## Copyright boundary

This repository contains **code, configuration, schemas, manifests, synthetic-fixture tests, and documentation only**. It must never contain source PDFs, rendered page images, OCR output, cleaned novel text, embeddings, or model weights.

`.gitignore` enforces this and `tests/test_gitignore.py` guards it. All commands print only metadata and counts — no novel text reaches the terminal, the test suite, the README, commits, or reports. Tests use synthetic fixtures exclusively.
