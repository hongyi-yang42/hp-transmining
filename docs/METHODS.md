# Methods

## Reproduction scope

This is an **extraction-method reproduction** of Bremmers et al. (2022) *"Translation Mining: Definiteness across Languages"* — not a parser-identical replication. We use the paper's published extraction logic verbatim; the parser differs because Bremmers et al. do not report which UD parser they used.

A 100% match against the paper's `FILTER_CONTRACTED_123` / `FILTER_PP` is therefore not achievable: residual differences come from parser variation (our Stanza vs the paper's unreported parser) and from edition-specific German PPs in the Carlsen 1998 text that the paper did not annotate.

## Parser

- **Paper upstream parser**: not reported in Bremmers et al. (2022).
- **This project**: Stanza 1.14, with `tokenize + mwt + pos + lemma + depparse`.

## Extraction logic (Step 1–3)

`scripts/run_full_novel_german_extraction.py` (backed by `src/hp_corpus/german_extraction.py`) follows the paper's `time-in-translation/conll-extractor` (vendored at `vendor/conll-extractor/`, gitignored). The older `scripts/run_paper_extractor.py` is a deprecated thin wrapper that delegates to it:

- **Contracted PPs**: tokens in `CONTRACTED` (`im`, `am`, `zum`, …) → extract `(prep, noun)` where the prep's head is an `NN`.
- **Uncontracted PPs**: `PREPOSITIONS` tokens whose same-head dependent is an `ART` in `DETERMINERS` (`der`/`dem`/`den`/…) → extract `(prep, det, noun)`.

Each row in `data/extracted/hp1_de_ch{NN}_{contracted,uncontracted}.tsv` carries two-level provenance (`parse_block_id`, `source_segment_id`) plus occurrence coordinates (`prep_token_id`, `det_token_id`, `noun_token_id`, `pp_token_start`, `pp_token_end`, `pp_surface`) so two PPs that share `(preposition, noun)` within one sentence can be told apart. Coordinates are CoNLL-U token IDs from the `_nomwt.conllu` files; `parse_block_id` (`<segment_id>#bNNN`) identifies the exact parsed block, `source_segment_id` the alignment-level sentence.

Hits are validated against `FILTER_CONTRACTED_123` and `FILTER_PP` from `conll_extractor.prepositions.data`.

## Step 4 — Cross-lingual annotation pack

**What a Step 4 datapoint is.** A *specific German PP occurrence* — a token range inside one sentence — together with the EN and ZH sentence(s) aligned to that German sentence. The annotator's task is to identify, inside each aligned sentence, the linguistic expression that corresponds to the German PP and to record its observable form.

**What the Step 4 master TSV is — and is not.**

- The master TSV (`full_novel_annotation_master.tsv`, built by `scripts/build_full_novel_annotation.py`) is the **machine master for the full novel (Ch.1–17)**: every extracted German PP occurrence, both forms, with its EN/ZH aligned anchor text and the retrieval-view context columns. Nothing is hand-picked at this stage.
- It is **not** the paper's final 96 trilingual contexts. The 96 are hand-selected from the full novel; membership in the 96 is a downstream human decision, not a property of any row. `paper_final_sample=false` on every row makes this explicit.
- It is **not** a (preposition, noun) type-level analysis. Two occurrences of `im Haus` in the same sentence are two datapoints.
- `FILTER_CONTRACTED_123` and `FILTER_PP` are extraction QA filters; they appear as the `author_resource_match` source column for visibility but do not gate membership in the annotation surface (the 13-item paired inventory is applied later, at eligible-pool build — see `FULL_NOVEL_SAMPLING.md`).
- It is **not** an automatic translation or definiteness classifier. EN/ZH counterpart spans and form labels are filled by a human annotator in the trilingual pair CSV (`ANNOTATION_CSV.md`). The builder never auto-translates or auto-classifies; `crosslingual_map` proposals live in a separate JSONL and never auto-populate annotation fields.
- It does **not** assign weak/strong definiteness, uniqueness, or familiarity. Only observable surface form is recorded.

### The 13-item operational inventory

The pool is restricted to canonical prepositions that appear in **both** of the author's published lists: the contracted `CONTRACTED` list (27 surface forms) and the uncontracted `PREPOSITIONS` list (13 forms). The intersection is 13 canonical prepositions: `an, auf, aus, bei, durch, für, gegen, hinter, in, über, unter, von, zu`. Contracted `ums` and `vorm` exist in German but are excluded because `um` and `vor` are not in `PREPOSITIONS`. This 13-item set is the operational paired inventory; it is not a universal German-grammar inventory.

The contraction table in `src/hp_corpus/step4.py` is pinned to the vendored `time-in-translation/conll-extractor` commit recorded in `vendor/conll-extractor.commit`. Two regression tests guard parity: a fixture-based test (runs on every checkout) and a vendor-based test (runs only when the vendor clone is present).

### Sentence alignment

The sentence alignments feeding Step 4 are produced by `src/hp_corpus/align.py` — **LaBSE** embeddings (local checkout under `models/LaBSE`, identity pinned in `config/embedding_models.yaml`) + a global banded DP with a lexical prior. The selection of LaBSE over e5-base and bge-m3, and the gold-standard audit behind it, are documented in [`ALIGNMENT_MODEL_DECISION.md`](./ALIGNMENT_MODEL_DECISION.md). The production pairs are **DE–EN and DE–ZH per chapter** (`scripts/run_alignments_v2.py`, Ch.1–17 by default; EN–ZH exists only as a `--diagnostics` extra with no downstream consumer).

Machine alignment quality is separated from annotation: alignment records scoring below the review threshold keep `method="embedding_dp"` and carry `needs_review=true` (nothing machine-made is ever labelled `manual`); on the annotation side the human's per-side `*_alignment_confidence` value (`not_aligned`) is the signal that the delivered context is not the right translation.

Misalignment and omission are distinct axes in the annotation CSV: `omitted` + `high`/`medium`/`low` confidence = the correctly-delivered context genuinely contains no counterpart (true omission); `omitted` + `not_aligned` = retrieval failed and the row needs repair, not an omission by the translator (see `ANNOTATION_CSV.md`).

### Builder input discipline

The full-novel builder takes its chapter list from the extraction manifest (`data/extracted/full_novel/manifest.json`) and fails closed on any gap: a missing manifest, a requested `(chapter, form)` without an `ok` / `zero_hits_ok` entry, or missing/empty segmented or alignment inputs raises before any output is written. The builder never emits a partial master.

### Module layout

- `src/hp_corpus/step4.py` — pure-Python candidate/master builder and TSV writer.
- `scripts/run_full_novel_german_extraction.py` — paper-faithful Ch.1–17 extraction + manifest.
- `scripts/build_full_novel_annotation.py` — builds the machine master TSV from extraction + segmented + alignment inputs.
- `scripts/build_annotation_csv.py` / `scripts/validate_annotation_csv.py` — the annotator-facing CSV and its return gate.
- `tests/test_step4.py`, `tests/test_full_novel_annotation.py` — synthetic-fixture tests; no novel text.

The annotator-facing surface is the trilingual pair CSV built from the
machine master — see [`ANNOTATION_CSV.md`](./ANNOTATION_CSV.md) and
[`FULL_NOVEL_SAMPLING.md`](./FULL_NOVEL_SAMPLING.md). The older Ch.1–3
10+10 pilot selector (`scripts/build_step4_annotation_pack.py`) remains
available for method piloting but is not the production path.

### Alignment record schema

Each alignment record is language-generic: `src`/`tgt` side lists plus `src_lang`/`tgt_lang`, `method` ∈ {`embedding_dp`, `manual`} (`manual` is reserved for genuinely human-created records and never auto-assigned), and a `needs_review` flag. Records written by the pre-rename serializer (`en`/`zh` field names for every pair, `vecalign_labse`) still parse — the schema normalizes them, and `src/hp_corpus/step4.py` identifies sides by segment-ID language prefix either way.

### Occurrence identity

`datapoint_id` = `dp_ch{NN}_{de_sentence_id}_t{token_start}-{token_end}`. Two occurrences in the same sentence with the same `(preposition, noun)` get different IDs because their token ranges differ.

### Source-row immutability

Each master row carries `source_row_sha256` over the immutable source columns (the retrieval-context columns are deliberately outside the hash, so the context window can be re-tuned without changing row identity). The returned-CSV validator binds rows by that hash **and** re-derives every machine column from the master for a cell-by-cell comparison — an edited context or German sentence fails closed even when the hash is untouched. Editable columns (annotation fields) live in separate columns and are not covered by the hash.

### Annotation schema

See [`ANNOTATION_CSV.md`](./ANNOTATION_CSV.md) for the annotator-facing
trilingual pair CSV (column-by-column guide and vocabularies).

## Output discipline

Per-PP detail goes only to TSV files under `data/extracted/` (gitignored). The machine master and annotation CSV go to `data/derived/` (also gitignored — contains novel text). **Stdout and summary JSON carry aggregate counts only** — never noun lemmas, surface forms, segment IDs, or sentence text. This is verified by:

- `tests/test_run_paper_extractor.py` — extractor stdout guard.
- `tests/test_step4.py::test_summary_no_token_text` — Step 4 summary guard.
- `tests/test_annotation_csv.py` / `tests/test_annotation_csv_validation.py` — annotation-CSV builder/validator stdout guards.
