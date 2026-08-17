# Full-novel German extraction and the eligible pool

Documents the pipeline from full-novel extraction (Ch.1–17) through the
German review to the formal eligible pool for the Bremmers
reproduction. The governing design is
`docs/ANALYSIS_SAMPLE_DESIGN.md`; this page describes the implemented
behavior.

Two scripts are involved:

  * `scripts/run_full_novel_german_extraction.py` — paper-faithful PP
    extraction with a per-chapter manifest. The single implementation
    of the paper's extractor (`src/hp_corpus/german_extraction.py`);
    emits `parse_block_id` + `source_segment_id` per row. The older
    `run_paper_extractor.py` is a deprecated thin wrapper that
    delegates here (kept only for invocation compatibility).
  * `scripts/build_eligible_pool.py` — the formal post-review
    selection path (`src/hp_corpus/sampling.py`). Runs **once** on a
    completed German review. The pool is a **derived** layer: the
    complete trilingual annotation table is retained permanently, and
    any later analysis set or proportional sample is derived from it —
    nothing downstream overwrites the annotation table.

## Extraction

The extraction algorithm is identical to the paper's
`time-in-translation/conll-extractor` (vendored at
`vendor/conll-extractor/`). For each chapter 1..17:

  * **Contracted PPs** — tokens in `CONTRACTED` (`im`, `am`, `zum`, …)
    whose head is an `NN` → extract `(prep, noun)`.
  * **Uncontracted PPs** — `PREPOSITIONS` tokens whose same-head
    dependent is an `ART` in `DETERMINERS` (`der`/`dem`/`den`/…) →
    extract `(prep, det, noun)`.

Each row carries occurrence coordinates (`prep_token_id`,
`det_token_id`, `noun_token_id`, `pp_token_start`, `pp_token_end`,
`pp_surface`) and the structural-gate metadata of the determiner token
(`det_xpos`, `det_deprel`; `-` on contracted rows, which have no det
token). `FILTER_CONTRACTED_123` and `FILTER_PP` results are written to
the `in_filter` column as **QA metadata only** — they never gate the
manifest's row count or pool membership, and they never appear on the
human annotation surface.

### Path conventions

All paths use `{chapter:02d}` zero-padding:

  * Input: `data/parsed/hp1_de_ch{NN}_nomwt.conllu`
  * Output: `data/extracted/full_novel/hp1_de_ch{NN}_{contracted,uncontracted}.tsv`

The extractor guarantees a TSV for every (chapter, form) pair — a
zero-hit form is a header-only file. The pool CLI relies on that: it
requires the **complete** extraction set and treats a missing file as
a hard failure.

### Fail-closed rules

  * Missing parsed input → exit non-zero by default
    (`MISSING_PARSED_INPUT`). `--allow-missing` opts into
    manifest-and-continue behavior, but only for missing inputs.
  * Empty parsed input (file exists but yields zero sentences) → exit
    non-zero always. An empty file is upstream corruption, not a
    missing one.
  * Block-provenance violations (unmigrated, duplicate, inconsistent,
    or content blocks without `# sent_id`) → exit non-zero.
  * Legitimate zero hits → `status="zero_hits_ok"`; a header-only TSV
    is written.

### Manifest

A per-chapter manifest entry captures `chapter`, `form`,
`parsed_path_sha256`, `parsed_path_size`,
`parsed_path_sentence_count`, block-provenance counts,
`extractor_version`, `status`, `row_count`, and (for error statuses)
`error`. The manifest is the operational record of what ran.

## German review + EN/ZH annotation: one trilingual CSV

The review covers the **full machine candidate pool** (the master's
1,392 rows), and it happens in the same single-pass document as the
EN/ZH counterpart marking: the **trilingual pair CSV**
(`annotation_pairs.csv`, built by `scripts/build_annotation_csv.py`,
one row per occurrence with the German PP, its sentence, and the
aligned English/Chinese contexts). The annotator fills, per row:

- `de_valid` — `include` / `exclude` (blank or `uncertain` = review not
  finished; the file is rejected until every row is decided);
- `de_corrected_lemma` — blank = machine lemma stands; non-blank =
  correction. Single formal lemma field, no priority chain;
- `de_exclusion_reason` (+ `de_notes`) when excluding;
- `en_/zh_counterpart`, `en_/zh_form`, `en_/zh_alignment_confidence`,
  `en_/zh_notes` — the counterpart marking (vocabularies and coupling
  rules in `docs/ANNOTATION_CSV.md`, enforced on return by
  `scripts/validate_annotation_csv.py`).

The machine-filled `id` and `row_hash` columns bind each returned row
to the exact master state.

## The eligible pool

`build_eligible_pool.py` reads the complete extraction set, the machine
master, and the returned annotator CSV (its `de_valid` column is the
final German decision), and applies the formal rule
(`docs/ANALYSIS_SAMPLE_DESIGN.md` §3):

1. **Preposition inventory** — canonical preposition ∈ the paper's
   13-item paired inventory. `um`/`vor` contractions are out (no
   uncontracted counterpart in the paper's lists).
2. **Structural gate** (uncontracted only) — `det_xpos == ART` and
   `det_deprel == det`. Wrong tags exclude the row; **missing**
   metadata is a hard failure of the whole run. Contracted rows are
   exempt.
3. **German review** — `exclude` rows are out; the review must be
   complete (see fail-closed rules below).
4. **Eligibility** (the paper's §2.2.1 "the same preposition and noun"):
   - reviewed-include **uncontracted**, Ch.1–17 →
     `uncontracted_all_chapters`;
   - reviewed-include **contracted**, Ch.1–3 → `contracted_ch1_3`;
   - reviewed-include **contracted**, Ch.4–17, only when its
     `(canonical_preposition, head_lemma)` pair — corrected lemma when
     non-blank, else machine lemma — occurs in the reviewed-include
     uncontracted set → `contracted_ch4_17_pair_matched`
     (otherwise `contracted_ch4_17_no_uncontracted_counterpart`).

Occurrence identity is `(chapter, source_segment_id, parse_block_id,
pp_token_start, pp_token_end)`. Exact duplicates collapse
deterministically to the first row (counted in the summary); the same
identity with conflicting core fields fails the run.

### Fail-closed rules (no partial output is ever written)

  * the run always reads the **complete** Ch.1–17 extraction set (34
    files — there is no chapter-subset option); any missing
    (chapter, form) file → `MISSING_EXTRACTION_INPUT`;
  * extraction header missing required columns →
    `EXTRACTION_SCHEMA_MISMATCH`;
  * duplicate `datapoint_id` in the machine master →
    `MASTER_DUPLICATE_ID`;
  * the extraction set and the master must correspond as **sets**:
    every inventory-eligible extraction row present in the master, and
    every master row present in the extraction (file existence is not
    file content — an extraction regression that drops rows must fail,
    not silently shrink the pool) → `EXTRACTION_MASTER_MISMATCH`;
  * duplicate `id` in the returned CSV, an id absent from the master,
    or a master row the returned CSV doesn't cover →
    `REVIEW_OVERLAY_INVALID`;
  * `row_hash` ≠ the master's hash for the same id →
    `REVIEW_OVERLAY_INVALID`;
  * any `de_valid` other than `include`/`exclude` (blank or `uncertain`
    = review not finished) → `REVIEW_OVERLAY_INVALID` from the CLI and
    `IncompleteReviewError` from the core rule itself — the public
    selector is fail-closed on its own, not only behind the CLI;
  * uncontracted row lacking structural metadata →
    `STRUCTURAL_METADATA_MISSING`;
  * identity conflict → `OCCURRENCE_IDENTITY_CONFLICT`;
  * existing outputs without `--force-output` → `OUTPUT_EXISTS`.

### Usage

```
uv run python scripts/build_eligible_pool.py \
    --extraction-dir data/extracted/full_novel \
    --master-tsv data/derived/step4/full_novel_annotation_master.tsv \
    --review-csv <returned annotation_pairs.csv> \
    --out-dir data/derived/eligible_pool
```

Outputs (gitignored — they carry corpus coordinates and lemmas):

  * `eligible_pool.tsv` — the eligible rows only, with form,
    canonical preposition, machine + corrected lemma, pool reason,
    identity columns, and source hash;
  * `eligible_pool_summary.json` — aggregate counts on the actual
    inputs: `extracted_total`, `duplicate_rows_collapsed`,
    `automatically_excluded` (inventory / structural),
    `human_review` (included / excluded), `eligible_pool`
    (`uncontracted_all_chapters`, `contracted_ch1_3`,
    `contracted_ch4_17_pair_matched`, `eligible_total`),
    `contracted_ch4_17_no_uncontracted_counterpart`, `by_reason`,
    plus `schema_version` and SHA-256 `input_hashes`.

Stdout carries aggregate counts only — never lemmas, datapoint or
segment IDs, surface forms, or sentence text.

## Data readiness

Extraction is complete and verified (34/34 chapter-form entries `ok`;
1,394 rows; 668/668 uncontracted rows carry `ART`+`det`). The German
review has not been run, so the eligible pool cannot be built yet —
running the CLI against the current state fails closed with
`REVIEW_CSV_ABSENT` / `REVIEW_OVERLAY_INVALID` and writes nothing. The
annotator CSV itself, however, is ready: it is machine-filled from the
master and carries no human judgment until the annotators fill it in
(`data/derived/annotation/annotation_pairs.csv`, gitignored).
Realized pool counts appear only after the completed review's single
run.
