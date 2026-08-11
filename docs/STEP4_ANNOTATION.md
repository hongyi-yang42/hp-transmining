# Step 4 — Annotation Guide

This document describes how to fill in the human-annotation columns of
`data/derived/step4/ch1_3_full_annotation.tsv` (the **Ch.1–3 annotation
target**). Read it once before annotating.

A separate file, `ch1_3_pilot_20.tsv`, is the original 10+10 method pilot
— a strict subset kept for method-record purposes. Annotate the
annotation-target TSV going forward.

## What this TSV is — and is not

- It is a **Ch.1–3 paper-eligible annotation pool**: every extracted
  German PP whose canonical preposition is in the paper's 13-item paired
  inventory (`an, auf, aus, bei, durch, für, gegen, hinter, in, über,
  unter, von, zu`).
- It is **not** the paper's final 96 trilingual contexts. The 96 are
  hand-selected from the full novel; no row in this TSV is part of the
  final 96 until a downstream human decision says so. `paper_final_sample
  = false` on every row makes this explicit.
- It does **not** assign weak/strong definiteness, uniqueness, or
  familiarity. Those are deferred to a later semantic pass.

## Goal of the annotation

For each row, you have:

- A **German PP occurrence** — a specific token range inside one German
  sentence (`de_pp_surface`, with `de_token_start`/`de_token_end`
  coordinates).
- The **English sentence(s)** that the alignment paired with that German
  sentence (`en_aligned_text`).
- The **Mandarin sentence(s)** that the alignment paired with that German
  sentence (`zh_aligned_text`).

Your job, in three layers:

1. **German candidate confirmation** — is this row actually a German PP
   we want to analyze?
2. **Alignment QC** — for each target language, is the aligned context
   the right one?
3. **Counterpart annotation** — if both alignments are confirmed,
   identify the expression that corresponds to the German PP and record
   its observable form.

Record **observable form only**. Do not classify definiteness as
weak/strong, assign semantic uniqueness/familiarity labels, or fill in
anything that requires interpretation beyond what is visible on the page.

## Source columns vs editable columns

The TSV has two zones:

- **Source columns** (everything from `datapoint_id` through
  `source_row_sha256`). Do **not** edit these. The validator computes a
  SHA-256 over them and will reject the file if any value changes.
- **Editable columns** (everything after `source_row_sha256`). These are
  yours to fill in. The builder pre-fills `{lang}_alignment_qc =
  assumed_ok` on every row; everything else starts blank.

If a source column looks wrong (e.g. the German sentence text is
malformed, or the EN/ZH alignment is obviously incorrect), do not patch
it in the TSV — see *When alignment is wrong* below.

## The 13-item operational inventory

The pool is restricted to canonical prepositions that appear in **both**
of the author's published lists:

- contracted forms (`zum`, `im`, `am`, `vom`, … — 27 surface forms)
- uncontracted forms (`zu`, `in`, `an`, `von`, … — 13 forms)

The intersection is **13 canonical prepositions**: `an, auf, aus, bei,
durch, für, gegen, hinter, in, über, unter, von, zu`. Contracted `ums`
and `vorm` exist in German but are excluded because `um` and `vor` are
not in the author's `PREPOSITIONS` list — they cannot pair with an
uncontracted form. This 13-item set is the operational paired inventory;
it is not a universal German-grammar inventory.

## German candidate confirmation

Three editable columns control whether a row stays in the analytic
dataset:

| `de_candidate_decision` | `de_exclusion_reason` | `de_candidate_notes` | EN/ZH annotation |
|---|---|---|---|
| (blank) | must be blank | must be blank | not yet started |
| `include` | **must be blank** | optional | required for completion |
| `exclude` | **must be filled** (controlled vocab) | optional | **not required** |
| `uncertain` | must be blank | **must be filled** | optional; row cannot be analysis-ready |

`de_exclusion_reason` controlled vocabulary:

| Value | Use when |
|---|---|
| `not_target_pp` | The extracted span is not actually a PP the paper is after (e.g. an idiomatic construction). |
| `not_definite` | The PP exists but its definiteness is structurally outside the paper's scope (rare; record details in `de_candidate_notes`). |
| `extraction_error` | The extractor mis-fired (wrong head, wrong span, parser error). |
| `duplicate` | Same occurrence as another row (record the duplicate's `datapoint_id` in notes). |
| `other` | None of the above (always record details in `de_candidate_notes`). |

The `author_resource_match` source column is **not** used for exclusion.
It is extraction-QA visibility only — a row that matches the paper's
filter is not automatically included, and a row that does not match is
not automatically excluded.

## Alignment QC

Two editable columns per language record whether the aligned context is
the right one:

| `{lang}_alignment_qc` | Meaning |
|---|---|
| `assumed_ok` | Builder default. The machine alignment is taken at face value; not yet checked. |
| `confirmed` | You have verified the alignment is correct (either located the counterpart or verified its absence). Required to mark `annotation_status=complete`. |
| `incorrect` | The alignment is wrong. Routes the row to the realignment queue. Blocks completion. |
| `uncertain` | You are unsure whether the alignment is right. Routes the row to the realignment queue. Requires `{lang}_alignment_notes`. |

`{lang}_alignment_notes` is free text and stays separate from
`{lang}_notes` (which covers span/form/relation decisions).

### Misalignment vs omission

These are distinct:

- **`{lang}_alignment_relation = omitted`** means the
  **correctly-aligned** target context genuinely contains no counterpart.
  Must be paired with `{lang}_form = omitted` and no span/ranges.
- **Misalignment** means the wrong target sentence was paired with the
  German source. Flag it via `{lang}_alignment_qc = incorrect` (and add
  a note). Do **not** mark the relation as `omitted`.

The validator rejects `{lang}_alignment_qc = incorrect` paired with
`{lang}_alignment_relation = omitted` as `MISALIGNED_NOT_OMISSION`. If
the alignment is wrong, the omission judgement is meaningless — fix the
alignment first.

### When alignment is wrong

Alignment correction is a **source-data rebuild**, not an editable-cell
fix. Correcting an alignment changes the source-side `{lang}_aligned_text`
and `source_row_sha256` on every affected row, so in-place correction
breaks the SHA-256 contract and leaves char ranges indexing stale text.

Workflow:

1. Flag bad-alignment rows by setting `{lang}_alignment_qc` to
   `incorrect` or `uncertain` and writing `{lang}_alignment_notes`.
2. Run `scripts/extract_realignment_queue.py` to collect flagged rows
   into `data/derived/step4/realignment_queue.tsv`.
3. Fix the underlying alignment (out of scope for this pass — the
   sentence-alignment algorithm in `src/hp_corpus/align.py` is
   untouched).
4. Re-run `scripts/build_ch1_3_full_annotation.py` to produce a new
   annotation-target TSV. The `datapoint_id` of each row is stable
   (it encodes only the DE sentence id and token range); the alignment
   fields and source hash change.
5. A follow-up migration script (not yet implemented) will carry over
   annotation fields from the old TSV to the new one by `datapoint_id`,
   skipping the side whose alignment changed so you re-do only that
   side.

## Counterpart annotation

Once the German candidate is `include` and both alignments are
`confirmed`, annotate the counterpart:

1. the relation between the German PP and its counterpart,
2. the surface text of the counterpart,
3. character ranges that pick the counterpart out of the aligned text,
4. the observable form of the counterpart,
5. your confidence.

### `{lang}_alignment_relation`

| Value | Meaning |
|---|---|
| `direct` | The aligned sentence contains an expression that translates the German PP directly (e.g. `im Haus` ↔ `in the house`). |
| `paraphrase` | The aligned sentence expresses the same meaning without a clean PP-to-PP mapping (e.g. a possessive construction where the German uses a PP). |
| `pronominal` | The counterpart is a pronoun (`darauf`, `there`, `it`, `他`). |
| `omitted` | The German PP has no counterpart in the **correctly-aligned** sentence. Must be paired with `{lang}_form = omitted` and **no** span/ranges. Requires `{lang}_alignment_qc = confirmed`. |
| `uncertain` | You cannot decide between the above. **Requires** a note in `{lang}_notes`. Blocks `annotation_status=complete`. |

### `en_form`

| Value | Use when the counterpart is… |
|---|---|
| `definite` | a singular or plural noun marked with a definite article (`the`). |
| `bare_singular` | a singular noun with no determiner (`to bed`, `at home`). |
| `demonstrative` | marked with a demonstrative (`this`, `that`, `those`). |
| `indefinite` | marked with `a`/`an`/`some`/`any`. |
| `possessive` | marked with a possessive (`my`, `Harry's`, `their`). |
| `pronoun` | a pronoun (`him`, `it`, `them`). |
| `proper_name` | a proper-name phrase (`Hogwarts`, `Privet Drive`). |
| `other` | none of the above (record details in `en_notes`). |
| `omitted` | no counterpart (paired with `en_alignment_relation=omitted`). |
| `uncertain` | cannot decide (record reasoning in `en_notes`). |

### `zh_form`

| Value | Use when the counterpart is… |
|---|---|
| `bare` | a bare noun with no determiner or classifier (`家`, `猫`). |
| `demonstrative` | marked with `这`/`那` (optionally with a classifier). |
| `numeral_classifier` | marked with a numeral-classifier phrase (`一只`, `三个`). |
| `possessive` | marked with a possessive (`我的`, `哈利的`). |
| `pronoun` | a pronoun (`他`, `它`, `它们`). |
| `proper_name` | a proper-name phrase (`霍格沃茨`, `女贞路`). |
| `other` | none of the above (record details in `zh_notes`). |
| `omitted` | no counterpart (paired with `zh_alignment_relation=omitted`). |
| `uncertain` | cannot decide (record reasoning in `zh_notes`). |

### `{lang}_confidence`

`high` · `medium` · `low`

### `annotation_status` (whole row)

| Value | Meaning |
|---|---|
| (blank) | Not yet started. |
| `unstarted` | Same as blank, explicit. |
| `in_progress` | You have begun but not finished. |
| `complete` | For **both** languages: `{lang}_alignment_qc = confirmed`, `{lang}_alignment_relation`, `{lang}_form`, and `{lang}_confidence` are all set, and the spans (when applicable) are filled. The validator enforces this. |

### `adjudication_status`

Leave blank during solo annotation. Use `pending` / `adjudicated` /
`disputed` only when a second annotator reviews.

## Character ranges

`en_char_ranges` and `zh_char_ranges` are JSON arrays of `[start, end]`
pairs. Each pair is a half-open interval over `en_aligned_text` /
`zh_aligned_text`. Concatenating the substrings — joined with a single
space when there is more than one pair — must reproduce `*_span_text`
exactly.

Examples (assume `en_aligned_text = "synth EN one"`, which is 12 chars:
`synth EN one`):

| What you want | `en_span_text` | `en_char_ranges` |
|---|---|---|
| whole sentence | `synth EN one` | `[[0,12]]` |
| just the first word | `synth` | `[[0,5]]` |
| two discontinuous words | `synth one` | `[[0,5],[9,12]]` |

Rules:

- Offsets are zero-based and `end` is exclusive.
- For discontinuous counterparts, list each contiguous span as a separate
  `[start, end]` pair. The validator joins them with a single space.
- If `*_alignment_relation = omitted`, leave `*_span_text` and
  `*_char_ranges` blank and set `*_form = omitted`.
- If `*_alignment_relation = uncertain`, you may leave the span blank but
  you must write a note.

## Workflow

1. `uv run python scripts/build_ch1_3_full_annotation.py` regenerates
   `data/derived/step4/ch1_3_full_annotation.tsv` from the latest
   extraction + alignment outputs. This **overwrites** any annotations
   in the file, so finish or stash your in-progress TSV before re-running.
2. Open the TSV in your editor of choice (any tab-aware TSV editor works;
   spreadsheet GUIs are fine if you set the column types to "text").
3. For each row:
   - Set `de_candidate_decision` (`include` / `exclude` / `uncertain`).
     Excluded rows skip the rest; uncertain rows need notes and stay out
     of the analytic dataset until adjudicated.
   - For each language, check the alignment and set `{lang}_alignment_qc`
     to `confirmed`, `incorrect`, or `uncertain`.
   - If `de_candidate_decision = include` and both alignments are
     `confirmed`: fill in `{lang}_alignment_relation`, `*_span_text`,
     `*_char_ranges`, `*_form`, `*_confidence`.
   - Set `annotation_status` to `in_progress` while working, `complete`
     once both sides are done.
4. Run `uv run python scripts/validate_step4_annotations.py
   data/derived/step4/ch1_3_full_annotation.tsv --annotation-pool`
   frequently. The validator prints only rule names and row indices —
   never source text — so its output is safe to share.
5. When you believe the dataset is research-complete, add
   `--require-complete`. The validator enforces every non-excluded row
   is `include + complete + confirmed` on both sides, and prints a
   rollup of completed / excluded / blocked / uncertain / pending /
   adjudicated counts.
6. Run `uv run python scripts/extract_realignment_queue.py
   data/derived/step4/ch1_3_full_annotation.tsv` to surface any rows
   flagged `incorrect` / `uncertain` for source-alignment work.

## What Step 4 deliberately does not produce

- No weak/strong definiteness classification.
- No semantic uniqueness / familiarity / specificity labels.
- No automatic counterpart detection or auto-form classification.
  (`crosslingual_map` proposes PP-shaped candidates only — it cannot
  surface NP, pronominal, paraphrastic, or omitted counterparts, and
  its proposals never auto-populate gold annotation fields.)
- No descriptive statistics, MDS, or visualization. (Those are deferred
  to Step 5.)
- No generalisation beyond Ch.1–3 in this phase.
