# Step 4 — Annotation Guide

This document describes how to fill in the human-annotation columns of
`data/derived/step4/ch1_3_pilot_20.tsv`. Read it once before annotating.

## Goal of the annotation

For each row, you have:

- A **German PP occurrence** — a specific token range inside one German
  sentence (`de_pp_surface`, with `de_token_start`/`de_token_end`
  coordinates).
- The **English sentence(s)** that the alignment paired with that German
  sentence (`en_aligned_text`).
- The **Mandarin sentence(s)** that the alignment paired with that German
  sentence (`zh_aligned_text`).

Your job: in each aligned sentence, identify the **expression that
corresponds to the German PP** and record:

1. the relation between the German PP and its counterpart,
2. the surface text of the counterpart,
3. character ranges that pick the counterpart out of the aligned text,
4. the observable form of the counterpart,
5. your confidence.

Do **not** classify definiteness as weak/strong, assign semantic
uniqueness/familiarity labels, or fill in anything that requires
interpretation beyond what is visible on the page. We are recording
observable linguistic form only.

## Source columns vs editable columns

The TSV has two zones:

- **Source columns** (everything from `datapoint_id` through
  `source_row_sha256`). Do **not** edit these. The validator computes a
  SHA-256 over them and will reject the file if any value changes.
- **Editable columns** (everything after `source_row_sha256`). These are
  yours to fill in.

If a source column looks wrong (e.g. the German sentence text is
malformed, or the EN/ZH alignment is obviously incorrect), do not patch
it in the TSV — note the problem in `general_notes` and flag the row in
`annotation_status` instead.

## Controlled vocabularies

### `*_alignment_relation` (EN and ZH)

| Value | Meaning |
|---|---|
| `direct` | The aligned sentence contains an expression that translates the German PP directly (e.g. `im Haus` ↔ `in the house`). |
| `paraphrase` | The aligned sentence expresses the same meaning without a clean PP-to-PP mapping (e.g. a possessive construction where the German uses a PP). |
| `pronominal` | The counterpart is a pronoun (`darauf`, `there`, `it`, `他`). |
| `omitted` | The German PP has no counterpart in the aligned sentence. Must be paired with `*_form = omitted` and **no** span/ranges. |
| `uncertain` | You cannot decide between the above. **Requires** a note in `*_notes`. |

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

### `*_confidence`

`high` · `medium` · `low`

### `annotation_status` (whole row)

| Value | Meaning |
|---|---|
| (blank) | Not yet started. |
| `unstarted` | Same as blank, explicit. |
| `in_progress` | You have begun but not finished. |
| `complete` | For **both** languages you have set `*_alignment_relation`, `*_form`, and `*_confidence`. The validator enforces this. |

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

1. `uv run python scripts/build_step4_annotation_pack.py` regenerates
   `data/derived/step4/ch1_3_pilot_20.tsv` from the latest extraction +
   alignment outputs. This **overwrites** any annotations in the file,
   so finish or stash your in-progress TSV before re-running.
2. Open the TSV in your editor of choice (any tab-aware TSV editor works;
   spreadsheet GUIs are fine if you set the column types to "text").
3. For each row, work through the EN side, then the ZH side.
4. Run `uv run python scripts/validate_step4_annotations.py
   data/derived/step4/ch1_3_pilot_20.tsv` frequently. The validator
   prints only rule names and row indices — never source text — so its
   output is safe to share.
5. When every row has `annotation_status = complete` and the validator
   returns `OK`, the pilot is ready for downstream analysis.

## What Step 4 deliberately does not produce

- No weak/strong definiteness classification.
- No semantic uniqueness / familiarity / specificity labels.
- No automatic counterpart detection or auto-form classification.
- No descriptive statistics, MDS, or visualization. (Those are deferred
  to Step 5.)
- No generalisation beyond Ch.1–3 in this phase.
