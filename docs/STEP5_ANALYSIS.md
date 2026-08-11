# Step 5 — paper-category derivation and analysis

Documents how a gold annotation TSV becomes the aggregate tables that
face the paper. The pipeline runs in two CLI stages so the
detailed-row projection is inspectable independently of the aggregate
roll-up.

```
gold TSV  ──▶  derive_paper_categories.py  ──▶  derived-category TSV
                                                        │
                                                        ▼
                                              analyze_step5.py
                                                        │
                                                        ▼
                            summary.json + distributions + cross-tabs + review list
```

## Eligibility

A row enters the analysis-ready denominator iff:

  * `de_candidate_decision = "include"`
  * `annotation_status = "complete"`
  * `adjudication_status` is blank or `"adjudicated"` (never
    `pending` / `disputed`)
  * both `en_aligned_text` and `zh_aligned_text` are non-empty

The caller is responsible for ensuring the gold TSV passes
`scripts/validate_step4_annotations.py --require-complete`. The
in-process predicate `is_analysis_ready()` in
`src/hp_corpus/step5.py` mirrors that rule for clarity, but the
validator is the authoritative gate.

Excluded, blocked, uncertain, or unresolved rows are **not** in the
denominator. Every percentage below uses the analysis-ready total as
its denominator.

## Category derivation

Coarse paper-facing categories are **derived**, never overwritten.
Fine annotator labels are preserved verbatim in the detailed TSV.

| Side | Coarse category | Fine labels that map to it |
|---|---|---|
| DE | `contracted` | (mirrors `de_form`) |
| DE | `uncontracted` | (mirrors `de_form`) |
| EN | `definite` | `definite` |
| EN | `bare_singular` | `bare_singular` |
| EN | `demonstrative` | `demonstrative` |
| EN | `other` | `indefinite`, `possessive`, `pronoun`, `proper_name`, `other`, `omitted`, `uncertain` |
| ZH | `bare` | `bare` |
| ZH | `demonstrative` | `demonstrative` |
| ZH | `other` | `numeral_classifier`, `possessive`, `pronoun`, `proper_name`, `other`, `omitted`, `uncertain` |

The fine labels that roll up to `other` are enumerated in the
aggregate summary's `source_labels_rolled_up` block, so nothing
disappears silently from the denominator. If a paper-facing table
collapses EN `other` into a single cell, the underlying counts are
recoverable from the detailed TSV (`en_form_fine`) and from the
summary JSON.

## Outputs

### `derive_paper_categories.py`

Reads a gold TSV and writes one row per analysis-ready datapoint:

```
datapoint_id
chapter
de_form
de_paper_category       # contracted | uncontracted
en_form_fine            # the annotator's fine label
en_paper_category       # definite | bare_singular | demonstrative | other
zh_form_fine            # the annotator's fine label
zh_paper_category       # bare | demonstrative | other
```

### `analyze_step5.py`

Reads the derived-category TSV and writes:

  * `summary.json` — top-level aggregate counts and the full
    distribution / cross-tab / roll-up objects.
  * `de_distribution.json`, `en_distribution.json`,
    `zh_distribution.json` — per-category count + row-percentage.
  * `de_x_zh_table.json`, `de_x_en_table.json` — two-way count +
    row-percentage tables. Row percentages are taken against the
    row-category subtotal; each row sums to ~100%. A `_denominators`
    block records each row category's total count and its share of the
    analysis-ready total.
  * `uncontracted_mandarin_bare_review.tsv` — one row per datapoint
    in the `uncontracted + Mandarin bare` cell. Columns are
    `datapoint_id` and `chapter` only — no sentence text. The user
    opens the gold TSV to read each row's full content.

## Deliberately not implemented

  * **Weak / strong definiteness labels.** The paper assigns
    weak/strong readings; this project records only observable surface
    form. Adding weak/strong would require a separate annotation pass
    and is out of scope for the current infrastructure.
  * **MDS visualization.** Category-distance definition is not
    finalized. The aggregate tables here are the input layer for any
    future MDS work, but the distance computation itself is deferred
    until the category-distance definition is settled.
  * **Auto-classification of EN/ZH counterparts.** The form labels
    are human annotations; this pipeline never auto-fills them.

## Privacy discipline

Stdout carries aggregate counts and output paths only. Datapoint IDs
appear in the review-list TSV (operational identifiers, not novel
text) but never on stdout. The detailed and gold TSVs carry sentence
text under `data/derived/` (gitignored); the review list does not.
