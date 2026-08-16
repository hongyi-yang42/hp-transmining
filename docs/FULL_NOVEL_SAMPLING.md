# Full-novel German extraction and sampling

Documents the methodology for expanding the Ch.1–3 method pilot to the
full novel (Ch.1–17), producing the analysis-target sample for the
Bremmers reproduction.

Two scripts are involved:

  * `scripts/run_full_novel_german_extraction.py` — paper-faithful PP
    extraction with a per-chapter manifest. The single implementation
    of the paper's extractor (`src/hp_corpus/german_extraction.py`);
    emits `parse_block_id` + `source_segment_id` per row. The older
    `run_paper_extractor.py` is a deprecated thin wrapper that
    delegates here (kept only for invocation compatibility).
  * `scripts/build_full_novel_sampling_ledger.py` — applies the U /
    C_early / C_late sampling rule and writes the ledger + target TSVs.

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
`pp_surface`) so two PPs that share `(preposition, noun)` within one
sentence can be told apart. `FILTER_CONTRACTED_123` and `FILTER_PP`
results are written to the `in_filter` column as **QA metadata only**;
they do not gate the manifest's row count or the sampling ledger's
membership.

### Path conventions

All paths use `{chapter:02d}` zero-padding:

  * Input: `data/parsed/hp1_de_ch{NN}_nomwt.conllu`
  * Output: `data/extracted/full_novel/hp1_de_ch{NN}_{contracted,uncontracted}.tsv`

This fixes two full-novel blockers in the older
`run_paper_extractor.py`: the `f"hp1_de_ch0{ch}_nomwt.conllu"` pattern
that mis-padded Ch.10–17 to `ch010`/`ch017`, and the silent skip of
missing parsed inputs.

### Fail-closed rules

  * Missing parsed input → exit non-zero by default
    (`MISSING_PARSED_INPUT`). `--allow-missing` opts into
    manifest-and-continue behavior, but only for missing inputs.
  * Empty parsed input (file exists but yields zero sentences) → exit
    non-zero always. An empty file is upstream corruption, not a
    missing one.
  * Extraction error (pyconl syntax error, etc.) → exit non-zero.
  * Legitimate zero hits (input exists, nonempty, extraction
    succeeded, row count = 0) → `status="zero_hits_ok"`; an empty TSV
    with the schema header is written.

### Manifest

A per-chapter manifest entry captures `chapter`, `form`,
`parsed_path_sha256`, `parsed_path_size`,
`parsed_path_sentence_count`, `extractor_version`,
`status ∈ {"ok", "zero_hits_ok", "missing_input", "empty_input",
"extraction_error"}`, `row_count`, and (for error statuses) `error`.
The manifest is the operational record of what ran.

## Sampling

The sampling rule is implemented as a pure, deterministic function in
`src/hp_corpus/sampling.py`:

  * **U** — every inventory-eligible, German-reviewed `include`
    uncontracted occurrence from Ch.1–17.
  * **C_early** — every inventory-eligible, German-reviewed `include`
    contracted occurrence from Ch.1–3.
  * **C_late** — every inventory-eligible, German-reviewed `include`
    contracted occurrence from Ch.4–17 whose effective head-noun lemma
    occurs in U.
  * **Final annotation target** = `U ∪ C_early ∪ C_late`.

`inventory_eligible` means the canonical preposition is in the paper's
13-item paired inventory (`PAPER_SHARED_PREPOSITIONS` in
`src/hp_corpus/step4.py`). Contracted `um`/`vor` forms are excluded
because neither `um` nor `vor` appears in the paper's uncontracted
`PREPOSITIONS` list, so they have no minimal-pair counterpart.

### C_late lemma-match policy

> **Forward design:** `docs/ANALYSIS_SAMPLE_DESIGN.md` (§3, S4)
> specifies a paper-literal **prep+noun** C_late match — the paper's
> §2.2.1 "same preposition and noun" — to replace the noun-only rule
> below via a future code change. Until it lands, the noun-only
> behavior described below is the only implemented rule, and any
> ledger built with it is provisional.

The C_late expansion is keyed on the **head-noun lemma alone**, not on
`canonical_preposition + lemma`. The paper's §3.2 describes the
expansion in terms of occurrences involving the same noun; canonical
preposition is retained in the ledger for audit and minimal-pair
analysis but does not gate membership. A C_late occurrence whose
canonical preposition differs from any matching U row still selects if
its head-noun lemma appears somewhere in U.

There is **no fuzzy or embedding-based lemma matching**. The match is
exact equality of the effective lemma, with three explicit hooks for
variation:

  1. **Reviewed lemma** — when an annotator-corrected lemma is
     available (`reviewed_head_lemma`), it overrides the machine
     lemma. This is the default mode.
  2. **Machine-lemma-only provisional mode** — `--use-machine-lemma`
     on the CLI builds a provisional ledger using only the machine
     lemma, clearly marked in the summary JSON. Useful before the
     German review is complete.
  3. **Manual lemma override** — for parser-lemma errors or genuine
     orthographic variants, the master annotation TSV may carry a
     `manual_lemma_override` column; non-blank values win over both
     reviewed and machine lemma.

### Effective lemma priority

```
manual_lemma_override  >  reviewed_head_lemma  >  machine_head_lemma  >  (blank)
```

A U or C_late row whose effective lemma is blank is blocked under
`blocked_lemma_review` — it cannot contribute to or match against U's
lemma set. C_early rows do not need a lemma (selection is on
form + chapter alone).

### Blocked and excluded rows

  * `de_candidate_decision = exclude` → `excluded_by_german_review`,
    not selected.
  * `de_candidate_decision = uncertain` or blank →
    `blocked_german_review`, blocked.
  * `de_candidate_decision = include` but missing required lemma →
    `blocked_lemma_review`, blocked.

Blocked and excluded rows are **never** in the analysis-ready sample.
They remain in the ledger for audit; every extracted occurrence
becomes one ledger row, including rejected ones.

### Ledger columns

```
datapoint_id
chapter
form
canonical_prep
machine_head_lemma
reviewed_head_lemma
effective_matching_lemma
german_candidate_decision
inventory_eligible
sampling_selected
sampling_reason              # one of SAMPLING_REASONS
sampling_status              # selected | not_selected | blocked
supports_late_contracted_ids # JSON list; populated for U rows
manual_lemma_override
source_hash
```

`supports_late_contracted_ids` is back-filled for each selected U row
with the datapoint IDs of the C_late rows it enabled — for audit
purposes, so a reviewer can see which Ch.4–17 PPs entered the sample
because of which Ch.1–17 uncontracted counterparts.

## Data readiness and fail-closed behavior

The pipeline is built to ship before the data: a missing parsed input
fails closed (`MISSING_PARSED_INPUT`) by default, and the sampling CLI
produces an empty ledger until the chapters it needs exist — once the
parses land, the ledger regenerates without code changes. At present
the Ch.1–17 machine corpus exists end-to-end, and every
inventory-eligible ledger row sits at `blocked_german_review`:
selection is gated on the German review, which has not been run.

Known provenance limitation: `sent_id` is not unique across CoNLL-U
blocks (one Segment can split into several Stanza sentence blocks), so
a `parse_block_id`-style migration is a hard gate before any annotator
batch (`docs/ANALYSIS_SAMPLE_DESIGN.md` §7.2).
