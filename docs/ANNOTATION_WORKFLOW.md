# Annotation workflow

End-to-end workflow for producing analysis-ready annotations on the
Step 4 annotation-target TSV. Covers context packs, template refresh,
multi-annotator batching, dual-annotation comparison, adjudication, and
gold-merge. The data structures this workflow operates on are defined in
[`STEP4_ANNOTATION.md`](./STEP4_ANNOTATION.md); this document is the
operational handoff.

The annotation target itself is produced by
`scripts/build_ch1_3_full_annotation.py` (Ch.1–3) or
`scripts/build_full_novel_sampling_ledger.py` (full novel). The
workflow below assumes one of those has already been run.

## Stage boundaries

```
                            ┌───────────────────────────────────┐
   master annotation TSV ──▶│ prepare_annotation_batches        │
                            │   (calibration + main + overlap)   │
                            └──────────────┬────────────────────┘
                                           │ per-annotator TSVs
                                           ▼
                            ┌───────────────────────────────────┐
                            │ build_annotation_context_pack      │
                            │   (annotator's reading view)       │
                            └──────────────┬────────────────────┘
                                           │
                                           ▼
                                 human annotation
                                           │
                                           ▼
                            ┌───────────────────────────────────┐
            new extraction ─▶│ refresh_annotation_template       │
                            │   (carry annotations forward)      │
                            └──────────────┬────────────────────┘
                                           │
                                           ▼
                            ┌───────────────────────────────────┐
            annotator A  ──▶│ compare_annotations                │
            annotator B  ──▶│ build_adjudication_ledger          │
                            └──────────────┬────────────────────┘
                                           │
                                           ▼
                            ┌───────────────────────────────────┐
                            │ merge_adjudicated                  │
                            │   (gold TSV + --require-complete)  │
                            └───────────────────────────────────┘
```

## 1. Context pack — `build_annotation_context_pack.py`

Generalises the pilot-only `build_alignment_review_pack.py` to any
annotation-target TSV. For each row, emits the aligned DE / EN / ZH
text plus a configurable number of preceding and following sentences
per language, so the annotator can read each PP in context.

```
uv run python scripts/build_annotation_context_pack.py \
    --input-tsv  data/derived/step4/ch1_3_full_annotation.tsv \
    --segmented-dir data/segmented \
    --output data/derived/step4/context_pack.tsv \
    --context-size 1
```

Fails closed on missing required columns, malformed `de_sentence_id`,
unresolved segment IDs, or malformed JSON in `en_sentence_ids` /
`zh_sentence_ids`. The TSV itself carries source text (gitignored);
stdout carries aggregate counts only.

## 2. Template refresh — `refresh_annotation_template.py`

When a new extraction / alignment run produces a new master TSV, this
script carries annotations forward from the old TSV. It copies
editable fields **only when `source_row_sha256` matches** between old
and new — a source-row change implies the row is no longer the same
datapoint, so the old annotations are not safe to copy.

```
uv run python scripts/refresh_annotation_template.py \
    --old-tsv data/derived/step4/ch1_3_old.tsv \
    --new-tsv data/derived/step4/ch1_3_new.tsv \
    --output  data/derived/step4/ch1_3_refreshed.tsv
```

Hash mismatches are routed to a conflict-ledger JSONL (no source text
in stdout). New IDs in the new TSV stay blank; removed IDs are listed
in the summary JSON. The script refuses to overwrite an existing output
without `--force-output`, and refuses any path collision between inputs
and output.

## 3. Batching — `prepare_annotation_batches.py`

Partitions a validated master TSV into per-annotator files with a
shared calibration set and configurable overlap. Assignment is fully
deterministic from `datapoint_id + seed`, so two runs with the same
inputs produce byte-identical batches.

```
uv run python scripts/prepare_annotation_batches.py \
    --master-tsv data/derived/step4/master.tsv \
    --out-dir   data/derived/step4/batches \
    --annotators alice bob \
    --seed hp-transmining-bremmers-v1 \
    --overlap 0.2 \
    --calibration-size 10
```

Rows with `de_candidate_decision=include` go to annotator batches
(plus the shared calibration set). `exclude` rows go to `excluded.tsv`
(or are dropped with `--drop-excluded`); blank or `uncertain` rows go
to `blocked_review.tsv`. The `batch_manifest.json` captures annotator
names, calibration IDs, per-annotator ID lists, and aggregate counts —
no novel text.

Enforced invariants: full coverage of eligible rows, no within-file
duplicates, calibration in every annotator file, byte-identical output
under same seed.

## 4. Comparison — `compare_annotations.py`

Joins two annotator TSVs by `datapoint_id` and surfaces linguistic
disagreements at field level on the research fields (EN/ZH alignment
QC + relation + span text + char ranges + form + confidence).
Workflow metadata (`annotator`, `annotation_status`,
`adjudication_status`) is NOT compared as linguistic content.

```
uv run python scripts/compare_annotations.py \
    --a data/derived/step4/annotator_a.tsv \
    --b data/derived/step4/annotator_b.tsv \
    --out-stem data/derived/step4/comparison
```

Refuses if `source_row_sha256` differs between A and B for the same
datapoint — annotators must be operating on the same template, else
the disagreement is methodological rather than linguistic. Blank vs
nonblank counts as a disagreement.

## 5. Adjudication ledger — `build_adjudication_ledger.py`

Reads the comparison TSV and emits a human-fillable ledger with one
row per disagreement: blank `adjudicated_value`, `resolution_status =
pending`. A sidecar provenance JSON records source-file basenames and
per-datapoint source-row hashes for traceability.

```
uv run python scripts/build_adjudication_ledger.py \
    --comparison data/derived/step4/comparison.disagreements.tsv \
    --a data/derived/step4/annotator_a.tsv \
    --b data/derived/step4/annotator_b.tsv \
    --output data/derived/step4/adjudication_ledger.tsv
```

The human adjudicator fills in `adjudicated_value` and sets
`resolution_status` to `adjudicated` (or `rejected`).

## 6. Gold merge — `merge_adjudicated.py`

Combines the master TSV, both annotator TSVs, and the adjudication
ledger into a single gold TSV. For each row:

  * Source columns come from the master.
  * Editable base is annotator-A's row.
  * For each field with an adjudicated disagreement, the
    `adjudicated_value` from the ledger overwrites A's value.
  * `adjudication_status = "adjudicated"` if any disagreement existed
    on the row; else stays blank.

Refuses source-hash mismatches across inputs, unresolved or
blank-value disagreements, and never guesses a winning annotation.
After writing the gold TSV, runs the Step 4 validator in
`--require-complete` mode and exits non-zero if it fails.

```
uv run python scripts/merge_adjudicated.py \
    --master data/derived/step4/master.tsv \
    --annotator-a data/derived/step4/annotator_a.tsv \
    --annotator-b data/derived/step4/annotator_b.tsv \
    --ledger    data/derived/step4/adjudication_ledger.tsv \
    --output    data/derived/step4/gold.tsv \
    --annotation-pool
```

The gold TSV is the analysis-ready input to Step 5 — see
[`STEP5_ANALYSIS.md`](./STEP5_ANALYSIS.md).

## Privacy discipline

Every script in this workflow prints only aggregate counts and output
paths to stdout. Sentence text, segment IDs, lemmas, surface forms,
annotation values, and `datapoint_id` values never appear on stdout —
they live only in the gitignored TSV / JSONL / JSON files the scripts
write under `data/derived/`. This is verified by the stdout-privacy
guard tests in each work package's test file.
