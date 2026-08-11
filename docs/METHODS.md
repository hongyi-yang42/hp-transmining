# Methods

## Reproduction scope

This is an **extraction-method reproduction** of Bremmers et al. (2022) *"Translation Mining: Definiteness across Languages"* — not a parser-identical replication. We use the paper's published extraction logic verbatim; the parser differs because Bremmers et al. do not report which UD parser they used.

A 100% match against the paper's `FILTER_CONTRACTED_123` / `FILTER_PP` is therefore not achievable: residual differences come from parser variation (our Stanza vs the paper's unreported parser) and from edition-specific German PPs in the Carlsen 1998 text that the paper did not annotate.

## Parser

- **Paper upstream parser**: not reported in Bremmers et al. (2022).
- **This project**: Stanza 1.14, with `tokenize + mwt + pos + lemma + depparse`.

## Extraction logic (Step 1–3)

`scripts/run_paper_extractor.py` follows the paper's `time-in-translation/conll-extractor` (vendored at `vendor/conll-extractor/`, gitignored):

- **Contracted PPs**: tokens in `CONTRACTED` (`im`, `am`, `zum`, …) → extract `(prep, noun)` where the prep's head is an `NN`.
- **Uncontracted PPs**: `PREPOSITIONS` tokens whose same-head dependent is an `ART` in `DETERMINERS` (`der`/`dem`/`den`/…) → extract `(prep, det, noun)`.

Each row in `data/extracted/hp1_de_ch{NN}_{contracted,uncontracted}.tsv` carries occurrence coordinates (`prep_token_id`, `det_token_id`, `noun_token_id`, `pp_token_start`, `pp_token_end`, `pp_surface`) so two PPs that share `(preposition, noun)` within one sentence can be told apart. Coordinates are CoNLL-U token IDs from the `_nomwt.conllu` files.

Hits are validated against `FILTER_CONTRACTED_123` and `FILTER_PP` from `conll_extractor.prepositions.data`.

## Step 4 — Cross-lingual annotation pack

**What a Step 4 datapoint is.** A *specific German PP occurrence* — a token range inside one sentence — together with the EN and ZH sentence(s) aligned to that German sentence. The annotator's task is to identify, inside each aligned sentence, the linguistic expression that corresponds to the German PP and to record its observable form.

**What the Step 4 TSV is — and is not.**

- The output TSV is a **Ch.1–3 paper-eligible annotation pool**. Every surviving occurrence whose canonical preposition is in the paper's 13-item paired inventory is included; nothing is hand-picked.
- It is **not** the paper's final 96 trilingual contexts. The 96 are hand-selected from the full novel; membership in the 96 is a downstream human decision, not a property of any row in this TSV. `paper_final_sample=false` on every row makes this explicit.
- It is **not** a (preposition, noun) type-level analysis. Two occurrences of `im Haus` in the same sentence are two datapoints.
- `FILTER_CONTRACTED_123` and `FILTER_PP` are extraction QA filters; they appear as the `author_resource_match` source column for visibility but do not gate membership in the pool.
- It is **not** an automatic translation or definiteness classifier. EN/ZH counterpart spans and form labels are filled by a human annotator. The builder never auto-translates or auto-classifies; `crosslingual_map` proposals live in a separate JSONL and never auto-populate annotation fields.
- It does **not** assign weak/strong definiteness, uniqueness, or familiarity. Only observable surface form is recorded.

### The 13-item operational inventory

The pool is restricted to canonical prepositions that appear in **both** of the author's published lists: the contracted `CONTRACTED` list (27 surface forms) and the uncontracted `PREPOSITIONS` list (13 forms). The intersection is 13 canonical prepositions: `an, auf, aus, bei, durch, für, gegen, hinter, in, über, unter, von, zu`. Contracted `ums` and `vorm` exist in German but are excluded because `um` and `vor` are not in `PREPOSITIONS`. This 13-item set is the operational paired inventory; it is not a universal German-grammar inventory.

The contraction table in `src/hp_corpus/step4.py` is pinned to the vendored `time-in-translation/conll-extractor` commit recorded in `vendor/conll-extractor.commit`. Two regression tests guard parity: a fixture-based test (runs on every checkout) and a vendor-based test (runs only when the vendor clone is present).

### Sentence alignment

The sentence alignments feeding Step 4 are produced by `src/hp_corpus/align.py` (e5-base embeddings + dynamic programming). They are **assumed correct initially** — the builder writes `{lang}_alignment_qc = assumed_ok` on every row — but checked opportunistically by the annotator while locating counterparts. The annotator may promote the QC flag to `confirmed` (after verifying a counterpart or its absence), or flag it as `incorrect` / `uncertain`, which routes the row to a realignment queue.

Misalignment and omission are distinct: `omitted` is reserved for "the **correctly-aligned** target context genuinely contains no counterpart". `incorrect` + `omitted` together is a hard validation error (`MISALIGNED_NOT_OMISSION`); a row in doubt must be flagged for realignment, not encoded as omission.

### Builder input discipline

For Ch.1–3 every requested chapter must contribute a full input set: both contracted and uncontracted extraction TSVs (non-empty body), DE/EN/ZH segmented JSONL, and DE–EN / DE–ZH alignment JSONL. Header-only, empty, or missing inputs raise `MissingInputsError`; an unresolved DE segment id raises `UnresolvedSegmentIdError`. The builder never emits a partial-corpus TSV. The header-only rule is scoped to Ch.1–3; the future Ch.4–17 extension will need an extraction-manifest mechanism to distinguish legitimately-empty chapters from missed runs (out of scope here).

### Module layout

- `src/hp_corpus/step4.py` — pure-Python builder, pilot selector, and TSV writer.
- `scripts/build_step4_annotation_pack.py` — CLI wrapper that wires file paths.
- `scripts/validate_step4_annotations.py` — CLI validator.
- `tests/test_step4.py` — synthetic-fixture tests; no novel text.

### Alignment compatibility

The current alignment serializer writes both sides of every record under the JSON keys `en` and `zh`, regardless of which languages the file actually pairs. `src/hp_corpus/step4.py` therefore identifies each side by **inspecting the segment IDs** (the language code is the second underscore-separated field) and additionally accepts `source`/`target` and `src`/`tgt` field names for forward compatibility.

### Occurrence identity

`datapoint_id` = `dp_ch{NN}_{de_sentence_id}_t{token_start}-{token_end}`. Two occurrences in the same sentence with the same `(preposition, noun)` get different IDs because their token ranges differ.

### Pilot selection (deterministic, no randomness)

The pilot is **10 contracted + 10 uncontracted**, chosen from candidates that have both EN and ZH alignment. Priority:

1. **`minimal_pair`** — groups `(de_prep_normalized, de_head_lemma)` that contain both forms. Each such group contributes at most one contracted + one uncontracted occurrence (the first by stable sort).
2. **`author_match`** — occurrences whose `author_resource_match` is True (i.e. `(prep, noun)` is in the paper's filter).
3. **`stable_fill`** — remainder in stable sort order `(chapter, de_sentence_id, de_token_start, de_token_end)`.

If the eligible pool cannot reach 10+10, the builder exits non-zero with a per-form shortfall report.

### Source-row immutability

Each pilot row carries `source_row_sha256` over the immutable source columns. The validator recomputes the hash and rejects any row whose source columns were edited. Editable columns (annotation fields) live in separate TSV columns and are not covered by the hash.

### Annotation schema

See [`STEP4_ANNOTATION.md`](./STEP4_ANNOTATION.md) for the column-by-column annotation guide.

## Output discipline

Per-PP detail goes only to TSV files under `data/extracted/` (gitignored). The Step 4 candidate JSONL and pilot TSV go to `data/derived/step4/` (also gitignored — contains novel text). **Stdout and summary JSON carry aggregate counts only** — never noun lemmas, surface forms, segment IDs, or sentence text. This is verified by:

- `tests/test_run_paper_extractor.py` — extractor stdout guard.
- `tests/test_step4.py::test_summary_no_token_text` — Step 4 summary guard.
- `tests/test_step4.py::test_validator_stdout_no_source_text` — validator stdout guard.
