# Analysis-sample design

Status: **the formal eligible-pool rule is implemented
(`src/hp_corpus/sampling.py` + `scripts/build_eligible_pool.py`, see
`docs/FULL_NOVEL_SAMPLING.md`); nothing downstream of it has been
executed.** The German review — which covers the full machine candidate
pool — has not been run, so no eligible pool, no EN/ZH annotation, and
no analysis exist yet. When the review is complete, the pool is built
**once** from it; there is no sampling layer between the reviewed pool
and annotation.

## 1. Reproduction target

- **Exact-96 parity: blocked — missing provenance.** The paper
  (Bremmers et al. 2022, §2.2.1/§3) publishes only aggregate counts
  (40 contracted / 56 uncontracted / 96) and no per-item list. No
  automatic reading of the published rule + inventories yields 96 on
  this text, so the authors' hand-curation step cannot be reconstructed.
  The 96 is an **external benchmark**, not a target.
- **Independent methodological replication: not blocked.** Re-implement
  the published extraction + selection rule on our own machine corpus
  and test the theoretical claims on the result.
- **Standing rule:** no candidate pool may be tuned toward 96 by
  excluding valid rows. Coincidence with 96, if it occurs, is reported
  — never engineered.

## 2. Corpus layers

| layer | definition | producer | cardinality |
|---|---|---|---|
| `machine_candidate_pool` | every paper-algorithm occurrence Ch.1–17 surviving the preposition-inventory gate | extraction (automatic) | 1,392 |
| annotation CSV (returned) | one final `include`/`exclude` decision + optional lemma correction + EN/ZH counterpart marking per pool row | annotators on the trilingual pair CSV (human, full pool, single pass) | pending |
| `eligible_pool` | reviewed-include occurrences meeting the paper-literal rule (§3) | `build_eligible_pool.py` (runs once, after review) | pending |
| `paper_sample_membership` | audit-only overlay marking membership in the authors' 96, applicable to any layer **if** the per-item list is ever obtained | — | blocked, empty |

Invariants:

1. Strict nesting: `eligible_pool ⊆ reviewed-include rows ⊆
   machine_candidate_pool`.
2. `paper_sample_membership` is orthogonal — it never gates, blocks, or
   reorders membership in any other layer.
3. Each layer is a separate artifact (its own file or column set).
4. Every layer transition reports retained / excluded counts (§6).

## 3. Selection procedure

Ordered gates; the first three are automatic, the fourth is human, the
last is rule-based. All use **German-side information only** — EN/ZH
gold must never enter the selection path.

**Extraction (done).** The paper's algorithm (contracted: `CONTRACTED`
token with NN head; uncontracted: `PREPOSITIONS` token with same-head
ART in `DETERMINERS`), per-chapter manifests all `ok`. 1,394
occurrences (726 contracted / 668 uncontracted).

**Preposition inventory.** Canonical preposition ∈ the 13-item paired
inventory (`an auf aus bei durch für gegen hinter in unter von zu
über`). `um`/`vor` contractions are excluded (2 rows): neither has an
uncontracted counterpart in the paper's lists, so no minimal pair is
possible. Retained: 1,392.

**Structural gate (uncontracted rows).** The det token must be
`xpos=ART` **and** `deprel=det`. Rationale: the paper's extractor gates
on `xpos == 'ART'`; the authors' organization publishes TreeTagger/STTS
tooling under which a relative-pronoun *der* is PRELS — outside ART —
**but whether that tooling produced the extractor's input is
unconfirmed, and the converter chain is unknown**. The deprel gate is
our Stanza-environment analogue of excluding non-article *der*, adopted
on tagset-convention grounds; it is not a claim about their pipeline.
On the machine pool 668/668 rows pass — a *structural* consistency
observation within the current Stanza representation, **not** proof
that semantic miscoding is absent (§7.1); the authoritative check is
the German review. Contracted rows have no det token; their
parser-error class is deferred to the review. Missing structural
metadata on an uncontracted row fails the whole pool build — the gate
never passes by default.

**German PP validity review + EN/ZH marking (human, full pool, one
pass).** Every `machine_candidate_pool` row appears in the trilingual
pair CSV with its German sentence and aligned EN/ZH contexts
(`docs/ANNOTATION_CSV.md`). Per row, judged **in sentential context**:
is it a genuine definite German PP (not a relative-pronoun or
demonstrative/pronominal reading, not a parse artifact), and is the
machine head lemma correct? The annotator records `de_valid`
(`include` / `exclude`), `de_corrected_lemma` (**blank** = "the
machine lemma stands"; non-blank = correction), an exclusion reason
when excluding, plus the EN/ZH counterpart span / form / alignment
confidence. `exclude` is the only human removal; exclusions require an
articulable reason code. A blank or `uncertain` decision means the
review is not finished — the eligible-pool build **fails closed**
rather than emit a partial pool. Duplicate returned ids, ids outside
the machine master, or a `row_hash` mismatch against the master
likewise fail closed.

**Eligible-pool rule (paper-literal, implemented).**

- every reviewed-`include` **uncontracted** occurrence, Ch.1–17;
- every reviewed-`include` **contracted** occurrence, Ch.1–3;
- a reviewed-`include` **contracted** occurrence, Ch.4–17, only when
  its `(canonical_preposition, head_lemma)` pair — corrected lemma when
  non-blank, else machine lemma — occurs in the reviewed-include
  uncontracted set (the paper's §2.2.1 "the same preposition and
  noun").

**Identity and dedup.** Occurrence identity = (chapter,
source_segment_id, **parse_block_id**, pp_token_start, pp_token_end).
The machine pool has 0 exact coordinate collisions (§7.2). Exact
duplicate rows collapse deterministically to the first; the same
identity with conflicting core fields fails the build. The
`parse_block_id` provenance migration and its centralized fail-closed
validation are implemented (§7.2); verifying that the real
`data/parsed/` files pass validation remains a **runtime hard gate**
before any annotator batch.

- Analysis unit = **occurrence** (token). No fuzzy lemma matching
  anywhere: orthographic variants are handled by the reviewer's
  corrected lemma, never by similarity.

## 4. Count waterfall (machine corpus, pre-review)

| # | step | rule | retained | excluded |
|---|---|---|---|---|
| 1 | extraction | paper algorithm, Ch.1–17 | 1,394 (726 contracted / 668 uncontracted) | — |
| 2 | preposition inventory | 13 paired prepositions | 1,392 | 2 (`um`/`vor`) |
| 3 | structural gate | det `ART` + `det` (uncontracted only) | 1,392 (668/668 pass) | 0 |
| 4 | German review | full pool, human validity + lemma | not started | not started |
| 5 | eligible pool | §3 rule on reviewed-include rows | pending review | pending review |

Steps 1–3 are verified in the production path every time the pool CLI
runs; steps 4–5 are filled by the single post-review run. There are no
projected or simulated numbers for steps 4–5 in this design.

## 5. Data-structure observations

### 5.1 Structural tag consistency — not absence of contamination

All 668 uncontracted rows resolve to a unique CoNLL-U block (fingerprint:
sent_id + det/prep surfaces at the recorded token ids), and every det
token is `xpos=ART, deprel=det` **in the current Stanza
representation**. Scope limit — this must **not** be read as "no
relative-pronoun contamination":

- The tags come from the same parse that produced the pool. Same-parser
  self-certification cannot rule out semantic miscoding: a relative
  pronoun mistagged `ART + det` passes every one of these automatic
  checks.
- What the observation supports: (a) the extractor's same-head
  constraint is structurally consistent with the tags, and (b) no
  *deprel-visible* relative-pronoun rows exist in the pool
  representation.
- Consequence: the contamination question remains **open — neither
  confirmed nor refuted**. Settlement requires an independent tagger or
  a human context audit; in this design the authoritative check is the
  German review (every row judged in sentential context). The
  uncontracted surplus against the paper's filter lists (+58) remains
  unattributed among parser-environment tag differences, edition
  differences, and the authors' curation.

### 5.2 sent_id is not a unique key into the parsed files

> **Status:** the numbers below are **pre-migration evidence**,
> measured on the frozen machine corpus before the block-provenance
> migration ran. The migration and its centralized fail-closed
> validation are now implemented (`hp_corpus.provenance`: missing,
> duplicate, or inconsistent provenance — including physical blocks
> with token content but no `# sent_id` — refuse parsing, extraction,
> and annotation-pack input). What remains is the **runtime hard
> gate**: before any annotator batch is generated, the real
> `data/parsed/` files must be verified migrated and passing that
> validation — an operational precondition of the batch run, not
> future code.

Structure (pre-migration): 6,279 DE CoNLL-U blocks shared 5,042
distinct sent_ids (1,076 duplicated; every chapter affected).
Mechanism: a Segment can be split into several Stanza sentence blocks
that all carried the Segment's sent_id; extracted-TSV coordinates are
block-relative, so sentence_id alone did not identify the reviewing
context.

Integrity checks on the pre-migration machine pool (all aggregate):

| check | result |
|---|---|
| duplicate `datapoint_id` in annotation master | 0 / 1,392 |
| duplicate `source_row_sha256` in master | 0 / 1,392 |
| exact (sentence_id, pp_token_start, pp_token_end) collisions across extracted TSVs | 0 / 1,394 |
| occurrence→block join multiplicity ≠ 1 | 0 / 1,394 (726 C + 668 U all unique) |
| block-partition: blocks sharing a sent_id concatenate (whitespace-stripped) to the segment text | 1,076 / 1,076 pass |

These are same-pipeline consistency checks, not independent proof, and
they could not protect a human reviewer who was pointed at "sentence_id
X" and landed on the wrong block — which is why the block-provenance
migration was the hard gate this evidence demanded.

### 5.3 The extracted TSV noun column carries the lemma, not the surface

form (e.g. `Riese` for `Riesen`). Harmless for the pool rule (matching
is on lemma) but relevant when eyeballing rows and when building
fingerprints.

## 6. Remaining requirements

Already implemented and no longer future work: the
`parse_block_id` / `source_segment_id` provenance migration with its
centralized fail-closed validation (§5.2) — including block-level
identity in extraction output, the annotation master, joins, and
annotation-pack input validation; and the **formal eligible-pool path**
(§3) — structural gate, full-pool review-overlay contract, paper-literal
prep+noun matching, single corrected-lemma schema, identity/dedup, and
the count summary emitted by the one production CLI
(`scripts/build_eligible_pool.py`).

Still outstanding:

1. **German review execution** — review the full machine candidate
   pool (1,392 rows) and produce the final overlay. No batches exist
   today.
2. **Runtime verification of the provenance gate** — confirm the real
   `data/parsed/` files are migrated and pass validation at batch time
   (§5.2).
3. **EN/ZH annotation** of the eligible pool and the downstream
   analysis.

## 7. Replication-claim boundary

Current status: a **transparent paper-literal re-implementation
design** with the selection rule implemented. No replication claim is
earned yet — the German review, EN/ZH annotation, and the downstream
analysis are all outstanding.

| claim | status |
|---|---|
| Methodological replication of extraction + selection | potentially claimable **after** the German review completes and the eligible pool is built (paper-literal rule; own parser environment documented as a deviation — Stanza here vs their unconfirmed environment, §3) |
| Conceptual replication of the theoretical conclusions | potentially claimable **after** our EN/ZH annotation of the eligible pool and the downstream analysis |
| Exact / direct reproduction of the 96 | **not claimable** — no per-item provenance; 40/56/96 remain benchmark aggregates |
| Author-filter agreement numbers | sensitivity statistic, not replication |

## 8. Provenance of the counts

All machine-corpus counts in this document were measured on the frozen
Ch.1–17 corpus (pre-annotation, 2026-08) and are re-verified by the
production pool CLI on every run. Per-item audit material (row-level
queues, matched/mismatch TSVs) is deliberately kept out of the
repository under the copyright boundary — only aggregate counts appear
here. An earlier sampling-parity audit (mode-based simulations,
2026-08-15) remains as historical evidence in gitignored local data
and git history; the mode framework it served has been removed from
the design.
