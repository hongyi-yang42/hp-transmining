# Analysis-sample design

Status: **design specification; nothing downstream has been executed.**
The selection rules below are not yet implemented in the pipeline (§8),
the German review has not been run, no analysis sample — including the
mode-C n = 400 draw — has been generated, no EN/ZH gold annotation
exists, and no analysis has been performed. Every count below that
depends on human review is an all-include **projection**. This document
specifies the target design; `docs/FULL_NOVEL_SAMPLING.md` describes
currently implemented behavior.

## 1. Reproduction target

- **Exact-96 parity: blocked — missing provenance.** The paper
  (Bremmers et al. 2022, §2.2.1/§3) publishes only aggregate counts
  (40 contracted / 56 uncontracted / 96) and no per-item list. No
  automatic reading of the published rule + inventories yields 96 on
  this text (simulations produce 858–1,062; §10), so the authors' hand
  curation step cannot be reconstructed. The 96 is an **external
  benchmark**, not a target.
- **Independent methodological replication: not blocked.** Re-implement
  the published extraction + selection rule on our own machine corpus
  and test the theoretical claims on the result.
- **Standing rule:** no candidate pool may be tuned toward 96 by
  excluding valid rows. Coincidence with 96, if it occurs, is reported
  — never engineered.

## 2. Corpus layers

| layer | definition | producer | cardinality |
|---|---|---|---|
| `machine_candidate_pool` | every paper-algorithm occurrence Ch.1–17 surviving the preposition-inventory gate (S1) | extraction (automatic) | 1,392 |
| `german_validated_pool` | pool rows with `de_candidate_decision = include` and a non-blank effective lemma | German review (human) | pending (≤ 1,392) |
| `analysis_sample` | mode-dependent subset of the validated pool (§4: mode-A rule or mode-C draw) | sampling ledger / stratified sampler | none generated |
| `paper_sample_membership` | audit-only overlay marking membership in the authors' 96, applicable to any layer **if** the per-item list is ever obtained | — | blocked, empty |

Invariants:

1. Strict nesting: `analysis_sample ⊆ german_validated_pool ⊆
   machine_candidate_pool`.
2. `paper_sample_membership` is orthogonal — it never gates, blocks, or
   reorders membership in layers 1–3.
3. Each layer is a separate artifact (its own file or column set).
   Collapsing layers — e.g. letting German review decisions act as the
   terminal sample-membership policy — is forbidden.
4. Every layer transition emits retained / excluded / blocked counts
   (§6).

## 3. Selection procedure

Ordered gates. S0–S2 automatic, S3 human, S4–S5 rule-based. All steps
use **German-side information only** — EN/ZH gold must never enter the
sampling path.

**S0 — extraction (done).** The paper's algorithm (contracted:
`CONTRACTED` token with NN head; uncontracted: `PREPOSITIONS` token
with same-head ART in `DETERMINERS`), per-chapter manifests all `ok`.
1,394 occurrences (726 contracted / 668 uncontracted).

**S1 — preposition inventory.** Canonical preposition ∈ the 13-item
paired inventory (`an auf aus bei durch für gegen hinter in unter von
zu über`). `um`/`vor` contractions are excluded (2 rows): neither has
an uncontracted counterpart in the paper's lists, so no minimal pair is
possible. Retained: 1,392.

**S2 — automatic parser-artifact gate (uncontracted rows).** The det
token must be `xpos=ART` **and** `deprel=det`. Rationale: the paper's
extractor gates on `xpos == 'ART'`; the authors' organization publishes
TreeTagger/STTS tooling under which a relative-pronoun *der* is PRELS —
outside ART — **but whether that tooling produced the extractor's input
is unconfirmed, and the converter chain is unknown**. The deprel gate
is our Stanza-environment analogue of excluding non-article *der*,
adopted on tagset-convention grounds; it is not a claim about their
pipeline. On the machine pool 668/668 rows pass — a *structural*
consistency observation within the current Stanza representation,
**not** proof that semantic miscoding is absent (§7.1); the
authoritative check is S3. Contracted rows have no det token; their
parser-error class is deferred to S3.

**S3 — German PP validity review (human).** Per row, judged **in
sentential context**: is it a genuine definite German PP (not a
relative-pronoun or demonstrative/pronominal reading, not a parse
artifact), and is the machine head lemma correct? Decisions:
`include` / `exclude` / `uncertain`, plus `reviewed_head_lemma`.
`exclude` is the only S3 removal. `uncertain` → blocked, never excluded
(conservative asymmetry). Exclusions require an articulable reason code.

**S4 — paper-literal selection rule (mode-A core; code change pending,
§8).**

- `U` = validated uncontracted occurrences, Ch.1–17.
- `C_early` = validated contracted occurrences, Ch.1–3.
- `C_late` = validated contracted occurrences, Ch.4–17 whose
  **(canonical_prep, effective_lemma)** pair occurs in `U` —
  paper-literal **prep+noun** matching (§2.2.1: "the same preposition
  and noun"). The currently implemented noun-only variant adds 204 rows
  on the machine corpus and is treated as a provisional deviation
  (`docs/FULL_NOVEL_SAMPLING.md`).

Effective-lemma priority:
`manual_lemma_override > reviewed_head_lemma > machine_head_lemma`.

**S5 — identity and dedup.**

- Occurrence identity = (chapter, source_segment_id, **parse_block_id**,
  pp_token_start, pp_token_end). Same-key rows collapse to one; the
  machine pool has 0 exact coordinate collisions (§7.2). The
  `parse_block_id` migration and its centralized fail-closed validation
  are implemented (§7.2); verifying that the real `data/parsed/` files
  are migrated and pass validation remains a **runtime hard gate**
  before annotator batches.
- Analysis unit = **occurrence** (token). Type = (form,
  canonical_prep, effective_lemma); type counts are always reported
  next to occurrence counts.
- No fuzzy lemma matching anywhere: orthographic variants are handled
  at S3 via `manual_lemma_override`, not at S4 via similarity.

**S6 — waterfall reporting.** Each sampler run writes
retained/excluded/blocked counts to its summary JSON; §6 is re-derived
on every regeneration.

## 4. Research modes

| | A — paper-literal full eligible set | B — author-filter sensitivity set | C — stratified analysis sample |
|---|---|---|---|
| definition | S0–S5, no further restriction | rows passing the extractor's runtime `check_filter` (FILTER_CONTRACTED / FILTER_PP) | pre-registered stratified draw from the machine-lemma projection of the mode-A rule (§5) |
| projected size | 858 all-include (668 U + 130 C_early + 60 C_late); realized after S3 | 139 (25 contracted + 114 uncontracted) | 400 target |
| tests | the paper's stated method end-to-end; the contracted/uncontracted translation-marking contrast at maximal power | agreement between our corpus/parser environment and the authors' curation | the same theoretical claims under annotation-budget constraints |
| role | full-set methodological reference | sensitivity / benchmark overlay — never the primary sample (inherits undocumented curation) | **primary analysis plan** |
| cannot claim | the authors' 96 (superset by construction) | parity (aggregate agreement only) | full-rule coverage (design-based inference only if the §5 freeze is respected) |

### 4.1 German-review scopes — three distinct objects, not to be conflated

- **Full-frame review (mode A only): 1,392 rows** — every
  machine_candidate_pool row gets a validity + lemma decision. Mode A's
  selection rule is defined over the whole validated pool, so it cannot
  be evaluated without reviewing the whole frame.
- **Mode C review scope: primary + reserve (≤ 500 rows) + support
  closure (≤ 29 further rows)** — mode C freezes the seeded
  primary+reserve ordering on the *machine* frame before any review,
  then sends only those rows to German review; the C_late
  support-closure rule (§5) may additionally require reviewing
  uncontracted witness rows outside primary+reserve. Full-frame review
  is not a prerequisite for mode C. Mode A needs no closure mechanism —
  its full-frame review covers every support row by construction.
- **B_forensic_union = 358 rows** — the union of any-whitelist hits:
  FILTER_CONTRACTED (25) ∪ FILTER_CONTRACTED_123 (219) ∪ FILTER_PP
  (114); the contracted sides are disjoint, the uncontracted side is
  shared. A review-*priority* scope for the mode-B sensitivity analysis
  — not a sample and not an annotation workload.
- **mismatch_queue ≈ 26 entry-level units** — 14 zero-occurrence
  whitelist entries + 10 near-miss rows + multiplicity decisions. A
  local forensic estimate (§10), not a design input.
- **B_runtime_filter = 139** is a derived count of rows passing
  `check_filter` under the running extractor — not a review scope.

### 4.2 Annotation workload (upper bounds)

`language-rows = 2 × occurrences` (one EN row + one ZH row per German
occurrence). Upper bounds sit before attrition (German-invalid or
alignment-unusable items: mode C replaces them from pre-registered
reserves §5; mode A shrinks and reports).

| mode | German review rows | German occurrences to annotate | EN+ZH language-rows (upper bound) |
|---|---|---|---|
| A | 1,392 (full frame) | ≤ 858 | ≤ 1,716 |
| B (runtime filter) | 358 (forensic union, §4.1) + ≈26-unit queue | 139 | ≤ 278 |
| C | ≤ 500 primary+reserve + ≤ 29 support-closure witnesses (§5; loose sum ≤ 529, the two sets overlap) | 400 target | ≤ 800 |

## 5. Mode C specification

- **Frame — frozen on the machine frame, before any review.** The
  pre-registerable frame is the **machine** paper-literal eligible set:
  machine_candidate_pool rows (post S1–S2) with the U / C_early /
  C_late structure computed on machine lemmas (all-include projection
  ≈ 858 rows; §6). Mode-A eligibility cannot be known before German
  review without full-frame review, so mode C does not wait for it —
  the machine-lemma projection is the frame, and the deviation from
  the post-review eligible set is reported after S3.
- **Order of operations (budget-saving timing):**
  1. Freeze frame + strata + the seeded full ordering per stratum
     (primary ranks 1..k_s, reserve ranks k_s+1..k_s+r_s) + the C_late
     support ladders — no review data is used.
  2. German review is sent for primary + reserve rows (upper bound
     n + Σr_s = 500 at α = 0.25), plus ladder witness rows as the
     support-closure rule consumes them (≤ 29 distinct).
  3. Realized sample = primary rows with S3 `include` **and** (for
     C_late primaries) a validated support witness; a primary that is
     German-invalid, alignment-unusable, or support-unreachable is
     replaced by the next unused same-stratum reserve, in rank order
     (never by EN/ZH outcomes).
  4. Invariant: the realized sample ⊆ `german_validated_pool` by
     construction — reserves and witnesses are reviewed before any
     promotion or closure decision.
  5. If a stratum's reserves are exhausted, its realized count shrinks;
     no top-up, no out-of-stratum substitution.
- **Strata:** form (contracted/uncontracted) × chapter band (1–3 / 4–9
  / 10–17) × minimal-pair group completeness (both forms present in the
  machine frame vs single form).
- **Group-preserving floor (both-form groups).** Both-form groups are
  the paper's core contrast; the machine pool has **22 groups holding
  104 occurrences (68 contracted + 36 uncontracted)**, each group with
  ≥ 1 occurrence of each form. The floor is **group-level**: every
  both-form group is guaranteed ≥ 1 contracted AND ≥ 1 uncontracted
  **primary** — 44 guaranteed slots at minimum. The remaining 60
  both-form occurrences compete in the general proportional draw like
  any other item. There is no occurrence-level floor and no
  both-form-specific cap: anti-domination is inherent in proportional
  allocation (both-form rows are 104/858 ≈ 12.1% of the frame), and the
  realized both-form share is reported post-draw. The floor is a
  draw-time rule only — after the draw, a floored primary is replaced
  under the same replacement rules as everything else (no re-flowering
  from non-drawn both-form items; if a group loses an entire form side
  to German exclusions, the guarantee is void for that group and the
  fact is logged).
- **C_late support closure (mode C only).** A C_late candidate's
  eligibility depends on an uncontracted counterpart existing — but
  under budget-saving timing that counterpart may never have been
  reviewed, so the candidate's rule-conformance would rest on machine
  data alone. Closure rule:
  1. **Ladder (frozen at the freeze event):** for each of the **60**
     paper-literal C_late candidates (machine frame, prep+noun
     matching), its ladder is the U rows of its (canonical_prep,
     machine lemma) group in frame sort order — shared per group
     (candidates of one group share rungs; the 60 candidates span **19
     groups** holding **29 distinct U rows**; ladder length 1–4, mean
     1.37).
  2. **Witness requirement:** a sampled C_late candidate (primary or
     promoted reserve) enters the realized sample only if at least one
     reviewed U row is a **witness**: S3 `include` **and** an effective
     lemma whose (canonical_prep, effective_lemma) key equals the
     candidate's effective key. Ladder rungs are reviewed in ladder
     order (deduped across candidates and against primary+reserve — a
     row is never reviewed twice); review stops at the first witness.
  3. **Exhaustion:** if a candidate's ladder is exhausted (all rungs
     excluded, uncertain-blocked, or re-keyed away), the candidate
     drops with reason `support_closure_failed` and is replaced from
     the same-stratum reserve. If the candidate's own lemma correction
     re-keys it to a group whose U rows were not reviewed, no witness
     is reachable — conservative drop under the same reason.
  4. **Cost bound:** ≤ 29 distinct witness-review rows beyond
     primary+reserve (reported separately, never folded into the ≤ 500
     claim). Expected cost is far lower: the group floor guarantees ≥ 1
     reviewed U primary in every both-form group ⊇ every C_late group,
     so extra reviews arise only when a group's reviewed U rows fail or
     re-key. A witness is a validity witness only — it is not itself
     part of the 400 and receives no EN/ZH annotation unless
     independently sampled.
  5. Mode A needs no closure mechanism: full-frame review witnesses
     every candidate by construction.
- **Allocation:** remainder (n − floored) split 50/50 across forms,
  proportional within form × chapter band, integerized by largest
  remainder.
- **Size — n = 400 (≈ 200 per form), a resource-constrained planning
  heuristic, not a power-backed calculation.** The heuristic
  arithmetic: a two-sided two-proportion z-test at α = .05, power .80,
  baseline p₁ = .50, Δ = .15 needs ≈ 170 annotatable items per form;
  × 2 forms = 340; ÷ mean ZH machine-alignment coverage 0.878
  (attrition treated as missing-at-random) ≈ 387, rounded to 400. What
  a genuine power analysis would still require: baseline marker
  proportions from pilot annotation; clustering adjustments (by
  chapter and by preposition type); stratification weights for this
  design; multiplicity control across marker categories. Permissible
  band 300–480; n is fixed before the draw and never adjusted after
  outcomes are seen. **n = 96 is not a design target.**
- **Reserve sizing:** r_s = ceil(α · k_s) with α = 0.25 — upper bound
  500 review rows total. Justification: alignment-unusable attrition is
  bounded near 12% by machine ZH coverage (0.878); German-invalid rate
  is unknown until a pilot review, so α carries that headroom. α is
  fixed at freeze time with its justification recorded.
- **Determinism:** frame sorted by (chapter, source_segment_id,
  parse_block_id, pp_token_start, pp_token_end, datapoint_id); one
  `random.Random(seed=20260815)` consuming strata in lexicographic
  order; each stratum's full seeded shuffle is computed once and fully
  recorded (the entire ordering is the pre-registered reserve ladder).
- **Replacement rules:** a primary item is replaced **only** for:
  (a) `german_review_excluded` — S3 rules the occurrence invalid,
  (b) `alignment_unusable` — the EN or ZH counterpart cannot be
  resolved by the machine alignment, or
  (c) `support_closure_failed` — a C_late candidate's witness ladder
  is exhausted (German-side only, like a and b).
  Replacement takes the next unused reserve in the **same stratum**, in
  rank order. Replacement is **never** triggered by EN/ZH annotation
  outcomes (that would condition selection on outcomes). Every
  replacement is logged (out_id, in_id, reason) in the sample summary
  JSON; items dropped for any other reason keep their status and shrink
  n without top-up.
- **Freeze checklist (before the first EN/ZH annotation):** frame file +
  SHA-256, seed, n, strata, allocation, group floor rule, C_late support
  ladders, α, reserve rule, and code commit recorded in the sample
  summary JSON. Post-freeze deviations are logged, never silent.

## 6. Count waterfall (machine corpus, pre-annotation)

| # | step | rule | + retained | − excluded |
|---|---|---|---|---|
| 1 | S0 extraction | paper algorithm, Ch.1–17 | 1,394 (726 C / 668 U) | — |
| 2 | S1 inventory | 13 paired prepositions | 1,392 | 2 (`um`/`vor`) |
| 3 | S2 parser-artifact gate | det `ART` + `det` (structural) | 1,392 | 0 |
| 4 | S3 German review | human validity + lemma | not started | not started |
| 5 | S4 `U` | valid uncontracted Ch.1–17 | [668] | — |
| 6 | S4 `C_early` | valid contracted Ch.1–3 | [130] | — |
| 7 | S4 `C_late` | (prep, lemma) ∈ U | [60] | [536 no-match — retained in pool, not selected] |
| 8 | **mode A total** | 5 + 6 + 7 | **[858]** | |
| 9 | mode B runtime filter | author filters | 139 (25 C / 114 U) | |
| 10 | B_forensic_union (review scope, §4.1 — not a selection) | any-whitelist union | 358 | |
| 11 | mode C | §5 draw | [400 target] | |

Bracketed values are **all-include projections** (every S3 decision
`include`, every lemma non-blank); realized counts replace them after
S3. Row 7's noun-only variant ([264], total [1,062], delta 204) is
reported for comparison with the current implementation only.
Provenance of all counts: §10.

## 7. Data-structure observations

### 7.1 Structural tag consistency — not absence of contamination

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
  a human context audit; in this design the authoritative check is S3
  (every row judged in sentential context). The uncontracted B-mode
  surplus against the paper's filter lists (+58) remains unattributed
  among parser-environment tag differences, edition differences, and
  the authors' curation.

### 7.2 sent_id is not a unique key into the parsed files

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

### 7.3 The extracted TSV noun column carries the lemma, not the surface

form (e.g. `Riese` for `Riesen`). Harmless for sampling (S4 matches on
lemma) but relevant when eyeballing rows and when building fingerprints.

## 8. Remaining requirements

Already implemented and no longer future work: the
`parse_block_id` / `source_segment_id` provenance migration with its
centralized fail-closed validation (§7.2) — including block-level
identity in extraction output, the annotation master, joins, and
annotation-pack input validation.

Still outstanding:

1. **S4 paper-literal selection** — prep+noun C_late matching in
   `src/hp_corpus/sampling.py` + CLI, replacing the noun-only rule; the
   waterfall counts of §6 are emitted as summary JSON by this one
   production implementation, giving the census numbers a tracked,
   testable reproduction path.
2. **Mode-C sampler** — frame freeze, seeded draw, group floor,
   support closure, reserve ladder: same code path, with
   synthetic-fixture tests covering the selection rule (prep+noun vs
   noun-only), the group floor, and the support-ladder/witness logic.
3. **Runtime verification of the provenance gate** — confirm the real
   `data/parsed/` files are migrated and pass validation at batch time
   (§7.2).
4. German review and EN/ZH annotation tooling build on the above; no
   batches exist today.

## 9. Replication-claim boundary

Current status: a **transparent paper-literal re-implementation
design**. No replication claim is earned yet — S4 selection, the
Mode-C sampler, German review, EN/ZH annotation, and the downstream
analysis are all outstanding.

| claim | status |
|---|---|
| Methodological replication of extraction + selection | potentially claimable **after** the S4 implementation lands, German review completes, and the mode-A sample is built (paper-literal rule; own parser environment documented as a deviation — Stanza here vs their unconfirmed environment, §3 S2) |
| Conceptual replication of the theoretical conclusions | potentially claimable **after** our EN/ZH annotation of a realized sample (mode A or C) and the downstream analysis |
| Exact / direct reproduction of the 96 | **not claimable** — no per-item provenance; 40/56/96 remain benchmark aggregates |
| Author-filter agreement numbers (mode B) | sensitivity statistic, not replication |

## 10. Provenance of the counts

All counts in this document were measured once on the frozen Ch.1–17
machine corpus (pre-annotation, 2026-08) during the sampling-parity
audit. Per-item audit material (row-level queues, matched/mismatch
TSVs) is deliberately kept out of the repository under the copyright
boundary — only aggregate counts appear here. Formal tracked
reproduction of the census/waterfall numbers is a requirement on the
production sampling CLI (§8), which will emit them from the single
implemented rule; until that lands, the numbers are recorded
observations, not a continuously verified invariant. The ≈26-unit
mismatch queue (§4.1) is a local forensic estimate — its near-miss
heuristics are not automated and it is not a design input.
