# PRE_REPORT_EVIDENCE — 2026-09-07

> **Post-date note (2026-09-26).** This is a frozen record of the 2026-09-07
> run. Two items have since changed: the rejoin path flagged as a design gap
> in §5.5 is now implemented (`scripts/merge_companion_review.py`, PR #7,
> main `1159516`), and `docs/AI_CODING_INTERACTION_EVIDENCE.md` (§5.3) is
> now committed alongside this record.

Independent pre-report verification run. Read-only with respect to research
semantics, human-annotation fields, and git history. No push, no merge, no
pipeline rebuild. No corpus text is quoted below (IDs, counts, hashes only).

## 1. Repository state

| item | value |
|---|---|
| `git rev-parse HEAD` | `85a1e01e8320cd20e187371c18b468b58a1e48fd` (== expected GitHub main HEAD) |
| `git status --short` | `?? docs/AI_CODING_INTERACTION_EVIDENCE.md` (only untracked file; known, intentionally uncommitted) |
| local `main` vs `origin/main` | equal (`85a1e01…` both) — **caveat**: `git fetch origin main` timed out (network), so the remote ref could not be refreshed this run; comparison is against the locally stored `origin/main` ref |
| last 5 commits | `85a1e01` bound widened context ±2/5-budget + defer unreliable-provenance rows; `2bb6eb8` confidence-split deliverable; `9fa3a79` heuristic window widening; `f3d4ee4` retrieval-context rescue + regression gate; `b691f9e` ZH ch17 end_page fix |

## 2. Gates

| gate | command | result |
|---|---|---|
| Lint | `uv run ruff check .` | **All checks passed** |
| Tests | `uv run pytest` | **432 passed**, 5 warnings (SWIG/DeprecationWarning, non-blocking), 1.14 s |
| Whitespace | `git diff --check` | clean (exit 0) |
| Annotation validator (template state, split deliverable) | `uv run python scripts/validate_annotation_csv.py data/derived/annotation/annotation_pairs.csv --master-tsv data/derived/step4/full_novel_annotation_master.tsv --companion-csv data/derived/annotation/annotation_pairs_low_confidence.csv --min-confidence 0.40` | `rows: 1303 / template_state: True / deferred_low_confidence: 89 / violations: 0`, exit 0 |
| Context-regression gate | `uv run python scripts/check_context_regression.py --manifest data/derived/annotation/context_regression.json --master-tsv data/derived/step4/full_novel_annotation_master.tsv` | `cases: 23 / sides checked: 26 / in-context: 25 / manual_review: 1 / violations: 0`, exit 0 |

The validator re-derives the low-confidence split from the master (never
trusts the file pair to define row sets) and re-derives every machine column
cell; it passing means the split and all machine columns are reproducible from
the master at threshold 0.40.

## 3. Production artifacts (gitignored, verified in place)

Inputs present and complete: `data/parsed/` (DE/EN/ZH × ch01–17 CoNLL-U, incl.
`_nomwt` variants for DE), `data/extracted/full_novel/` (17 × contracted +
17 × uncontracted TSVs + `manifest.json`, summary reports `manifest_by_status: ok=34`),
`data/aligned/` (DE–EN and DE–ZH × ch01–17 JSONL).

Split deliverable (master → main + companion):

| file | rows | SHA-256 |
|---|---|---|
| `data/derived/step4/full_novel_annotation_master.tsv` | 1392 | `90d11bf68faf68c21379355da12d914e051db0e128477a5ab0abff71e891571a` |
| `data/derived/annotation/annotation_pairs.csv` | 1303 | `f878d6cb24f7125dbd3410aef0140af9e448a310f0bd2f26fd72b89d7501f625` |
| `data/derived/annotation/annotation_pairs_low_confidence.csv` | 89 | `daf2172241f4ccb4bf2636260cf05a450f449c883d3a4e5c6a91546f9f85d0ce` |

Set relations (machine-checked this run):

- main ∪ companion == master IDs exactly (no missing, no extra); 1303 + 89 = 1392.
- main ∩ companion = ∅ (disjoint); no duplicate IDs within any file.

Coverage and composition:

- Chapters: master and main cover ch1–17 with no gaps; companion spans
  ch{1–10, 12, 14–17} (no ch11/ch13 rows — those chapters produced no
  deferred candidates, not a coverage failure, since the union is exact).
- Form: master contracted 724 / uncontracted 668; main 680/623; companion
  44/45 (sums match master exactly).
- 13-preposition inventory confirmed in master summary (`shared_prepositions`,
  13 entries); `pool_total: 1392` of `candidate_total: 1394` (2 ineligible
  contracted rows excluded by the build).

Context provenance (master totals; main/companion sum to these exactly):

| provenance | EN | ZH |
|---|---|---|
| anchor_window | 997 | 896 |
| heuristic_widened | 197 | 218 |
| merge_widened | 188 | 269 |
| neighbor_fallback | 10 | 8 |
| manual_review | 0 | 1 |

Companion deferral reasons (`machine_low_conf_sides`): zh=70, en=14, en,zh=5
(companion also absorbs all neighbor_fallback/manual_review rows).

## 4. Reading-budget verification (≤5 target segments, ±2 widening)

Constants in code: `WIDENED_CONTEXT_WINDOW = 2`, `WIDENED_CONTEXT_BUDGET = 5`,
`FALLBACK_BRACKET_CAP = 6`, base anchor window ±1 (`src/hp_corpus/step4.py:236-240,854`).

Production distribution of context-segment counts per side:

- `anchor_window`: 2–5 segments typically (mode 3 = anchor ±1); **one ZH row
  has 6** (`dp_ch02_…p0020_s004#b001_t6-8`, alignment 1:4). Cause: the anchor
  group itself is 4 contiguous segments and is *always preserved in full*
  (documented rule, `src/hp_corpus/candidates.py:95-108`); ±1 around a
  4-anchor group = 6. The 5-segment budget bounds the *widened* view and the
  *added* neighbours, never the anchor group — so this is per-design, not a
  budget violation. No anchor_window row on either side otherwise exceeds 5.
- `heuristic_widened` / `merge_widened`: max exactly 5, never more — the
  ±2/5 budget is hard-respected in production data.
- `neighbor_fallback`: max 2 observed (cap 6).
- `manual_review` (1 ZH row, in companion): context ids present (5) but
  provenance forced to manual_review by the human-reviewed fallback plan when
  a reviewed locus falls outside the machine window
  (`src/hp_corpus/step4.py:972-977`); correctly deferred to the companion.

## 5. Discrepancies / caveats

1. `git fetch` to GitHub timed out — origin/main equality is asserted against
   the locally stored ref, not a live remote read.
2. One anchor_window row carries 6 context segments (see §4) — documented
   anchor-group-preservation rule, flagged here for transparency.
3. `docs/AI_CODING_INTERACTION_EVIDENCE.md` remains untracked/uncommitted
   (pre-existing state, intentional).
4. Main CSV's human-editable confidence columns are blank — expected in
   template state (human annotation not yet started).
5. Known design gap: no implemented rejoin path from (main + companion)
   reviewed CSVs to the single complete review CSV that
   `build_eligible_pool.py` requires — traced in Phase B of this run
   (not implemented, per instructions).

## 6. Classification

**machine-ready** — all machine stages (extraction, alignment, master build,
split deliverable, validators, gates) are reproducible and green; human
annotation has not started (CSV in template state), so the project is not
human-validated, analysis-ready, or at paper-conclusion stage. LLM-assisted
alignment QC in this repo is judge/QC evidence, **not** human gold.
