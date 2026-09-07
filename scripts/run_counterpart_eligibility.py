"""Run the simplified counterpart-eligibility gate (pilot).

Experimental side path — NOT the production annotation workflow, and
**zero model calls**: this consumes the same saved artifacts as the
hybrid arbitration runner (saved GLM proposals, saved constituent-v2
routing + candidate dump, eflomal-only baseline for context, machine
master, CoNLL-U parses) and asks exactly one research-facing question
per target side:

    is there an identifiable nominal/referential expression in the
    reliable translation locus usable as the counterpart of the German
    PP for surface-form comparison?

Per side: ``comparable_counterpart`` / ``no_comparable_nominal_counterpart``
/ ``unresolved``. GLM spans are semantic anchors: the full containing
nominal constituent is recovered from the parse before the form is
classified. A GLM ``omitted`` claim is never honored as ``omitted``;
verbalizations and restructurings are simply not nominal counterparts
for the core table. Row level: ``core_tuple_eligible`` /
``excluded_no_comparable_counterpart`` / ``unresolved`` — excluded and
unresolved rows are preserved with reasons (eligibility funnel), never
deleted.

Outputs (gitignored): eligibility CSV + run report with aggregate
counts only. Machine diagnostics, never claimed accuracy.

Stdout discipline: aggregate counts only — never corpus text, spans,
lemmas, or datapoint ids.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_deterministic_tuples", REPO_ROOT / "scripts" / "run_deterministic_tuples.py"
)
det = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(det)

from hp_corpus.constituent_candidates import classify_locus  # noqa: E402
from hp_corpus.deterministic_tuples import build_side_layout  # noqa: E402
from hp_corpus.hybrid_arbitration import (  # noqa: E402
    ELIGIBILITY_COLUMNS,
    CandidateRef,
    EligibilityDecision,
    GlmSide,
    LocalSide,
    SideFacts,
    build_eligibility_row,
    eligibility_side,
    map_span_to_tokens,
    row_eligibility_for,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
SIDES = (("en", "de_en", "en"), ("zh", "de_zh", "zh"))


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _load_links(links_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for pair in det.PAIRS:
        records = []
        with open(links_dir / f"{pair}.links.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        out[pair] = records
    return out


def _load_csv(path: Path, key: str = "id") -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def _candidate_refs(dump_side: dict) -> list[CandidateRef]:
    if not dump_side:
        return []
    return [
        CandidateRef(
            span=c.get("span", ""),
            category=c.get("category", ""),
            eflomal=bool(c.get("eflomal")),
            ctx_rank=c.get("ctx_rank"),
        )
        for c in dump_side.get("candidates") or []
    ]


def cmd_run(args: argparse.Namespace) -> int:
    paths = {
        "master": args.master_tsv,
        "glm_csv": args.glm_csv,
        "v2_csv": args.constituent_csv,
        "candidates": args.candidates_json,
        "baseline_csv": args.baseline_csv,
    }
    for name, path in paths.items():
        if not path.exists():
            return _fail("INPUT_ABSENT", f"{name}: {path}")
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    try:
        master = det.read_master(args.master_tsv)
    except ValueError as exc:
        return _fail("MASTER_INVALID", str(exc))
    master_by_id = {r["datapoint_id"]: r for r in master}
    glm_rows = _load_csv(args.glm_csv)
    v2_rows = _load_csv(args.constituent_csv)
    if set(glm_rows) != set(v2_rows):
        return _fail("ID_SETS_DIFFER", f"glm={len(glm_rows)} v2={len(v2_rows)}")
    with open(args.candidates_json, encoding="utf-8") as f:
        dump = json.load(f)

    links = _load_links(args.links_dir)
    indexes = {pair: det._anchor_record_index(links[pair]) for pair in det.PAIRS}
    cache = det.ParseCache(args.parsed_dir)

    elig_rows: list[dict[str, str]] = []
    state_counts = {s: Counter() for s, _, _ in SIDES}
    evidence_counts = {s: Counter() for s, _, _ in SIDES}
    review_counts = {s: Counter() for s, _, _ in SIDES}
    row_counts: Counter = Counter()
    funnel_side_pairs = Counter()
    omission_claims = {s: Counter() for s, _, _ in SIDES}

    for dp in sorted(glm_rows):
        glm_row, v2_row = glm_rows[dp], v2_rows[dp]
        mrow = master_by_id.get(dp)
        if mrow is None:
            return _fail("MASTER_ROW_ABSENT", dp[:24])
        dump_row = dump.get(dp, {})

        decisions: dict[str, EligibilityDecision] = {}
        for side, pair, lang in SIDES:
            glm = GlmSide(
                status=glm_row.get(f"machine_{side}_status", ""),
                span=glm_row.get(f"machine_{side}_counterpart", ""),
                paper_form=glm_row.get(f"machine_{side}_paper_form", ""),
                detail_form=glm_row.get(f"machine_{side}_detail_form", ""),
            )
            local = LocalSide(
                state=v2_row.get(f"machine_{side}_status", ""),
                evidence=v2_row.get(f"machine_{side}_evidence", "none"),
                chosen_span=v2_row.get(f"machine_{side}_counterpart", ""),
                form=(
                    v2_row.get(f"machine_{side}_paper_form", ""),
                    v2_row.get(f"machine_{side}_detail_form", ""),
                ),
                category="",
            )

            anchor = json.loads(mrow.get(f"{side}_sentence_ids") or "[]")
            provenance = (mrow.get(f"{side}_context_provenance") or "").strip()
            context_ids = json.loads(mrow.get(f"{side}_context_ids") or "[]")
            rec = (
                next(
                    (
                        r
                        for r in indexes[pair].get(tuple(anchor), [])
                        if mrow["de_source_segment_id"] in r["de_segments"]
                    ),
                    None,
                )
                if anchor
                else None
            )
            locus = classify_locus(anchor, provenance, rec is not None, context_ids)

            context_text = mrow.get(f"{side}_context_text") or ""
            span_in_context = bool(glm.span.strip()) and glm.span in context_text
            layout = None
            mapping = None
            in_locus_blocks = False
            if glm.proposes_span or local.has_overt_choice:
                locus_segments = rec["tgt_segments"] if locus == "reliable" else context_ids
                layout = build_side_layout(
                    locus_segments, det._merged_sentences(cache, lang, locus_segments)
                )
            if glm.proposes_span and layout is not None:
                in_locus_blocks = any(glm.span in bt for bt in layout.block_texts)
                mapping = map_span_to_tokens(layout, glm.span)
            candidates = _candidate_refs(dump_row.get(side))
            facts = SideFacts(
                locus=locus,
                span_in_context=span_in_context,
                mapping=mapping,
                in_locus_blocks=in_locus_blocks,
                eflomal_covers_span=any(
                    c.eflomal and _related(glm.span, c.span) for c in candidates
                ),
                contextual_top1_covers=any(
                    c.ctx_rank == 1 and _related(glm.span, c.span) for c in candidates
                ),
                layout=layout,
            )
            decision = eligibility_side(glm, local, facts, candidates, lang)
            decisions[side] = decision

            state_counts[side][decision.eligibility] += 1
            evidence_counts[side][decision.evidence] += 1
            review_counts[side]["yes" if decision.requires_review else "no"] += 1
            if glm.omission_claim:
                omission_claims[side][decision.eligibility] += 1

        de_valid = glm_row.get("machine_de_valid_proposal", "") == "include"
        row_elig = row_eligibility_for(de_valid, decisions)
        row_counts[row_elig] += 1
        funnel_side_pairs[
            f"en={decisions['en'].eligibility[:10]}|zh={decisions['zh'].eligibility[:10]}"
        ] += 1
        elig_rows.append(build_eligibility_row(glm_row, decisions, row_elig))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ELIGIBILITY_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(elig_rows)

    report = {
        "schema_version": "counterpart-eligibility-run-v1",
        "rows": len(elig_rows),
        "api_calls": 0,
        "states": {s: dict(sorted(v.items())) for s, v in state_counts.items()},
        "evidence": {s: dict(sorted(v.items())) for s, v in evidence_counts.items()},
        "requires_review": {s: dict(sorted(v.items())) for s, v in review_counts.items()},
        "row_eligibility_funnel": dict(sorted(row_counts.items())),
        "side_state_pairs": dict(sorted(funnel_side_pairs.items())),
        "glm_omission_claims_final_states": {
            s: dict(sorted(v.items())) for s, v in omission_claims.items()
        },
        "note": "Machine diagnostics, never claimed accuracy. "
        "no_comparable_nominal_counterpart is NOT an omitted claim; "
        "unresolved is never a linguistic absence. Excluded rows are "
        "preserved with reasons for the eligibility funnel.",
    }
    report_path = args.output.with_name(args.output.stem + "_run_report.json")
    det.write_json(report_path, report)

    print(f"rows: {len(elig_rows)}  api_calls: 0")
    for s, _, _ in SIDES:
        print(f"{s} states: {dict(sorted(state_counts[s].items()))}")
        print(f"{s} evidence: {dict(sorted(evidence_counts[s].items()))}")
    print(f"row funnel: {dict(sorted(row_counts.items()))}")
    print(
        f"omission_claims→state: en={dict(sorted(omission_claims['en'].items()))} "
        f"zh={dict(sorted(omission_claims['zh'].items()))}"
    )
    print(f"output: {args.output}")
    print(f"report: {report_path}")
    return EXIT_OK


def _related(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-tsv", type=Path, required=True)
    ap.add_argument("--glm-csv", type=Path, required=True)
    ap.add_argument("--constituent-csv", type=Path, required=True)
    ap.add_argument("--candidates-json", type=Path, required=True)
    ap.add_argument("--baseline-csv", type=Path, required=True)
    ap.add_argument("--links-dir", type=Path, required=True)
    ap.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force-output", action="store_true")
    args = ap.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
