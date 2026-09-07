"""Run hybrid arbitration over the saved machine tuple evidence (pilot).

Experimental side path — NOT the production annotation workflow. This
runner makes **zero model calls**: it joins three already-saved pilot
artifacts and the production master,

  * the saved GLM proposal CSV (semantic counterpart proposer);
  * the saved constituent-v2 CSV + its per-row candidate dump (Stanza
    structure, eflomal lexical flags, LaBSE contextual ranks);
  * the saved eflomal-only baseline CSV (four-way diagnostic only);
  * the machine master TSV (retrieval contexts, provenance, anchors)

and writes one hybrid proposal per language side per row via
``hp_corpus.hybrid_arbitration``. The GLM form labels are recomputed
deterministically from the span's parse tokens; GLM ``omitted`` claims
are guarded (never final); GLM spans outside the strict DP-aligned
locus are routed as semantic disagreement. The GLM and deterministic
artifacts are read-only inputs — outputs land in new files only.

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
    HYBRID_COLUMNS,
    HYBRID_METHOD_ID,
    GlmSide,
    HybridDecision,
    LocalSide,
    SideFacts,
    arbitrate_side,
    build_hybrid_row,
    map_span_to_tokens,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
SIDES = (("en", "de_en", "en"), ("zh", "de_zh", "zh"))
_RELATIONS = ("exact", "containment")


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


def _related(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _dump_candidates(dump_side: dict) -> list[dict]:
    return dump_side.get("candidates") or [] if dump_side else []


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
    baseline_rows = _load_csv(args.baseline_csv)
    if not (set(glm_rows) == set(v2_rows) == set(baseline_rows)):
        return _fail(
            "ID_SETS_DIFFER",
            f"glm={len(glm_rows)} v2={len(v2_rows)} baseline={len(baseline_rows)}",
        )
    with open(args.candidates_json, encoding="utf-8") as f:
        dump = json.load(f)

    links = _load_links(args.links_dir)
    indexes = {pair: det._anchor_record_index(links[pair]) for pair in det.PAIRS}
    cache = det.ParseCache(args.parsed_dir)

    hybrid_rows: list[dict[str, str]] = []
    diagnostics: dict[str, dict] = {}
    route_counts = {s: Counter() for s, _, _ in SIDES}
    review_counts = {s: Counter() for s, _, _ in SIDES}
    evidence_counts = {s: Counter() for s, _, _ in SIDES}
    omission_outcomes = {s: Counter() for s, _, _ in SIDES}
    correction_kinds = Counter()
    correction_rows = []
    coverage = {s: Counter() for s, _, _ in SIDES}
    four_way = {
        "glm": {s: Counter() for s, _, _ in SIDES},
        "eflomal_baseline": {s: Counter() for s, _, _ in SIDES},
        "constituent_v2": {s: Counter() for s, _, _ in SIDES},
        "hybrid": {s: Counter() for s, _, _ in SIDES},
    }

    for dp in sorted(glm_rows):
        glm_row, v2_row, base_row = glm_rows[dp], v2_rows[dp], baseline_rows[dp]
        mrow = master_by_id.get(dp)
        if mrow is None:
            return _fail("MASTER_ROW_ABSENT", dp[:24])
        dump_row = dump.get(dp, {})

        decisions: dict[str, HybridDecision] = {}
        locals_: dict[str, LocalSide] = {}
        diag = {"german_pp": glm_row.get("german_pp", ""), "sides": {}}
        for side, pair, lang in SIDES:
            glm = GlmSide(
                status=glm_row.get(f"machine_{side}_status", ""),
                span=glm_row.get(f"machine_{side}_counterpart", ""),
                paper_form=glm_row.get(f"machine_{side}_paper_form", ""),
                detail_form=glm_row.get(f"machine_{side}_detail_form", ""),
            )
            v2_state = v2_row.get(f"machine_{side}_status", "")
            v2_evidence = v2_row.get(f"machine_{side}_evidence", "none")
            chosen = v2_row.get(f"machine_{side}_counterpart", "")
            category = ""
            for cand in _dump_candidates(dump_row.get(side)):
                if cand.get("span") == chosen:
                    category = cand.get("category") or ""
                    break
            local = LocalSide(
                state=v2_state,
                evidence=v2_evidence,
                chosen_span=chosen,
                form=(
                    v2_row.get(f"machine_{side}_paper_form", ""),
                    v2_row.get(f"machine_{side}_detail_form", ""),
                ),
                category=category,
            )
            locals_[side] = local

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
            mapping = None
            if glm.proposes_span:
                locus_segments = rec["tgt_segments"] if locus == "reliable" else context_ids
                layout = build_side_layout(
                    locus_segments, det._merged_sentences(cache, lang, locus_segments)
                )
                mapping = map_span_to_tokens(layout, glm.span)
            cands = _dump_candidates(dump_row.get(side))
            eflomal_covers = any(
                c.get("eflomal") and _related(glm.span, c.get("span", "")) for c in cands
            )
            top1_covers = any(
                c.get("ctx_rank") == 1 and _related(glm.span, c.get("span", "")) for c in cands
            )
            facts = SideFacts(
                locus=locus,
                span_in_context=span_in_context,
                mapping=mapping,
                in_locus_blocks=mapping is not None,
                eflomal_covers_span=eflomal_covers,
                contextual_top1_covers=top1_covers,
            )
            decision = arbitrate_side(glm, local, facts, lang)
            decisions[side] = decision

            route_counts[side][decision.route] += 1
            review_counts[side]["yes" if decision.requires_review else "no"] += 1
            evidence_counts[side][decision.evidence] += 1
            coverage[side]["with_span" if decision.counterpart else "without_span"] += 1
            if glm.omission_claim:
                omission_outcomes[side][decision.route] += 1
            if decision.form_correction is not None:
                old = decision.form_correction
                new = (decision.paper_form, decision.detail_form)
                correction_kinds[f"{old}→{new}"] += 1
                correction_rows.append(
                    {
                        "id": dp,
                        "side": side,
                        "glm_paper": old[0],
                        "glm_detail": old[1],
                        "hybrid_paper": new[0],
                        "hybrid_detail": new[1],
                    }
                )
            four_way["glm"][side][glm.status] += 1
            four_way["eflomal_baseline"][side][base_row.get(f"machine_{side}_status", "")] += 1
            four_way["constituent_v2"][side][v2_state] += 1
            four_way["hybrid"][side][decision.route] += 1
            diag["sides"][side] = {
                "glm": {
                    "status": glm.status,
                    "counterpart": glm.span,
                    "paper_form": glm.paper_form,
                    "detail_form": glm.detail_form,
                },
                "local": {
                    "state": local.state,
                    "evidence": local.evidence,
                    "counterpart": local.chosen_span,
                },
                "hybrid": {
                    "route": decision.route,
                    "counterpart": decision.counterpart,
                    "realization_type": decision.realization_type,
                    "paper_form": decision.paper_form,
                    "detail_form": decision.detail_form,
                    "evidence": decision.evidence,
                    "requires_review": decision.requires_review,
                    "form_correction": (
                        None
                        if decision.form_correction is None
                        else {
                            "glm": list(decision.form_correction),
                            "hybrid": [decision.paper_form, decision.detail_form],
                        }
                    ),
                },
            }
        diagnostics[dp] = diag
        hybrid_rows.append(build_hybrid_row(glm_row, decisions, locals_))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HYBRID_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(hybrid_rows)
    diag_path = args.output.with_name(args.output.stem + "_diagnostics.json")
    det.write_json(diag_path, diagnostics)

    report = {
        "schema_version": "hybrid-arbitration-run-v1",
        "method": HYBRID_METHOD_ID,
        "rows": len(hybrid_rows),
        "api_calls": 0,
        "routes": {s: dict(sorted(v.items())) for s, v in route_counts.items()},
        "requires_review": {s: dict(sorted(v.items())) for s, v in review_counts.items()},
        "evidence": {s: dict(sorted(v.items())) for s, v in evidence_counts.items()},
        "coverage": {s: dict(sorted(v.items())) for s, v in coverage.items()},
        "omission_claims_routed_never_final": {
            s: dict(sorted(v.items())) for s, v in omission_outcomes.items()
        },
        "form_corrections_applied_to_glm_spans": {
            "total": len(correction_rows),
            "by_transition": dict(sorted(correction_kinds.items())),
            "rows": correction_rows,
        },
        "semantic_drift_flagged": {
            s: route_counts[s]["semantic_disagreement"] for s, _, _ in SIDES
        },
        "four_way_state_distributions": {
            k: {s: dict(sorted(v.items())) for s, v in vv.items()} for k, vv in four_way.items()
        },
        "note": "Hybrid routes are machine proposal tiers, never claimed "
        "accuracy; GLM agreement is a signal, not gold.",
    }
    report_path = args.output.with_name(args.output.stem + "_run_report.json")
    det.write_json(report_path, report)

    print(f"rows: {len(hybrid_rows)}  api_calls: 0")
    for s, _, _ in SIDES:
        print(f"{s} routes: {dict(sorted(route_counts[s].items()))}")
        print(f"{s} requires_review: {dict(sorted(review_counts[s].items()))}")
        print(f"{s} omission_claims→route: {dict(sorted(omission_outcomes[s].items()))}")
    print(f"form_corrections: {len(correction_rows)} {dict(sorted(correction_kinds.items()))}")
    print(f"output: {args.output}")
    print(f"diagnostics: {diag_path}")
    print(f"report: {report_path}")
    return EXIT_OK


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
