"""Run the constituent-constrained deterministic approach (pilot v2).

Experimental side path — NOT the production annotation workflow.
Chains the pieces the pilot already has:

  * the frozen eflomal links (lexical anchor evidence);
  * the existing EN/ZH Stanza parses (constituent candidates);
  * the local LaBSE encoder as one contextual similarity signal
    (frozen, CPU fp32, no fine-tuning, no generative API);
  * the agreement-based routing policy and the frozen paper-form
    codebook from ``hp_corpus.constituent_candidates``.

Subcommands:

  ``annotate``  Run the approach over the development pack manifest and
                write the machine tuple CSV (shared schema + evidence
                columns), a per-row candidate dump (inspectable
                evidence), and a run report.

  ``compare``   Three-way comparison on the same pack: the saved GLM
                baseline (no new API calls), the saved eflomal-only
                deterministic baseline, and this approach. Span
                relations use clean denominators (exact /
                containment-boundary / divergent among rows where both
                methods produced a span).

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

from hp_corpus.constituent_candidates import (  # noqa: E402
    CONSTITUENT_COLUMNS,
    CONSTITUENT_METHOD_ID,
    RouteResult,
    build_constituent_row,
    eflomal_supported_indices,
    form_for_candidate,
    generate_candidates,
    route_side,
)
from hp_corpus.contextual_similarity import (  # noqa: E402
    ContextualSimilarity,
    config_summary,
    rank_candidates,
)
from hp_corpus.deterministic_tuples import (  # noqa: E402
    build_side_layout,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

METHOD_PARAMS_ID = "eflomal-frozen+stanza-parses+labse-cpu-v1"
CHOSEN_STATES = ("nominal_counterpart", "non_nominal_counterpart")


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _load_links(links_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for pair in det.PAIRS:
        path = links_dir / f"{pair}.links.jsonl"
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        out[pair] = records
    return out


def cmd_annotate(args: argparse.Namespace) -> int:
    if not args.master_tsv.exists():
        return _fail("MASTER_TSV_ABSENT", str(args.master_tsv))
    if not args.pack_manifest.exists():
        return _fail("PACK_MANIFEST_ABSENT", str(args.pack_manifest))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    try:
        master = det.read_master(args.master_tsv)
    except ValueError as exc:
        return _fail("MASTER_INVALID", str(exc))
    with open(args.pack_manifest, encoding="utf-8") as f:
        pack = json.load(f)
    strata = {r["datapoint_id"]: r["pack_stratum"] for r in pack.get("rows") or []}
    if not strata:
        return _fail("PACK_EMPTY", str(args.pack_manifest))

    try:
        similarity = ContextualSimilarity()
    except FileNotFoundError as exc:
        return _fail("CONTEXTUAL_MODEL_MISSING", str(exc))
    contextual_config = config_summary()

    links = _load_links(args.links_dir)
    indexes = {pair: det._anchor_record_index(links[pair]) for pair in det.PAIRS}
    cache = det.ParseCache(args.parsed_dir)
    vector_cache: dict[tuple[str, ...], list[list[float]]] = {}

    def layout_with_vectors(lang: str, segments: list[str]):
        layout = build_side_layout(
            segments, det._merged_sentences(cache, lang, segments)
        )
        key = (lang, *layout.block_ids)
        if key not in vector_cache:
            vector_cache[key] = similarity.word_vectors(
                [tok.form for tok in layout.tokens]
            )
        return layout, vector_cache[key]

    machine_rows: list[dict[str, str]] = []
    candidate_dump: dict[str, dict] = {}
    state_counter = {s: Counter() for s in ("en", "zh")}
    evidence_counter = {s: Counter() for s in ("en", "zh")}
    reason_counter: Counter = Counter()

    for row in master:
        dp = row["datapoint_id"]
        if dp not in strata:
            continue
        routed: dict[str, RouteResult] = {}
        forms: dict[str, tuple[str, str]] = {}
        counts: dict[str, int] = {}
        dump: dict[str, dict] = {}
        for side, pair, lang in (("en", "de_en", "en"), ("zh", "de_zh", "zh")):
            anchor = json.loads(row.get(f"{side}_sentence_ids") or "[]")
            provenance = (row.get(f"{side}_context_provenance") or "").strip()
            # Unreliable locus = no anchor at all, or the machine could
            # not retrieve the side (manual_review — the reviewed true
            # locus sits outside the machine window). neighbor_fallback
            # sides are bracket contexts whose locus may well be present
            # (the recovered regression cases): proceed, and let the
            # two-signal agreement decide.
            locus_reliable = bool(anchor) and provenance != "manual_review"
            if not anchor:
                routed[side] = RouteResult("not_aligned", reason="no_anchor")
                counts[side] = 0
                continue
            if not locus_reliable:
                routed[side] = RouteResult("not_aligned", reason="unreliable_locus")
                counts[side] = 0
                continue
            rec = next(
                (
                    r
                    for r in indexes[pair].get(tuple(anchor), [])
                    if row["de_source_segment_id"] in r["de_segments"]
                ),
                None,
            )
            if rec is None:
                routed[side] = RouteResult("unresolved", reason="anchor_record_not_found")
                counts[side] = 0
                reason_counter[f"{side}.anchor_record_not_found"] += 1
                continue
            de_layout, de_vecs = layout_with_vectors("de", rec["de_segments"])
            tgt_layout, tgt_vecs = layout_with_vectors(lang, rec["tgt_segments"])
            pp = det._pp_slice(
                de_layout,
                row["de_parse_block_id"],
                int(row["de_token_start"]),
                int(row["de_token_end"]),
            )
            if pp is None:
                routed[side] = RouteResult("unresolved", reason="pp_block_not_in_record")
                counts[side] = 0
                reason_counter[f"{side}.pp_block_not_in_record"] += 1
                continue
            candidates = generate_candidates(tgt_layout, lang)
            counts[side] = len(candidates)
            supported = eflomal_supported_indices(
                candidates, pp, set(map(tuple, rec["fwd"])), set(map(tuple, rec["rev"]))
            )
            contextual = rank_candidates(de_vecs, pp, tgt_vecs, candidates)
            result = route_side(candidates, supported, contextual, locus_reliable=True)
            routed[side] = result
            if result.chosen is not None:
                forms[side] = form_for_candidate(result.chosen, tgt_layout, lang)
            ranked = sorted(contextual.items(), key=lambda kv: -kv[1])
            rank_of = {idx: i + 1 for i, (idx, _) in enumerate(ranked)}
            dump[side] = {
                "state": result.state,
                "evidence": result.evidence,
                "contextual_margin": result.contextual_margin,
                "candidates": [
                    {
                        "span": c.span_text,
                        "category": c.category,
                        "eflomal": i in supported,
                        "ctx_sim": contextual.get(i),
                        "ctx_rank": rank_of.get(i),
                    }
                    for i, c in enumerate(candidates)
                ],
            }
            if result.state not in CHOSEN_STATES:
                reason_counter[f"{side}.{result.state}"] += 1

        for s in ("en", "zh"):
            state_counter[s][routed[s].state if routed.get(s) else "not_aligned"] += 1
            evidence_counter[s][routed[s].evidence if routed.get(s) else "none"] += 1
        machine_rows.append(
            build_constituent_row(
                row, routed, forms, counts,
                pack_stratum=strata[dp], method_params_id=METHOD_PARAMS_ID,
            )
        )
        candidate_dump[dp] = dump

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(CONSTITUENT_COLUMNS), lineterminator="\r\n"
        )
        w.writeheader()
        w.writerows(machine_rows)
    dump_path = args.output.with_name(args.output.stem + "_candidates.json")
    det.write_json(dump_path, candidate_dump)

    form_agg = {f"{s}_{k}": Counter() for s in ("en", "zh") for k in ("paper", "detail")}
    for r in machine_rows:
        for s in ("en", "zh"):
            if r[f"machine_{s}_status"] in CHOSEN_STATES:
                form_agg[f"{s}_paper"][r[f"machine_{s}_paper_form"]] += 1
                form_agg[f"{s}_detail"][r[f"machine_{s}_detail_form"] or "(blank)"] += 1
    report = {
        "schema_version": "constituent-tuples-run-v1",
        "method": CONSTITUENT_METHOD_ID,
        "method_params_id": METHOD_PARAMS_ID,
        "contextual_config": contextual_config,
        "rows": len(machine_rows),
        "states": {k: dict(sorted(v.items())) for k, v in state_counter.items()},
        "evidence": {k: dict(sorted(v.items())) for k, v in evidence_counter.items()},
        "forms": {k: dict(sorted(v.items())) for k, v in form_agg.items()},
        "non_chosen_reasons": dict(sorted(reason_counter.items())),
    }
    report_path = args.output.with_name(args.output.stem + "_run_report.json")
    det.write_json(report_path, report)

    print(f"rows: {len(machine_rows)}")
    for s in ("en", "zh"):
        print(f"{s}_status: {dict(sorted(state_counter[s].items()))}")
        print(f"{s}_evidence: {dict(sorted(evidence_counter[s].items()))}")
        print(f"{s}_paper_form: {dict(sorted(form_agg[f'{s}_paper'].items()))}")
    print(f"output: {args.output}")
    print(f"candidates: {dump_path}")
    print(f"report: {report_path}")
    return EXIT_OK


def _span_relation(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return "both_unavailable"
    if not a or not b:
        return "unavailable"
    if a == b:
        return "exact"
    if a in b or b in a:
        return "containment"
    return "divergent"


def cmd_compare(args: argparse.Namespace) -> int:
    paths = {
        "glm": args.glm_csv,
        "baseline": args.baseline_csv,
        "new": args.new_csv,
    }
    for name, path in paths.items():
        if not path.exists():
            return _fail("CSV_ABSENT", f"{name}: {path}")
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    def load(path: Path) -> dict[str, dict[str, str]]:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    runs = {name: load(path) for name, path in paths.items()}
    id_sets = [set(v) for v in runs.values()]
    if not id_sets[0] == id_sets[1] == id_sets[2]:
        return _fail("ID_SETS_DIFFER", str([len(s) for s in id_sets]))

    n = len(runs["new"])
    relation = {f"{a}_vs_{b}": Counter() for a, b in (("new", "glm"), ("new", "baseline"))}
    fallback_rows = []
    evidence_pairs = Counter()
    for dp in sorted(runs["new"]):
        new = runs["new"][dp]
        need_fallback = False
        for s in ("en", "zh"):
            new_span = new[f"machine_{s}_counterpart"]
            for other in ("glm", "baseline"):
                other_span = runs[other][dp][f"machine_{s}_counterpart"]
                relation[f"new_vs_{other}"][_span_relation(new_span, other_span)] += 1
            if new[f"machine_{s}_status"] not in CHOSEN_STATES:
                need_fallback = True
        evidence_pairs[(new["machine_en_evidence"], new["machine_zh_evidence"])] += 1
        if need_fallback:
            fallback_rows.append(dp)

    glm_state = {s: Counter() for s in ("en", "zh")}
    base_state = {s: Counter() for s in ("en", "zh")}
    for dp in runs["new"]:
        for s in ("en", "zh"):
            glm_state[s][runs["glm"][dp][f"machine_{s}_status"]] += 1
            base_state[s][runs["baseline"][dp][f"machine_{s}_status"]] += 1

    report = {
        "schema_version": "constituent-three-way-v1",
        "rows": n,
        "state_distributions": {
            "glm": {k: dict(sorted(v.items())) for k, v in glm_state.items()},
            "eflomal_baseline": {k: dict(sorted(v.items())) for k, v in base_state.items()},
        },
        "span_relations": {k: dict(sorted(v.items())) for k, v in relation.items()},
        "evidence_pairs": {f"en={a}|zh={b}": c for (a, b), c in sorted(evidence_pairs.items())},
        "fallback_rows": fallback_rows,
        "fallback_share": round(len(fallback_rows) / n, 4) if n else 0.0,
        "note": "GLM agreement is a signal, never gold.",
    }
    det.write_json(args.output, report)
    print(f"rows: {n}")
    for k, v in relation.items():
        print(f"{k}: {dict(sorted(v.items()))}")
    print(f"fallback_rows: {len(fallback_rows)} ({report['fallback_share']:.0%})")
    print(f"evidence_pairs: {report['evidence_pairs']}")
    print(f"output: {args.output}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("annotate", help="constituent-constrained tuples for a pack")
    p.add_argument("--master-tsv", type=Path, required=True)
    p.add_argument("--pack-manifest", type=Path, required=True)
    p.add_argument("--links-dir", type=Path, required=True)
    p.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("compare", help="three-way: GLM vs eflomal baseline vs new")
    p.add_argument("--glm-csv", type=Path, required=True)
    p.add_argument("--baseline-csv", type=Path, required=True)
    p.add_argument("--new-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
