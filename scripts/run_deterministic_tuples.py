"""Run the deterministic-first local baseline beside the GLM pilot.

Experimental side path — NOT the production annotation workflow. Reads
only production artifacts (CoNLL-U parses, LaBSE+DP alignment records,
the machine master) plus the frozen eflomal links; writes its own
gitignored outputs under ``data/derived/machine_preannotation/``.

Subcommands:

  ``train``     Build the DE-EN / DE-ZH bitext from the production
                alignment records (token lines from the parses), run
                eflomal (model 3, both directions), and freeze the
                per-record links as a JSONL artifact with sha256s —
                the reproducibility anchor (eflomal sampling has no
                user seed; downstream steps are pure functions of the
                frozen links).

  ``annotate``  Run the deterministic path over a pack manifest: locate
                each row's anchor alignment record, intersect the
                eflomal links onto the row's DE PP token span, recover
                the smallest defensible target span (exact substring of
                the target block text AND of the master retrieval
                context), classify EN/ZH paper forms by the frozen
                deterministic rules, and write a machine tuple CSV on
                the shared schema with the deterministic method id.

  ``compare``   Join the deterministic artifact with the GLM pilot CSV
                on row id and report status/span/form agreement.
                GLM agreement is a signal, never gold.

Stdout discipline: aggregate counts only — never corpus text, spans,
lemmas, or datapoint ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from hp_corpus.deterministic_tuples import (
    DETERMINISTIC_METHOD_ID,
    EFLOMAL_PARAMETERS,
    LINKS_SCHEMA_VERSION,
    SpanRecovery,
    build_bitext,
    build_deterministic_row,
    build_side_layout,
    chinese_form,
    eflomal_input_line,
    english_form,
    load_alignment_records,
    parse_links_file,
    read_conllu,
    recover_target_span,
    sha256_file,
)
from hp_corpus.machine_preannotation import MACHINE_TUPLE_COLUMNS

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

PAIRS = ("de_en", "de_zh")
CHAPTERS = range(1, 18)
METHOD_PARAMS_ID = "eflomal2.0-m3-s1-intersect-v1"

_ID_CHAPTER_RE = re.compile(r"_ch(\d{2})_")


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def read_master(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    seen: set[str] = set()
    for r in rows:
        dp = r.get("datapoint_id", "")
        if not dp:
            raise ValueError("master row with empty datapoint_id")
        if dp in seen:
            raise ValueError(f"duplicate datapoint_id in master: {dp}")
        seen.add(dp)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MACHINE_TUPLE_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _chapter_of(segment_id: str) -> int:
    m = _ID_CHAPTER_RE.search(segment_id)
    if not m:
        raise ValueError(f"cannot parse chapter from id: {segment_id[:24]}…")
    return int(m.group(1))


class ParseCache:
    """Lazy per-(lang, chapter) CoNLL-U reader."""

    def __init__(self, parsed_dir: Path):
        self.parsed_dir = parsed_dir
        self._cache: dict[tuple[str, int], dict] = {}

    def get(self, lang: str, chapter: int) -> dict:
        key = (lang, chapter)
        if key not in self._cache:
            suffix = "_nomwt" if lang == "de" else ""
            self._cache[key] = read_conllu(
                self.parsed_dir / f"hp1_{lang}_ch{chapter:02d}{suffix}.conllu"
            )
        return self._cache[key]


# --- train -------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    try:
        from eflomal import Aligner
    except ImportError:
        return _fail("EFLOMAL_MISSING", "install the wordalign extra (eflomal)")

    out_dir: Path = args.output_dir
    for path in [out_dir / f"{p}.links.jsonl" for p in PAIRS] + [
        out_dir / f"{p}.manifest.json" for p in PAIRS
    ]:
        if path.exists() and not args.force_output:
            return _fail("OUTPUT_EXISTS", f"{path}; pass --force-output to overwrite")

    aligner = Aligner(
        model=EFLOMAL_PARAMETERS["model"],
        n_samplers=EFLOMAL_PARAMETERS["n_samplers"],
    )
    for pair in PAIRS:
        src_lang, tgt_lang = pair.split("_")
        records: list[dict] = []
        for ch in CHAPTERS:
            records.extend(
                load_alignment_records(args.aligned_dir / f"hp1_{pair}_ch{ch:02d}.jsonl")
            )
        bitext = build_bitext(records)
        if not bitext:
            return _fail("BITEXT_EMPTY", pair)

        de_cache = ParseCache(args.parsed_dir)
        tgt_cache = ParseCache(args.parsed_dir)
        src_lines: list[str] = []
        tgt_lines: list[str] = []
        line_lengths: list[tuple[int, int]] = []
        for entry in bitext:
            de_layout = build_side_layout(
                entry["de_segments"],
                _merged_sentences(de_cache, src_lang, entry["de_segments"]),
            )
            tgt_layout = build_side_layout(
                entry["tgt_segments"],
                _merged_sentences(tgt_cache, tgt_lang, entry["tgt_segments"]),
            )
            line_lengths.append((len(de_layout.tokens), len(tgt_layout.tokens)))
            if not de_layout.tokens or not tgt_layout.tokens:
                src_lines.append("")
                tgt_lines.append("")
                continue
            src_lines.append(eflomal_input_line(de_layout))
            tgt_lines.append(eflomal_input_line(tgt_layout))

        fwd_path = out_dir / f"{pair}.fwd.links"
        rev_path = out_dir / f"{pair}.rev.links"
        out_dir.mkdir(parents=True, exist_ok=True)
        aligner.align(
            iter(src_lines),
            iter(tgt_lines),
            links_filename_fwd=str(fwd_path),
            links_filename_rev=str(rev_path),
            quiet=True,
        )
        fwd = parse_links_file(fwd_path)
        rev = parse_links_file(rev_path)
        if len(fwd) != len(bitext) or len(rev) != len(bitext):
            return _fail(
                "LINKS_COUNT_MISMATCH",
                f"{pair}: {len(fwd)}/{len(rev)} vs {len(bitext)} records",
            )

        # Fail closed on any link addressing a token outside its
        # sentence — this is exactly the class of bug created by token
        # forms containing whitespace, and it must never reach the
        # frozen links artifact.
        out_of_range = 0
        for (n_src, n_tgt), f_links, r_links in zip(line_lengths, fwd, rev, strict=True):
            for links in (f_links, r_links):
                for s, t in links:
                    if s >= n_src or t >= n_tgt:
                        out_of_range += 1
        if out_of_range:
            return _fail("LINKS_OUT_OF_RANGE", f"{pair}: {out_of_range} link(s)")

        links_path = out_dir / f"{pair}.links.jsonl"
        with open(links_path, "w", encoding="utf-8") as f:
            for entry, f_links, r_links in zip(bitext, fwd, rev, strict=True):
                if not entry["de_segments"]:
                    continue
                f.write(
                    json.dumps(
                        {
                            "de_segments": entry["de_segments"],
                            "tgt_segments": entry["tgt_segments"],
                            "fwd": sorted(f_links),
                            "rev": sorted(r_links),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": LINKS_SCHEMA_VERSION,
            "pair": pair,
            "records": len(bitext),
            "parameters": EFLOMAL_PARAMETERS,
            "links_sha256": {
                "fwd": sha256_file(fwd_path),
                "rev": sha256_file(rev_path),
            },
        }
        write_json(out_dir / f"{pair}.manifest.json", manifest)
        print(f"{pair}: records={len(bitext)} fwd_links={sum(len(s) for s in fwd)}")
    print(f"output_dir: {out_dir}")
    return EXIT_OK


def _merged_sentences(cache: ParseCache, lang: str, segment_ids: list[str]) -> dict:
    merged: dict = {}
    for seg in segment_ids:
        merged.update(cache.get(lang, _chapter_of(seg)))
    return merged


# --- annotate ------------------------------------------------------------------------


def _load_links(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _anchor_record_index(links: list[dict]) -> dict[tuple[str, ...], list[dict]]:
    index: dict[tuple[str, ...], list[dict]] = {}
    for rec in links:
        index.setdefault(tuple(rec["tgt_segments"]), []).append(rec)
    return index


def _pp_slice(de_layout, de_parse_block_id: str, token_start: int, token_end: int):
    """0-based PP token positions within the record's DE concatenation."""
    if de_parse_block_id not in de_layout.block_ids:
        return None
    slot = de_layout.block_ids.index(de_parse_block_id)
    block_start = _block_token_start(de_layout, slot)
    lo = block_start + token_start - 1
    hi = block_start + token_end - 1
    if not (0 <= lo <= hi < len(de_layout.tokens)):
        return None
    return range(lo, hi + 1)


def _block_token_start(de_layout, slot: int) -> int:
    start = 0
    for s in de_layout.token_block:
        if s == slot:
            return start
        start += 1
    return start


def cmd_annotate(args: argparse.Namespace) -> int:
    if not args.master_tsv.exists():
        return _fail("MASTER_TSV_ABSENT", str(args.master_tsv))
    if not args.pack_manifest.exists():
        return _fail("PACK_MANIFEST_ABSENT", str(args.pack_manifest))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    try:
        master = read_master(args.master_tsv)
    except ValueError as exc:
        return _fail("MASTER_INVALID", str(exc))
    with open(args.pack_manifest, encoding="utf-8") as f:
        pack = json.load(f)
    strata = {r["datapoint_id"]: r["pack_stratum"] for r in pack.get("rows") or []}
    if not strata:
        return _fail("PACK_EMPTY", str(args.pack_manifest))

    links = {
        pair: _load_links(args.links_dir / f"{pair}.links.jsonl") for pair in PAIRS
    }
    indexes = {pair: _anchor_record_index(links[pair]) for pair in PAIRS}
    cache = ParseCache(args.parsed_dir)

    machine_rows: list[dict[str, str]] = []
    reasons: Counter = Counter()
    absorbed: Counter = Counter()
    downgraded: Counter = Counter()

    for row in master:
        dp = row["datapoint_id"]
        if dp not in strata:
            continue
        recoveries: dict[str, SpanRecovery | None] = {}
        forms: dict[str, tuple[str, str] | None] = {}
        for side, pair, lang in (("en", "de_en", "en"), ("zh", "de_zh", "zh")):
            anchor = json.loads(row.get(f"{side}_sentence_ids") or "[]")
            if not anchor:
                recoveries[side] = None
                forms[side] = None
                continue
            candidates = indexes[pair].get(tuple(anchor), [])
            rec = next(
                (r for r in candidates if row["de_source_segment_id"] in r["de_segments"]),
                None,
            )
            if rec is None:
                recoveries[side] = SpanRecovery("unresolved", reason="anchor_record_not_found")
                forms[side] = None
                continue
            de_layout = build_side_layout(
                rec["de_segments"],
                _merged_sentences(cache, "de", rec["de_segments"]),
            )
            tgt_layout = build_side_layout(
                rec["tgt_segments"],
                _merged_sentences(cache, lang, rec["tgt_segments"]),
            )
            pp = _pp_slice(
                de_layout,
                row["de_parse_block_id"],
                int(row["de_token_start"]),
                int(row["de_token_end"]),
            )
            if pp is None:
                recoveries[side] = SpanRecovery("unresolved", reason="pp_block_not_in_record")
                forms[side] = None
                continue
            recovery = recover_target_span(
                pp,
                de_layout,
                tgt_layout,
                set(map(tuple, rec["fwd"])),
                set(map(tuple, rec["rev"])),
                lang=lang,
            )
            if recovery.status == "aligned":
                context = row.get(f"{side}_context_text") or ""
                if recovery.span_text not in context:
                    downgraded[side] += 1
                    recovery = SpanRecovery("unresolved", reason="not_in_context")
            if recovery.status != "aligned":
                reasons[f"{side}.{recovery.reason}"] += 1
            if recovery.absorbed_determiner:
                absorbed[side] += 1
            recoveries[side] = recovery
            forms[side] = (
                english_form(recovery.span_tokens)
                if lang == "en"
                else chinese_form(recovery.span_tokens)
            ) if recovery.status == "aligned" else None

        machine_rows.append(
            build_deterministic_row(
                row,
                recoveries["en"],
                recoveries["zh"],
                forms["en"],
                forms["zh"],
                pack_stratum=strata[dp],
                method_params_id=METHOD_PARAMS_ID,
            )
        )

    write_csv(args.output, machine_rows)

    agg: dict[str, Counter] = {f"{s}_status": Counter() for s in ("en", "zh")}
    form_agg: dict[str, Counter] = {
        f"{s}_{k}": Counter() for s in ("en", "zh") for k in ("paper_form", "detail_form")
    }
    de_agg: Counter = Counter()
    for r in machine_rows:
        de_agg[r["machine_de_valid_proposal"]] += 1
        for s in ("en", "zh"):
            agg[f"{s}_status"][r[f"machine_{s}_status"]] += 1
            if r[f"machine_{s}_status"] == "aligned":
                form_agg[f"{s}_paper_form"][r[f"machine_{s}_paper_form"]] += 1
                form_agg[f"{s}_detail_form"][r[f"machine_{s}_detail_form"] or "(blank)"] += 1

    report = {
        "schema_version": "deterministic-tuples-run-v1",
        "method": DETERMINISTIC_METHOD_ID,
        "method_params_id": METHOD_PARAMS_ID,
        "parameters": EFLOMAL_PARAMETERS,
        "rows": len(machine_rows),
        "de_proposal": dict(sorted(de_agg.items())),
        "statuses": {k: dict(sorted(v.items())) for k, v in agg.items()},
        "forms": {k: dict(sorted(v.items())) for k, v in form_agg.items()},
        "unresolved_reasons": dict(sorted(reasons.items())),
        "absorbed_determiner": dict(sorted(absorbed.items())),
        "context_gate_downgrades": dict(sorted(downgraded.items())),
    }
    report_path = args.output.with_name(args.output.stem + "_run_report.json")
    write_json(report_path, report)

    print(f"rows: {len(machine_rows)}")
    for s in ("en", "zh"):
        print(f"{s}_status: {dict(sorted(agg[f'{s}_status'].items()))}")
    print(f"de_proposal: {dict(sorted(de_agg.items()))}")
    print(f"en_paper_form: {dict(sorted(form_agg['en_paper_form'].items()))}")
    print(f"zh_paper_form: {dict(sorted(form_agg['zh_paper_form'].items()))}")
    print(f"unresolved_reasons: {dict(sorted(reasons.items()))}")
    print(f"output: {args.output}")
    print(f"report: {report_path}")
    return EXIT_OK


# --- compare ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    for path in (args.glm_csv, args.deterministic_csv):
        if not path.exists():
            return _fail("CSV_ABSENT", str(path))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    def load(path: Path) -> dict[str, dict[str, str]]:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    glm = load(args.glm_csv)
    det = load(args.deterministic_csv)
    if set(glm) != set(det):
        return _fail(
            "ID_SETS_DIFFER",
            f"glm={len(glm)} det={len(det)} shared={len(set(glm) & set(det))}",
        )

    status_matrix = {s: Counter() for s in ("en", "zh")}
    span_agree = Counter()
    form_agree = Counter()
    both_aligned = Counter()
    disagreement_ids = {s: [] for s in ("en", "zh")}
    api_free_naive = 0
    api_free_conservative = 0
    for dp in sorted(glm):
        g, d = glm[dp], det[dp]
        sides_fully_aligned = True
        agrees = True
        for s in ("en", "zh"):
            gs, ds = g[f"machine_{s}_status"], d[f"machine_{s}_status"]
            status_matrix[s][f"glm={gs}|det={ds}"] += 1
            if ds != "aligned":
                sides_fully_aligned = False
            if ds == "aligned":
                if gs == "aligned":
                    both_aligned[s] += 1
                    gspan = (g[f"machine_{s}_counterpart"] or "").strip()
                    dspan = (d[f"machine_{s}_counterpart"] or "").strip()
                    if gspan == dspan:
                        span_agree[s] += 1
                    else:
                        agrees = False
                        disagreement_ids[s].append(dp)
                    gform = g[f"machine_{s}_paper_form"]
                    dform = d[f"machine_{s}_paper_form"]
                    if gform == dform:
                        form_agree[s] += 1
                    else:
                        agrees = False
                        if dp not in disagreement_ids[s]:
                            disagreement_ids[s].append(dp)
                else:
                    agrees = False
                    disagreement_ids[s].append(dp)
        if sides_fully_aligned:
            api_free_naive += 1
            if agrees:
                api_free_conservative += 1

    report = {
        "schema_version": "deterministic-vs-glm-v1",
        "rows": len(glm),
        "status_matrix": {k: dict(sorted(v.items())) for k, v in status_matrix.items()},
        "span_exact_agreement_among_both_aligned": dict(sorted(span_agree.items())),
        "paper_form_agreement_among_both_aligned": dict(sorted(form_agree.items())),
        "both_aligned": dict(sorted(both_aligned.items())),
        "api_free_rows_naive_both_sides_aligned": api_free_naive,
        "api_free_rows_conservative_plus_glm_agreement": api_free_conservative,
        "share_naive": round(api_free_naive / len(glm), 4) if glm else 0.0,
        "share_conservative": round(api_free_conservative / len(glm), 4) if glm else 0.0,
        "disagreement_ids": disagreement_ids,
        "note": "GLM agreement is a signal, not gold; disagreements are "
        "candidates for LLM/human fallback, not automatic errors.",
    }
    write_json(args.output, report)
    print(f"rows: {len(glm)}")
    for s in ("en", "zh"):
        print(f"{s} status_matrix: {dict(sorted(status_matrix[s].items()))}")
        print(f"{s} span_agree(both aligned): {span_agree[s]}/{both_aligned[s]}")
        print(f"{s} form_agree(both aligned): {form_agree[s]}/{both_aligned[s]}")
    print(f"api_free_naive: {api_free_naive} ({report['share_naive']:.0%})")
    print(f"api_free_conservative: {api_free_conservative} ({report['share_conservative']:.0%})")
    print(f"output: {args.output}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train eflomal on the production bitext and freeze links")
    p.add_argument("--aligned-dir", type=Path, default=Path("data/aligned"))
    p.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("annotate", help="deterministic tuples for a pack manifest")
    p.add_argument("--master-tsv", type=Path, required=True)
    p.add_argument("--pack-manifest", type=Path, required=True)
    p.add_argument("--links-dir", type=Path, required=True)
    p.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("compare", help="deterministic vs GLM pilot agreement")
    p.add_argument("--glm-csv", type=Path, required=True)
    p.add_argument("--deterministic-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
