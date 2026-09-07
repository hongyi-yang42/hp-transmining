"""Run the constrained machine pre-annotation pilot (experimental side path).

NOT the production annotation workflow — this never touches
``annotation_pairs*.csv``, ``build_annotation_csv.py``, or
``build_eligible_pool.py``. It reads the machine master plus the
retrieval-context regression manifest and produces a separate,
gitignored machine-proposal artifact under ``data/derived/``.

Subcommands:

  ``dev-pack``    Select the ~30-40-row development/calibration pack
                  (deliberate strata: routine, widened-context,
                  demonstrative-cue, companion-difficult, regression
                  cases incl. the two named Ch.14/Ch.15 regressions).
                  Writes a pack manifest (ids + strata + composition).
                  This is explicitly NOT a validation sample.

  ``annotate``    Run the model over a pack, one row per request,
                  temperature 0, validate every response
                  deterministically (exact-substring spans, frozen
                  enums, id round-trip, one response per row), retry
                  invalid responses once with the violated rule names,
                  then write ``machine_tuple_dev_calibration.csv``
                  (failed rows carried with status ``invalid`` — never
                  silently dropped) plus a local run report.

  ``review-sheet`` Add blank human-validation columns (correct /
                  incorrect / uncertain + corrections) to a machine
                  tuple CSV — the human checks the machine proposal
                  against the fixed codebook; machine and human cells
                  stay separate.

  ``sample``      The future frozen random validation pack sampler
                  (explicit seed, stable ids, seed + source hash
                  recorded, no hand-picking). Implemented for the
                  later freeze — do NOT run it on the real master
                  before the pilot review; a synthetic unit test is
                  the intended exercise.

Stdout discipline: aggregate counts only — never corpus text, spans,
lemmas, or datapoint ids.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from hp_corpus.annotation_csv import MASTER_TO_CSV_COLUMNS
from hp_corpus.llm_client import ModelGatewayError, call_model, parse_json_object
from hp_corpus.machine_preannotation import (
    DEV_PACK_STRATA,
    MACHINE_TUPLE_COLUMNS,
    PROMPT_VERSION,
    RESPONSE_KEYS,
    REVIEW_SHEET_COLUMNS,
    SYSTEM_PROMPT,
    aggregate_outcomes,
    build_machine_row,
    build_reviewer_sheet,
    build_user_prompt,
    prompt_sha256,
    sample_validation_pack,
    select_dev_pack,
    validate_machine_response,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

PACK_SCHEMA_VERSION = "machine-tuple-dev-pack-v1"
RUN_SCHEMA_VERSION = "machine-tuple-run-v1"

# The two known retrieval-context regressions, forced into every dev pack.
FORCED_REGRESSION_IDS = (
    "dp_ch14_hp1_de_ch14_p0042_s001#b001_t8-9",  # Ch.14 "im Schrank" (zh manual_review)
    "dp_ch15_hp1_de_ch15_p0059_s001#b001_t15-17",  # Ch.15 "bei dem Aufruhr" (zh neighbor_fallback)
)


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_master(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        header = list(reader.fieldnames or [])
    required = list(MASTER_TO_CSV_COLUMNS) + [
        "en_alignment_confidence",
        "zh_alignment_confidence",
    ]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"master header is missing required columns: {missing}")
    seen: set[str] = set()
    for r in rows:
        dp = r.get("datapoint_id", "")
        if not dp:
            raise ValueError("master row with empty datapoint_id")
        if dp in seen:
            raise ValueError(f"duplicate datapoint_id in master: {dp}")
        seen.add(dp)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _load_regression_ids(manifest_path: Path) -> set[str]:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(cases, dict):
        raise ValueError("regression manifest carries no cases")
    return set(cases)


def cmd_dev_pack(args: argparse.Namespace) -> int:
    if not args.master_tsv.exists():
        return _fail("MASTER_TSV_ABSENT", str(args.master_tsv))
    if not args.regression_json.exists():
        return _fail("REGRESSION_JSON_ABSENT", str(args.regression_json))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")
    try:
        master = read_master(args.master_tsv)
        regression_ids = _load_regression_ids(args.regression_json)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return _fail("INPUT_INVALID", str(exc))

    master_ids = {r["datapoint_id"] for r in master}
    unknown_forced = [dp for dp in FORCED_REGRESSION_IDS if dp not in master_ids]
    if unknown_forced:
        return _fail("FORCED_REGRESSION_NOT_IN_MASTER", f"{len(unknown_forced)} id(s) missing")

    chosen, composition = select_dev_pack(
        master,
        regression_ids=regression_ids,
        forced_regression_ids=FORCED_REGRESSION_IDS,
    )
    if not chosen:
        return _fail("NO_ROWS", "pack selection is empty")
    unexplained = set(composition) - set(DEV_PACK_STRATA)
    if unexplained:
        return _fail("STRATUM_UNKNOWN", str(sorted(unexplained)))

    manifest = {
        "schema_version": PACK_SCHEMA_VERSION,
        "purpose": "development/calibration pack — NOT a validation sample, NOT gold",
        "master_sha256": _sha256(args.master_tsv),
        "regression_manifest": str(args.regression_json),
        "forced_regression_ids": list(FORCED_REGRESSION_IDS),
        "composition": composition,
        "rows": [
            {"datapoint_id": dp, "pack_stratum": stratum}
            for dp, stratum in sorted(chosen.items())
        ],
    }
    write_json(args.output, manifest)
    total = len(chosen)
    print(f"rows: {total}")
    for stratum in DEV_PACK_STRATA:
        print(f"{stratum}: {composition.get(stratum, 0)}")
    print(f"output: {args.output}")
    return EXIT_OK


def _annotate_one(
    master_row: dict[str, str],
    *,
    model: str,
    retries: int,
    max_tokens: int,
) -> tuple[dict | None, list[str], list[str]]:
    """One row through prompt → call → parse → validate (+ retries).

    Returns (accepted response or None, rule names of the LAST attempt,
    gateway error descriptions — rule names / error classes only, no
    content)."""
    user = build_user_prompt(master_row)
    last_errors: list[str] = ["NOT_RUN"]
    gateway_errors: list[str] = []
    for attempt in range(retries + 1):
        try:
            text = call_model(
                SYSTEM_PROMPT, user, model=model,
                temperature=0.0, max_tokens=max_tokens,
            )
        except ModelGatewayError as exc:
            gateway_errors.append(str(exc))
            last_errors = ["GATEWAY_ERROR"]
            continue
        obj = parse_json_object(text)
        if obj is None:
            last_errors = ["UNPARSEABLE_RESPONSE"]
        else:
            last_errors = validate_machine_response(obj, master_row)
            if not last_errors:
                return obj, [], gateway_errors
        if attempt < retries:
            user = (
                build_user_prompt(master_row)
                + "\n\nYour previous reply was rejected by deterministic validation "
                f"for these rules: {', '.join(sorted(set(last_errors)))}. Reply again "
                "with one corrected JSON object only, following every rule."
            )
    return None, sorted(set(last_errors)), gateway_errors


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
    rows_meta = pack.get("rows") or []
    if not rows_meta:
        return _fail("PACK_EMPTY", str(args.pack_manifest))

    master_by_id = {r["datapoint_id"]: r for r in master}
    strata: dict[str, str] = {}
    missing = []
    for entry in rows_meta:
        dp, stratum = entry["datapoint_id"], entry["pack_stratum"]
        if dp not in master_by_id:
            missing.append(dp)
        strata[dp] = stratum
    if missing:
        return _fail("PACK_ID_NOT_IN_MASTER", f"{len(missing)} id(s) missing")

    machine_rows: list[dict[str, str]] = []
    invalid_by_rule: Counter[str] = Counter()
    invalid_ids: list[str] = []
    gateway_failure_count = 0
    started = time.monotonic()
    for i, (dp, stratum) in enumerate(strata.items(), start=1):
        source = master_by_id[dp]
        response, errors, gateway_errors = _annotate_one(
            source, model=args.model, retries=args.retries, max_tokens=args.max_tokens
        )
        if gateway_errors:
            gateway_failure_count += len(gateway_errors)
        if response is None:
            invalid_ids.append(dp)
            for rule in errors:
                invalid_by_rule[rule] += 1
        machine_rows.append(
            build_machine_row(
                source,
                response,
                model=args.model,
                prompt_version=PROMPT_VERSION,
                pack_stratum=stratum,
            )
        )
        if i % 10 == 0 or i == len(strata):
            print(f"progress: {i}/{len(strata)}")

    write_csv(args.output, machine_rows, MACHINE_TUPLE_COLUMNS)

    aggregates = aggregate_outcomes(machine_rows)
    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "parameters": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "retries": args.retries,
        },
        "master_sha256": _sha256(args.master_tsv),
        "pack_manifest": str(args.pack_manifest),
        "pack_composition": pack.get("composition", {}),
        "rows": len(machine_rows),
        "aggregates": aggregates,
        "invalid_model_outputs": {
            "count": len(invalid_ids),
            "by_rule": dict(sorted(invalid_by_rule.items())),
            "datapoint_ids": invalid_ids,
        },
        "gateway_failures": gateway_failure_count,
        "response_keys_allowed": sorted(RESPONSE_KEYS),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    report_path = args.output.with_name(args.output.stem + "_run_report.json")
    write_json(report_path, report)

    print(f"rows: {len(machine_rows)}")
    for side in ("en", "zh"):
        print(f"{side}_status: {dict(sorted(aggregates[f'{side}_status'].items()))}")
    print(f"de_proposal: {dict(sorted(aggregates['de_proposal'].items()))}")
    print(f"en_paper_form: {dict(sorted(aggregates['en_paper_form'].items()))}")
    print(f"zh_paper_form: {dict(sorted(aggregates['zh_paper_form'].items()))}")
    print(f"invalid_model_outputs: {len(invalid_ids)}")
    print(f"gateway_failures: {gateway_failure_count}")
    print(f"output: {args.output}")
    print(f"report: {report_path}")
    return EXIT_OK


def cmd_review_sheet(args: argparse.Namespace) -> int:
    if not args.machine_csv.exists():
        return _fail("MACHINE_CSV_ABSENT", str(args.machine_csv))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")
    with open(args.machine_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if header != list(MACHINE_TUPLE_COLUMNS):
        return _fail("MACHINE_CSV_HEADER_MISMATCH", str(args.machine_csv))
    sheet = build_reviewer_sheet(rows)
    write_csv(args.output, sheet, REVIEW_SHEET_COLUMNS)
    n_machine = len(MACHINE_TUPLE_COLUMNS)
    print(f"rows: {len(sheet)}")
    print(
        f"columns: {len(REVIEW_SHEET_COLUMNS)} "
        f"({n_machine} machine + {len(REVIEW_SHEET_COLUMNS) - n_machine} human)"
    )
    print(f"output: {args.output}")
    return EXIT_OK


def cmd_sample(args: argparse.Namespace) -> int:
    """The future frozen-sample draw. Implemented but deliberately not
    exercised on the real master in this pilot — the freeze happens only
    after the pilot review."""
    if not args.master_tsv.exists():
        return _fail("MASTER_TSV_ABSENT", str(args.master_tsv))
    if args.output.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")
    try:
        master = read_master(args.master_tsv)
    except ValueError as exc:
        return _fail("MASTER_INVALID", str(exc))
    ids = [r["datapoint_id"] for r in master]
    try:
        manifest = sample_validation_pack(
            ids, n=args.n, seed=args.seed, source_sha256=_sha256(args.master_tsv)
        )
    except ValueError as exc:
        return _fail("SAMPLE_INVALID", str(exc))
    write_json(args.output, manifest)
    print(f"n: {manifest['n']}")
    print(f"seed: {manifest['seed']}")
    print(f"source_sha256: {manifest['source_sha256'][:16]}…")
    print(f"output: {args.output}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dev-pack", help="select the development/calibration pack")
    p.add_argument("--master-tsv", type=Path, required=True)
    p.add_argument("--regression-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_dev_pack)

    p = sub.add_parser("annotate", help="run the machine pre-annotation over a pack")
    p.add_argument("--master-tsv", type=Path, required=True)
    p.add_argument("--pack-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="model name recorded in every row")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("review-sheet", help="add blank human-validation columns")
    p.add_argument("--machine-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_review_sheet)

    p = sub.add_parser(
        "sample", help="deterministic seeded validation-pack draw (do NOT freeze yet)"
    )
    p.add_argument("--master-tsv", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force-output", action="store_true")
    p.set_defaults(func=cmd_sample)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
