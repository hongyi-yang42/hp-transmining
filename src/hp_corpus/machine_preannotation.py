"""Constrained machine pre-annotation of trilingual form tuples (pilot).

Experimental side path — NOT the production annotation workflow. The
production path (``annotation_csv`` → ``build_annotation_csv`` →
``validate_annotation_csv`` → ``build_eligible_pool``) is untouched;
this module builds a *separate* machine-proposal artifact whose purpose
is to test whether

    machine alignment/retrieval → constrained machine proposal →
    human validation

can become a reliable pre-annotation layer. Machine proposals are not
human gold and must never be described as such.

Frozen codebook. The paper-comparable **core** forms are fixed here and
the language model may only choose among them; ``other`` is a deliberate
member so the model never has to force a translation shift into a
paper-reported major category. Operational **detail** forms live in a
separate vocabulary that must never be presented as the published
Bremmers et al. form inventory, and semantic/theoretical interpretations
(weak/strong, anaphoric, bridging, familiar) are excluded from both
vocabularies by construction.

Deterministic response validation runs around every model response: the
counterpart span must be an exact substring of the supplied retrieval
context (no paraphrase, normalization, or translation), enums are
closed, ids round-trip, and a genuine translator omission (aligned +
blank span + ``other`` + ``omitted``) stays distinguishable from a
retrieval failure (``not_aligned``). Rows whose response fails
validation after retries are written with ``machine_{en,zh}_status =
invalid`` — present, never silently dropped, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from hp_corpus.annotation_csv import (
    DEFER_PROVENANCES,
    EXCLUSION_REASONS,
    MASTER_TO_CSV_COLUMNS,
    excel_safe,
)

PROMPT_VERSION = "mpa-v1"

# --- frozen paper-comparable core codebook -----------------------------------
#
# German core form is machine-derived upstream (contracted /
# uncontracted) and passed through; the model never re-derives it.

EN_PAPER_FORMS: frozenset[str] = frozenset(
    {
        "definite",
        "bare_singular",
        "demonstrative",
        "other",
    }
)

ZH_PAPER_FORMS: frozenset[str] = frozenset(
    {
        "bare",
        "demonstrative",
        "other",
    }
)

# --- operational detail vocabulary (NOT the paper's inventory) ---------------
#
# Optional finer surface-form information, kept strictly separate from
# the paper-comparable labels. Blank = no detail beyond the paper form.

EN_DETAIL_FORMS: frozenset[str] = frozenset(
    {
        "indefinite",
        "possessive",
        "pronoun",
        "proper_name",
        "omitted",
    }
)

ZH_DETAIL_FORMS: frozenset[str] = frozenset(
    {
        "numeral_classifier",
        "possessive",
        "pronoun",
        "proper_name",
        "omitted",
    }
)

# --- machine statuses and proposals -------------------------------------------

# What a model response may carry. ``invalid`` is artifact-only: the
# writer marks rows whose response failed deterministic validation.
ALIGNMENT_STATUSES: frozenset[str] = frozenset({"aligned", "not_aligned", "uncertain"})
ARTIFACT_STATUS_VALUES: frozenset[str] = ALIGNMENT_STATUSES | {"invalid"}

DE_PROPOSALS: frozenset[str] = frozenset({"include", "exclude", "uncertain"})

# --- machine artifact schema --------------------------------------------------

# Passthrough of the master's machine cells — identical projection to
# the production annotator CSV, so any join back to the master or the
# annotation CSV works on the same values.
MACHINE_SOURCE_COLUMNS: tuple[str, ...] = tuple(MASTER_TO_CSV_COLUMNS.values())

# The machine proposal columns (filled by this pipeline, blank where the
# model returned nothing usable).
MACHINE_PROPOSAL_COLUMNS: tuple[str, ...] = (
    "machine_de_valid_proposal",
    "machine_de_exclusion_reason",
    "machine_en_counterpart",
    "machine_en_paper_form",
    "machine_en_detail_form",
    "machine_en_status",
    "machine_zh_counterpart",
    "machine_zh_paper_form",
    "machine_zh_detail_form",
    "machine_zh_status",
    "machine_annotation_model",
    "machine_annotation_prompt_version",
)

MACHINE_TUPLE_COLUMNS: tuple[str, ...] = (
    MACHINE_SOURCE_COLUMNS + ("pack_stratum",) + MACHINE_PROPOSAL_COLUMNS
)

# Machine columns that carry free corpus text (counterpart spans) and
# get the Excel-safe prefix at write time, like the production writer.
_MACHINE_EXCEL_SAFE_COLUMNS = frozenset(
    {"machine_en_counterpart", "machine_zh_counterpart"}
)

# The exact JSON keys a model response may carry.
RESPONSE_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "de_valid_proposal",
        "de_exclusion_reason",
        "en_counterpart",
        "en_paper_form",
        "en_detail_form",
        "en_status",
        "zh_counterpart",
        "zh_paper_form",
        "zh_detail_form",
        "zh_status",
    }
)

# --- human validation sheet (design for the later reviewer) -------------------

HUMAN_CHECK_VALUES: frozenset[str] = frozenset({"correct", "incorrect", "uncertain"})

HUMAN_COLUMNS: tuple[str, ...] = (
    "human_de_valid_check",
    "human_de_valid_correction",
    "human_de_exclusion_reason_correction",
    "human_en_counterpart_check",
    "human_en_counterpart_correction",
    "human_en_form_check",
    "human_en_paper_form_correction",
    "human_en_detail_form_correction",
    "human_zh_counterpart_check",
    "human_zh_counterpart_correction",
    "human_zh_form_check",
    "human_zh_paper_form_correction",
    "human_zh_detail_form_correction",
    "human_notes",
)

REVIEW_SHEET_COLUMNS: tuple[str, ...] = MACHINE_TUPLE_COLUMNS + HUMAN_COLUMNS

# --- prompt -------------------------------------------------------------------

SYSTEM_PROMPT = """You are a constrained pre-annotation engine for a translation-mining corpus.
For each German prepositional phrase (PP) you receive one row with the German PP,
its German sentence, and an English and a Chinese retrieval context (the aligned
translation sentences plus neighbouring sentences). You propose the English and
Mandarin counterpart of the German PP and their surface forms.

Absolute rules:
1. Identify the translation counterpart of the German PP ONLY within the
   supplied context for that language. Never use outside knowledge of the
   novel, never guess from memory.
2. Copy the counterpart span EXACTLY as it appears in the context — an exact
   substring, character for character. No paraphrase, no normalization, no
   retranslation, no fixing typos. If you cannot delimit the counterpart
   confidently, leave it empty and mark that language "uncertain" rather
   than guess.
3. Choose paper_form ONLY from the frozen vocabulary for that language (given
   in the schema). Never invent a new category. When none of the specific
   categories fits, use "other" — that is exactly what it is for.
4. Put finer operational information only in detail_form (indefinite,
   possessive, pronoun, proper_name, numeral_classifier, omitted). Never put
   a detail value into paper_form and never treat a detail value as a
   paper-comparable category.
5. If the provided context is not the translation locus of this German
   sentence (the context simply does not contain the translation), set that
   language's status to "not_aligned" and leave its counterpart, paper_form,
   and detail_form empty. Do not hallucinate a counterpart.
6. If the context IS the right translation locus but the German PP has
   genuinely no counterpart in it (a true omission by the translator), set
   status "aligned", leave the counterpart empty, set paper_form "other",
   and detail_form "omitted". This is a translation omission, not a
   retrieval failure, and the two must never be conflated.
7. Do NOT infer semantic definiteness strength (weak/strong), anaphoricity,
   bridging, familiarity, or discourse salience. Surface form only. Those
   interpretations are a later, separate stage.
8. German validity: propose "include" when the row is a genuine definite
   German PP of the kind the study targets; "exclude" when it is a
   relative-pronoun or demonstrative reading, a parse artifact, or a wrong
   span (with a reason from the schema vocabulary); "uncertain" when you
   cannot tell.
9. Respond with exactly ONE JSON object matching the requested schema — no
   commentary, no markdown fences, no keys beyond the schema."""

RESPONSE_SCHEMA_TEMPLATE = {
    "id": "<the row id, copied verbatim>",
    "de_valid_proposal": "include | exclude | uncertain",
    "de_exclusion_reason": (
        "not_target_pp | not_definite | extraction_error | duplicate | other "
        "| (empty unless exclude)"
    ),
    "en_counterpart": "<exact substring of english_context, or empty>",
    "en_paper_form": (
        "definite | bare_singular | demonstrative | other "
        "| (empty when not_aligned/uncertain)"
    ),
    "en_detail_form": (
        "indefinite | possessive | pronoun | proper_name | omitted | (empty)"
    ),
    "en_status": "aligned | not_aligned | uncertain",
    "zh_counterpart": "<exact substring of chinese_context, or empty>",
    "zh_paper_form": (
        "bare | demonstrative | other | (empty when not_aligned/uncertain)"
    ),
    "zh_detail_form": (
        "numeral_classifier | possessive | pronoun | proper_name | omitted | (empty)"
    ),
    "zh_status": "aligned | not_aligned | uncertain",
}


def prompt_sha256() -> str:
    """Stable hash of the full prompt (system + schema + user template)."""
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode("utf-8"))
    h.update(json.dumps(RESPONSE_SCHEMA_TEMPLATE, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_user_prompt(source_row: dict[str, str]) -> str:
    """The per-row user message: the row payload plus the response schema."""
    payload = {
        "id": source_row["datapoint_id"],
        "de_form": source_row["de_form"],
        "german_pp": source_row["de_pp_surface"],
        "german_sentence": source_row["de_sentence_text"],
        "english_context": source_row["en_context_text"],
        "chinese_context": source_row["zh_context_text"],
    }
    return (
        "Annotate this row. Respond with one JSON object with exactly these keys:\n"
        + json.dumps(RESPONSE_SCHEMA_TEMPLATE, ensure_ascii=False, indent=1)
        + "\n\nRow:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


# --- deterministic response validation ----------------------------------------


def validate_machine_response(
    response: dict, source_row: dict[str, str]
) -> list[str]:
    """Validate one parsed model response against its source row.

    Returns rule-level error names only (no row content, no ids) so
    aggregates stay privacy-safe. An empty list means the response is
    acceptable for the artifact.
    """
    errors: list[str] = []
    side_lang = {"en": "EN", "zh": "ZH"}

    if response.get("id") != source_row.get("datapoint_id"):
        errors.append("ID_MISMATCH")

    unexpected = set(response) - RESPONSE_KEYS
    if unexpected:
        errors.append("UNEXPECTED_RESPONSE_KEY")

    proposal = response.get("de_valid_proposal")
    if proposal not in DE_PROPOSALS:
        errors.append("DE_PROPOSAL_INVALID")
    reason = response.get("de_exclusion_reason") or ""
    if proposal == "exclude" and reason not in EXCLUSION_REASONS:
        errors.append("DE_EXCLUSION_REASON_INVALID")
    if proposal in ("include", "uncertain") and reason:
        errors.append("DE_REASON_WITHOUT_EXCLUDE")

    for side, paper_forms, detail_forms in (
        ("en", EN_PAPER_FORMS, EN_DETAIL_FORMS),
        ("zh", ZH_PAPER_FORMS, ZH_DETAIL_FORMS),
    ):
        lang = side_lang[side]
        status = response.get(f"{side}_status")
        span = response.get(f"{side}_counterpart") or ""
        paper = response.get(f"{side}_paper_form") or ""
        detail = response.get(f"{side}_detail_form") or ""

        if paper and paper not in paper_forms:
            errors.append(f"{lang}_PAPER_FORM_INVALID")
        if detail and detail not in detail_forms:
            errors.append(f"{lang}_DETAIL_FORM_INVALID")

        if span:
            # The constrained-span rule: a non-empty counterpart must be
            # an exact substring of the supplied retrieval context.
            context = source_row.get(f"{side}_context_text") or ""
            if span not in context:
                errors.append(f"{lang}_SPAN_NOT_SUBSTRING")

        if status not in ALIGNMENT_STATUSES:
            errors.append(f"{lang}_STATUS_INVALID")
            continue
        if status == "not_aligned":
            if span or paper or detail:
                errors.append(f"{lang}_NOT_ALIGNED_WITH_CONTENT")
        elif status == "uncertain":
            if paper or detail:
                errors.append(f"{lang}_UNCERTAIN_WITH_FORM")
        else:  # aligned
            if detail == "omitted":
                # A true translator omission: blank span, paper form
                # "other" (this codebook maps omission to other), status
                # stays aligned.
                if span:
                    errors.append(f"{lang}_OMITTED_WITH_SPAN")
                if paper != "other":
                    errors.append(f"{lang}_OMITTED_PAPER_NOT_OTHER")
            elif span:
                if not paper:
                    errors.append(f"{lang}_SPAN_WITHOUT_PAPER_FORM")
            else:
                errors.append(f"{lang}_ALIGNED_WITHOUT_SPAN_OR_OMISSION")

    return errors


def validate_machine_responses(
    responses: list[dict], source_rows: dict[str, dict[str, str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate a response batch against the requested source rows.

    Returns (per-row errors keyed by response id, batch-level errors).
    Fails closed on duplicates, unknown ids, and silently dropped rows —
    every requested row must have exactly one response.
    """
    batch_errors: list[str] = []
    seen: set[str] = set()
    per_row: dict[str, list[str]] = {}
    for resp in responses:
        rid = resp.get("id")
        if not isinstance(rid, str) or not rid:
            batch_errors.append("RESPONSE_ID_BLANK")
            continue
        if rid in seen:
            batch_errors.append("DUPLICATE_RESPONSE_ID")
            continue
        seen.add(rid)
        if rid not in source_rows:
            batch_errors.append("RESPONSE_ID_UNKNOWN")
            continue
        per_row[rid] = validate_machine_response(resp, source_rows[rid])
    missing = sorted(set(source_rows) - seen)
    if missing:
        batch_errors.append("RESPONSE_ROW_MISSING")
    return per_row, batch_errors


# --- artifact row construction -------------------------------------------------


def project_source_cells(master_row: dict[str, str]) -> dict[str, str]:
    """Project a master row into the passthrough machine cells, with the
    same Excel-safe treatment the production writer applies."""
    cells: dict[str, str] = {}
    for master_col, csv_col in MASTER_TO_CSV_COLUMNS.items():
        value = (master_row.get(master_col) or "").strip()
        if csv_col in ("english_context", "chinese_context", "german_sentence",
                       "german_pp", "german_head_lemma"):
            value = excel_safe(value)
        cells[csv_col] = value
    return cells


def build_machine_row(
    source_row: dict[str, str],
    response: dict | None,
    *,
    model: str,
    prompt_version: str,
    pack_stratum: str,
) -> dict[str, str]:
    """Assemble one machine artifact row.

    ``response=None`` (or a response that failed validation) yields the
    ``invalid`` marker status with proposal cells blank — the row stays
    present and countable, never silently dropped and never guessed.
    """
    row = project_source_cells(source_row)
    row["pack_stratum"] = pack_stratum
    row["machine_annotation_model"] = model
    row["machine_annotation_prompt_version"] = prompt_version

    def cell(key: str, value) -> str:
        value = "" if value is None else str(value)
        if key in _MACHINE_EXCEL_SAFE_COLUMNS:
            value = excel_safe(value)
        return value

    if response is None:
        row["machine_de_valid_proposal"] = "uncertain"
        row["machine_de_exclusion_reason"] = ""
        for side in ("en", "zh"):
            row[f"machine_{side}_counterpart"] = ""
            row[f"machine_{side}_paper_form"] = ""
            row[f"machine_{side}_detail_form"] = ""
            row[f"machine_{side}_status"] = "invalid"
        return {c: row.get(c, "") for c in MACHINE_TUPLE_COLUMNS}

    mapping = {
        "machine_de_valid_proposal": response.get("de_valid_proposal"),
        "machine_de_exclusion_reason": response.get("de_exclusion_reason"),
        "machine_en_counterpart": response.get("en_counterpart"),
        "machine_en_paper_form": response.get("en_paper_form"),
        "machine_en_detail_form": response.get("en_detail_form"),
        "machine_en_status": response.get("en_status"),
        "machine_zh_counterpart": response.get("zh_counterpart"),
        "machine_zh_paper_form": response.get("zh_paper_form"),
        "machine_zh_detail_form": response.get("zh_detail_form"),
        "machine_zh_status": response.get("zh_status"),
    }
    for col, value in mapping.items():
        row[col] = cell(col, value)
    return {c: row.get(c, "") for c in MACHINE_TUPLE_COLUMNS}


# --- development calibration pack ---------------------------------------------

_DEMONSTRATIVE_CUE_EN = re.compile(r"\b(this|that|these|those)\b", re.IGNORECASE)
_DEMONSTRATIVE_CUE_ZH = re.compile(r"[这那]")

ROUTINE_MIN_CONFIDENCE = 0.40  # the production split threshold

# Strata of the development calibration pack, in selection priority
# order. Deterministic: candidates are iterated in datapoint_id order
# and each row lands in at most one stratum.
DEV_PACK_STRATA: tuple[str, ...] = (
    "regression_case",
    "companion_difficult",
    "widened_context",
    "demonstrative_cue",
    "routine",
)


def _confidence(row: dict[str, str], side: str) -> float:
    try:
        return float(row.get(f"{side}_alignment_confidence") or "nan")
    except ValueError:
        return float("nan")


def select_dev_pack(
    master_rows: list[dict[str, str]],
    *,
    regression_ids: set[str],
    forced_regression_ids: tuple[str, ...] = (),
    stratum_sizes: dict[str, int] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Select the development/calibration pack from the master.

    Deliberately constructed for mechanism and failure-mode coverage —
    NOT statistically representative, NOT a validation sample. Returns
    ``{datapoint_id: stratum}`` plus the composition counts.

    Routine rows are exactly what the production split would deliver to
    the annotator main file (both sides ``anchor_window``, both machine
    confidences ≥ the production threshold): rows with a defer
    provenance or low confidence can never land in the routine stratum,
    so known-difficult rows are never silently promoted to reliable
    routine rows by this pipeline.
    """
    sizes = {
        "regression_case": 5,
        "companion_difficult": 6,
        "widened_context": 9,
        "demonstrative_cue": 6,
        "routine": 10,
    }
    if stratum_sizes:
        sizes.update(stratum_sizes)

    order = sorted(master_rows, key=lambda r: r.get("datapoint_id", ""))
    chosen: dict[str, str] = {}

    def take(candidates: list[dict[str, str]], stratum: str, k: int) -> None:
        added = 0
        for r in candidates:
            if added >= k:
                return
            dp = r.get("datapoint_id", "")
            if dp and dp not in chosen:
                chosen[dp] = stratum
                added += 1

    by_id = {r.get("datapoint_id", ""): r for r in order}

    # 1. Forced regression cases first (the two named known regressions).
    for dp in forced_regression_ids:
        if dp in by_id and dp not in chosen:
            chosen[dp] = "regression_case"
    take(
        [r for r in order if r.get("datapoint_id", "") in regression_ids],
        "regression_case",
        max(0, sizes["regression_case"] - len(chosen)),
    )

    def deferred(r: dict[str, str]) -> bool:
        return (
            r.get("en_context_provenance") in DEFER_PROVENANCES
            or r.get("zh_context_provenance") in DEFER_PROVENANCES
            or _confidence(r, "en") < ROUTINE_MIN_CONFIDENCE
            or _confidence(r, "zh") < ROUTINE_MIN_CONFIDENCE
        )

    def both_anchor(r: dict[str, str]) -> bool:
        return (
            r.get("en_context_provenance") == "anchor_window"
            and r.get("zh_context_provenance") == "anchor_window"
        )

    def has_cue(r: dict[str, str]) -> bool:
        en = r.get("en_context_text") or ""
        zh = r.get("zh_context_text") or ""
        return bool(_DEMONSTRATIVE_CUE_EN.search(en) or _DEMONSTRATIVE_CUE_ZH.search(zh))

    take([r for r in order if deferred(r)], "companion_difficult", sizes["companion_difficult"])
    take(
        [
            r
            for r in order
            if not deferred(r)
            and (
                r.get("en_context_provenance") in ("merge_widened", "heuristic_widened")
                or r.get("zh_context_provenance") in ("merge_widened", "heuristic_widened")
            )
        ],
        "widened_context",
        sizes["widened_context"],
    )
    take(
        [r for r in order if not deferred(r) and both_anchor(r) and has_cue(r)],
        "demonstrative_cue",
        sizes["demonstrative_cue"],
    )
    routine_target = sizes["routine"]
    routine_rows = [
        r
        for r in order
        if not deferred(r) and both_anchor(r) and not has_cue(r)
        and _confidence(r, "en") >= 0.80
        and _confidence(r, "zh") >= 0.80
    ]
    half = routine_target // 2
    take([r for r in routine_rows if r.get("de_form") == "contracted"], "routine", half)
    take(
        [r for r in routine_rows if r.get("de_form") == "uncontracted"],
        "routine",
        routine_target - half,
    )

    composition = {s: 0 for s in DEV_PACK_STRATA}
    for stratum in chosen.values():
        composition[stratum] += 1
    return chosen, composition


# --- future frozen validation sample (implemented, NOT to be run yet) ----------

SAMPLE_SCHEMA_VERSION = "machine-tuple-validation-sample-v1"


def sample_validation_pack(
    datapoint_ids: list[str], *, n: int, seed: int, source_sha256: str
) -> dict:
    """Deterministic seeded draw for the future frozen random validation
    pack. Same input list + seed ⇒ same sample, always. The manifest
    records the seed and the source hash; there is no hand-picking step
    after the draw. Do NOT freeze the real random-50 with this before
    the pilot review — a synthetic unit test is the intended exercise.
    """
    ids = sorted(set(datapoint_ids))
    if not ids:
        raise ValueError("no datapoint ids to sample from")
    if not 0 < n <= len(ids):
        raise ValueError(f"sample size {n} out of range for {len(ids)} ids")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ids, n))
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "seed": seed,
        "n": n,
        "source_sha256": source_sha256,
        "datapoint_ids": chosen,
    }


# --- reviewer sheet -------------------------------------------------------------


def build_reviewer_sheet(machine_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add the blank human-validation columns to machine artifact rows.

    The human never invents categories: every check is correct /
    incorrect / uncertain against the fixed codebook shown alongside,
    with correction cells used when a check is incorrect. Machine cells
    and human cells stay strictly separate.
    """
    out: list[dict[str, str]] = []
    for row in machine_rows:
        extended = {c: row.get(c, "") for c in MACHINE_TUPLE_COLUMNS}
        for c in HUMAN_COLUMNS:
            extended[c] = ""
        out.append({c: extended[c] for c in REVIEW_SHEET_COLUMNS})
    return out


def validate_reviewer_sheet(rows: list[dict[str, str]]) -> list[str]:
    """Light fail-closed checks on a returned reviewer sheet: check
    enums, correction coupling, and the constrained-span rule for human
    counterpart corrections (a non-blank correction span must be an
    exact substring of the row's retrieval context). Rule-level error
    names only.
    """
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        dp = (row.get("id", "") or "").strip()
        if not dp:
            errors.append("BLANK_ID")
            continue
        if dp in seen:
            errors.append("DUPLICATE_ID")
        seen.add(dp)

        for col in (
            "human_de_valid_check",
            "human_en_counterpart_check",
            "human_en_form_check",
            "human_zh_counterpart_check",
            "human_zh_form_check",
        ):
            value = (row.get(col, "") or "").strip()
            if value and value not in HUMAN_CHECK_VALUES:
                errors.append("HUMAN_CHECK_INVALID")

        de_check = (row.get("human_de_valid_check", "") or "").strip()
        de_correction = (row.get("human_de_valid_correction", "") or "").strip()
        de_reason = (row.get("human_de_exclusion_reason_correction", "") or "").strip()
        if de_check == "incorrect" and not de_correction:
            errors.append("DE_CHECK_INCORRECT_WITHOUT_CORRECTION")
        if de_check in ("correct", "uncertain") and (de_correction or de_reason):
            errors.append("DE_CORRECTION_WITHOUT_INCORRECT")
        if de_correction and de_correction not in ("include", "exclude"):
            errors.append("DE_CORRECTION_INVALID")

        for side, paper_forms, detail_forms in (
            ("en", EN_PAPER_FORMS, EN_DETAIL_FORMS),
            ("zh", ZH_PAPER_FORMS, ZH_DETAIL_FORMS),
        ):
            lang = side.upper()
            span_check = (row.get(f"human_{side}_counterpart_check", "") or "").strip()
            span_correction = (row.get(f"human_{side}_counterpart_correction", "") or "").strip()
            form_check = (row.get(f"human_{side}_form_check", "") or "").strip()
            paper_correction = (row.get(f"human_{side}_paper_form_correction", "") or "").strip()
            detail_correction = (row.get(f"human_{side}_detail_form_correction", "") or "").strip()

            if span_check == "incorrect" and not span_correction:
                errors.append(f"{lang}_SPAN_CHECK_INCORRECT_WITHOUT_CORRECTION")
            if span_check in ("correct", "uncertain") and span_correction:
                errors.append(f"{lang}_SPAN_CORRECTION_WITHOUT_INCORRECT")
            if span_correction:
                context_col = "english_context" if side == "en" else "chinese_context"
                context = (row.get(context_col, "") or "").strip()
                if span_correction.strip() not in context:
                    errors.append(f"{lang}_SPAN_CORRECTION_NOT_SUBSTRING")
            if form_check == "incorrect" and not (paper_correction or detail_correction):
                errors.append(f"{lang}_FORM_CHECK_INCORRECT_WITHOUT_CORRECTION")
            if form_check in ("correct", "uncertain") and (paper_correction or detail_correction):
                errors.append(f"{lang}_FORM_CORRECTION_WITHOUT_INCORRECT")
            if paper_correction and paper_correction not in paper_forms:
                errors.append(f"{lang}_PAPER_CORRECTION_INVALID")
            if detail_correction and detail_correction not in detail_forms:
                errors.append(f"{lang}_DETAIL_CORRECTION_INVALID")
    return errors


# --- aggregation ----------------------------------------------------------------


def aggregate_outcomes(machine_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Aggregate machine outcome counts for the run report. Counts only —
    no row content."""
    agg: dict[str, dict[str, int]] = {
        "rows": {"total": len(machine_rows)},
        "pack_stratum": {},
        "de_proposal": {},
        "de_exclusion_reason": {},
        "en_status": {},
        "en_paper_form": {},
        "en_detail_form": {},
        "zh_status": {},
        "zh_paper_form": {},
        "zh_detail_form": {},
    }
    for row in machine_rows:
        for agg_col, row_col in (
            ("pack_stratum", "pack_stratum"),
            ("de_proposal", "machine_de_valid_proposal"),
            ("de_exclusion_reason", "machine_de_exclusion_reason"),
            ("en_status", "machine_en_status"),
            ("en_paper_form", "machine_en_paper_form"),
            ("en_detail_form", "machine_en_detail_form"),
            ("zh_status", "machine_zh_status"),
            ("zh_paper_form", "machine_zh_paper_form"),
            ("zh_detail_form", "machine_zh_detail_form"),
        ):
            value = (row.get(row_col, "") or "").strip()
            key = value if value else "(blank)"
            agg[agg_col][key] = agg[agg_col].get(key, 0) + 1
    return agg


__all__ = [
    "PROMPT_VERSION",
    "EN_PAPER_FORMS",
    "ZH_PAPER_FORMS",
    "EN_DETAIL_FORMS",
    "ZH_DETAIL_FORMS",
    "ALIGNMENT_STATUSES",
    "ARTIFACT_STATUS_VALUES",
    "DE_PROPOSALS",
    "MACHINE_SOURCE_COLUMNS",
    "MACHINE_PROPOSAL_COLUMNS",
    "MACHINE_TUPLE_COLUMNS",
    "RESPONSE_KEYS",
    "HUMAN_CHECK_VALUES",
    "HUMAN_COLUMNS",
    "REVIEW_SHEET_COLUMNS",
    "DEV_PACK_STRATA",
    "ROUTINE_MIN_CONFIDENCE",
    "SAMPLE_SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "RESPONSE_SCHEMA_TEMPLATE",
    "prompt_sha256",
    "build_user_prompt",
    "validate_machine_response",
    "validate_machine_responses",
    "project_source_cells",
    "build_machine_row",
    "select_dev_pack",
    "sample_validation_pack",
    "build_reviewer_sheet",
    "validate_reviewer_sheet",
    "aggregate_outcomes",
]
