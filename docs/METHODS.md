# Methods

## Reproduction scope

This is an **extraction-method reproduction** of Bremmers et al. (2022) *"Translation Mining: Definiteness across Languages"* — not a parser-identical replication. We use the paper's published extraction logic verbatim; the parser differs because Bremmers et al. do not report which UD parser they used.

A 100% match against the paper's `FILTER_CONTRACTED_123` / `FILTER_PP` is therefore not achievable: residual differences come from parser variation (our Stanza vs the paper's unreported parser) and from edition-specific German PPs in the Carlsen 2013 text that the paper did not annotate.

## Parser

- **Paper upstream parser**: not reported in Bremmers et al. (2022).
- **This project**: Stanza 1.14, with `tokenize + mwt + pos + lemma + depparse`.

## Extraction logic

`scripts/run_paper_extractor.py` follows the paper's `time-in-translation/conll-extractor` (vendored at `vendor/conll-extractor/`, gitignored):

- **Contracted PPs**: tokens in `CONTRACTED` (`im`, `am`, `zum`, …) → extract `(prep, noun)` where the prep's head is an `NN`.
- **Uncontracted PPs**: `PREPOSITIONS` tokens whose same-head dependent is an `ART` in `DETERMINERS` (`der`/`dem`/`den`/…) → extract `(prep, det, noun)`.

Hits are validated against `FILTER_CONTRACTED_123` and `FILTER_PP` from `conll_extractor.prepositions.data`.

## Output discipline

The script writes per-PP detail (sentence id, prep, det, noun, in_filter) only to TSV files under `data/extracted/` (gitignored). **Stdout carries aggregate counts only** — never noun lemmas or other token-level data from the source text. This is verified by `tests/test_run_paper_extractor.py`.
