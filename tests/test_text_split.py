"""Synthetic-fixture tests for hp_corpus.text_split.

Every test case uses invented German-like tokens, not novel text. The
splitter is conservative — it should never split real words, even when
real compounds happen to be missing from the wordlist (provided no
function-word / camelCase signal exists).
"""

from __future__ import annotations

from hp_corpus.text_split import (
    FUNCTION_WORDS,
    build_wordlist_from_text,
    split_concat,
)

# --------------------------------------------------------------------- helpers


def _wl(*tokens: str) -> set[str]:
    """Build a small wordlist (with case variants) from explicit tokens."""
    out: set[str] = set()
    for t in tokens:
        out.add(t)
        out.add(t.lower())
    return out


# --------------------------------------------------------------------- no-split cases


def test_short_token_is_never_split() -> None:
    """Tokens below the minimum length are returned as-is."""
    wl = _wl("Haus")
    assert split_concat("Haus", wl) == ["Haus"]
    assert split_concat("xx", wl) == ["xx"]


def test_known_word_is_never_split() -> None:
    """Tokens that are in the wordlist (real compounds) survive untouched."""
    wl = _wl("Apfelbaum", "Apfel", "Baum", "Schlüssel")
    # Apfelbaum is in the wordlist → returned as-is, even though both
    # Apfel and Baum are also in the wordlist and could in principle be
    # re-joined as a compound.
    assert split_concat("Apfelbaum", wl) == ["Apfelbaum"]


def test_unknown_long_token_without_indicator_is_not_split() -> None:
    """A long token that isn't in the wordlist but has no function-word,
    camelCase, or length-based indicator should not be split — we'd rather
    miss an artifact than corrupt a rare valid word."""
    wl = _wl("Haus")
    # "Xyzabcde" — 8 chars, no German indicator, lowercase, unknown.
    # Below _MIN_LEN_TO_CONSIDER (9), so no split.
    assert split_concat("Xyzabcde", wl) == ["Xyzabcde"]


def test_mcdonald_pattern_not_split() -> None:
    """``McGonagall`` has a lower→upper transition at position 2 (c→G),
    which is the Mc-pattern and should be preserved."""
    wl = _wl()
    assert split_concat("McGonagall", wl) == ["McGonagall"]


# --------------------------------------------------------------------- function-word prefix splits


def test_split_contraction_prefix() -> None:
    """``beimersten`` → ``beim`` + ``ersten`` (contracted prep + adjective)."""
    wl = _wl("ersten")
    assert split_concat("beimersten", wl) == ["beim", "ersten"]


def test_split_article_prefix() -> None:
    """``dieBohrmaschinen`` → ``die`` + ``Bohrmaschinen`` (article + Capital noun).

    Note this is also a camelCase transition, but the function-word prefix
    strategy is tried first and succeeds.
    """
    wl = _wl("Bohrmaschinen")
    assert split_concat("dieBohrmaschinen", wl) == ["die", "Bohrmaschinen"]


def test_split_conjunction_prefix() -> None:
    """``undgeheimnisvolle`` → ``und`` + ``geheimnisvolle``."""
    wl = _wl("geheimnisvolle")
    assert split_concat("undgeheimnisvolle", wl) == ["und", "geheimnisvolle"]


def test_unknown_prefix_function_word_still_splits_if_rest_is_plausible() -> None:
    """Even if the rest is not in the wordlist, accept the split when the
    rest is a plausibly-German capitalized noun (≥ 3 chars)."""
    wl = _wl()
    # "derMysterioese" — der is a function word; rest is capitalized
    # noun-like form not in wordlist but accepted by morphology.
    assert split_concat("derMysterioese", wl) == ["der", "Mysterioese"]


# --------------------------------------------------------------------- function-word suffix splits


def test_split_auxiliary_suffix() -> None:
    """``Fleckist`` → ``Fleck`` + ``ist``."""
    wl = _wl("Fleck")
    assert split_concat("Fleckist", wl) == ["Fleck", "ist"]


def test_split_preposition_suffix() -> None:
    """``Vergleichzu`` → ``Vergleich`` + ``zu``."""
    wl = _wl("Vergleich")
    assert split_concat("Vergleichzu", wl) == ["Vergleich", "zu"]


def test_split_article_suffix() -> None:
    """``Endeder`` → ``Ende`` + ``der``."""
    wl = _wl("Ende")
    assert split_concat("Endeder", wl) == ["Ende", "der"]


def test_split_long_auxiliary_suffix() -> None:
    """``Mittagspausewar`` → ``Mittagspause`` + ``war``."""
    wl = _wl("Mittagspause")
    assert split_concat("Mittagspausewar", wl) == ["Mittagspause", "war"]


# --------------------------------------------------------------------- camelCase splits


def test_camel_case_split_when_no_function_word_matches() -> None:
    """``derenNichtsnutz`` — ``deren`` is not in our function-word set, but
    the lower→upper transition at position 5 triggers camelCase split."""
    wl = _wl("deren", "Nichtsnutz")
    # Prefers function-word strategies first. Neither matches (deren not
    # in FUNCTION_WORDS), so falls through to camelCase.
    assert "deren" not in FUNCTION_WORDS  # sanity
    assert split_concat("derenNichtsnutz", wl) == ["deren", "Nichtsnutz"]


def test_camel_case_does_not_split_when_halves_dont_validate() -> None:
    """A camelCase transition where one half is implausible (1 char, mixed
    digits, etc.) should not be split."""
    wl = _wl()
    # "xZ" — both halves too short; min length check rejects.
    assert split_concat("axZbc", wl) == ["axZbc"]


# --------------------------------------------------------------------- long-token fallback


def test_long_token_splits_when_both_halves_valid() -> None:
    """Long tokens without a function-word/camelCase indicator fall through
    to the any-valid-split strategy."""
    wl = _wl("Schlüsselloch", "lauschen")
    assert split_concat("Schlüssellochlauschen", wl) == ["Schlüsselloch", "lauschen"]


def test_long_token_does_not_split_when_only_one_half_valid() -> None:
    """If only one half of a long token validates, leave it alone."""
    wl = _wl("Haus")  # only "Haus" is known
    # 22 chars, no valid split → untouched (other half isn't in wordlist
    # and isn't a Capital+lower noun form)
    assert split_concat("Hausgeisterhaftxyz", wl) == ["Hausgeisterhaftxyz"]


# --------------------------------------------------------------------- recursion


def test_recursive_split_three_way() -> None:
    """Triple concat like ``undHausblau`` should split recursively into 3."""
    wl = _wl("Haus", "blau")
    # "und" prefix → split → "Hausblau" → recursively → "Haus" + "blau"
    assert split_concat("undHausblau", wl) == ["und", "Haus", "blau"]


def test_recursive_split_via_function_word_then_camelcase() -> None:
    """``beimHausblau`` → ``beim`` + ``Hausblau`` → ``beim`` + ``Haus`` + ``blau``."""
    wl = _wl("Haus", "blau")
    assert split_concat("beimHausblau", wl) == ["beim", "Haus", "blau"]


# --------------------------------------------------------------------- morphology fallback


def test_capitalized_unknown_noun_accepted_as_plausible() -> None:
    """A capitalized noun-like form (≥ 3 chars) is accepted as plausible
    even when not in the wordlist — protects rare proper nouns."""
    wl = _wl("Ende")
    # "EndeXylos" → Ende + Xylos (Xylos is Capital+lower, accepted by morphology)
    assert split_concat("EndeXylos", wl) == ["Ende", "Xylos"]


# --------------------------------------------------------------------- build_wordlist_from_text


def test_build_wordlist_handles_german_umlauts() -> None:
    """The wordlist regex must include German umlauts (ä ö ü ß Ä Ö Ü)."""
    text = "Der Bär aß einen Äpfel mit übler Laune."
    wl = build_wordlist_from_text(text)
    for w in ("Der", "Bär", "aß", "einen", "Äpfel", "übler", "Laune"):
        assert w in wl
        assert w.lower() in wl


def test_build_wordlist_excludes_single_chars() -> None:
    """Single-char tokens are too ambiguous to keep — drop them."""
    wl = build_wordlist_from_text("a b c Haus")
    assert "a" not in wl
    assert "b" not in wl
    assert "Haus" in wl


def test_build_wordlist_strips_punctuation() -> None:
    """Punctuation is not part of wordlist entries."""
    wl = build_wordlist_from_text("Haus, Baum. Wald!")
    assert "Haus" in wl
    assert "Baum" in wl
    assert "Wald" in wl
