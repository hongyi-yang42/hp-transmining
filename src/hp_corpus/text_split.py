"""Conservative splitter for text-layer concat artifacts.

Born-digital PDFs occasionally emit two adjacent words as a single token
(no whitespace between them in the text layer). For the 1998 Carlsen DE
PDF this happens often enough — ``beimersten``, ``Endeder``, ``dieBohrmaschinen``,
``Schlüssellochlauschen`` — that Stanza parses the merged token as one
unknown noun, and downstream PP extraction fails the paper-filter lemma
match.

This module exposes one pure function, :func:`split_concat`, that takes a
token and a wordlist and returns a list of sub-tokens (``[token]`` if no
split is made). The algorithm is deliberately conservative:

  * Tokens already in the wordlist are never split (so real compounds like
    ``Apfelbaum`` survive untouched, provided they appear at least once in
    the source the wordlist was built from).
  * Splits are attempted only at:
      1. Function-word prefix boundaries (``beim`` + ``ersten``).
      2. Function-word suffix boundaries (``Fleck`` + ``ist``).
      3. Lower→upper camelCase transitions, excluding the ``Mc``/``Mac``
         pattern at position 2 (``McGonagall`` survives).
      4. Long-token fallback (len ≥ 18): any binary split where both
         halves are valid words.
  * Both halves must independently validate against the wordlist (or be a
    plausibly-German token by morphology) before a split is accepted.

The wordlist is a plain ``set[str]`` so callers can build it from any
convenient corpus (we use the 2013 cleaned-text Ch.1-3 vocab in
``scripts/fix_1998_concat.py``).
"""

from __future__ import annotations

# --------------------------------------------------------------------- types

#: Common German function words. Used as split indicators. Order doesn't
#: matter — the splitter tries longest-first so multi-char entries win.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        # definite articles
        "der", "die", "das", "den", "dem", "des",
        # indefinite articles
        "ein", "eine", "einen", "einem", "einer", "eines",
        # negation
        "kein", "keine", "keinen", "keinem", "keiner", "keines",
        # prepositions (uncontracted)
        "in", "an", "auf", "mit", "bei", "zu", "von", "nach",
        "aus", "vor", "über", "unter", "durch", "ohne", "für",
        "um", "ab", "seit", "gegen", "trotz", "während", "wegen",
        # contracted prepositions (definite-article fusion)
        "im", "am", "beim", "zum", "zur", "vom", "ins", "ans",
        "aufs", "übers", "unters", "fürs",
        # personal pronouns
        "ich", "du", "er", "sie", "es", "wir", "ihr",
        "mich", "dich", "uns", "euch", "sich",
        # possessive pronouns
        "mein", "dein", "sein", "unser", "euer",
        # demonstratives / relatives
        "dieser", "diese", "dieses", "jener", "jene", "jenes",
        "welcher", "welche", "welches", "wer", "was",
        # conjunctions
        "und", "oder", "aber", "denn", "sondern", "als", "wenn",
        "weil", "damit", "obwohl", "dass", "ob",
        # auxiliaries (often enclitic to nouns in concat artifacts)
        "ist", "war", "sind", "waren", "wird", "werden",
        "wurde", "wurden", "hat", "hatte", "haben", "hatten",
        "kann", "könnte", "soll", "muss", "will", "mag",
        # common particles / adverbs
        "nicht", "auch", "nur", "schon", "noch", "immer",
        "wieder", "dann", "dort", "hier", "so", "ja", "nein",
    }
)

#: Longest-first ordering so prefix/suffix checks try the most-specific
#: function word before shorter ones (e.g., "beim" before "bei").
_FUNCTION_WORDS_BY_LEN: tuple[str, ...] = tuple(
    sorted(FUNCTION_WORDS, key=lambda w: (-len(w), w))
)

#: Tokens below this length are never examined. Below 5 chars, no strategy
#: can produce a valid split (each strategy requires ≥3 chars per half).
_MIN_LEN_TO_CONSIDER = 5


# --------------------------------------------------------------------- public


def split_concat(token: str, wordlist: set[str]) -> list[str]:
    """Conservatively split ``token`` into sub-tokens.

    Returns ``[token]`` unchanged when no confident split is found. The
    wordlist is the caller-supplied vocabulary of known German words
    (lowercase entries match case-insensitively; capitalized entries
    match verbatim — so the wordlist should usually contain both forms).
    """
    if len(token) < _MIN_LEN_TO_CONSIDER:
        return [token]
    # Never split tokens that are themselves in the wordlist. This is the
    # main safeguard against breaking real compounds (Apfelbaum, etc.).
    if _is_known(token, wordlist):
        return [token]

    # Strategy 1: function-word prefix
    head_tail = _try_function_word_prefix(token, wordlist)
    if head_tail is not None:
        head, tail = head_tail
        return split_concat(head, wordlist) + split_concat(tail, wordlist)

    # Strategy 2: function-word suffix
    head_tail = _try_function_word_suffix(token, wordlist)
    if head_tail is not None:
        head, tail = head_tail
        return split_concat(head, wordlist) + split_concat(tail, wordlist)

    # Strategy 3: lower→upper camelCase transition (excluding Mc/Mac)
    head_tail = _try_camel_case(token, wordlist)
    if head_tail is not None:
        head, tail = head_tail
        return split_concat(head, wordlist) + split_concat(tail, wordlist)

    # Strategy 4: any-binary-split fallback. For unknown tokens, accept any
    # split where both halves are plausible words. The early ``_is_known``
    # check protects real compounds that appear in the wordlist.
    head_tail = _try_any_valid_split(token, wordlist)
    if head_tail is not None:
        head, tail = head_tail
        return split_concat(head, wordlist) + split_concat(tail, wordlist)

    return [token]


def build_wordlist_from_text(text: str) -> set[str]:
    """Build a wordlist (set of tokens, original case + lowercase) from a
    text corpus. Tokens are word characters including German umlauts,
    stripped of leading/trailing punctuation.
    """
    import re

    wordlist: set[str] = set()
    for match in re.finditer(r"[A-Za-zÄÖÜäöüß]+", text):
        tok = match.group()
        if len(tok) >= 2:
            wordlist.add(tok)
            wordlist.add(tok.lower())
    return wordlist


# --------------------------------------------------------------------- private


def _is_known(token: str, wordlist: set[str]) -> bool:
    """Token is in the wordlist (case-sensitive or lowercased)."""
    return token in wordlist or token.lower() in wordlist


def _is_plausible_word(token: str, wordlist: set[str]) -> bool:
    """Token validates as a plausible German word.

    A token is plausible if any of:
      * It's in the wordlist (case-insensitive).
      * It's a capitalized noun-like form (Cap + lower letters, ≥ 3 chars).
        This accepts rare proper nouns and out-of-corpus vocabulary that
        the concatenation accidentally merged with.
    """
    if len(token) < 2:
        return False
    if _is_known(token, wordlist):
        return True
    if (
        len(token) >= 3
        and token[0].isupper()
        and token[1:].islower()
        and token.isalpha()
    ):
        return True
    return False


def _try_function_word_prefix(
    token: str, wordlist: set[str]
) -> tuple[str, str] | None:
    """Try splitting off a known function-word prefix.

    Returns (prefix, rest) if the rest is a plausible word; otherwise None.
    """
    lower = token.lower()
    for fw in _FUNCTION_WORDS_BY_LEN:
        n = len(fw)
        # Require at least 3 chars in the rest to avoid trivial splits.
        # fws are sorted longest-first; we still want to try shorter fws
        # when the longest ones don't fit, so use ``continue`` not ``break``.
        if len(token) - n < 3:
            continue
        if lower.startswith(fw):
            rest = token[n:]
            if _is_plausible_word(rest, wordlist):
                return token[:n], rest
    return None


def _try_function_word_suffix(
    token: str, wordlist: set[str]
) -> tuple[str, str] | None:
    """Try splitting off a known function-word suffix."""
    lower = token.lower()
    for fw in _FUNCTION_WORDS_BY_LEN:
        n = len(fw)
        if len(token) - n < 3:
            continue
        if lower.endswith(fw):
            head = token[:-n]
            if _is_plausible_word(head, wordlist):
                return head, token[-n:]
    return None


def _try_camel_case(
    token: str, wordlist: set[str]
) -> tuple[str, str] | None:
    """Try splitting at the first lower→upper camelCase transition.

    Excludes transitions at position 2 (Mc/Mac pattern, e.g., McGonagall)
    and at position 1 (single lowercase prefix, unlikely in German).
    """
    for i in range(3, len(token) - 1):
        if token[i - 1].islower() and token[i].isupper():
            left = token[:i]
            right = token[i:]
            if _is_plausible_word(left, wordlist) and _is_plausible_word(
                right, wordlist
            ):
                return left, right
            # Found a camelCase transition but the halves don't validate.
            # Don't keep scanning — later transitions are even less likely
            # to be the right split point.
            return None
    return None


def _try_any_valid_split(
    token: str, wordlist: set[str]
) -> tuple[str, str] | None:
    """For unknown tokens, accept any binary split where both halves are
    plausible German words. Prefers the split that maximizes the smaller
    half's length (most balanced).

    The early ``_is_known`` check in :func:`split_concat` protects real
    compounds that appear in the wordlist, so this fallback only fires on
    tokens the wordlist doesn't recognize — which is exactly the concat
    artifacts we want to fix.
    """
    best: tuple[str, str] | None = None
    best_min_len = 0
    for i in range(3, len(token) - 2):
        left = token[:i]
        right = token[i:]
        if _is_plausible_word(left, wordlist) and _is_plausible_word(
            right, wordlist
        ):
            smaller = min(len(left), len(right))
            if smaller > best_min_len:
                best_min_len = smaller
                best = (left, right)
    return best
