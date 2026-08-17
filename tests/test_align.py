"""Regression tests for the content-addressed embedding cache.

History: ``align_segments`` originally hardcoded cache keys as ``"en"`` and
``"zh"`` (cache v0); a second bug scoped keys only by ``lang + model``,
causing chapter-to-chapter contamination (cache v1). The current implementation
(cache v2) is content-addressed: the cache identity is SHA-256(schema_version,
model_name, lang, scope, ordered_segment_ids, ordered_sentence_texts).

These tests pin:
  * cache identity is sensitive to text, ID order, and row count
  * malformed / mixed-lang / mixed-chapter / duplicate-ID inputs raise
    ``CacheValidationError`` rather than falling back to a shared key
  * different language pairs and different chapters produce different caches
  * cache identity is invariant to the order in which alignments are run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hp_corpus.align import (
    CacheValidationError,
    _build_cache_identity,
    _global_dp_align,
    _lang_of_segments,
    embed_sentences,
)
from hp_corpus.schema import Alignment, Segment

_ID_LANG_RE = __import__("re").compile(r"^[a-z0-9]+_([a-z]{2,3})_ch\d{2}_p\d{4}_s\d{3}$")


def _lang_of_id(seg_id: str) -> str:
    m = _ID_LANG_RE.match(seg_id)
    return m.group(1) if m else ""


def _mk_seg(seg_id: str, text: str) -> Segment:
    return Segment(
        id=seg_id,
        book="hp1",
        lang=_lang_of_id(seg_id),
        chapter=1,
        source_pages=[1],
        sentence_ordinal=1,
        paragraph=1,
        sentence=1,
        text=text,
    )


@pytest.fixture
def fake_encoder(monkeypatch):
    """Install a fake ``sentence_transformers.SentenceTransformer`` so tests
    exercise the cache logic without loading the real ~1.1 GB e5 model.

    The fake produces a deterministic 4-dim vector per input string (hash-based),
    so identical inputs yield identical vectors across calls."""

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, inputs, **kwargs):  # type: ignore[no-untyped-def]
            return np.array(
                [
                    [
                        float((hash(s + f"|{i}") % 1000) / 1000.0)
                        for i in range(4)
                    ]
                    for s in inputs
                ],
                dtype=np.float32,
            )

    fake_module = type(sys)("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return FakeModel


# --------------------------------------------------------------------- helpers


def _ids(lang: str, ch: int, n: int) -> list[str]:
    return [f"hp1_{lang}_ch{ch:02d}_p0001_s{i:03d}" for i in range(1, n + 1)]


def _texts(n: int, prefix: str = "synth") -> list[str]:
    return [f"{prefix} sentence {i}" for i in range(1, n + 1)]


# --------------------------------------------------------------------- existing


def test_lang_of_segments_extracts_from_first_id() -> None:
    segs = [
        _mk_seg("hp1_de_ch01_p0001_s001", "x"),
        _mk_seg("hp1_de_ch01_p0001_s002", "y"),
    ]
    assert _lang_of_segments(segs) == "de"


def test_lang_of_segments_handles_unknown_id_format() -> None:
    """A malformed ID yields an empty language code (no fake fallback)."""

    class FakeSeg:
        id = "garbage_id"

    assert _lang_of_segments([FakeSeg()]) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------- 10 cache tests


def test_cache_miss_when_text_changes(tmp_path: Path, fake_encoder) -> None:
    """Requirement 1: changing any sentence text must produce a different
    cache file (digest change → different filename)."""
    ids = _ids("de", 1, 3)
    texts_a = _texts(3, "alpha")
    texts_b = _texts(3, "beta")

    embed_sentences(texts_a, ids, tmp_path, model_name="fake/M")
    embed_sentences(texts_b, ids, tmp_path, model_name="fake/M")

    # Two distinct .npy files under the same scope/lang/ subdirectory.
    npy_files = list((tmp_path / "v2" / "de" / "hp1_de_ch01").glob("*.npy"))
    assert len(npy_files) == 2, f"expected 2 caches (text-a vs text-b), got {npy_files}"


def test_cache_miss_when_id_order_changes(tmp_path: Path, fake_encoder) -> None:
    """Requirement 2: reordering segment IDs (without changing texts) must
    produce a different cache. Order is part of the identity."""
    ids_a = _ids("de", 1, 3)
    ids_b = list(reversed(ids_a))
    texts = _texts(3, "alpha")

    id_a = _build_cache_identity(ids_a, texts, "fake/M")
    id_b = _build_cache_identity(ids_b, texts, "fake/M")
    assert id_a.digest != id_b.digest, "reordering IDs must change digest"


def test_stale_row_count_does_not_silently_load(tmp_path: Path, fake_encoder) -> None:
    """Requirement 3: if a .npy exists but its row count mismatches the input
    (e.g. a previous version left a stale file), embed_sentences must NOT
    silently load it. It must re-encode."""
    ids = _ids("de", 1, 3)
    texts = _texts(3, "alpha")

    # First call: writes a valid cache.
    embed_sentences(texts, ids, tmp_path, model_name="fake/M")
    # Sanity: cache exists.
    npy_files = list((tmp_path / "v2" / "de" / "hp1_de_ch01").glob("*.npy"))
    assert len(npy_files) == 1

    # Corrupt the .npy by overwriting with a wrong-shape array under the
    # same filename. The digest-based filename stays valid, but the contents
    # no longer match what the identity expects.
    npy_path = npy_files[0]
    stale = np.zeros((99, 4), dtype=np.float32)  # wrong row count
    np.save(npy_path, stale)

    # Re-encode: must detect the row-count mismatch and overwrite with a
    # correctly-shaped array (no silent use of the 99-row file).
    out = embed_sentences(texts, ids, tmp_path, model_name="fake/M")
    assert out.shape == (3, 4), f"expected (3,4), got {out.shape}"


def test_malformed_id_raises(tmp_path: Path, fake_encoder) -> None:
    """Requirement 4: a malformed segment ID must raise CacheValidationError."""
    bad_ids = ["garbage_id", "hp1_de_ch01_p0001_s002"]
    texts = _texts(2)
    with pytest.raises(CacheValidationError, match="malformed segment ID"):
        embed_sentences(texts, bad_ids, tmp_path, model_name="fake/M")


def test_mixed_language_raises(tmp_path: Path, fake_encoder) -> None:
    """Requirement 5: mixing languages in one input must raise."""
    mixed = ["hp1_de_ch01_p0001_s001", "hp1_en_ch01_p0001_s002"]
    texts = _texts(2)
    with pytest.raises(CacheValidationError, match="mixed language"):
        embed_sentences(texts, mixed, tmp_path, model_name="fake/M")


def test_mixed_chapter_raises(tmp_path: Path, fake_encoder) -> None:
    """Requirement 6: mixing chapter scope in one input must raise."""
    mixed = ["hp1_de_ch01_p0001_s001", "hp1_de_ch02_p0001_s002"]
    texts = _texts(2)
    with pytest.raises(CacheValidationError, match="mixed chapter scope"):
        embed_sentences(texts, mixed, tmp_path, model_name="fake/M")


def test_duplicate_segment_id_raises(tmp_path: Path, fake_encoder) -> None:
    """Requirement 7: a duplicate segment ID must raise."""
    dup = ["hp1_de_ch01_p0001_s001", "hp1_de_ch01_p0001_s001"]
    texts = _texts(2)
    with pytest.raises(CacheValidationError, match="duplicate segment IDs"):
        embed_sentences(texts, dup, tmp_path, model_name="fake/M")


def test_de_en_and_de_zh_use_different_target_caches(
    tmp_path: Path, fake_encoder, monkeypatch
) -> None:
    """Requirement 8: when aligning DE→EN and DE→ZH, the target-side caches
    must live under different paths (one for EN, one for ZH). Previously the
    cache key was hardcoded "zh" so DE→ZH silently reused EN vectors."""
    from hp_corpus.align import AlignmentConfig, align_segments

    de = [_mk_seg("hp1_de_ch01_p0001_s001", "alpha de")]
    en = [_mk_seg("hp1_en_ch01_p0001_s001", "alpha en")]
    zh = [_mk_seg("hp1_zh_ch01_p0001_s001", "alpha zh")]

    cfg = AlignmentConfig(embed_cache_dir=tmp_path, model_name="fake/M")
    align_segments(de, en, cfg)
    align_segments(de, zh, cfg)

    # en cache must exist under v2/en/hp1_en_ch01/, zh under v2/zh/hp1_zh_ch01/.
    assert list((tmp_path / "v2" / "en" / "hp1_en_ch01").glob("*.npy"))
    assert list((tmp_path / "v2" / "zh" / "hp1_zh_ch01").glob("*.npy"))
    # And no en cache file under zh/ or vice versa.
    assert not list((tmp_path / "v2" / "en").rglob("*hp1_zh*"))
    assert not list((tmp_path / "v2" / "zh").rglob("*hp1_en*"))


def test_different_chapters_use_different_caches(
    tmp_path: Path, fake_encoder
) -> None:
    """Requirement 9: chapter 1 and chapter 2 of the same language must not
    share a cache file. Different segment counts/IDs → different digests."""
    ids_ch01 = _ids("de", 1, 3)
    ids_ch02 = _ids("de", 2, 3)
    texts = _texts(3, "alpha")

    embed_sentences(texts, ids_ch01, tmp_path, model_name="fake/M")
    embed_sentences(texts, ids_ch02, tmp_path, model_name="fake/M")

    ch1_files = list((tmp_path / "v2" / "de" / "hp1_de_ch01").glob("*.npy"))
    ch2_files = list((tmp_path / "v2" / "de" / "hp1_de_ch02").glob("*.npy"))
    assert len(ch1_files) == 1
    assert len(ch2_files) == 1
    assert ch1_files[0].name != ch2_files[0].name


def test_cache_identity_invariant_to_call_order(tmp_path: Path, fake_encoder) -> None:
    """Requirement 10: aligning (A,B) then (C,D) must produce the same set of
    cache files as aligning (C,D) then (A,B). The cache layout has no
    dependency on call sequence — only on the actual segment content."""
    from hp_corpus.align import AlignmentConfig, align_segments

    def _build(segs1, segs2):
        """Run two alignments in a given order on a fresh cache dir."""
        cfg = AlignmentConfig(embed_cache_dir=tmp_path, model_name="fake/M")
        align_segments(segs1[0], segs1[1], cfg)
        align_segments(segs2[0], segs2[1], cfg)

    a = [_mk_seg("hp1_de_ch01_p0001_s001", "a")]
    b = [_mk_seg("hp1_en_ch01_p0001_s001", "b")]
    c = [_mk_seg("hp1_zh_ch01_p0001_s001", "c")]
    d = [_mk_seg("hp1_de_ch02_p0001_s001", "d")]

    # Order 1: (a,b) then (c,d)
    _build(([a[0]], [b[0]]), ([c[0]], [d[0]]))
    files_order1 = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.npy"))

    # Wipe cache, redo in reversed order.
    for p in tmp_path.rglob("*.npy"):
        p.unlink()
    for p in tmp_path.rglob("*.meta.json"):
        p.unlink()

    # Order 2: (c,d) then (a,b)
    _build(([c[0]], [d[0]]), ([a[0]], [b[0]]))
    files_order2 = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.npy"))

    assert files_order1 == files_order2, (
        f"cache layout depends on call order:\n order1={files_order1}\n order2={files_order2}"
    )


# --------------------------------------------------------------------- DP scoring
#
# Mean-pair scoring + margin: regression tests for the alignment-DP rework.
# Under the old max-single-pair scoring (×0.95/0.90 discounts), a 1:1 on the
# best-matching half always dominated the 1:2 that also covered the remaining
# half, so partial-context alignments won structurally.


def _unit(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([[np.cos(a), np.sin(a)]], dtype=np.float64)


def test_dp_prefers_1_to_2_over_1_to_1_plus_gap() -> None:
    """One src sentence genuinely covering two tgt sentences: the 1:2 must beat
    a 1:1 on the better half + a gap for the other (0.825 mean vs 0.9 - 0.2)."""
    src = _unit(0.0)
    tgt = np.vstack([_unit(np.rad2deg(np.arccos(0.90))), _unit(np.rad2deg(np.arccos(0.75)))])
    matches = _global_dp_align(src, tgt, band=0.5)
    assert [(s, t) for s, t, _, _ in matches] == [([0], [0, 1])]
    score, margin = matches[0][2], matches[0][3]
    assert score == pytest.approx((0.90 + 0.75) / 2 - 0.02, abs=1e-6)
    # Alternative was 1:1 on the better half + 0:1 gap: 0.90 - 0.2
    assert margin == pytest.approx(0.805 - 0.70, abs=1e-6)


def test_dp_does_not_absorb_unrelated_neighbor() -> None:
    """A second tgt sentence clearly unrelated to the src (sim 0.5 vs 0.95)
    must NOT be absorbed into a spurious 1:2 — it stays gapped."""
    src = _unit(0.0)
    tgt = np.vstack([_unit(np.rad2deg(np.arccos(0.95))), _unit(np.rad2deg(np.arccos(0.50)))])
    matches = _global_dp_align(src, tgt, band=0.5)
    assert [(s, t) for s, t, _, _ in matches] == [([0], [0]), ([], [1])]


def test_dp_margin_nonnegative_and_present() -> None:
    """Every match on a competitive cell carries a >= 0 margin."""
    rng = np.random.default_rng(7)
    src = rng.normal(size=(6, 8))
    tgt = rng.normal(size=(7, 8))
    matches = _global_dp_align(src, tgt, band=0.5)
    assert matches
    margins = [m for _, _, _, m in matches if m is not None]
    assert margins and all(m >= -1e-9 for m in margins)


def test_alignment_schema_margin_optional_for_old_records() -> None:
    """Alignment records written before the margin field existed (no margin
    key) must still validate, with margin defaulting to None."""
    rec = Alignment.model_validate_json(
        '{"align_id":"a0001","en":["hp1_en_ch01_p0001_s001"],'
        '"zh":["hp1_zh_ch01_p0001_s001"],"type":"1:1","confidence":0.8,'
        '"method":"vecalign_labse","validated":false}'
    )
    assert rec.margin is None
    rec2 = Alignment(
        align_id="a0002",
        en=["hp1_en_ch01_p0001_s002"],
        zh=["hp1_zh_ch01_p0001_s002"],
        type="1:1",
        confidence=0.8,
        method="vecalign_labse",
        margin=0.12,
    )
    assert rec2.margin == pytest.approx(0.12)
