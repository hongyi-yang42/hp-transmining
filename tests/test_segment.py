"""Tests for sentence segmentation and ID stability."""

from __future__ import annotations

import re

from hp_corpus.schema import CleanSentence
from hp_corpus.segment import _split_en, _split_zh, make_id, segment_all

_ID_PATTERN = re.compile(r"^hp1_[a-z]+_ch\d{2}_p\d{4}_s\d{3}$")


def test_make_id_format() -> None:
    i = make_id("hp1", "zh", 1, 3, 12)
    assert i == "hp1_zh_ch01_p0003_s012"
    assert _ID_PATTERN.match(i)


def test_zh_splits_on_terminators() -> None:
    parts = _split_zh("第一句。第二句！第三句？")
    assert parts == ["第一句。", "第二句！", "第三句？"]


def test_zh_does_not_split_on_comma() -> None:
    parts = _split_zh("他来了，看见了，征服了。")
    assert len(parts) == 1
    assert parts[0] == "他来了，看见了，征服了。"


def test_zh_preserves_quoted_terminator() -> None:
    """A terminator inside a quote pair should NOT split the sentence."""
    parts = _split_zh("他说“你好。”然后离开。")
    assert len(parts) == 2
    assert parts[0] == "他说“你好。”"
    assert parts[1] == "然后离开。"


def test_zh_dialogue_final_quote_still_splits() -> None:
    """Quote content ends in a terminator → the quoted sentence stands alone,
    attribution becomes its own segment (matches the EN splitter)."""
    parts = _split_zh("「到这儿来。」她说。")
    assert parts == ["「到这儿来。」", "她说。"]


def test_zh_mid_sentence_quote_close_does_not_split() -> None:
    """「好吧，」-style quotes close mid-sentence; splitting there strands a
    content-free attribution fragment (他说。) downstream. Must stay whole."""
    parts = _split_zh("「好吧，」他说。")
    assert parts == ["「好吧，」他说。"]


def test_zh_inner_quote_close_never_splits() -> None:
    """Only the outermost quote's close can split, and only on a terminator."""
    parts = _split_zh("他说：「我说『走吧。』然后离开。」之后再见。")
    assert parts == ["他说：「我说『走吧。』然后离开。」", "之后再见。"]


def test_de_splits_after_closing_guillemet() -> None:
    """German dialogue puts the sentence-final period inside the closing «.
    The split must fall after « so dialogue and narration separate."""
    parts = _split_en("»Komm hierher.« Er stand auf.", abbreviations=[])
    assert parts == ["»Komm hierher.«", "Er stand auf."]


def test_de_splits_between_consecutive_dialogue_turns() -> None:
    """Two speakers' turns must not merge into one segment."""
    parts = _split_en(
        "»Ich habe ihn verbrannt.« »Es war kein Versehen«, rief Harry.",
        abbreviations=[],
    )
    assert parts == [
        "»Ich habe ihn verbrannt.«",
        "»Es war kein Versehen«, rief Harry.",
    ]


def test_en_splits_on_terminator_followed_by_space() -> None:
    parts = _split_en("First sentence. Second sentence! Third?", abbreviations=[])
    assert parts == ["First sentence.", "Second sentence!", "Third?"]


def test_en_preserves_abbreviations() -> None:
    parts = _split_en(
        "Mr. Dursley met Mrs. Figg. They spoke briefly.",
        abbreviations=["Mr.", "Mrs."],
    )
    assert parts == ["Mr. Dursley met Mrs. Figg.", "They spoke briefly."]


def test_en_preserves_ellipsis_mid_sentence() -> None:
    parts = _split_en("He paused... then continued. End.", abbreviations=[])
    assert parts == ["He paused... then continued.", "End."]


def test_segment_assigns_deterministic_ids(en_config) -> None:
    cleans = [
        CleanSentence(page=1, paragraph=1, text="First. Second.", source_pages=[1]),
        CleanSentence(page=1, paragraph=2, text="Third!", source_pages=[1]),
    ]
    segs = segment_all(cleans, "hp1", "en", 1, en_config)
    ids = [s.id for s in segs]
    assert ids == ["hp1_en_ch01_p0001_s001", "hp1_en_ch01_p0001_s002", "hp1_en_ch01_p0002_s001"]
    assert all(s.source_pages == [1] for s in segs)


def test_segment_reproducible_across_runs(en_config) -> None:
    cleans = [
        CleanSentence(page=1, paragraph=1, text="One. Two.", source_pages=[1]),
    ]
    a = segment_all(cleans, "hp1", "en", 1, en_config)
    b = segment_all(cleans, "hp1", "en", 1, en_config)
    assert [s.id for s in a] == [s.id for s in b]


def test_segment_zh(zh_config) -> None:
    cleans = [
        CleanSentence(page=1, paragraph=1, text="第一句。第二句！", source_pages=[1]),
    ]
    segs = segment_all(cleans, "hp1", "zh", 1, zh_config)
    assert len(segs) == 2
    assert segs[0].text == "第一句。"
    assert segs[1].text == "第二句！"
