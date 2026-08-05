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
