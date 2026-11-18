"""Native text mapping, shared cue policy, serialization, and metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from transcription_pipeline.result_metadata import ResultMetadata
from transcription_pipeline.results import TranscriptionResult
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.results import AlignmentToken, ResultBuilder
from transcription_pipeline.subtitles import SubtitleConfig, SubtitleRenderer
from transcription_pipeline.subtitles.renderer import Cue

pytestmark = pytest.mark.usefixtures("sentence_tokenizer")


@pytest.mark.parametrize(
    "text,aligned",
    [
        ("Hello, how are you?", ["hello", "how", "are", "you"]),
        ("¿Dónde está la llave?", ["dónde", "está", "la", "llave"]),
        ("¡Necesito ayuda!", ["necesito", "ayuda"]),
        ("Don't open the door.", ["don", "t", "open", "the", "door"]),
        ("The apartment is 204-B.", ["the", "apartment", "is", "204", "B"]),
        ("There's no A/C.", ["there", "s", "no", "A", "C"]),
    ],
)
def test_text_is_never_reconstructed_from_aligner(
    text: str, aligned: list[str]
) -> None:
    builder = ResultBuilder()
    builder.append(
        text,
        [AlignmentToken(word, i, i + 1) for i, word in enumerate(aligned)],
        0,
        len(aligned),
        "es",
    )
    result = builder.build(len(aligned), "qwen3", "asr", "aligner")
    assert [(word.start, word.end) for word in result.words] == [
        (i, i + 1) for i in range(len(aligned))
    ]
    cues = SubtitleRenderer().cues(result)
    assert " ".join(cue.text for cue in cues) == text
    assert ResultMetadata(result).export(len(cues))["average_word_confidence"] is None


@pytest.mark.parametrize(
    "text",
    [
        r"{\\b1} literal \N \n \h <i>words</i>",
        "¿Está allí? ¡Sí!",
        "TRANSCRIPTIONCUE00000001PLACEHOLDER",
    ],
)
@pytest.mark.parametrize("format_", ["srt", "vtt"])
def test_serializer_preserves_literal_text(text: str, format_: str) -> None:
    contents = SubtitleRenderer.serialize(
        (Cue(59999, 61001, text), Cue(62000, 63000, "second")), format_
    )
    assert text in contents
    assert "second" in contents
    if format_ == "srt":
        assert "00:00:59,999 --> 00:01:01,001" in contents
        assert contents.startswith("1\n")
        assert "\n2\n" in contents
    else:
        assert contents.startswith("WEBVTT")


@pytest.mark.parametrize("following_start", [1, 2])
def test_missing_word_timing_remains_missing(following_start: float) -> None:
    builder = ResultBuilder()
    builder.append(
        "Hello missing world",
        [
            AlignmentToken("Hello", 0, 1),
            AlignmentToken("missing"),
            AlignmentToken("world", following_start, 3),
        ],
        0,
        3,
        "en",
    )
    result = builder.build(3, "qwen3", "asr", "aligner")
    assert (result.words[1].start, result.words[1].end) == (None, None)
    assert ResultMetadata(result).export(1)["unaligned_word_count"] == 1
    assert [cue.text for cue in SubtitleRenderer().cues(result)] == [
        "Hello missing world"
    ]


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [AlignmentToken("hello")],
        [AlignmentToken("different", 0, 1)],
        [AlignmentToken("hello", 0, 0), AlignmentToken("world", 0, 0)],
    ],
)
def test_unusable_alignment_fails(tokens: list[AlignmentToken]) -> None:
    with pytest.raises(TranscriptionError):
        builder = ResultBuilder()
        builder.append("hello world", tokens, 0, 3, "en")
        builder.build(3, "qwen3", "asr", "aligner")


@pytest.mark.parametrize(
    "start,end", [(-1, 1), (2, 1), (0, float("inf")), (float("nan"), 1), (0, 4)]
)
def test_invalid_timestamps_fail(start: float, end: float) -> None:
    with pytest.raises(TranscriptionError):
        builder = ResultBuilder()
        builder.append("hello", [AlignmentToken("hello", start, end)], 0, 3, "en")
        builder.build(3, "qwen3", "asr", "aligner")


@pytest.mark.parametrize("speaker", [None, "B"])
def test_supplied_speakers_take_priority_over_pause_boundaries(
    speaker: str | None,
) -> None:
    builder = ResultBuilder()
    builder.append(
        "Hello there world",
        [
            AlignmentToken("Hello", 0, 0.2, "A"),
            AlignmentToken("there", 0.3, 0.5, speaker),
            AlignmentToken("world", 3, 4, speaker),
        ],
        0,
        4,
        "en",
    )
    cues = SubtitleRenderer().cues(builder.build(4, "whisperx", "asr", "align"))
    assert [cue.text for cue in cues] == (
        ["[A] Hello", "[B] there world"] if speaker else ["[A] Hello", "there", "world"]
    )


def test_long_model_labelled_turn_is_not_split_by_sentences_or_length() -> None:
    """Keep a known speaker's whole turn even beyond the former width/time limits."""
    text = "The maintenance person checked the sink yesterday. There is still a lot of water underneath it today. Please send someone to check it again."
    tokens = [
        AlignmentToken(word, index, index + 0.5, "A")
        for index, word in enumerate(text.split())
    ]
    builder = ResultBuilder()
    builder.append(text, tokens, 0, len(tokens), "en")
    result = builder.build(len(tokens), "whisperx", "asr", "aligner")
    assert [cue.text for cue in SubtitleRenderer().cues(result)] == [f"[A] {text}"]


def test_short_call_exchange_preserves_sentences_including_collapsed_name() -> None:
    """Regression for blocks 18–23: real sentence times, with names anonymized."""
    sentences = [
        ("All right, hon.", 43.28, 43.76),
        ("What's your apartment number?", 43.76, 44.72),
        ("Seven twenty eight.", 45.76, 46.64),
        ("And your name and phone number.", 47.44, 48.64),
        ("Maria.", 49.52, 49.52),
        ("Is my phone number you said?", 49.92, 52.08),
        ("Yes, ma'am.", 52.56, 53.12),
        ("Maria Lopez.", 54.08, 55.12),
    ]
    builder = ResultBuilder()
    builder.append(
        " ".join(text for text, _, _ in sentences),
        [AlignmentToken(text, start, end) for text, start, end in sentences],
        43,
        56,
        "en",
    )
    result = builder.build(67.54, "qwen3", "asr", "aligner")
    before = ResultMetadata(result).export(0)
    cues = SubtitleRenderer().cues(result)
    assert [cue.text for cue in cues] == [text for text, _, _ in sentences]
    assert (cues[4].start, cues[4].end) == (49520, 49920)
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))
    assert all(cue.start < cue.end for cue in cues)
    assert all(word.speaker is None for word in result.words)
    assert ResultMetadata(result).export(0) == before


def test_unlabelled_source_segments_are_not_joined_without_punctuation() -> None:
    """Retain backend boundaries even when no punctuation or speaker ID was supplied."""
    builder = ResultBuilder()
    builder.append(
        "your apartment number",
        [AlignmentToken("your apartment number", 0, 1)],
        0,
        1,
        "en",
    )
    builder.append(
        "seven twenty eight", [AlignmentToken("seven twenty eight", 1, 2)], 1, 2, "en"
    )
    result = builder.build(2, "whisperx", "asr", "aligner")
    assert [cue.text for cue in SubtitleRenderer().cues(result)] == [
        "your apartment number",
        "seven twenty eight",
    ]


def test_rounding_collapse_merges_without_losing_words() -> None:
    builder = ResultBuilder()
    builder.append(
        "hello tail",
        [AlignmentToken("hello", 0, 1), AlignmentToken("tail", 1.00001, 1.00004)],
        0,
        1.00004,
        "en",
    )
    cues = SubtitleRenderer(SubtitleConfig(max_characters=1)).cues(
        builder.build(2, "qwen3", "asr", "align")
    )
    assert len(cues) == 1
    assert cues[0].text == "hello tail"
    assert cues[0].end > cues[0].start


@pytest.mark.parametrize(
    "intervals",
    [
        [(0, 0), (2, 3)],
        [(0, 1), (1, 1), (1, 2)],
        [(0, 1), (3, 3)],
        [(0, 1), (3, 3), (5, 6)],
        [(0.9996, 0.9997)],
        [(0, 3), (2, 4)],
    ],
)
def test_native_points_only_change_cue_layout(
    intervals: list[tuple[float, float]],
) -> None:
    """Protect native observations, EOF rounding, and speaker labels during layout."""
    tokens = [
        AlignmentToken(f"word{index}", start, end, f"speaker{index}", 0.9)
        for index, (start, end) in enumerate(intervals)
    ]
    text = " ".join(token.text for token in tokens)
    duration = max(1, intervals[-1][1])
    builder = ResultBuilder()
    builder.append(text, tokens, 0, duration, "en")
    result = builder.build(duration, "whisperx", "asr", "aligner")
    before = ResultMetadata(result).export(0)
    cues = SubtitleRenderer(SubtitleConfig(max_characters=1)).cues(result)
    assert len(cues) == len(tokens)
    assert all(0 <= cue.start < cue.end <= duration * 1000 for cue in cues)
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))
    assert " ".join(cue.text for cue in cues) == " ".join(
        f"[{token.speaker}] {token.text}" for token in tokens
    )
    assert [(word.start, word.end) for word in result.words] == intervals
    assert ResultMetadata(result).export(0) == before
    assert before["average_word_confidence"] == pytest.approx(0.9)
    assert before["unaligned_word_count"] == 0


def test_metadata_source_intervals_are_independent_of_cues(
    sample_transcription_result: TranscriptionResult,
) -> None:
    result = sample_transcription_result
    short = SubtitleRenderer(SubtitleConfig(max_characters=1)).cues(result)
    long = SubtitleRenderer().cues(result)
    a, b = ResultMetadata(result).export(len(short)), ResultMetadata(result).export(
        len(long)
    )
    a.pop("subtitle_cue_count")
    b.pop("subtitle_cue_count")
    assert a == b
    assert a["total_duration"] == 2
    assert a["audio_duration"] == 5
    assert a["speaking_duration"] == 2
    assert a["detected_languages"] == "en"


def test_same_normalized_result_produces_identical_backend_outputs(
    sample_transcription_result: TranscriptionResult,
) -> None:
    renderer = SubtitleRenderer()
    assert renderer.render(sample_transcription_result) == renderer.render(
        replace(sample_transcription_result, backend="qwen3")
    )


def test_qwen_source_speech_intervals_exclude_silence() -> None:
    builder = ResultBuilder()
    builder.append(
        "hello world",
        [AlignmentToken("hello", 1, 2), AlignmentToken("world", 8, 9)],
        0,
        10,
        "en",
    )
    result = builder.build(10, "qwen3", "asr", "align")
    assert ResultMetadata(result).export(2)["speaking_duration"] == 2
    assert ResultMetadata(result).export(2)["total_duration"] == 9
    assert ResultMetadata(result).export(2)["segments"] == 2


def test_missing_prefix_uses_original_chunk_bounds() -> None:
    builder = ResultBuilder()
    builder.append(
        "the apartment",
        [AlignmentToken("the"), AlignmentToken("apartment", 2, 3)],
        0,
        5,
        "en",
    )
    result = builder.build(5, "qwen3", "asr", "align")
    assert (result.words[0].start, result.words[0].end) == (None, None)
    cue = SubtitleRenderer().cues(result)[0]
    assert (cue.text, cue.start, cue.end) == ("the apartment", 0, 3000)


def test_overlapping_source_intervals_are_counted_once() -> None:
    builder = ResultBuilder()
    builder.append("hello", [AlignmentToken("hello", 0, 3)], 0, 3, "en")
    builder.append("world", [AlignmentToken("world", 2, 4)], 2, 4, "en")
    result = builder.build(5, "whisperx", "asr", "align")
    assert ResultMetadata(result).export(1)["speaking_duration"] == 4


def test_standalone_punctuation_belongs_to_a_spoken_word() -> None:
    builder = ResultBuilder()
    builder.append(
        "— hello — world !",
        [AlignmentToken("hello", 0, 1), AlignmentToken("world", 1, 2)],
        0,
        2,
        "en",
    )
    result = builder.build(2, "qwen3", "asr", "align")
    assert len(result.words) == 2
    assert (
        " ".join(cue.text for cue in SubtitleRenderer().cues(result))
        == "— hello — world !"
    )


@pytest.mark.parametrize(
    "token",
    [AlignmentToken("hello", float("nan"), None), AlignmentToken("hello", None, -1)],
)
def test_missing_timestamp_does_not_hide_an_invalid_bound(token: Any) -> None:
    with pytest.raises(TranscriptionError, match="timestamp"):
        builder = ResultBuilder()
        builder.append(
            "hello world", [token, AlignmentToken("world", 1, 2)], 0, 3, "en"
        )
        builder.build(3, "qwen3", "asr", "aligner")


def test_multiple_line_configuration_preserves_complete_text() -> None:
    builder = ResultBuilder()
    text = "Hello world again today"
    builder.append(
        text,
        [
            AlignmentToken(word, index * 0.1, (index + 1) * 0.1)
            for index, word in enumerate(text.split())
        ],
        0,
        1,
        "en",
    )
    result = builder.build(5, "qwen3", "asr", "align")
    renderer = SubtitleRenderer(SubtitleConfig(max_characters=12, max_lines=2))
    cues = renderer.cues(result)
    assert len(cues) == 1
    assert cues[0].text == "Hello world\nagain today"
    assert " ".join(cues[0].text.split()) == text
    assert "Hello world\nagain today" in renderer.serialize(cues)


@pytest.mark.parametrize(
    "cue", [Cue(-1, 1, "word"), Cue(1, 1, "word"), Cue(2, 1, "word"), Cue(0, 1, "")]
)
def test_serializer_rejects_invalid_cue(cue: Cue) -> None:
    with pytest.raises(TranscriptionError):
        SubtitleRenderer.serialize((cue,))


def test_float_rounding_crosses_minute_boundary() -> None:
    builder = ResultBuilder()
    builder.append("hello", [AlignmentToken("hello", 59.9996, 60.2006)], 59, 61, "en")
    output = SubtitleRenderer().render(builder.build(62, "qwen3", "asr", "align"))
    assert "00:01:00,000 --> 00:01:00,201" in output


@pytest.mark.parametrize(
    "aligned", ["call the manager", "call call manager", "call the call manager"]
)
def test_word_omissions_or_reordering_are_rejected_instead_of_fuzzy_matched(
    aligned: str,
) -> None:
    """A mostly matching transcript must not silently borrow nearby word timings."""
    builder = ResultBuilder()
    with pytest.raises(TranscriptionError, match="text discrepancy"):
        builder.append(
            "call call the manager",
            [AlignmentToken(word, i, i + 1) for i, word in enumerate(aligned.split())],
            0,
            4,
            "en",
        )


def test_timing_quality_warnings_do_not_rewrite_native_alignment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose the audited long-word and collapsed-cue failures without guessing timing."""
    builder = ResultBuilder()
    tokens = [
        AlignmentToken("What?", 0, 49),
        AlignmentToken("Yes.", 49, 49),
        AlignmentToken("Okay.", 49, 49),
        AlignmentToken("Bye.", 49, 50),
    ]
    builder.append("What? Yes. Okay. Bye.", tokens, 0, 50, "en")
    result = builder.build(50, "qwen3", "asr", "aligner")
    diagnostics = ResultMetadata(result).diagnostics()
    assert diagnostics["suspicious_word_duration_count"] == 1
    assert diagnostics["alignment_warnings"] == 1
    assert ResultMetadata(result).export(4)["alignment_warnings"] == 1
    SubtitleRenderer().cues(result)
    assert "review source alignment" in caplog.text
    assert [(word.start, word.end) for word in result.words] == [
        (t.start, t.end) for t in tokens
    ]
