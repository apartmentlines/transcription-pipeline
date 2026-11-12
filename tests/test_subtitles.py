"""Lossless reconciliation, shared cue policy, serialization, and metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from transcription_pipeline.results import TranscriptionResult
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.results import AlignmentToken, ResultBuilder, TextReconciler
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
    cues = SubtitleRenderer().cues(result)
    assert " ".join(cue.text for cue in cues) == text
    assert result.metadata(len(cues))["average_word_confidence"] is None


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


def test_unaligned_word_retained_with_bounded_inference() -> None:
    words = TextReconciler.reconcile(
        "Hello missing world",
        [
            AlignmentToken("Hello", 0, 1),
            AlignmentToken("missing"),
            AlignmentToken("world", 2, 3),
        ],
        0,
        3,
    )
    assert words[1].provenance == "inferred"
    assert (words[1].start, words[1].end) == (1, 2)


@pytest.mark.parametrize(
    "tokens", [[], [AlignmentToken("hello")], [AlignmentToken("different", 0, 1)]]
)
def test_unusable_alignment_fails(tokens: list[AlignmentToken]) -> None:
    with pytest.raises(TranscriptionError):
        TextReconciler.reconcile("hello world", tokens, 0, 3)


@pytest.mark.parametrize(
    "start,end", [(-1, 1), (0, 0), (2, 1), (0, float("inf")), (float("nan"), 1), (0, 4)]
)
def test_invalid_timestamps_fail(start: float, end: float) -> None:
    with pytest.raises(TranscriptionError):
        TextReconciler.reconcile("hello", [AlignmentToken("hello", start, end)], 0, 3)


def test_long_pause_and_speaker_change_force_breaks() -> None:
    builder = ResultBuilder()
    builder.append(
        "Hello there world",
        [
            AlignmentToken("Hello", 0, 0.2, "A"),
            AlignmentToken("there", 0.3, 0.5, "B"),
            AlignmentToken("world", 3, 4, "B"),
        ],
        0,
        4,
        "en",
    )
    cues = SubtitleRenderer().cues(builder.build(4, "whisperx", "asr", "align"))
    assert len(cues) == 3
    assert [cue.text for cue in cues] == ["[A] Hello", "[B] there", "[B] world"]


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


def test_metadata_source_intervals_are_independent_of_cues(
    sample_transcription_result: TranscriptionResult,
) -> None:
    result = sample_transcription_result
    short = SubtitleRenderer(SubtitleConfig(max_characters=1)).cues(result)
    long = SubtitleRenderer().cues(result)
    a, b = result.metadata(len(short)), result.metadata(len(long))
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
        derive_speech_bounds=True,
    )
    result = builder.build(10, "qwen3", "asr", "align")
    assert result.metadata(2)["speaking_duration"] == 2
    assert result.metadata(2)["total_duration"] == 9
    assert result.metadata(2)["segments"] == 2


def test_missing_prefix_uses_original_chunk_bounds() -> None:
    builder = ResultBuilder()
    builder.append(
        "the apartment",
        [AlignmentToken("the"), AlignmentToken("apartment", 2, 3)],
        0,
        5,
        "en",
        derive_speech_bounds=True,
    )
    result = builder.build(5, "qwen3", "asr", "align")
    assert result.words[0].provenance == "inferred"
    assert (result.words[0].start, result.words[0].end) == (0, 2)


def test_overlapping_source_intervals_are_counted_once() -> None:
    builder = ResultBuilder()
    builder.append("hello", [AlignmentToken("hello", 0, 3)], 0, 3, "en")
    builder.append("world", [AlignmentToken("world", 2, 4)], 2, 4, "en")
    result = builder.build(5, "whisperx", "asr", "align")
    assert result.metadata(1)["speaking_duration"] == 4


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
        TextReconciler.reconcile(
            "hello world", [token, AlignmentToken("world", 1, 2)], 0, 3
        )


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
