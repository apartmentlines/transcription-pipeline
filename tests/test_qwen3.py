"""Native Qwen interfaces and complete sample coverage without model downloads."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from typing import Any

import numpy as np
import pytest

from transcription_pipeline.result_metadata import ResultMetadata
from transcription_pipeline.constants import (
    QWEN_SAMPLE_RATE,
)
from transcription_pipeline.deployment import Deployment
from transcription_pipeline.backends.qwen3 import AudioChunker, Qwen3Transcriber
from transcription_pipeline.deployment import ModelAsset
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.results import (
    AlignmentToken,
    ResultBuilder,
    TranscriptionContext,
)


@pytest.fixture
def qwen(deployment: Deployment, monkeypatch: pytest.MonkeyPatch):
    config = replace(
        deployment,
        backend="qwen3",
        models=tuple(
            ModelAsset(
                role, f"Qwen/{role}", "a" * 40, deployment.manifest_path.parent / role
            )
            for role in ("asr", "aligner")
        ),
    )
    torch, transformers, vllm = Mock(), Mock(), Mock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_capability.return_value = (8, 6)
    torch.inference_mode.side_effect = nullcontext
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    monkeypatch.setattr(
        Qwen3Transcriber, "_dependencies", staticmethod(lambda: (torch, transformers))
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    backend = Qwen3Transcriber(config)

    def format_prompt(messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return messages[0]["content"]

    backend.asr_processor.apply_chat_template.side_effect = format_prompt
    backend.aligner_processor.timestamp_segment_time = 80
    yield backend, torch, transformers
    backend.close()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "language English<asr_text>Hello, hello hello hello hello!",
            ("en", "Hello, hello hello hello hello!", "English"),
        ),
        (
            "language Spanish<asr_text>¿Dónde está la llave?",
            ("es", "¿Dónde está la llave?", "Spanish"),
        ),
        ("language None<asr_text>", (None, "", "None")),
        ("", (None, "", "")),
    ],
)
def test_raw_protocol_preserves_spoken_text(raw: str, expected: Any) -> None:
    assert Qwen3Transcriber.parse_transcript(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "language French<asr_text>Bonjour",
        "language None<asr_text>Hello",
        "Hello without header",
    ],
)
def test_unknown_language_and_protocol_fail(raw: str) -> None:
    with pytest.raises(TranscriptionError):
        Qwen3Transcriber.parse_transcript(raw)


def test_aligner_api_and_timestamp_offset(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, _, _ = qwen
    inputs = MagicMock()
    inputs.get.return_value.sum.return_value.item.return_value = 500
    inputs.to.return_value = inputs
    backend.aligner_processor.feature_extractor.hop_length = 160
    backend.aligner_processor.prepare_forced_aligner_inputs.return_value = (
        inputs,
        [["hola"]],
    )
    backend.aligner_processor.decode_forced_alignment.return_value = [
        [{"text": "hola", "start_time": 1.25, "end_time": 2.5}]
    ]
    tokens = backend._align(np.zeros(80000, dtype=np.float32), "¡Hola!", "es", 179.99)
    assert backend.aligner_model.call_args.kwargs["use_cache"] is False
    assert tokens[0].start == pytest.approx(181.24)
    assert tokens[0].end == pytest.approx(182.49)
    kwargs = backend.aligner_processor.prepare_forced_aligner_inputs.call_args.kwargs
    assert kwargs["transcript"] == "¡Hola!"
    assert kwargs["language"] == "es"
    assert kwargs["processor_kwargs"]["audio_kwargs"] == {
        "sampling_rate": 16000,
        "truncation": False,
    }
    decode = backend.aligner_processor.decode_forced_alignment.call_args.kwargs
    assert decode["word_lists"] == [["hola"]]
    assert (
        decode["timestamp_token_id"] is backend.aligner_model.config.timestamp_token_id
    )


@pytest.mark.parametrize(
    "intervals",
    [
        [(0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 1), (1, 2)],
        [(0, 0), (0, 0), (0, 2)],
        [(0, 2), (2, 2), (2, 2)],
        [(0, 0.5), (1, 1), (1.5, 2)],
    ],
)
def test_collapsed_qwen_timestamps_preserve_text_and_bounded_timing(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
    intervals: list[tuple[float, float]],
) -> None:
    backend, _, _ = qwen
    offset = 179.99
    original = [
        AlignmentToken(f"word{index}", start + offset, end + offset)
        for index, (start, end) in enumerate(intervals)
    ]
    inputs = MagicMock()
    inputs.get.return_value.sum.return_value.item.return_value = 200
    inputs.to.return_value = inputs
    backend.aligner_processor.feature_extractor.hop_length = 160
    backend.aligner_processor.prepare_forced_aligner_inputs.return_value = (
        inputs,
        [[token.text for token in original]],
    )
    text = " ".join(token.text for token in original)
    backend.aligner_processor.decode_forced_alignment.return_value = [
        [
            {"text": token.text, "start_time": start, "end_time": end}
            for token, (start, end) in zip(original, intervals)
        ]
    ]
    tokens = backend._align(
        np.zeros(2 * QWEN_SAMPLE_RATE, dtype=np.float32), text, "en", offset
    )
    builder = ResultBuilder()
    assert tokens == original
    builder.append(text, tokens, offset, offset + 2, "en")
    result = builder.build(offset + 2, "qwen3", "asr", "aligner")
    assert result.text == text
    assert len(result.words) == len(original)
    assert ResultMetadata(result).export(1)["unaligned_word_count"] == 0
    assert ResultMetadata(result).diagnostics()["zero_duration_word_count"] == sum(
        a == b for a, b in intervals
    )
    for source, word in zip(original, result.words):
        assert (word.start, word.end) == (source.start, source.end)
    assert all(
        a.end is not None and b.start is not None and a.end <= b.start
        for a, b in zip(result.words, result.words[1:])
    )


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, -1),
        (6, 6),
        (4.8, 5.51),
        (4.8, 8.54),
        (2, 1),
        (float("nan"), 1),
        (float("inf"), float("inf")),
    ],
)
def test_invalid_qwen_timestamps_are_not_repaired(
    qwen: tuple[Qwen3Transcriber, Mock, Mock], start: float, end: float
) -> None:
    backend, _, _ = qwen
    inputs = MagicMock()
    inputs.get.return_value.sum.return_value.item.return_value = 500
    inputs.to.return_value = inputs
    backend.aligner_processor.feature_extractor.hop_length = 160
    backend.aligner_processor.prepare_forced_aligner_inputs.return_value = (
        inputs,
        [["hello"]],
    )
    backend.aligner_processor.decode_forced_alignment.return_value = [
        [{"text": "hello", "start_time": start, "end_time": end}]
    ]
    with pytest.raises(
        TranscriptionError, match="Invalid alignment timestamp interval"
    ):
        backend._align(np.zeros(80000, dtype=np.float32), "hello", "en", 0)


@pytest.mark.parametrize("start,end", [(4.8, 5.02), (5.02, 5.02), (4.8, 5.08)])
def test_qwen_endpoint_quantization_is_bounded_and_reported(
    qwen: tuple[Qwen3Transcriber, Mock, Mock], start: float, end: float
) -> None:
    """Retain a final word without accepting larger out-of-audio predictions."""
    backend, _, _ = qwen
    inputs = MagicMock()
    inputs.get.return_value.sum.return_value.item.return_value = 500
    inputs.to.return_value = inputs
    backend.aligner_processor.feature_extractor.hop_length = 160
    backend.aligner_processor.prepare_forced_aligner_inputs.return_value = (
        inputs,
        [["bye"]],
    )
    backend.aligner_processor.decode_forced_alignment.return_value = [
        [{"text": "bye", "start_time": start, "end_time": end}]
    ]
    diagnostics: dict[str, Any] = {}
    tokens = backend._align(
        np.zeros(80000, dtype=np.float32), "bye", "en", 10, diagnostics
    )
    assert tokens == [AlignmentToken("bye", min(start, 5) + 10, 15)]
    assert diagnostics["alignment_endpoint_clamps"] == [
        {"word_index": 0, "start": start, "end": end, "duration": 5}
    ]


def completion(text: str, finish: str = "stop") -> Any:
    """A native vLLM completion with no inference runtime dependency."""
    return SimpleNamespace(
        text=f"language English<asr_text>{text}", finish_reason=finish, token_ids=[1, 2]
    )


def aligned_word(
    _audio: Any, text: str, _language: str, offset: float, _details: Any
) -> list[AlignmentToken]:
    return [AlignmentToken(text, offset, offset + 1)]


@pytest.mark.parametrize("seconds", [1, 30, 60, 90])
def test_configured_chunks_cover_every_sample_without_truncation(seconds: int) -> None:
    waveform = np.ones(181 * QWEN_SAMPLE_RATE + 17, dtype=np.float32)
    waveform[(seconds - 1) * QWEN_SAMPLE_RATE : seconds * QWEN_SAMPLE_RATE] = 0
    chunks = AudioChunker.ranges(waveform, seconds)
    assert chunks[0].start == 0 and chunks[-1].end == len(waveform)
    assert all(
        0 < chunk.end - chunk.start <= seconds * QWEN_SAMPLE_RATE for chunk in chunks
    )
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))
    assert chunks[0].end < seconds * QWEN_SAMPLE_RATE


def test_vllm_uses_configured_batch_precision_and_visible_device(qwen: Any) -> None:
    backend, _, transformers = qwen
    config = replace(
        backend.deployment, device="cuda:1", precision="float16", qwen_batch_size=8
    )
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    other = Qwen3Transcriber(config)
    try:
        kwargs = sys.modules["vllm"].LLM.call_args.kwargs
        assert kwargs["device_ids"] == [1]
        assert kwargs["dtype"] == "float16"
        assert kwargs["max_num_seqs"] == 8
        assert kwargs["mm_processor_kwargs"]["truncation"] is False
        assert transformers.AutoModelForMultimodalLM.from_pretrained.call_count == 0
    finally:
        other.close()


def test_batch_preserves_context_and_isolates_recording_failures(
    qwen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _ = qwen
    monkeypatch.setattr(
        AudioChunker,
        "decode",
        Mock(
            side_effect=[
                np.zeros(80000, dtype=np.float32),
                ValueError("bad audio"),
                np.zeros(80000, dtype=np.float32),
            ]
        ),
    )
    backend.llm.generate.return_value = [
        SimpleNamespace(outputs=[completion("First")]),
        SimpleNamespace(outputs=[completion("Third")]),
    ]
    backend._align = Mock(side_effect=aligned_word)
    results = backend.transcribe_batch(
        [
            ("a.wav", TranscriptionContext("Alpha", "A term")),
            ("bad.wav", TranscriptionContext()),
            ("c.wav", TranscriptionContext("Gamma", "C term")),
        ]
    )
    assert results[0].text == "First"
    assert isinstance(results[1], ValueError)
    assert results[2].text == "Third"
    requests = backend.llm.generate.call_args.args[0]
    assert "Alpha" in requests[0]["prompt"] and "A term" in requests[0]["prompt"]
    assert "Gamma" in requests[1]["prompt"] and "C term" in requests[1]["prompt"]
    assert "Alpha" not in requests[1]["prompt"]


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_retry_replaces_neighbor_window_without_duplicate_text_or_timing(
    qwen: Any, monkeypatch: pytest.MonkeyPatch, failed_index: int
) -> None:
    backend, _, _ = qwen
    waveform = np.zeros(65 * QWEN_SAMPLE_RATE, dtype=np.float32)
    chunks = AudioChunker.ranges(waveform)
    monkeypatch.setattr(AudioChunker, "decode", Mock(return_value=waveform))
    original = [
        completion(f"Original{i}", "length" if i == failed_index else "stop")
        for i in range(3)
    ]
    backend._generate = Mock(side_effect=[original, [completion("Recovered")]])
    backend._align = Mock(side_effect=aligned_word)
    result = backend.transcribe("audio.wav", TranscriptionContext("Context"))
    assert result.text == (
        "Original0 Recovered" if failed_index == 2 else "Recovered Original2"
    )
    windows = result.diagnostics["chunks"]
    assert windows[0]["start_sample"] == 0
    assert windows[0]["end_sample"] == windows[1]["start_sample"]
    assert windows[1]["end_sample"] == len(waveform)
    retry = windows[1 if failed_index == 2 else 0]
    assert len(retry["recognition_retry"]["original_chunks"]) == 2
    assert backend._align.call_count == 2
    assert backend._generate.call_count == 2
    assert (
        result.words[1].start
        == (chunks[1].start if failed_index == 2 else chunks[2].start)
        / QWEN_SAMPLE_RATE
    )


def test_failed_retry_is_bounded_and_does_not_fail_other_recordings(
    qwen: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _ = qwen
    monkeypatch.setattr(
        AudioChunker, "decode", Mock(return_value=np.zeros(80000, dtype=np.float32))
    )
    backend._generate = Mock(
        side_effect=[
            [completion("Repeat", "length"), completion("Good")],
            [completion("Repeat", "length")],
        ]
    )
    backend._align = Mock(return_value=[AlignmentToken("Good", 0, 1)])
    results = backend.transcribe_batch(
        [("bad.wav", TranscriptionContext()), ("good.wav", TranscriptionContext())]
    )
    assert isinstance(results[0], TranscriptionError)
    assert results[0].code == "generation_incomplete"
    assert results[1].text == "Good"
    assert backend._generate.call_count == 2
