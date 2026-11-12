"""WhisperX model reuse, context, alignment, speakers, and standalone output."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock
from typing import Any

import numpy as np
import pytest

from transcription_pipeline.deployment import Deployment
from transcription_pipeline.results import TranscriptionResult
from transcription_pipeline.backends.whisperx import WhisperXTranscriber
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.results import TranscriptionContext
from transcription_pipeline.transcriber import TranscriptionCommand


@pytest.fixture
def whisperx_backend(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    torch, whisperx = Mock(), Mock()
    torch.inference_mode.side_effect = nullcontext
    whisperx.load_audio.return_value = np.zeros(80000, dtype=np.float32)
    whisperx.load_align_model.return_value = (Mock(), {})
    whisperx.load_model.return_value.transcribe.return_value = {
        "language": "en",
        "segments": [{"text": "Hello, world!", "start": 0, "end": 2}],
    }
    whisperx.align.return_value = {
        "segments": [
            {
                "words": [
                    {"word": "hello", "start": 0, "end": 1, "score": 0.8},
                    {"word": "world", "start": 1, "end": 2, "score": 1},
                ]
            }
        ]
    }
    monkeypatch.setattr(
        "transcription_pipeline.backends.whisperx.importlib.import_module", Mock()
    )
    monkeypatch.setattr(
        WhisperXTranscriber, "_dependencies", staticmethod(lambda: (torch, whisperx))
    )
    audio = tmp_path / "audio.wav"
    audio.touch()
    return WhisperXTranscriber(deployment), whisperx, audio


def test_whisperx_transcribes_and_reuses_models(whisperx_backend: Any) -> None:
    backend, whisperx, path = whisperx_backend
    for _ in range(2):
        result = backend.transcribe(
            path, TranscriptionContext("Test Apartments", "HVAC")
        )
        assert result.text == "Hello, world!"
        assert result.metadata(1)["average_word_confidence"] == 0.9
    assert whisperx.load_model.call_count == 1
    assert whisperx.load_align_model.call_count == 1
    assert whisperx.load_model.call_args.kwargs["local_files_only"] is True
    assert whisperx.load_align_model.call_args.kwargs["model_cache_only"] is True
    assert "Test Apartments" in backend.model.options.initial_prompt
    assert "HVAC" in backend.model.options.initial_prompt


def test_whisperx_spanish_and_language_rejection(whisperx_backend: Any) -> None:
    backend, _, path = whisperx_backend
    backend.model.transcribe.return_value["language"] = "es"
    result = backend.transcribe(path, TranscriptionContext())
    assert result.detected_languages == ("es",)
    backend.model.transcribe.return_value["language"] = "fr"
    with pytest.raises(TranscriptionError, match="Unsupported"):
        backend.transcribe(path, TranscriptionContext())


def test_whisperx_optional_diarization_preserves_word_labels(
    whisperx_backend: Any,
) -> None:
    backend, whisperx, path = whisperx_backend
    backend.diarization_model = Mock()
    aligned = whisperx.align.return_value
    for word in aligned["segments"][0]["words"]:
        word["speaker"] = "SPEAKER_01"
    whisperx.assign_word_speakers.return_value = aligned
    result = backend.transcribe(path, TranscriptionContext())
    assert all(word.speaker == "SPEAKER_01" for word in result.words)
    backend.diarization_model.assert_called_once()
    whisperx.assign_word_speakers.assert_called_once()


def test_whisperx_gpu_failures_remain_retryable(whisperx_backend: Any) -> None:
    backend, _, path = whisperx_backend
    backend.model.transcribe.side_effect = RuntimeError("CUDA out of memory")
    with pytest.raises(TranscriptionError) as caught:
        backend.transcribe(path, TranscriptionContext())
    assert caught.value.is_transient_error()


@pytest.mark.usefixtures("sentence_tokenizer")
def test_standalone_shared_output_and_debug_artifacts(
    deployment: Deployment,
    sample_transcription_result: TranscriptionResult,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "transcription_pipeline.transcriber.ManifestStore.load",
        Mock(return_value=deployment),
    )
    factory = Mock()
    factory.return_value.transcribe.return_value = sample_transcription_result
    monkeypatch.setattr(
        "transcription_pipeline.transcriber.BackendFactory.create", factory
    )
    args = TranscriptionCommand.parser().parse_args(
        ["audio.wav", "--output-dir", str(tmp_path), "--debug-output"]
    )
    destination = TranscriptionCommand.run(args)
    assert (
        destination.read_text() == "1\n00:00:00,000 --> 00:00:02,000\nHello world\n\n"
    )
    assert (
        destination.with_suffix(".txt").read_text() == sample_transcription_result.text
    )
    assert destination.with_suffix(".diagnostics.json").is_file()
    factory.return_value.transcribe.assert_called_once()


def test_unsupported_format_rejected_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock()
    monkeypatch.setattr(
        "transcription_pipeline.transcriber.BackendFactory.create", factory
    )
    with pytest.raises(SystemExit):
        TranscriptionCommand.parser().parse_args(
            ["audio.wav", "--output-format", "pdf"]
        )
    factory.assert_not_called()


def test_workers_can_initialize_cached_models_concurrently(
    whisperx_backend: Any,
) -> None:
    """The backend factory must not serialize independent model loading."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    original, whisperx, _ = whisperx_backend
    barrier = Barrier(2)

    def load(*_args: Any, **kwargs: Any):
        assert kwargs["local_files_only"] is True
        barrier.wait(timeout=5)
        return Mock()

    whisperx.load_model.side_effect = load
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(WhisperXTranscriber, original.deployment) for _ in range(2)
        ]
        workers = [future.result(timeout=10) for future in futures]
    assert workers[0].model is not workers[1].model


def test_whisperx_uses_shared_fallback_for_library_interpolated_words(
    whisperx_backend: Any,
) -> None:
    backend, whisperx, path = whisperx_backend
    backend.model.transcribe.return_value["segments"][0]["text"] = "Hello 204-B world!"
    backend.model.transcribe.return_value["segments"][0]["end"] = 3
    whisperx.align.return_value = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0, "end": 1, "score": 0.9},
                    {"word": "204-B", "start": 1, "end": 1},
                    {"word": "world", "start": 2, "end": 3, "score": 0.9},
                ]
            }
        ]
    }
    result = backend.transcribe(path, TranscriptionContext())
    assert result.text == "Hello 204-B world!"
    assert result.words[1].provenance == "inferred"
    assert (result.words[1].start, result.words[1].end) == (1, 2)
