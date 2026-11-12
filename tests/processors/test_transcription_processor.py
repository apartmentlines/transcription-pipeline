"""Worker configuration binding and the preserved external result contract."""

from __future__ import annotations

from unittest.mock import MagicMock
from dataclasses import replace
from typing import Any

from download_pipeline_processor.file_data import FileData
import pytest

from transcription_pipeline.deployment import Deployment
from transcription_pipeline.results import TranscriptionResult
from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from transcription_pipeline.results import TranscriptionContext
from transcription_pipeline.subtitles import SubtitleRenderer


@pytest.mark.usefixtures("sentence_tokenizer")
def test_worker_contract_and_shared_output(
    deployment: Deployment,
    mock_transcriber: MagicMock,
    sample_transcription_result: TranscriptionResult,
    file_data: FileData,
) -> None:
    mock_transcriber.return_value.transcribe.return_value = sample_transcription_result
    worker_class = TranscriptionProcessor.bind(deployment)
    processor = worker_class(debug=True)
    result = processor.process(file_data)
    assert result["id"] == file_data.id
    assert result["success"] is True
    assert result["transcription"] == SubtitleRenderer().render(
        sample_transcription_result
    )
    assert result["metadata"]["total_words"] == 2
    mock_transcriber.assert_called_once_with(deployment, True)
    mock_transcriber.return_value.transcribe.assert_called_once_with(
        file_data.local_path, TranscriptionContext("Test Apartments")
    )


def test_bound_classes_do_not_share_mutable_backend_selection(
    deployment: Deployment, mock_transcriber: MagicMock
) -> None:
    first = TranscriptionProcessor.bind(deployment)
    other = replace(deployment, backend="qwen3", qwen_attention="flash_attention_2")
    second = TranscriptionProcessor.bind(other)
    first(debug=False)
    second(debug=True)
    assert mock_transcriber.call_args_list[0].args == (deployment, False)
    assert mock_transcriber.call_args_list[1].args == (other, True)


@pytest.mark.parametrize(
    "error",
    [TranscriptionError("CUDA failed"), ValueError("invalid"), RuntimeError("failed")],
)
def test_worker_preserves_error_classification(
    deployment: Deployment,
    mock_transcriber: MagicMock,
    file_data: FileData,
    error: Exception,
) -> None:
    mock_transcriber.return_value.transcribe.side_effect = error
    processor = TranscriptionProcessor(deployment)
    with pytest.raises(type(error)) as caught:
        processor.process(file_data)
    assert caught.value is error


def test_queued_failure_does_not_retain_inference_tensors(
    deployment: Deployment, mock_transcriber: MagicMock, file_data: FileData
) -> None:
    """Unwound model frames must not keep allocations alive through queued errors."""
    import weakref

    class Allocation:
        """Stand-in for a large tensor whose lifetime is observable."""

    observed = []

    def fail(*_args: Any):
        allocation = Allocation()
        observed.append(weakref.ref(allocation))
        try:
            raise RuntimeError("inner inference failure")
        except RuntimeError as cause:
            raise TranscriptionError("inference failed") from cause

    mock_transcriber.return_value.transcribe.side_effect = fail
    processor = TranscriptionProcessor(deployment)
    with pytest.raises(TranscriptionError) as caught:
        processor.process(file_data)
    assert caught.value.__cause__ is not None
    assert observed[0]() is None
