import os
import pytest
from unittest.mock import Mock, patch
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from transcription_pipeline.transcriber import TranscriptionError


def test_transcription_processor_instantiation(mock_transcriber):
    processor = TranscriptionProcessor()
    assert hasattr(processor, "process")
    assert hasattr(processor, "_format_transcription")
    assert hasattr(processor, "transcriber")


def test_format_transcription(mock_transcriber, mock_get_writer, sample_srt):
    mock_writer = Mock()
    mock_get_writer.return_value = mock_writer

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.read.return_value = sample_srt
        temp_file = mock_temp.return_value.__enter__.return_value
        temp_file.name = "/tmp/mock_temp.srt"

        processor = TranscriptionProcessor()
        result = processor._format_transcription(
            {"segments": [{"text": "Test subtitle"}]}
        )

        assert result == sample_srt
        mock_writer.assert_called_once()
        mock_get_writer.assert_called_once_with(
            "srt", os.path.dirname("/tmp/mock_temp.srt")
        )


def test_transcription_processor_process(
    mock_transcriber,
    mock_get_writer,
    file_data,
    sample_transcription_result,
    sample_srt,
):
    # Setup mocks
    mock_transcriber_instance = mock_transcriber.return_value
    mock_transcriber_instance.transcribe.return_value = sample_transcription_result

    mock_writer = Mock()
    mock_get_writer.return_value = mock_writer

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.read.return_value = sample_srt
        temp_file = mock_temp.return_value.__enter__.return_value
        temp_file.name = "/tmp/mock_temp.srt"

        # Create processor and process file
        processor = TranscriptionProcessor()
        result = processor.process(file_data)

        # Verify transcription was called
        mock_transcriber_instance.transcribe.assert_called_once_with(
            file_data.local_path
        )

        # Verify writer was configured correctly
        mock_get_writer.assert_called_once_with(
            "srt", os.path.dirname("/tmp/mock_temp.srt")
        )

        # Verify result structure
        assert result["id"] == file_data.id
        assert result["success"] is True
        assert result["transcription"] == sample_srt
        assert result["metadata"]["language"] == sample_transcription_result["language"]
        assert result["metadata"]["segments"] == len(
            sample_transcription_result["segments"]
        )


def test_process_failed_transcription(mock_transcriber, file_data):
    mock_transcriber_instance = mock_transcriber.return_value
    mock_transcriber_instance.transcribe.side_effect = Exception("Transcription failed")

    processor = TranscriptionProcessor()
    with pytest.raises(Exception) as exc_info:
        processor.process(file_data)

    assert str(exc_info.value) == "Transcription failed"


def test_process_propagates_transcription_error(mock_transcriber, file_data):
    mock_transcriber_instance = mock_transcriber.return_value
    original_error = TranscriptionError(ValueError("Invalid audio format"))
    mock_transcriber_instance.transcribe.side_effect = original_error

    processor = TranscriptionProcessor()
    with pytest.raises(TranscriptionError) as exc_info:
        processor.process(file_data)

    assert exc_info.value is original_error


def test_process_propagates_value_error(mock_transcriber, file_data):
    mock_transcriber_instance = mock_transcriber.return_value
    original_error = ValueError("Invalid parameter")
    mock_transcriber_instance.transcribe.side_effect = original_error

    processor = TranscriptionProcessor()
    with pytest.raises(ValueError) as exc_info:
        processor.process(file_data)

    assert exc_info.value is original_error
