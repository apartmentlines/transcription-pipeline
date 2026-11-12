from __future__ import annotations

import logging
from unittest.mock import Mock

from download_pipeline_processor.file_data import FileData
import pytest

from transcription_pipeline.processors.transcription_pre_processor import (
    TranscriptionPreProcessor,
)
from transcription_pipeline.audio_file_validator import AudioFileLengthError


def test_transcription_pre_processor_instantiation():
    processor = TranscriptionPreProcessor()
    assert hasattr(processor, "audio_file_validator")
    assert processor.audio_file_validator is not None


def test_transcription_pre_processor_instantiation_with_debug():
    processor = TranscriptionPreProcessor(debug=True)
    assert processor.audio_file_validator.log.getEffectiveLevel() == logging.DEBUG


def test_pre_process_success(file_data: FileData):
    processor = TranscriptionPreProcessor()
    processor.audio_file_validator.validate = Mock()
    result = processor.pre_process(file_data)
    processor.audio_file_validator.validate.assert_called_once_with(
        file_data.local_path
    )
    assert result == file_data


def test_pre_process_validation_failure(file_data: FileData):
    processor = TranscriptionPreProcessor()
    processor.audio_file_validator.validate = Mock(
        side_effect=AudioFileLengthError("File too short")
    )
    with pytest.raises(AudioFileLengthError, match="File too short"):
        processor.pre_process(file_data)
    processor.audio_file_validator.validate.assert_called_once_with(
        file_data.local_path
    )
