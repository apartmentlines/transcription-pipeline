import pytest
from unittest.mock import patch, Mock
import numpy as np
from download_pipeline_processor.file_data import (  # pyright: ignore[reportMissingImports]
    FileData,
)
from transcription_pipeline.constants import INITIAL_PROMPT


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all heavy dependencies for faster test execution."""
    mock_whisperx = Mock()
    mock_torch = Mock()

    class MockTensor:
        def cpu(self):
            pass

    mock_torch.Tensor = MockTensor

    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.memory_allocated.return_value = 0
    mock_torch.cuda.memory_reserved.return_value = 0
    mock_torch.cuda.empty_cache = Mock()

    # Mock torch.inference_mode as a context manager
    inference_mode_mock = Mock()
    inference_mode_mock.__enter__ = Mock(return_value=None)
    inference_mode_mock.__exit__ = Mock(return_value=None)
    mock_torch.inference_mode.return_value = inference_mode_mock

    with patch(
        "transcription_pipeline.transcriber._import_dependencies"
    ) as mock_import:
        mock_import.return_value = (np, mock_whisperx, mock_torch)
        yield mock_import


@pytest.fixture
def file_data() -> FileData:
    file_data = FileData(id="123", name="test_file", url="http://example.com/file")
    file_data.record_name = "Test Apartments"
    file_data.local_path = "/path/to/test_file"
    return file_data


@pytest.fixture
def mock_transcriber():
    with patch(
        "transcription_pipeline.processors.transcription_processor.Transcriber"
    ) as mock:
        yield mock


@pytest.fixture
def mock_get_writer():
    with patch(
        "transcription_pipeline.processors.transcription_processor.get_writer"
    ) as mock:
        yield mock


@pytest.fixture
def sample_transcription_result():
    return {
        "language": "en",
        "segments": [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "This is a test", "start": 2.0, "end": 4.0},
        ],
    }


@pytest.fixture
def sample_srt():
    return "1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n"


@pytest.fixture
def initial_prompt():
    return INITIAL_PROMPT % "Test Apartments"
