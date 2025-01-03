import pytest
from unittest.mock import patch
from download_pipeline_processor.file_data import FileData


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
