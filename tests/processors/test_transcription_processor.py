from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from download_pipeline_processor.file_data import FileData


def test_transcription_processor_instantiation():
    processor = TranscriptionProcessor()
    assert hasattr(processor, "process")


def test_transcription_processor_process():
    processor = TranscriptionProcessor()
    mock_file_data = FileData(id="123", name="test_file", url="http://example.com/file")
    result = processor.process(mock_file_data)
    assert result["id"] == "123"
    assert result["success"] is True
    assert "transcription" in result
    assert "metadata" in result
