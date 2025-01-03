from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)


def test_transcription_processor_instantiation():
    processor = TranscriptionProcessor()
    assert hasattr(processor, "process")


def test_transcription_processor_process(file_data):
    processor = TranscriptionProcessor()
    result = processor.process(file_data)
    assert result["id"] == "123"
    assert result["success"] is True
    assert "transcription" in result
    assert "metadata" in result
