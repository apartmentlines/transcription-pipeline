import pytest

from transcription_pipeline.transcriber import TranscriptionError


def test_transcription_error_message():
    original_exception = ValueError("Invalid input format")
    with pytest.raises(TranscriptionError) as exc_info:
        raise TranscriptionError(original_exception)
    assert str(exc_info.value) == "Transcription error: Invalid input format"
