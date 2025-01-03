import os
from unittest.mock import patch, Mock
from transcription_pipeline.processors.transcription_post_processor import (
    TranscriptionPostProcessor,
)


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_transcription_post_processor_instantiation():
    processor = TranscriptionPostProcessor()
    assert processor.api_key == "test_api_key"
    assert processor.domain == "test_domain"


@patch("transcription_pipeline.processors.transcription_post_processor.post_request")
def test_transcription_post_processor_post_process_success(
    mock_post_request, file_data
):
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_post_request.return_value = mock_response
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
        "metadata": {"key": "value"},
    }
    processor.post_process(result, file_data)
    mock_post_request.assert_called_once()


@patch("transcription_pipeline.processors.transcription_post_processor.post_request")
def test_transcription_post_processor_post_process_failure(
    mock_post_request, file_data
):
    mock_post_request.side_effect = Exception("Post request failed")
    processor = TranscriptionPostProcessor()
    result = {"id": "123", "success": False}
    processor.post_process(result, file_data)
    mock_post_request.assert_called_once()


def test_determine_result_state_success(file_data):
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
        "metadata": {"key": "value"},
    }
    state = processor.determine_result_state(result, file_data)
    assert state == result


def test_determine_result_state_with_error(file_data):
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
    }
    file_data.add_error("download", "Download failed")

    state = processor.determine_result_state(result, file_data)
    assert state == {"id": "123", "success": False}


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_format_request_payload_success():
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
        "metadata": {"key": "value"},
    }
    data = processor.format_request_payload(result)
    assert data == {
        "api_key": "test_api_key",
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
        "metadata": '{"key": "value"}',
    }


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_format_request_payload_failure():
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": False,
    }
    data = processor.format_request_payload(result)
    assert data == {
        "api_key": "test_api_key",
        "id": "123",
        "success": False,
    }


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_build_update_url():
    processor = TranscriptionPostProcessor()
    url = processor.build_update_url()
    assert url == "https://test_domain/al/transcriptions/update/operator-recording"


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_handle_response_success():
    processor = TranscriptionPostProcessor()
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    result = {"id": "123"}
    processor.handle_response(mock_response, result)


@patch.dict(
    os.environ,
    {"TRANSCRIPTION_API_KEY": "test_api_key", "TRANSCRIPTION_DOMAIN": "test_domain"},
)
def test_handle_response_failure():
    processor = TranscriptionPostProcessor()
    mock_response = Mock()
    mock_response.json.return_value = {"success": False, "message": "Error message"}
    result = {"id": "123"}
    processor.handle_response(mock_response, result)
