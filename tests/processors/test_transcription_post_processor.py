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


def test_determine_result_state_with_file_data_error(file_data):
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
    }
    file_data.add_error("download", "Download failed")

    state = processor.determine_result_state(result, file_data)
    assert state == {"id": "123", "success": False}


def test_determine_result_state_with_missing_transcription_error(file_data):
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
        "transcription": "",
    }
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
        "transcription_state": "active",
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


import pytest
import requests
from download_pipeline_processor.error import TransientPipelineError
from transcription_pipeline.transcriber import TranscriptionError


def test_is_transient_download_error(file_data):
    processor = TranscriptionPostProcessor()
    http_error = requests.exceptions.HTTPError()
    http_error.response = type("Response", (), {"status_code": None})

    transient_status_codes = [429, 500, 502, 503, 504, 599]
    for status_code in transient_status_codes:
        http_error.response.status_code = status_code
        file_data.add_error("download", http_error)
        assert processor.is_transient_download_error(file_data) is True

    non_transient_status_codes = [400, 401, 403, 404, 422]
    for status_code in non_transient_status_codes:
        http_error.response.status_code = status_code
        file_data.add_error("download", http_error)
        assert processor.is_transient_download_error(file_data) is False

    transient_exceptions = [
        requests.exceptions.ConnectionError(),
        requests.exceptions.Timeout(),
        requests.exceptions.ChunkedEncodingError(),
        requests.exceptions.ContentDecodingError(),
        requests.exceptions.SSLError(),
    ]
    for exception in transient_exceptions:
        file_data.add_error("download", exception)
        assert processor.is_transient_download_error(file_data) is True

    non_transient_exceptions = [
        requests.exceptions.URLRequired(),
        requests.exceptions.TooManyRedirects(),
        requests.exceptions.MissingSchema(),
        requests.exceptions.InvalidSchema(),
        requests.exceptions.InvalidURL(),
    ]
    for exception in non_transient_exceptions:
        file_data.add_error("download", exception)
        assert processor.is_transient_download_error(file_data) is False

    file_data.add_error("download", ValueError())
    assert processor.is_transient_download_error(file_data) is False

    file_data.add_error("process", RuntimeError())
    assert processor.is_transient_download_error(file_data) is False


def test_is_transient_processing_error(file_data):
    processor = TranscriptionPostProcessor()

    file_data.add_error("process", TranscriptionError(ValueError("Bad format")))
    assert processor.is_transient_processing_error(file_data) is False

    file_data.add_error("process", RuntimeError("Unexpected error"))
    assert processor.is_transient_processing_error(file_data) is True

    file_data.add_error("download", Exception())
    assert processor.is_transient_processing_error(file_data) is False


def test_determine_result_state_transient_error(file_data):
    processor = TranscriptionPostProcessor()

    file_data.add_error("download", requests.exceptions.ConnectionError())
    with pytest.raises(TransientPipelineError):
        processor.determine_result_state(None, file_data)

    file_data.add_error("process", ValueError())
    with pytest.raises(TransientPipelineError):
        processor.determine_result_state(None, file_data)

    file_data.add_error("process", TranscriptionError(ValueError()))
    result_state = processor.determine_result_state(None, file_data)
    assert result_state == {"id": file_data.id, "success": False}
