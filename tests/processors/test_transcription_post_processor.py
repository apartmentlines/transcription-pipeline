import pytest
import requests
import os

from unittest.mock import patch, Mock
from download_pipeline_processor.error import (  # pyright: ignore[reportMissingImports]
    TransientPipelineError,
)
from download_pipeline_processor.file_data import (  # pyright: ignore[reportMissingImports]
    FileData,
)
from transcription_pipeline.transcriber import TranscriptionError
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

    with patch.object(processor, "handle_response") as mock_handle_response:
        processor.post_process(result, file_data)
        mock_post_request.assert_called_once()
        # Verify handle_response got the transformed state
        expected_state = processor.determine_result_state(result, file_data)
        assert mock_handle_response.call_args[0][1] == expected_state


@patch("transcription_pipeline.processors.transcription_post_processor.post_request")
def test_transcription_post_processor_post_process_failure(
    mock_post_request, file_data
):
    mock_post_request.side_effect = Exception("Post request failed")
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": False,
        "metadata": {"error_stage": "process", "error": "Test error"},
    }
    processor.post_process(result, file_data)
    mock_post_request.assert_called_once()


@patch("transcription_pipeline.processors.transcription_post_processor.post_request")
def test_post_process_error_state_flow(mock_post_request, file_data):
    """Test that post_process correctly handles and transforms error states"""
    mock_response = Mock()
    mock_response.json.return_value = {"success": True}
    mock_post_request.return_value = mock_response

    processor = TranscriptionPostProcessor()

    # Add a non-GPU TranscriptionError (which should be non-transient)
    error_msg = "File format error"
    file_data.add_error("process", TranscriptionError(error_msg))

    original_result = {
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
    }

    with patch.object(processor, "handle_response") as mock_handle_response:
        processor.post_process(original_result, file_data)

        # Verify handle_response received the error state
        called_result = mock_handle_response.call_args[0][1]
        assert called_result["success"] is False
        assert called_result["metadata"]["error"] == f"Transcription error: {error_msg}"
        assert called_result["metadata"]["error_stage"] == "process"


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
    error_msg = "Download failed"
    file_data.add_error("download", error_msg)

    result_state = processor.determine_result_state(result, file_data)
    assert result_state == {
        "id": file_data.id,
        "success": False,
        "metadata": {"error_stage": "download", "error": error_msg},
    }


def test_determine_result_state_with_missing_transcription_error(file_data):
    processor = TranscriptionPostProcessor()
    result = {
        "id": "123",
        "success": True,
        "transcription": "",
    }
    result_state = processor.determine_result_state(result, file_data)
    expected_message = f"No transcription found for {file_data.name}"
    assert result_state == {
        "id": "123",
        "success": False,
        "metadata": {"error_stage": "process", "error": expected_message},
    }


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
        "metadata": {
            "language": "en",
            "segments": 2,
            "total_words": 10,
            "total_duration": 5.0,
            "speaking_duration": 4.5,
            "average_word_confidence": 0.8512,
        },
    }
    data = processor.format_request_payload(result)
    assert data == {
        "api_key": "test_api_key",
        "id": "123",
        "success": True,
        "transcription": "Test transcription",
        "metadata": '{"language": "en", "segments": 2, "total_words": 10, "total_duration": 5.0, "speaking_duration": 4.5, "average_word_confidence": 0.8512}',
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
        "metadata": {"error_stage": "process", "error": "Test error"},
    }
    data = processor.format_request_payload(result)
    assert data == {
        "api_key": "test_api_key",
        "id": "123",
        "success": False,
        "metadata": '{"error_stage": "process", "error": "Test error"}',
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

    # Test GPU-related TranscriptionError (should be transient)
    file_data.add_error("process", TranscriptionError("CUDA failed to initialize"))
    assert processor.is_transient_processing_error(file_data) is True

    # Test cuBLAS GPU error (should be transient)
    file_data.add_error("process", TranscriptionError("cuBLAS failed during operation"))
    assert processor.is_transient_processing_error(file_data) is True

    # Test non-GPU TranscriptionError (should not be transient)
    file_data.add_error("process", TranscriptionError("Invalid audio format"))
    assert processor.is_transient_processing_error(file_data) is False

    # Test other error types (should be transient)
    file_data.add_error("process", RuntimeError("Unexpected error"))
    assert processor.is_transient_processing_error(file_data) is True

    # Test non-process stage error (should not be transient)
    file_data.add_error("download", Exception())
    assert processor.is_transient_processing_error(file_data) is False


def test_determine_result_state_transient_error():
    processor = TranscriptionPostProcessor()
    result = {"success": False}

    # Test transient download error
    download_file = FileData(id="123", name="test_file", url="http://example.com/file")
    download_file.add_error("download", requests.exceptions.ConnectionError())
    with pytest.raises(TransientPipelineError):
        processor.determine_result_state(result, download_file)

    # Test transient processing error
    process_file = FileData(id="123", name="test_file", url="http://example.com/file")
    process_file.add_error("process", ValueError())
    with pytest.raises(TransientPipelineError):
        processor.determine_result_state(result, process_file)

    # Test GPU-related error (should be transient)
    gpu_error_file = FileData(id="123", name="test_file", url="http://example.com/file")
    gpu_error_file.add_error("process", TranscriptionError("CUDA failed to initialize"))
    with pytest.raises(TransientPipelineError):
        processor.determine_result_state(result, gpu_error_file)

    # Test non-GPU TranscriptionError (should not be transient)
    non_transient_file = FileData(
        id="123", name="test_file", url="http://example.com/file"
    )
    non_transient_file.add_error("process", TranscriptionError("Invalid audio format"))
    result_state = processor.determine_result_state(result, non_transient_file)
    assert result_state == {
        "id": non_transient_file.id,
        "success": False,
        "metadata": {
            "error_stage": "process",
            "error": str(non_transient_file.error.error),
        },
    }
