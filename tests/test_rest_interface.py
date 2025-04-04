import pytest
import logging
import requests
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock, ANY
from flask import Flask

from transcription_pipeline.rest_interface import (
    PipelineConfigValidator,
    PipelineLifecycleManager,
    send_callback,
    create_app,
    parse_arguments,
    TranscriptionRestInterface,
    main,
)
from transcription_pipeline.constants import (
    DEFAULT_REST_FAILURE_EXIT_CODE,
    DEFAULT_REST_HOST,
    DEFAULT_REST_PORT,
)
from transcription_pipeline.main import TranscriptionPipeline # Needed for mocking
from transcription_pipeline import utils # Needed for mocking


# --- Fixtures ---

@pytest.fixture
def minimal_valid_data() -> dict[str, Any]:
    """Minimal valid data for the /run request."""
    return {"api_key": "test_key", "domain": "test.com"}

@pytest.fixture
def full_valid_data(tmp_path) -> dict[str, Any]:
    """Valid data including all optional fields."""
    cache_dir = tmp_path / "cache"
    return {
        "api_key": "test_key",
        "domain": "test.com",
        "limit": 100,
        "min_id": 10,
        "max_id": 500,
        "processing_limit": 5,
        "download_queue_size": 20,
        "download_cache": str(cache_dir),
        "debug": True,
        "simulate_downloads": True,
        "callback_url": "http://localhost:9999/callback",
    }

@pytest.fixture
def validator() -> PipelineConfigValidator:
    """Fixture for the PipelineConfigValidator instance."""
    return PipelineConfigValidator()

# --- Test Cases ---

def test_validator_instantiation(validator: PipelineConfigValidator):
    """Test that the validator can be instantiated."""
    assert validator is not None

def test_validate_minimal_success(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test successful validation with minimal required data."""
    expected_kwargs = {
        "api_key": "test_key",
        "domain": "test.com",
        "debug": False, # Default
        "simulate_downloads": False, # Default
    }
    expected_callback_url = None

    kwargs, callback_url = validator.validate(minimal_valid_data, expected_api_key="test_key")

    assert kwargs == expected_kwargs
    assert callback_url == expected_callback_url

def test_validate_full_success(validator: PipelineConfigValidator, full_valid_data: dict[str, Any], tmp_path):
    """Test successful validation with all fields provided."""
    cache_dir = tmp_path / "cache"
    expected_kwargs = {
        "api_key": "test_key",
        "domain": "test.com",
        "limit": 100,
        "min_id": 10,
        "max_id": 500,
        "processing_limit": 5,
        "download_queue_size": 20,
        "download_cache": cache_dir,
        "debug": True,
        "simulate_downloads": True,
    }
    expected_callback_url = "http://localhost:9999/callback"

    kwargs, callback_url = validator.validate(full_valid_data, expected_api_key="test_key")

    assert kwargs == expected_kwargs
    assert callback_url == expected_callback_url

def test_validate_debug_flag_propagation(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test that the debug flag from the request overrides the default."""
    data = minimal_valid_data.copy()
    data["debug"] = True
    kwargs, _ = validator.validate(data, expected_api_key="test_key")
    assert kwargs["debug"] is True

def test_validate_simulate_downloads_flag(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test that the simulate_downloads flag is correctly parsed."""
    data = minimal_valid_data.copy()
    data["simulate_downloads"] = True
    kwargs, _ = validator.validate(data, expected_api_key="test_key")
    assert kwargs["simulate_downloads"] is True

def test_validate_optional_fields_none(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test validation when optional numeric/path fields are explicitly None."""
    data = minimal_valid_data.copy()
    data["limit"] = None
    data["min_id"] = None
    data["max_id"] = None
    data["processing_limit"] = None
    data["download_queue_size"] = None
    data["download_cache"] = None
    data["callback_url"] = None

    expected_kwargs = {
        "api_key": "test_key",
        "domain": "test.com",
        "debug": False,
        "simulate_downloads": False,
    }
    expected_callback_url = None

    kwargs, callback_url = validator.validate(data, expected_api_key="test_key")

    assert kwargs == expected_kwargs
    assert callback_url == expected_callback_url

def test_validate_missing_api_key(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test validation failure when api_key is missing."""
    data = minimal_valid_data.copy()
    del data["api_key"]
    with pytest.raises(ValueError, match="api_key and domain are required parameters."):
        validator.validate(data, expected_api_key="test_key")

def test_validate_missing_domain(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test validation failure when domain is missing."""
    data = minimal_valid_data.copy()
    del data["domain"]
    with pytest.raises(ValueError, match="api_key and domain are required parameters."):
        validator.validate(data, expected_api_key="test_key")

def test_validate_invalid_api_key(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test validation failure when provided api_key doesn't match expected."""
    with pytest.raises(ValueError, match="Invalid API key."):
        validator.validate(minimal_valid_data, expected_api_key="wrong_key")

def test_validate_no_expected_api_key(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test validation succeeds when no API key is expected by the server."""
    kwargs, _ = validator.validate(minimal_valid_data, expected_api_key=None)
    assert kwargs["api_key"] == "test_key" # Still included in kwargs

@pytest.mark.parametrize(
    "field, value, error_match",
    [
        ("limit", 0, "limit must be a positive integer."),
        ("limit", -1, "limit must be a positive integer."),
        ("limit", "abc", "Invalid value for limit: invalid literal for int"),
        ("min_id", 0, "min_id must be a positive integer."),
        ("min_id", -5, "min_id must be a positive integer."),
        ("min_id", "xyz", "Invalid value for min_id: invalid literal for int"),
        ("max_id", 0, "max_id must be a positive integer."),
        ("max_id", -10, "max_id must be a positive integer."),
        ("max_id", [], "Invalid value for max_id: expected an integer or string convertible to integer."),
        ("processing_limit", 0, "processing_limit must be a positive integer."),
        ("processing_limit", -2, "processing_limit must be a positive integer."),
        ("processing_limit", {}, "Invalid value for processing_limit: expected an integer or string convertible to integer."),
        ("download_queue_size", 0, "download_queue_size must be a positive integer."),
        ("download_queue_size", -3, "download_queue_size must be a positive integer."),
        ("download_queue_size", "1 2", "Invalid value for download_queue_size: invalid literal for int"),
    ],
)
def test_validate_invalid_positive_int_fields(
    validator: PipelineConfigValidator,
    minimal_valid_data: dict[str, Any],
    field: str,
    value: Any,
    error_match: str,
):
    """Test validation failure for various invalid positive integer fields."""
    data = minimal_valid_data.copy()
    data[field] = value
    with pytest.raises((ValueError, TypeError), match=error_match):
        validator.validate(data, expected_api_key="test_key")

def test_validate_download_cache_path_conversion(
    validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any], tmp_path
):
    """Test that download_cache string is converted to Path object."""
    cache_dir = tmp_path / "my_cache"
    data = minimal_valid_data.copy()
    data["download_cache"] = str(cache_dir)

    kwargs, _ = validator.validate(data, expected_api_key="test_key")

    assert "download_cache" in kwargs
    assert isinstance(kwargs["download_cache"], Path)
    assert kwargs["download_cache"] == cache_dir

def test_validate_callback_url_present(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test that callback_url is correctly extracted."""
    url = "http://example.com/notify"
    data = minimal_valid_data.copy()
    data["callback_url"] = url

    _, callback_url = validator.validate(data, expected_api_key="test_key")
    assert callback_url == url

def test_validate_callback_url_absent(validator: PipelineConfigValidator, minimal_valid_data: dict[str, Any]):
    """Test that callback_url is None when not provided."""
    _, callback_url = validator.validate(data=minimal_valid_data, expected_api_key="test_key")
    assert callback_url is None


# --- Tests for send_callback ---

@patch("transcription_pipeline.rest_interface.utils.post_request")
def test_send_callback_success(mock_post_request: MagicMock, caplog):
    """Test successful callback sending."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_post_request.return_value = mock_response

    url = "http://test.com/callback"
    payload = {"status": "success", "message": "done"}
    logger = logging.getLogger("test_logger")

    with caplog.at_level(logging.INFO):
        send_callback(url, payload, logger, debug=False)

    mock_post_request.assert_called_once_with(url, payload, True) # json=True
    assert f"Sending callback to {url} with payload: {payload}" in caplog.text
    assert f"Callback to {url} successful (Status: 200)" in caplog.text

@patch("transcription_pipeline.rest_interface.utils.post_request")
def test_send_callback_request_exception(mock_post_request: MagicMock, caplog):
    """Test callback sending when post_request raises RequestException."""
    url = "http://test.com/callback"
    payload = {"status": "error", "message": "failed"}
    logger = logging.getLogger("test_logger")
    exception = requests.exceptions.RequestException("Connection error")
    mock_post_request.side_effect = exception

    with caplog.at_level(logging.ERROR):
        send_callback(url, payload, logger, debug=False)

    mock_post_request.assert_called_once_with(url, payload, True)
    assert f"Failed to send callback to {url}: {exception}" in caplog.text
    # Ensure no unexpected exceptions were raised by send_callback itself

@patch("transcription_pipeline.rest_interface.utils.post_request")
def test_send_callback_unexpected_exception(mock_post_request: MagicMock, caplog):
    """Test callback sending when post_request raises an unexpected error."""
    url = "http://test.com/callback"
    payload = {"status": "error", "message": "unexpected"}
    logger = logging.getLogger("test_logger")
    exception = TypeError("Something weird happened")
    mock_post_request.side_effect = exception

    with caplog.at_level(logging.ERROR):
        send_callback(url, payload, logger, debug=True) # Enable debug for exc_info

    mock_post_request.assert_called_once_with(url, payload, True)
    assert f"Unexpected error sending callback to {url}: {exception}" in caplog.text
    assert "Traceback" in caplog.text # Check if exc_info was logged

@patch("transcription_pipeline.rest_interface.utils.post_request")
def test_send_callback_debug_logging(mock_post_request: MagicMock, caplog):
    """Test that exc_info is included in logs only when debug=True."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_post_request.return_value = mock_response

    url = "http://test.com/callback"
    payload = {"status": "success"}
    logger = logging.getLogger("test_logger")

    # Test with debug=False
    exception = requests.exceptions.RequestException("Connection error")
    mock_post_request.side_effect = exception
    with caplog.at_level(logging.ERROR):
        send_callback(url, payload, logger, debug=False)
    assert f"Failed to send callback to {url}: {exception}" in caplog.text
    assert "Traceback" not in caplog.text
    caplog.clear() # Reset logs

    # Test with debug=True
    exception = requests.exceptions.RequestException("Another connection error")
    mock_post_request.side_effect = exception
    with caplog.at_level(logging.ERROR):
        send_callback(url, payload, logger, debug=True)
    assert f"Failed to send callback to {url}: {exception}" in caplog.text
    assert "Traceback" in caplog.text


# --- Fixtures for PipelineLifecycleManager ---

@pytest.fixture
def mock_send_callback():
    """Fixture for a mocked send_callback function."""
    with patch("transcription_pipeline.rest_interface.send_callback") as mock:
        yield mock

@pytest.fixture
def mock_transcription_pipeline():
    """Fixture for a mocked TranscriptionPipeline class."""
    with patch("transcription_pipeline.rest_interface.TranscriptionPipeline") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock, mock_instance

@pytest.fixture
def mock_thread():
    """Fixture for a mocked threading.Thread class.

    The returned mock class will produce mock instances that have a mock `start` method.
    """
    with patch("transcription_pipeline.rest_interface.threading.Thread") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.start = MagicMock()
        yield mock_class, mock_instance

@pytest.fixture
def mock_tempfile():
    """Fixture for mocking tempfile.NamedTemporaryFile."""
    with patch("transcription_pipeline.rest_interface.tempfile.NamedTemporaryFile") as mock:
        mock_file = MagicMock()
        mock_file.name = "/tmp/fake_log_file.log"
        mock_file.close = MagicMock()
        # Make the mock usable as a context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_file
        mock_context.__exit__.return_value = None
        mock.return_value = mock_context
        yield mock, mock_file

@pytest.fixture
def manager(mock_send_callback) -> PipelineLifecycleManager:
    """Fixture for PipelineLifecycleManager instance."""
    logger = logging.getLogger("test_manager_logger")
    return PipelineLifecycleManager(
        callback_notifier=mock_send_callback,
        logger=logger,
        debug=False
    )

@pytest.fixture
def manager_debug(mock_send_callback) -> PipelineLifecycleManager:
    """Fixture for PipelineLifecycleManager instance with debug=True."""
    logger = logging.getLogger("test_manager_logger_debug")
    return PipelineLifecycleManager(
        callback_notifier=mock_send_callback,
        logger=logger,
        debug=True
    )

# --- Tests for PipelineLifecycleManager ---

def test_manager_initial_state(manager: PipelineLifecycleManager):
    """Test the initial state of the manager."""
    assert not manager.is_running()
    assert manager.get_status() == {"status": "idle", "exit_code": None}
    assert manager.get_exit_code() == DEFAULT_REST_FAILURE_EXIT_CODE # Default until success
    assert manager.pipeline_log_file_path is None

def test_manager_start_pipeline_success(
    manager: PipelineLifecycleManager,
    mock_thread: tuple[MagicMock, MagicMock],
    mock_tempfile: tuple[MagicMock, MagicMock],
    mock_transcription_pipeline: tuple[MagicMock, MagicMock], # Keep mocks unused here
    mock_send_callback: MagicMock, # Keep mock unused here
):
    """Test successfully starting the pipeline."""
    mock_thread_class, mock_thread_instance = mock_thread
    mock_tempfile_func, mock_tempfile_file = mock_tempfile
    pipeline_kwargs = {"domain": "test.com", "api_key": "key"}
    callback_url = "http://callback.test"

    # Mock os.environ.pop to avoid side effects if test fails mid-way
    with patch.dict(os.environ, {}, clear=True):
        result = manager.start_pipeline(pipeline_kwargs, callback_url)

        assert result is True
        assert manager.is_running()
        assert manager.get_status() == {"status": "running", "exit_code": None}
        assert manager.pipeline_log_file_path == mock_tempfile_file.name
        assert os.environ.get("DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE") == mock_tempfile_file.name

        # Check tempfile was called correctly
        mock_tempfile_func.assert_called_once_with(
            prefix="pipeline_manager_", suffix=".log", delete=False, mode="w"
        )

        # Check thread was created and started correctly
        mock_thread_class.assert_called_once_with(
            target=manager._run_pipeline_task,
            args=(pipeline_kwargs, callback_url),
            daemon=True # Check if daemon=True is set
        )
        mock_thread_instance.start.assert_called_once()

        # Check state variables
        assert manager._pipeline_started is True
        assert manager._pipeline_exit_code == DEFAULT_REST_FAILURE_EXIT_CODE # Initial failure code
        assert not manager._pipeline_completion_event.is_set() # Should be reset

def test_manager_start_pipeline_already_running(manager: PipelineLifecycleManager, mock_thread):
    """Test attempting to start the pipeline when it's already running."""
    mock_thread_class, _ = mock_thread
    pipeline_kwargs = {"domain": "test.com", "api_key": "key"}
    callback_url = "http://callback.test"

    mock_file = MagicMock()
    mock_file.name = "/tmp/fake_log_1.log"
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_context.__exit__.return_value = None

    # Start it once
    with patch.dict(os.environ, {}, clear=True), \
         patch("transcription_pipeline.rest_interface.tempfile.NamedTemporaryFile", return_value=mock_context):
        first_result = manager.start_pipeline(pipeline_kwargs, callback_url)
        assert first_result is True # Ensure first start succeeded

    # Try starting again
    result = manager.start_pipeline(pipeline_kwargs, callback_url)

    assert result is False
    assert manager.is_running() # Still running
    # Ensure thread wasn't started a second time
    assert mock_thread_class.call_count == 1
    assert mock_thread_class.return_value.start.call_count == 1

@patch("transcription_pipeline.rest_interface.threading.Event")
def test_manager_wait_for_completion(mock_event_class, manager: PipelineLifecycleManager):
    """Test that wait_for_completion calls wait() on the event."""
    mock_event_instance = MagicMock()
    manager._pipeline_completion_event = mock_event_instance # Replace the real event

    manager.wait_for_completion()

    mock_event_instance.wait.assert_called_once()

def test_manager_get_exit_code(manager: PipelineLifecycleManager):
    """Test retrieving the exit code."""
    manager._pipeline_exit_code = 0 # Simulate success
    assert manager.get_exit_code() == 0

    manager._pipeline_exit_code = 5 # Simulate specific failure
    assert manager.get_exit_code() == 5

# --- Tests for _run_pipeline_task logic ---

def test_run_pipeline_task_success(
    manager: PipelineLifecycleManager,
    mock_transcription_pipeline: tuple[MagicMock, MagicMock],
    mock_send_callback: MagicMock,
    mock_tempfile: tuple[MagicMock, MagicMock],
    caplog
):
    """Test the logic within _run_pipeline_task for a successful run."""
    mock_pipeline_class, mock_pipeline_instance = mock_transcription_pipeline
    _, mock_tempfile_file = mock_tempfile
    pipeline_kwargs = {"domain": "test.com", "api_key": "key", "debug": False}
    callback_url = "http://callback.test"
    log_content = "Pipeline ran well."

    # --- Setup preconditions for _run_pipeline_task ---
    manager.pipeline_log_file_path = mock_tempfile_file.name # Set log path manually
    # Ensure the environment variable is set for the duration of the task run
    log_env_var = "DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"
    env_patch = patch.dict(os.environ, {log_env_var: mock_tempfile_file.name}, clear=True)

    # Mock file reading for _fetch_pipeline_logs within _trigger_callback
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = log_content
    mock_open_context = MagicMock()
    mock_open_context.__enter__.return_value = mock_file_handle
    mock_open_context.__exit__.return_value = None
    open_patch = patch("builtins.open", return_value=mock_open_context)
    # --- End Setup ---

    with env_patch, open_patch as mock_open_func, caplog.at_level(logging.INFO):
        # Directly execute the task method
        manager._run_pipeline_task(pipeline_kwargs, callback_url)

    # --- Assertions ---
    mock_pipeline_class.assert_called_once_with(**pipeline_kwargs)
    mock_pipeline_instance.run.assert_called_once()

    # Check that logs were fetched for the callback
    mock_open_func.assert_called_once_with(mock_tempfile_file.name, 'r')
    mock_file_handle.read.assert_called_once()

    # Check that the correct callback was sent
    expected_payload = {"status": "success", "message": log_content}
    mock_send_callback.assert_called_once_with(
        callback_url, expected_payload, manager.log, manager.debug
    )

    # Check final state (set within _run_pipeline_task)
    assert not manager.is_running() # Should be complete now
    assert manager.get_status() == {"status": "complete", "exit_code": 0}
    assert manager.get_exit_code() == 0
    assert manager._pipeline_completion_event.is_set()
    assert "Pipeline run finished successfully" in caplog.text
    # Check environment variable was cleaned up
    assert log_env_var not in os.environ

def test_run_pipeline_task_pipeline_error(
    manager_debug: PipelineLifecycleManager, # Use debug manager for exc_info check
    mock_transcription_pipeline: tuple[MagicMock, MagicMock],
    mock_send_callback: MagicMock,
    mock_tempfile: tuple[MagicMock, MagicMock],
    caplog
):
    """Test the logic within _run_pipeline_task when the pipeline fails."""
    mock_pipeline_class, mock_pipeline_instance = mock_transcription_pipeline
    _, mock_tempfile_file = mock_tempfile # We need the mock file name
    pipeline_kwargs = {"domain": "test.com", "api_key": "key", "debug": True}
    callback_url = "http://callback.test"
    error_message = "Pipeline failed!"
    mock_pipeline_instance.run.side_effect = Exception(error_message)

    # --- Setup preconditions for _run_pipeline_task ---
    manager_debug.pipeline_log_file_path = mock_tempfile_file.name # Set log path manually
    log_env_var = "DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"
    env_patch = patch.dict(os.environ, {log_env_var: mock_tempfile_file.name}, clear=True)
    open_patch = patch("builtins.open") # Mock open to prevent actual file access attempts
    # --- End Setup ---

    with env_patch, open_patch as mock_open_func, caplog.at_level(logging.ERROR):
         # Directly execute the task method
        manager_debug._run_pipeline_task(pipeline_kwargs, callback_url)

    # --- Assertions ---
    mock_pipeline_class.assert_called_once_with(**pipeline_kwargs)
    mock_pipeline_instance.run.assert_called_once()

    # Check logs were NOT fetched on error
    mock_open_func.assert_not_called()

    # Check that the correct error callback was sent
    expected_payload = {"status": "error", "message": error_message}
    mock_send_callback.assert_called_once_with(
        callback_url, expected_payload, manager_debug.log, manager_debug.debug
    )

    # Check final state
    assert not manager_debug.is_running()
    assert manager_debug.get_status() == {"status": "complete", "exit_code": DEFAULT_REST_FAILURE_EXIT_CODE}
    assert manager_debug.get_exit_code() == DEFAULT_REST_FAILURE_EXIT_CODE
    assert manager_debug._pipeline_completion_event.is_set()
    assert f"Error during pipeline execution: {error_message}" in caplog.text
    assert "Traceback" in caplog.text # Because debug=True
    # Check environment variable was cleaned up
    assert log_env_var not in os.environ

def test_run_pipeline_task_no_callback_url(
    manager: PipelineLifecycleManager,
    mock_transcription_pipeline: tuple[MagicMock, MagicMock],
    mock_send_callback: MagicMock,
    mock_tempfile: tuple[MagicMock, MagicMock],
):
    """Test the logic within _run_pipeline_task when no callback URL is provided."""
    mock_pipeline_class, mock_pipeline_instance = mock_transcription_pipeline
    _, mock_tempfile_file = mock_tempfile # We need the mock file name
    pipeline_kwargs = {"domain": "test.com", "api_key": "key"}
    callback_url = None # No callback

    # --- Setup preconditions for _run_pipeline_task ---
    manager.pipeline_log_file_path = mock_tempfile_file.name # Set log path manually
    log_env_var = "DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"
    env_patch = patch.dict(os.environ, {log_env_var: mock_tempfile_file.name}, clear=True)
    # --- End Setup ---

    with env_patch:
        # Directly execute the task method
        manager._run_pipeline_task(pipeline_kwargs, callback_url)

    # --- Assertions ---
    mock_pipeline_class.assert_called_once_with(**pipeline_kwargs)
    mock_pipeline_instance.run.assert_called_once()

    # Check callback was not called
    mock_send_callback.assert_not_called()

    # Check final state (should be success)
    assert not manager.is_running()
    assert manager.get_status() == {"status": "complete", "exit_code": 0}
    assert manager.get_exit_code() == 0
    assert manager._pipeline_completion_event.is_set()
    # Check environment variable was cleaned up
    assert log_env_var not in os.environ

# --- Tests for _fetch_pipeline_logs ---

def test_fetch_pipeline_logs_success(manager: PipelineLifecycleManager, mock_tempfile):
    """Test fetching logs successfully."""
    _, mock_tempfile_file = mock_tempfile
    manager.pipeline_log_file_path = mock_tempfile_file.name
    log_content = "Line 1\nLine 2"

    # Mock the file handle returned by open's context manager
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = log_content

    # Mock the context manager returned by open()
    mock_open_context = MagicMock()
    mock_open_context.__enter__.return_value = mock_file_handle
    mock_open_context.__exit__.return_value = None

    # Patch builtins.open to return our mock context manager
    with patch("builtins.open", return_value=mock_open_context) as mock_open_func:
        result = manager._fetch_pipeline_logs()

    assert result == log_content
    # Check that open was called correctly
    mock_open_func.assert_called_once_with(mock_tempfile_file.name, 'r')
    # Check that read was called on the file handle
    mock_file_handle.read.assert_called_once()

def test_fetch_pipeline_logs_empty(manager: PipelineLifecycleManager, mock_tempfile):
    """Test fetching logs when the file is empty."""
    _, mock_tempfile_file = mock_tempfile
    manager.pipeline_log_file_path = mock_tempfile_file.name

    # Mock the file handle returned by open's context manager
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = ""

    # Mock the context manager returned by open()
    mock_open_context = MagicMock()
    mock_open_context.__enter__.return_value = mock_file_handle
    mock_open_context.__exit__.return_value = None

    with patch("builtins.open", return_value=mock_open_context) as mock_open_func:
        result = manager._fetch_pipeline_logs()

    assert result == "Log file is empty."
    mock_open_func.assert_called_once_with(mock_tempfile_file.name, 'r')
    mock_file_handle.read.assert_called_once()

def test_fetch_pipeline_logs_file_not_found(manager: PipelineLifecycleManager, caplog):
    """Test fetching logs when the file doesn't exist."""
    manager.pipeline_log_file_path = "/non/existent/path.log"
    error = FileNotFoundError("No such file")

    with patch("builtins.open", side_effect=error), caplog.at_level(logging.ERROR):
        result = manager._fetch_pipeline_logs()

    assert "Error reading log file" in result
    assert manager.pipeline_log_file_path in result
    assert "Error reading log file" in caplog.text
    assert manager.pipeline_log_file_path in caplog.text

def test_fetch_pipeline_logs_no_path_set(manager: PipelineLifecycleManager, caplog):
    """Test fetching logs when the path was never set."""
    manager.pipeline_log_file_path = None

    with caplog.at_level(logging.ERROR):
        result = manager._fetch_pipeline_logs()

    assert "Log file path not set" in result
    assert "Log file path not set" in caplog.text


# --- Fixtures for create_app ---

@pytest.fixture
def mock_manager() -> MagicMock:
    """Fixture for a mocked PipelineLifecycleManager."""
    mock = MagicMock(spec=PipelineLifecycleManager)
    mock.is_running.return_value = False
    mock.get_status.return_value = {"status": "idle", "exit_code": None}
    mock.start_pipeline.return_value = True # Default success
    return mock

@pytest.fixture
def mock_validator() -> MagicMock:
    """Fixture for a mocked PipelineConfigValidator."""
    mock = MagicMock(spec=PipelineConfigValidator)
    # Default validation success
    mock.validate.return_value = ({"arg1": "val1"}, "http://callback.test")
    return mock

@pytest.fixture(scope="function") # Ensure fresh app per test
def test_app(mock_manager: MagicMock, mock_validator: MagicMock) -> Flask:
    """Fixture to create a Flask test app instance using the factory."""
    logger = logging.getLogger("test_app_logger")
    app = create_app(
        pipeline_manager=mock_manager,
        validator=mock_validator,
        api_key="server_key", # Example API key for testing
        logger=logger,
        debug=False
    )
    app.config.update({"TESTING": True})
    return app

@pytest.fixture
def client(test_app: Flask):
    """Fixture for the Flask test client."""
    return test_app.test_client()


# --- Tests for create_app ---

def test_create_app_status_endpoint(client, mock_manager: MagicMock):
    """Test the /status endpoint."""
    mock_manager.get_status.return_value = {"status": "running", "exit_code": None}
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json == {"status": "running", "exit_code": None}
    mock_manager.get_status.assert_called_once()

def test_create_app_run_endpoint_success(client, mock_manager: MagicMock, mock_validator: MagicMock):
    """Test the /run endpoint with a successful request."""
    request_data = {"api_key": "server_key", "domain": "example.com"}
    validated_kwargs = {"domain": "example.com", "api_key": "server_key"}
    callback_url = "http://callback.example.com"
    mock_validator.validate.return_value = (validated_kwargs, callback_url)
    mock_manager.start_pipeline.return_value = True

    response = client.post("/run", json=request_data)

    assert response.status_code == 202
    assert response.json == {"message": "Transcription pipeline accepted for processing."}
    mock_validator.validate.assert_called_once_with(request_data, "server_key")
    mock_manager.start_pipeline.assert_called_once_with(validated_kwargs, callback_url)

def test_create_app_run_endpoint_not_json(client, mock_manager: MagicMock, mock_validator: MagicMock):
    """Test the /run endpoint when the request is not JSON."""
    response = client.post("/run", data="not json")
    assert response.status_code == 400
    assert response.json == {"error": "Request must be JSON"}
    mock_validator.validate.assert_not_called()
    mock_manager.start_pipeline.assert_not_called()

def test_create_app_run_endpoint_validation_error(client, mock_manager: MagicMock, mock_validator: MagicMock):
    """Test the /run endpoint when validation fails."""
    request_data = {"domain": "example.com"} # Missing api_key
    error_message = "api_key and domain are required parameters."
    mock_validator.validate.side_effect = ValueError(error_message)

    response = client.post("/run", json=request_data)

    assert response.status_code == 400
    assert response.json == {"error": error_message}
    mock_validator.validate.assert_called_once_with(request_data, "server_key")
    mock_manager.start_pipeline.assert_not_called()

def test_create_app_run_endpoint_already_running(client, mock_manager: MagicMock, mock_validator: MagicMock):
    """Test the /run endpoint when the pipeline manager reports it's already running."""
    request_data = {"api_key": "server_key", "domain": "example.com"}
    validated_kwargs = {"domain": "example.com", "api_key": "server_key"}
    callback_url = "http://callback.example.com"
    mock_validator.validate.return_value = (validated_kwargs, callback_url)
    mock_manager.start_pipeline.return_value = False # Simulate already running

    response = client.post("/run", json=request_data)

    assert response.status_code == 409
    assert response.json == {"error": "Pipeline run has already been triggered or failed to start."}
    mock_validator.validate.assert_called_once_with(request_data, "server_key")
    mock_manager.start_pipeline.assert_called_once_with(validated_kwargs, callback_url)

def test_create_app_run_endpoint_unexpected_error(client, mock_manager: MagicMock, mock_validator: MagicMock):
    """Test the /run endpoint when an unexpected error occurs during validation or start."""
    request_data = {"api_key": "server_key", "domain": "example.com"}
    error_message = "Something broke!"
    mock_validator.validate.side_effect = Exception(error_message) # Simulate error during validation

    response = client.post("/run", json=request_data)

    assert response.status_code == 500
    assert "error" in response.json
    assert error_message in response.json["error"]
    mock_validator.validate.assert_called_once_with(request_data, "server_key")
    mock_manager.start_pipeline.assert_not_called()


# --- Tests for parse_arguments ---

@patch("sys.argv", ["script_name"])
def test_parse_arguments_defaults():
    """Test parsing arguments with default values."""
    args = parse_arguments()
    assert args.host == DEFAULT_REST_HOST
    assert args.port == DEFAULT_REST_PORT
    assert args.api_key is None
    assert args.debug is False

@patch("sys.argv", ["script_name", "--host", "192.168.1.100"])
def test_parse_arguments_custom_host():
    """Test parsing arguments with a custom host."""
    args = parse_arguments()
    assert args.host == "192.168.1.100"
    assert args.port == DEFAULT_REST_PORT
    assert args.api_key is None
    assert args.debug is False

@patch("sys.argv", ["script_name", "--port", "9000"])
def test_parse_arguments_custom_port():
    """Test parsing arguments with a custom port."""
    args = parse_arguments()
    assert args.host == DEFAULT_REST_HOST
    assert args.port == 9000
    assert args.api_key is None
    assert args.debug is False

@patch("sys.argv", ["script_name", "--api-key", "mysecretkey"])
def test_parse_arguments_custom_api_key():
    """Test parsing arguments with a custom API key."""
    args = parse_arguments()
    assert args.host == DEFAULT_REST_HOST
    assert args.port == DEFAULT_REST_PORT
    assert args.api_key == "mysecretkey"
    assert args.debug is False

@patch("sys.argv", ["script_name", "--debug"])
def test_parse_arguments_debug_flag():
    """Test parsing arguments with the debug flag."""
    args = parse_arguments()
    assert args.host == DEFAULT_REST_HOST
    assert args.port == DEFAULT_REST_PORT
    assert args.api_key is None
    assert args.debug is True

@patch("sys.argv", ["script_name", "--host", "0.0.0.0", "--port", "8888", "--api-key", "testkey123", "--debug"])
def test_parse_arguments_all_custom():
    """Test parsing arguments with all custom values."""
    args = parse_arguments()
    assert args.host == "0.0.0.0"
    assert args.port == 8888
    assert args.api_key == "testkey123"
    assert args.debug is True


# --- Tests for TranscriptionRestInterface ---

@patch("transcription_pipeline.rest_interface.PipelineLifecycleManager")
@patch("transcription_pipeline.rest_interface.PipelineConfigValidator")
@patch("transcription_pipeline.rest_interface.Logger")
@patch("transcription_pipeline.rest_interface.send_callback") # Need to patch where it's used
def test_transcription_rest_interface_init(
    mock_send_callback: MagicMock,
    mock_logger_class: MagicMock,
    mock_validator_class: MagicMock,
    mock_manager_class: MagicMock,
):
    """Test the __init__ method of TranscriptionRestInterface."""
    mock_logger_instance = MagicMock()
    mock_logger_class.return_value = mock_logger_instance
    mock_validator_instance = MagicMock()
    mock_validator_class.return_value = mock_validator_instance
    mock_manager_instance = MagicMock()
    mock_manager_class.return_value = mock_manager_instance

    # Test with debug=False
    interface_no_debug = TranscriptionRestInterface(debug=False)

    mock_logger_class.assert_called_once_with("TranscriptionRESTInterface", debug=False)
    mock_validator_class.assert_called_once()
    mock_manager_class.assert_called_once_with(
        callback_notifier=mock_send_callback,
        logger=mock_logger_instance,
        debug=False
    )
    assert interface_no_debug.debug is False
    assert interface_no_debug.log is mock_logger_instance
    assert interface_no_debug.validator is mock_validator_instance
    assert interface_no_debug.manager is mock_manager_instance

    # Reset mocks for the next test case
    mock_logger_class.reset_mock()
    mock_validator_class.reset_mock()
    mock_manager_class.reset_mock()

    # Test with debug=True
    interface_debug = TranscriptionRestInterface(debug=True)

    mock_logger_class.assert_called_once_with("TranscriptionRESTInterface", debug=True)
    mock_validator_class.assert_called_once()
    mock_manager_class.assert_called_once_with(
        callback_notifier=mock_send_callback,
        logger=mock_logger_instance,
        debug=True
    )
    assert interface_debug.debug is True
    assert interface_debug.log is mock_logger_instance
    assert interface_debug.validator is mock_validator_instance
    assert interface_debug.manager is mock_manager_instance


# --- Tests for main function ---

@patch("transcription_pipeline.rest_interface.TranscriptionRestInterface")
@patch("sys.argv", ["script_name", "--host", "test_host", "--port", "1234"])
def test_main_no_api_key_no_debug(mock_interface_class: MagicMock):
    """Test main with default API key (None) and debug=False."""
    mock_instance = MagicMock()
    mock_interface_class.return_value = mock_instance

    with patch.dict(os.environ, {}, clear=True): # Ensure no env var
        main()

    mock_interface_class.assert_called_once_with(debug=False)
    mock_instance.run_server.assert_called_once_with(host="test_host", port=1234, api_key=None)

@patch("transcription_pipeline.rest_interface.TranscriptionRestInterface")
@patch("sys.argv", ["script_name", "--api-key", "arg_key", "--debug"])
def test_main_api_key_from_args_debug(mock_interface_class: MagicMock):
    """Test main with API key from args and debug=True."""
    mock_instance = MagicMock()
    mock_interface_class.return_value = mock_instance

    with patch.dict(os.environ, {"TRANSCRIPTION_API_KEY": "env_key"}, clear=True): # Env var exists but arg takes precedence
        main()

    mock_interface_class.assert_called_once_with(debug=True)
    mock_instance.run_server.assert_called_once_with(host=DEFAULT_REST_HOST, port=DEFAULT_REST_PORT, api_key="arg_key")

@patch("transcription_pipeline.rest_interface.TranscriptionRestInterface")
@patch("sys.argv", ["script_name"]) # No API key arg
def test_main_api_key_from_env(mock_interface_class: MagicMock):
    """Test main with API key from environment variable."""
    mock_instance = MagicMock()
    mock_interface_class.return_value = mock_instance
    env_api_key = "environment_secret_key"

    with patch.dict(os.environ, {"TRANSCRIPTION_API_KEY": env_api_key}, clear=True):
        main()

    mock_interface_class.assert_called_once_with(debug=False)
    mock_instance.run_server.assert_called_once_with(host=DEFAULT_REST_HOST, port=DEFAULT_REST_PORT, api_key=env_api_key)

@patch("transcription_pipeline.rest_interface.TranscriptionRestInterface")
@patch("sys.argv", ["script_name", "--debug"]) # No API key arg or env var
def test_main_no_api_key_debug(mock_interface_class: MagicMock):
    """Test main with no API key provided and debug=True."""
    mock_instance = MagicMock()
    mock_interface_class.return_value = mock_instance

    with patch.dict(os.environ, {}, clear=True): # Ensure no env var
        main()

    mock_interface_class.assert_called_once_with(debug=True)
    mock_instance.run_server.assert_called_once_with(host=DEFAULT_REST_HOST, port=DEFAULT_REST_PORT, api_key=None)


# --- Integration Tests for TranscriptionRestInterface.run_server ---

@patch("transcription_pipeline.rest_interface.sys.exit")
@patch("transcription_pipeline.rest_interface.threading.Thread")
@patch("transcription_pipeline.rest_interface.create_app")
def test_run_server_flow_success(
    mock_create_app: MagicMock,
    mock_thread_class: MagicMock,
    mock_sys_exit: MagicMock,
):
    """Test the orchestration flow of run_server for a successful exit."""
    mock_flask_app = MagicMock(spec=Flask)
    mock_create_app.return_value = mock_flask_app

    mock_thread_instance = mock_thread_class.return_value
    mock_thread_instance.start = MagicMock()

    # --- Instantiate Real Interface ---
    # We need a real interface to test its run_server method
    # Its internal manager and validator will be real, but we mock manager methods
    interface = TranscriptionRestInterface(debug=False)

    interface.manager.wait_for_completion = MagicMock()
    interface.manager.get_exit_code = MagicMock(return_value=0) # Simulate success exit code

    test_host = "127.0.0.1"
    test_port = 5000
    test_api_key = "success_key"
    interface.run_server(host=test_host, port=test_port, api_key=test_api_key)

    mock_create_app.assert_called_once_with(
        pipeline_manager=interface.manager,
        validator=interface.validator,
        api_key=test_api_key,
        logger=interface.log,
        debug=False
    )
    # Check thread creation: Use ANY for the lambda target as it's hard to match exactly
    mock_thread_class.assert_called_once_with(target=ANY, daemon=True)
    # Check the lambda target calls app.run with correct args (requires inspecting the ANY target's args)
    # This is complex, let's focus on the fact that Thread was called with daemon=True
    # and that start() was called.
    mock_thread_instance.start.assert_called_once()

    interface.manager.wait_for_completion.assert_called_once()
    interface.manager.get_exit_code.assert_called_once()
    mock_sys_exit.assert_called_once_with(0) # Assert exit with success code

@patch("transcription_pipeline.rest_interface.sys.exit")
@patch("transcription_pipeline.rest_interface.threading.Thread")
@patch("transcription_pipeline.rest_interface.create_app")
def test_run_server_flow_failure_exit(
    mock_create_app: MagicMock,
    mock_thread_class: MagicMock,
    mock_sys_exit: MagicMock,
):
    """Test the orchestration flow of run_server for a failure exit code."""
    mock_flask_app = MagicMock(spec=Flask)
    mock_create_app.return_value = mock_flask_app

    mock_thread_instance = mock_thread_class.return_value
    mock_thread_instance.start = MagicMock()

    # --- Instantiate Real Interface ---
    interface = TranscriptionRestInterface(debug=True) # Test with debug=True

    interface.manager.wait_for_completion = MagicMock()
    failure_code = 5 # Simulate failure exit code
    interface.manager.get_exit_code = MagicMock(return_value=failure_code)

    test_host = "0.0.0.0"
    test_port = 8080
    test_api_key = None
    interface.run_server(host=test_host, port=test_port, api_key=test_api_key)

    mock_create_app.assert_called_once_with(
        pipeline_manager=interface.manager,
        validator=interface.validator,
        api_key=test_api_key,
        logger=interface.log,
        debug=True # Check debug flag propagation
    )
    mock_thread_class.assert_called_once_with(target=ANY, daemon=True)
    mock_thread_instance.start.assert_called_once()
    interface.manager.wait_for_completion.assert_called_once()
    interface.manager.get_exit_code.assert_called_once()
    mock_sys_exit.assert_called_once_with(failure_code) # Assert exit with failure code
