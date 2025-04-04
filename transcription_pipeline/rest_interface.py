"""REST interface for controlling the transcription pipeline.

This interface accepts a single request to trigger the pipeline and then
signals completion before the process exits. Designed for one-shot container execution.
"""

import os
import argparse
import sys
import tempfile
import logging
import threading
import subprocess
import traceback
from pathlib import Path
from typing import Any

import requests
from flask import Flask, request, jsonify, Response
from download_pipeline_processor.logger import (
    Logger,
)

from transcription_pipeline.constants import (
    DEFAULT_REST_HOST,
    DEFAULT_REST_PORT,
    DEFAULT_REST_FAILURE_EXIT_CODE,
)
from transcription_pipeline.main import TranscriptionPipeline
from transcription_pipeline import utils


class TranscriptionRestInterface:
    """Manages the Flask REST interface for triggering the transcription pipeline.

    This class encapsulates the Flask application, state management for one-shot
    execution, and the logic for handling pipeline runs in a background thread.
    It ensures that only one pipeline run can be triggered per instance and
    coordinates the shutdown process upon completion.

    :param debug: Enable debug logging for the interface and potentially the pipeline.
    :type debug: bool
    """
    def __init__(self, debug: bool = False) -> None:
        """Initialize the REST interface components and state."""
        self.debug: bool = debug
        self.log: logging.Logger = Logger("TranscriptionRESTInterface", debug=self.debug)
        self.validator: PipelineConfigValidator = PipelineConfigValidator()
        self.manager: PipelineLifecycleManager = PipelineLifecycleManager(
            callback_notifier=send_callback,
            logger=self.log,
            debug=self.debug
        )

    def run_server(self, host: str, port: int, api_key: str | None) -> None:
        """Run the Flask development server and wait for pipeline completion.

        :param host: Host to bind the server to.
        :type host: str
        :param port: Port to bind the server to.
        :type port: int
        :param api_key: The API key the server should expect, or None.
        :type api_key: str | None
        """
        # Create the Flask app using the factory function and internal components
        app: Flask = create_app(
            pipeline_manager=self.manager,
            validator=self.validator,
            api_key=api_key,
            logger=self.log,
            debug=self.debug
        )

        flask_thread: threading.Thread = threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=self.debug, use_reloader=False),
            daemon=True
        )
        flask_thread.start()
        self.log.info(f"Flask server started in background thread on http://{host}:{port}")

        # Wait for the pipeline manager to signal completion
        self.log.info("Waiting for pipeline completion signal...")
        self.manager.wait_for_completion() # Blocks here

        exit_code: int = self.manager.get_exit_code()
        self.log.info(f"Pipeline completion signal received. Exiting with code {exit_code}.")
        sys.exit(exit_code)


class PipelineConfigValidator:
    """Validates the configuration data received for a pipeline run request.

    Ensures required fields are present, performs type checks, validates
    positive integers, and checks the API key if required.
    """

    def validate(
        self, data: dict[str, Any], expected_api_key: str | None
    ) -> tuple[dict[str, Any], str | None]:
        """Parse and validate the JSON payload from the /run request.

        :param data: The JSON data received from the request.
        :type data: dict[str, Any]
        :param expected_api_key: The API key the server expects, or None if no key check is needed.
        :type expected_api_key: str | None
        :return: A tuple containing the validated pipeline keyword arguments
                 and the callback URL (if provided).
        :rtype: tuple[dict[str, Any], str | None]
        :raises ValueError: If required parameters are missing or validation fails.
        """
        api_key: str | None = data.get("api_key")
        domain: str | None = data.get("domain")

        if not api_key or not domain:
            raise ValueError("api_key and domain are required parameters.")

        self._validate_api_key(api_key, expected_api_key)

        debug: bool = data.get("debug", False)
        simulate_downloads: bool = data.get("simulate_downloads", False)
        callback_url: str | None = data.get("callback_url")

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "domain": domain,
            "debug": debug,
            "simulate_downloads": simulate_downloads,
        }

        limit: int | None = self._validate_positive_int(data.get("limit"), "limit")
        if limit is not None:
            kwargs["limit"] = limit
        min_id: int | None = self._validate_positive_int(data.get("min_id"), "min_id")
        if min_id is not None:
            kwargs["min_id"] = min_id
        max_id: int | None = self._validate_positive_int(data.get("max_id"), "max_id")
        if max_id is not None:
            kwargs["max_id"] = max_id
        processing_limit: int | None = self._validate_positive_int(data.get("processing_limit"), "processing_limit")
        if processing_limit is not None:
            kwargs["processing_limit"] = processing_limit
        download_queue_size: int | None = self._validate_positive_int(data.get("download_queue_size"),"download_queue_size")
        if download_queue_size is not None:
            kwargs["download_queue_size"] = download_queue_size

        download_cache_str: str | None = data.get("download_cache")
        if download_cache_str is not None:
            try:
                kwargs["download_cache"] = Path(download_cache_str)
            except TypeError as e:
                raise ValueError(f"Invalid value for download_cache: {e}") from e

        return kwargs, callback_url

    def _validate_positive_int(
        self, value: str | int | None, name: str
    ) -> int | None:
        """Validate if a value is a positive integer.

        :param value: The value to validate.
        :type value: str | int | None
        :param name: The name of the parameter for error messages.
        :type name: str
        :return: The validated positive integer or None if input is None.
        :rtype: int | None
        :raises ValueError: If the value is not a positive integer or cannot be converted.
        """
        if value is None:
            return None
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
            return int_value
        except (ValueError, TypeError) as e:
            if isinstance(e, TypeError):
                 raise ValueError(f"Invalid value for {name}: expected an integer or string convertible to integer.") from e
            raise ValueError(f"Invalid value for {name}: {e}") from e


    def _validate_api_key(
        self, provided_key: str | None, expected_key: str | None
    ) -> None:
        """Validate the provided API key against the expected key.

        Does nothing if the key is valid or if no expected key is set.

        :param provided_key: The API key from the request data.
        :type provided_key: str | None
        :param expected_key: The API key the server expects. If None, validation passes.
        :type expected_key: str | None
        :raises ValueError: If the expected key is set and the provided key does not match.
        """
        if expected_key is None:
            return
        if provided_key != expected_key:
            raise ValueError("Invalid API key.")


def send_callback(url: str, payload: dict[str, Any], logger: logging.Logger, debug: bool = False) -> None:
    """Send a POST request to the callback URL using utils.post_request.

    Logs errors but does not raise exceptions.

    :param url: The URL to send the callback to.
    :type url: str
    :param payload: The JSON payload to send.
    :type payload: dict[str, Any]
    :param logger: The logger instance to use for logging.
    :type logger: logging.Logger
    :param debug: If True, include exception info in error logs.
    :type debug: bool
    """
    try:
        logger.info(f"Sending callback to {url} with payload: {payload}")
        response: requests.Response = utils.post_request(url, payload, True)
        logger.info(f"Callback to {url} successful (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send callback to {url}: {e}", exc_info=debug)
    except Exception as e:
        logger.error(f"Unexpected error sending callback to {url}: {e}", exc_info=debug)


class PipelineLifecycleManager:
    """Manages the state and execution lifecycle of a single pipeline run.

    Handles starting the pipeline in a background thread, managing state
    (idle, running, complete), setting up pipeline-specific logging,
    triggering callbacks on completion/error, and providing the final exit code.

    :param callback_notifier: Function to call for sending completion/error notifications.
    :type callback_notifier: Callable[[str, dict[str, Any], logging.Logger, bool], None]
    :param logger: Logger instance for manager-specific logging.
    :type logger: logging.Logger
    :param debug: Enable debug logging.
    :type debug: bool
    """
    def __init__(
        self,
        callback_notifier: callable, # Use 'callable' for type hint
        logger: logging.Logger,
        debug: bool = False
    ) -> None:
        """Initialize the pipeline lifecycle manager."""
        self.callback_notifier: callable = callback_notifier
        self.log: logging.Logger = logger
        self.debug: bool = debug

        self._lock: threading.Lock = threading.Lock()
        self._pipeline_started: bool = False
        self._pipeline_completed: bool = False # Track completion separately
        self._pipeline_completion_event: threading.Event = threading.Event()
        self._pipeline_exit_code: int = DEFAULT_REST_FAILURE_EXIT_CODE # Default failure
        self.pipeline_log_file_path: str | None = None

    def is_running(self) -> bool:
        """Check if the pipeline is currently running.

        :return: True if the pipeline has started but not yet completed, False otherwise.
        :rtype: bool
        """
        with self._lock:
            return self._pipeline_started and not self._pipeline_completed

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the pipeline managed by this instance.

        :return: Dictionary containing status ('idle', 'running', 'complete') and exit_code (int | None).
        :rtype: dict[str, Any]
        """
        with self._lock:
            if self._pipeline_completed:
                status = "complete"
                exit_code = self._pipeline_exit_code
            elif self._pipeline_started:
                status = "running"
                exit_code = None
            else:
                status = "idle"
                exit_code = None
            return {"status": status, "exit_code": exit_code}

    def start_pipeline(self, pipeline_kwargs: dict[str, Any], callback_url: str | None) -> bool:
        """Start the transcription pipeline in a background thread.

        Sets up logging, starts the thread, and updates state.

        :param pipeline_kwargs: Keyword arguments to pass to the TranscriptionPipeline constructor.
        :type pipeline_kwargs: dict[str, Any]
        :param callback_url: The URL to notify upon completion or error, or None.
        :type callback_url: str | None
        :return: True if the pipeline was started successfully, False if it was already running.
        :rtype: bool
        """
        with self._lock:
            if self._pipeline_started:
                self.log.warning("Pipeline start requested, but already started.")
                return False

            # Reset state for a new run
            self._pipeline_started = True
            self._pipeline_completed = False
            self._pipeline_exit_code = DEFAULT_REST_FAILURE_EXIT_CODE # Reset to default failure
            self._pipeline_completion_event.clear()
            self.pipeline_log_file_path = None # Reset log path

            try:
                self._setup_pipeline_logging()
            except Exception as e:
                self.log.error(f"Failed to set up pipeline logging: {e}", exc_info=self.debug)
                # Mark as failed immediately if logging setup fails
                self._pipeline_completed = True
                self._pipeline_completion_event.set()
                return False # Indicate start failure

            # Start the background task
            thread: threading.Thread = threading.Thread(
                target=self._run_pipeline_task,
                args=(pipeline_kwargs, callback_url),
                daemon=True # Ensure thread doesn't block exit
            )
            thread.start()
            self.log.info("Pipeline execution thread started.")
            return True

    def _setup_pipeline_logging(self) -> None:
        """Create a temporary log file for the pipeline run and set the environment variable.

        Stores the path in `self.pipeline_log_file_path`.
        """
        # Set up logging to a temporary file for the pipeline.
        # This context manager handles closing the file descriptor correctly.
        with tempfile.NamedTemporaryFile(
            prefix="pipeline_manager_", suffix=".log", delete=False, mode="w"
        ) as log_file:
            self.pipeline_log_file_path = log_file.name
        # Set environment variable for the pipeline process to use
        # Use pop to avoid leaving it set if the process fails early
        os.environ["DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"] = self.pipeline_log_file_path
        self.log.info(f"Pipeline logging configured to: {self.pipeline_log_file_path}")


    def _run_pipeline_task(self, pipeline_kwargs: dict[str, Any], callback_url: str | None) -> None:
        """The target function for the background pipeline execution thread.

        Instantiates and runs the TranscriptionPipeline, handles exceptions,
        triggers the callback, and sets the completion event.

        :param pipeline_kwargs: Keyword arguments for TranscriptionPipeline.
        :type pipeline_kwargs: dict[str, Any]
        :param callback_url: Optional callback URL.
        :type callback_url: str | None
        """
        task_success: bool = False
        error_message: str = ""
        exit_code: int = DEFAULT_REST_FAILURE_EXIT_CODE

        try:
            self.log.info("Instantiating and starting pipeline run in background thread...")
            # Ensure the log file env var is set (should be by start_pipeline)
            if not os.environ.get("DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"):
                 raise RuntimeError("Pipeline log file environment variable not set.")

            pipeline: TranscriptionPipeline = TranscriptionPipeline(**pipeline_kwargs)
            pipeline.run() # This blocks until the pipeline is done
            self.log.info("Pipeline run finished successfully in background thread.")
            task_success = True
            exit_code = 0 # Success code
        except Exception as e:
            error_message = str(e)
            self.log.error(f"Error during pipeline execution: {error_message}", exc_info=self.debug)
            # Keep default failure exit code
        finally:
            # Clean up environment variable
            log_env_var = "DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"
            if log_env_var in os.environ:
                del os.environ[log_env_var]

            # Update state within lock
            with self._lock:
                self._pipeline_exit_code = exit_code
                self._pipeline_completed = True

            # Trigger callback outside the lock to avoid holding it during network I/O
            self._trigger_callback(task_success, error_message, callback_url)

            # Signal completion last
            self.log.debug("Signaling pipeline completion event.")
            self._pipeline_completion_event.set()

    def _fetch_pipeline_logs(self) -> str:
        """Read and return the contents of the pipeline's temporary log file.

        :return: Log content as a string, or an error message/empty message.
        :rtype: str
        """
        log_path = self.pipeline_log_file_path
        if not log_path:
            self.log.error("Log file path not set, cannot fetch logs.")
            return "Log file path not set."
        try:
            with open(log_path, 'r') as log_file:
                log_contents = log_file.read()
            # Optionally delete the log file after reading if desired
            # os.remove(log_path)
            # self.pipeline_log_file_path = None
            return log_contents if log_contents else "Log file is empty."
        except FileNotFoundError:
            msg = f"Error reading log file: File not found at {log_path}"
            self.log.error(msg)
            return msg
        except Exception as e:
            error_message = f"Error reading log file: {str(e)}"
            self.log.error(error_message, exc_info=self.debug)
            return error_message

    def _trigger_callback(self, task_success: bool, error_message: str, callback_url: str | None) -> None:
        """Construct the payload and send the completion/error callback.

        Fetches logs if the task was successful.

        :param task_success: Whether the pipeline task completed successfully.
        :type task_success: bool
        :param error_message: Error message if task failed, empty otherwise.
        :type error_message: str
        :param callback_url: The URL to send the callback to, or None.
        :type callback_url: str | None
        """
        if not callback_url:
            self.log.debug("No callback URL provided, skipping notification.")
            return

        payload: dict[str, str] = {"status": "success" if task_success else "error"}
        if error_message:
            payload["message"] = error_message
        elif task_success:
            # Fetch logs only on success
            logs: str = self._fetch_pipeline_logs()
            payload["message"] = logs
            if self.debug:
                self.log.debug(f"Pipeline logs being sent in callback:\n{logs[:500]}...") # Log snippet
        else:
             # Should not happen if error_message is empty and task failed, but handle defensively
             payload["message"] = "Pipeline failed with unspecified error."

        # Use the injected callback notifier function
        self.callback_notifier(callback_url, payload, self.log, self.debug)

    def wait_for_completion(self) -> None:
        """Block until the pipeline completion event is set."""
        self.log.debug("Waiting for pipeline completion event...")
        self._pipeline_completion_event.wait()
        self.log.debug("Pipeline completion event received.")

    def get_exit_code(self) -> int:
        """Get the final exit code of the pipeline run.

        Defaults to a failure code if the pipeline never completed successfully.

        :return: The exit code (0 for success, non-zero for failure).
        :rtype: int
        """
        with self._lock:
            # Return the stored exit code, which defaults to failure
            # and is updated upon task completion.
            return self._pipeline_exit_code


def create_app(
    pipeline_manager: PipelineLifecycleManager,
    validator: PipelineConfigValidator,
    api_key: str | None,
    logger: logging.Logger,
    debug: bool = False
) -> Flask:
    """Create and configure the Flask application instance.

    Registers the /status and /run endpoints, wiring them to the provided
    pipeline manager and validator.

    :param pipeline_manager: The initialized PipelineLifecycleManager instance.
    :type pipeline_manager: PipelineLifecycleManager
    :param validator: The initialized PipelineConfigValidator instance.
    :type validator: PipelineConfigValidator
    :param api_key: The API key the server expects for validation, or None.
    :type api_key: str | None
    :param logger: The logger instance to use for request logging.
    :type logger: logging.Logger
    :param debug: Enable Flask debug mode (primarily for tracebacks).
    :type debug: bool
    :return: The configured Flask application instance.
    :rtype: Flask
    """
    app: Flask = Flask(__name__)
    app.config["DEBUG"] = debug # Set Flask debug mode if requested

    @app.route("/status", methods=["GET"])
    def status() -> tuple[Response, int]:
        """Endpoint to check the status of the pipeline manager."""
        logger.debug("Status endpoint called")
        current_status: dict[str, Any] = pipeline_manager.get_status()
        return jsonify(current_status), 200

    @app.route("/run", methods=["POST"])
    def run_pipeline() -> tuple[Response, int]:
        """Endpoint to trigger the transcription pipeline execution."""
        logger.info("Run endpoint called")

        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data: dict[str, Any] | None = request.get_json()
        if data is None:
            logger.error("Failed to get JSON data from request")
            return jsonify({"error": "Failed to parse JSON data"}), 400

        logger.debug(f"Received run request: {data}")

        try:
            pipeline_kwargs, callback_url = validator.validate(data, api_key)
            started: bool = pipeline_manager.start_pipeline(pipeline_kwargs, callback_url)
            if started:
                logger.info("Pipeline accepted for processing.")
                return jsonify({"message": "Transcription pipeline accepted for processing."}), 202
            else:
                logger.warning("Pipeline start requested, but manager reported it's already running or failed to start.")
                return jsonify({"error": "Pipeline run has already been triggered or failed to start."}), 409
        except ValueError as ve:
            logger.error(f"Validation error processing run request: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            message: str = f"An unexpected error occurred processing the run request: {e}"
            logger.error(message, exc_info=debug)
            return jsonify({"error": message}), 500
    return app


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the REST interface server.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run the Transcription REST Interface.")
    parser.add_argument(
        "--host", type=str, default=DEFAULT_REST_HOST, help="Host to bind the server to, default: %(default)s"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_REST_PORT, help="Port to bind the server to, default: %(default)s"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key used to validate requests (also can be provided as TRANSCRIPTION_API_KEY environment variable)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug mode and verbose logging."
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments, initialize the interface, and run the server."""
    args: argparse.Namespace = parse_arguments()
    api_key: str | None = args.api_key or os.environ.get("TRANSCRIPTION_API_KEY")
    interface: TranscriptionRestInterface = TranscriptionRestInterface(debug=args.debug)
    interface.run_server(host=args.host, port=args.port, api_key=api_key)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
