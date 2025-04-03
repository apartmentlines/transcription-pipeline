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
        self.app: Flask = Flask(__name__)
        self._lock: threading.Lock = threading.Lock()
        self._pipeline_started: bool = False
        self._pipeline_completion_event: threading.Event = threading.Event()
        self._pipeline_exit_code: int = DEFAULT_REST_FAILURE_EXIT_CODE
        self.pipeline_log_file_path: str = ""
        self._set_api_key()
        self._register_routes()

    def setup_pipeline_logging(self) -> None:
        """Set up file logging for the pioeline.
        """
        # Set up logging to a temporary file for the pipeline.
        log_file = tempfile.NamedTemporaryFile(
            prefix="transcription_rest_interface_",
            suffix=".log",
            delete=False,
            mode="w"
        )
        self.pipeline_log_file_path = log_file.name
        log_file.close()
        os.environ["DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE"] = self.pipeline_log_file_path
        self.log.info(f"Logging transcription operations to temporary file: {self.pipeline_log_file_path}")

    def _register_routes(self) -> None:
        """Register Flask URL rules."""
        self.app.add_url_rule("/status", view_func=self.status, methods=["GET"])
        self.app.add_url_rule("/run", view_func=self.run_pipeline, methods=["POST"])

    def _send_callback(self, url: str, payload: dict[str, Any]) -> None:
        """Send a POST request to the callback URL.

        Logs errors but does not raise exceptions.

        :param url: The URL to send the callback to.
        :type url: str
        :param payload: The JSON payload to send.
        :type payload: dict[str, Any]
        :raises requests.exceptions.RequestException: If the callback request fails.
        """
        try:
            self.log.info(f"Sending callback to {url} with payload: {payload}")
            response: requests.Response = utils.post_request(url, payload)
            self.log.info(f"Callback to {url} successful (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            self.log.error(f"Failed to send callback to {url}: {e}", exc_info=self.debug)
        except Exception as e:
            self.log.error(f"Unexpected error sending callback to {url}: {e}", exc_info=self.debug)

    def _validate_positive_int(self, value: str | int | None, name: str) -> int | None:
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
            raise ValueError(f"Invalid value for {name}: {e}") from e

    def _parse_and_validate_run_request(self, data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Parse and validate the JSON payload from the /run request.

        :param data: The JSON data received from the request.
        :type data: dict[str, Any]
        :return: A tuple containing the validated pipeline keyword arguments
                 and the callback URL (if provided).
        :rtype: tuple[dict[str, Any], str | None]
        :raises ValueError: If required parameters are missing or validation fails.
        """
        api_key: str | None = data.get("api_key")
        domain: str | None = data.get("domain")
        debug: bool = self.debug or data.get("debug", False)
        simulate_downloads: bool = data.get("simulate_downloads", False)
        callback_url: str | None = data.get("callback_url")

        if debug:
            self.log.info("Debug logging enabled via API request.")
        if not api_key or not domain:
            raise ValueError("api_key and domain are required parameters.")
        if not self._validate_api_key(api_key):
            raise ValueError("Invalid API key.")

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
            kwargs["download_cache"] = Path(download_cache_str)

        return kwargs, callback_url

    def status(self) -> tuple[Response, int]:
        """Check the status of the REST interface.

        :return: A JSON response indicating the service is running.
        :rtype: tuple[Response, int]
        """
        self.log.debug("Status endpoint called")
        return jsonify({"status": "ok", "started": self._pipeline_started}), 200

    def run_pipeline(self) -> tuple[Response, int]:
        """Trigger the transcription pipeline execution.

        Expects a JSON payload with configuration options (see
        `_parse_and_validate_run_request`). Runs the pipeline in a
        background thread.

        :return: A JSON response indicating success (202 Accepted), conflict (409),
                 bad request (400), or internal server error (500).
        :rtype: tuple[Response, int]
        :raises ValueError: If validation of the request payload fails.
        """
        self.log.info("Run endpoint called")
        with self._lock:
            if self._pipeline_started:
                self.log.warning("Pipeline run already triggered. Rejecting request.")
                return jsonify({"error": "Pipeline run has already been triggered for this instance."}), 409
        if not request.is_json:
            self.log.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data: dict[str, Any] | None = request.get_json()
        if data is None:
             self.log.error("Failed to get JSON data from request")
             return jsonify({"error": "Failed to parse JSON data"}), 400

        self.log.debug(f"Received run request: {data}")
        try:
            with self._lock:
                if self._pipeline_started:
                    self.log.warning("Pipeline run already triggered (race condition?). Rejecting request.")
                    return jsonify({"error": "Pipeline run has already been triggered for this instance."}), 409
                self._pipeline_started = True

            pipeline_kwargs, callback_url = self._parse_and_validate_run_request(data)

            thread: threading.Thread = threading.Thread(
                target=self._run_pipeline_task,
                args=(pipeline_kwargs, callback_url)
            )
            thread.start()
            self.log.info("Pipeline execution thread started.")
            return jsonify({"message": "Transcription pipeline accepted for processing."}), 202
        except ValueError as ve:
            self.log.error(f"Validation error processing run request: {ve}")
            with self._lock:
                self._pipeline_started = False
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            message: str = f"An unexpected error occurred processing the run request: {e}"
            self.log.error(message, exc_info=self.debug)
            with self._lock:
                 self._pipeline_started = False
                 self._pipeline_exit_code = DEFAULT_REST_FAILURE_EXIT_CODE
            self._pipeline_completion_event.set()
            return jsonify({"error": message}), 500

    def _run_pipeline_task(self, pipeline_kwargs: dict[str, Any], cb_url: str | None) -> None:
        """Run the pipeline in a background thread, handle callback, and signal completion.

        This method is intended to be the target of the pipeline execution thread. It
        handles exceptions during the pipeline run itself and ensures the completion
        event is always set.

        :param pipeline_kwargs: Dictionary of keyword arguments for TranscriptionPipeline.
        :type pipeline_kwargs: dict[str, Any]
        :param cb_url: Optional callback URL to notify upon completion.
        :type cb_url: str | None
        """
        task_success: bool = False
        error_message: str = ""
        try:
            self.log.info("Instantiating and starting pipeline run in background thread...")
            self.setup_pipeline_logging()
            pipeline: TranscriptionPipeline = TranscriptionPipeline(**pipeline_kwargs)
            pipeline.run()
            self.log.info("Pipeline run finished successfully in background thread.")
            task_success = True
            with self._lock:
                self._pipeline_exit_code = 0
        except Exception as e:
            error_message = str(e)
            self.log.error(f"Error during pipeline execution: {error_message}", exc_info=self.debug)
            with self._lock:
                self._pipeline_exit_code = DEFAULT_REST_FAILURE_EXIT_CODE
        finally:
            if cb_url:
                payload: dict[str, str] = {"status": "success" if task_success else "error"}
                if error_message:
                    payload["message"] = error_message
                else:
                    logs: str = self.fetch_pipeline_logs()
                    payload["message"] = logs
                    if self.debug:
                        self.log.debug("Pipeline logs")
                        print(logs)
                self._send_callback(cb_url, payload)
            self.log.debug("Signaling pipeline completion event.")
            self._pipeline_completion_event.set()

    def _set_api_key(self, api_key: str | None = None) -> None:
        """Set the API key for the REST interface.

        :param api_key: The API key to set.
        :type api_key: str | None
        """
        self.api_key: str | None = api_key

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate the API key against the key for the REST interface.

        :param api_key: The API key to validate.
        :type api_key: str
        """
        return self.api_key is None or self.api_key == api_key

    def fetch_pipeline_logs(self) -> str:
        """Fetch the contents of the pipeline log file.

        :return: The contents of the log file as a string, or an error message if the file cannot be read.
        :rtype: str
        """
        try:
            with open(self.pipeline_log_file_path, 'r') as log_file:
                log_contents = log_file.read()
            return log_contents if log_contents else "Log file is empty."
        except Exception as e:
            error_message = f"Error reading log file: {str(e)}"
            self.log.error(error_message, exc_info=self.debug)
            return error_message

    def run_server(self, host: str, port: int, api_key: str | None) -> None:
        """Run the Flask development server and wait for pipeline completion.

        :param host: Host to bind the server to.
        :type host: str
        :param port: Port to bind the server to.
        :type port: int
        """
        self._set_api_key(api_key)
        flask_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "debug": self.debug,
            "use_reloader": False
        }
        flask_thread: threading.Thread = threading.Thread(
            target=lambda: self.app.run(**flask_kwargs),
            daemon=True
        )
        flask_thread.start()
        self.log.info(f"Flask server started in background thread on http://{host}:{port}")
        self.log.info("Waiting for /run request and pipeline completion...")
        self._pipeline_completion_event.wait()
        self.log.info(f"Pipeline completion signal received. Exiting with code {self._pipeline_exit_code}.")
        sys.exit(self._pipeline_exit_code)


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


def stop_pod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        subprocess.run(["runpodctl", "stop", "pod", pod_id])


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
    finally:
        stop_pod()
