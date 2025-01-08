import os
import json
import requests
from download_pipeline_processor.processors.base_post_processor import BasePostProcessor
from download_pipeline_processor.file_data import FileData
from download_pipeline_processor.error import TransientPipelineError
from transcription_pipeline.transcriber import TranscriptionError
from transcription_pipeline.utils import post_request

from ..constants import (
    TRANSCRIPTION_STATE_ACTIVE,
)


class TranscriptionPostProcessor(BasePostProcessor):
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.api_key = os.environ.get("TRANSCRIPTION_API_KEY")
        self.domain = os.environ.get("TRANSCRIPTION_DOMAIN")

    def is_transient_download_error(self, file_data: FileData) -> bool:
        if file_data.error.stage == "download":
            error = file_data.error.error
            if isinstance(error, requests.exceptions.HTTPError):
                status_code = error.response.status_code
                return status_code == 429 or (500 <= status_code < 600)
            return isinstance(
                error,
                (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError,
                    requests.exceptions.SSLError,
                ),
            )
        return False

    def is_transient_processing_error(self, file_data: FileData) -> bool:
        if file_data.error.stage == "process":
            error = file_data.error.error
            return not isinstance(error, (TranscriptionError,))
        return False

    def post_process(self, result: dict, file_data: FileData) -> None:
        url = self.build_update_url()
        result_state = self.determine_result_state(result, file_data)
        data = self.format_request_payload(result_state)
        try:
            response = post_request(url, data)
            self.handle_response(response, result)
        except Exception as e:
            self.log.error(f"Failed to post-process result for ID {file_data.id}: {e}")

    def determine_result_state(self, result: dict, file_data: FileData) -> dict:
        if file_data.has_error:
            self.log.warning(
                f"Processing failed in stage '{file_data.error.stage}' with error: {file_data.error.error}"
            )
            if self.is_transient_download_error(
                file_data
            ) or self.is_transient_processing_error(file_data):
                self.log.debug(
                    f"Skipping post-processing for {file_data.name} due to transient error"
                )
                raise TransientPipelineError(file_data.error.error)
            return {
                "id": file_data.id,
                "success": False,
                "metadata": {
                    "error_stage": file_data.error.stage,
                    "error": str(file_data.error.error),
                },
            }
        if result["success"]:
            if not result["transcription"]:
                message = f"No transcription found for {file_data.name}"
                self.log.warning(message)
                return {
                    "id": file_data.id,
                    "success": False,
                    "metadata": {
                        "error_stage": "process",
                        "error": message,
                    },
                }
        return result

    def format_request_payload(self, result: dict) -> dict:
        metadata = result.get("metadata", {})
        data = {
            "api_key": self.api_key,
            "id": result["id"],
            "success": result["success"],
            "metadata": json.dumps(metadata),
        }
        if result["success"]:
            data["transcription_state"] = TRANSCRIPTION_STATE_ACTIVE
            data["transcription"] = result["transcription"]
        return data

    def build_update_url(self) -> str:
        return f"https://{self.domain}/al/transcriptions/update/operator-recording"

    def handle_response(self, response, result) -> None:
        resp_json = response.json()
        if resp_json.get("success"):
            self.log.info(
                f"Successfully updated transcription for ID {result.get('id')}"
            )
        else:
            self.log.error(
                f"Failed to update transcription: {resp_json.get('message')}"
            )
