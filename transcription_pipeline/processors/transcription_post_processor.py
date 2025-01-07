import os
import json
from download_pipeline_processor.processors.base_post_processor import BasePostProcessor
from download_pipeline_processor.file_data import FileData
from transcription_pipeline.utils import post_request

from ..constants import (
    TRANSCRIPTION_STATE_ACTIVE,
)


class TranscriptionPostProcessor(BasePostProcessor):
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.api_key = os.environ.get("TRANSCRIPTION_API_KEY")
        self.domain = os.environ.get("TRANSCRIPTION_DOMAIN")

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
            return {"id": file_data.id, "success": False}
        if result["success"]:
            if not result["transcription"]:
                self.log.warning(
                    f"No transcription found for {file_data.name}"
                )
                return {"id": file_data.id, "success": False}
        return result

    def format_request_payload(self, result: dict) -> dict:
        data = {
            "api_key": self.api_key,
            "id": result["id"],
            "success": result["success"],
            "transcription_state": TRANSCRIPTION_STATE_ACTIVE,
        }
        if result["success"]:
            data["transcription"] = result["transcription"]
            data["metadata"] = json.dumps(result["metadata"])
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
