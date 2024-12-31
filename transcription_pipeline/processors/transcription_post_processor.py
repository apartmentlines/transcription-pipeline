import os
from download_pipeline_processor.processors.base_post_processor import BasePostProcessor
from transcription_pipeline.utils import post_request


class TranscriptionPostProcessor(BasePostProcessor):
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.api_key = os.environ.get("TRANSCRIPTION_API_KEY")
        self.domain = os.environ.get("TRANSCRIPTION_DOMAIN")

    def post_process(self, result: dict) -> None:
        url = self.build_update_url()
        data = self.construct_post_data(result)
        try:
            response = post_request(url, data)
            self.handle_response(response)
        except Exception as e:
            self.log.error(
                f"Failed to post-process result for ID {result.get('id')}: {e}"
            )

    def construct_post_data(self, result: dict) -> dict:
        data = {
            "api_key": self.api_key,
            "id": result["id"],
            "success": result["success"],
        }
        if result["success"]:
            data["transcription"] = result["transcription"]
            data["metadata"] = result["metadata"]
        return data

    def build_update_url(self) -> str:
        return f"https://{self.domain}/al/transcriptions/update/operator-recording"

    def handle_response(self, response) -> None:
        resp_json = response.json()
        if resp_json.get("success"):
            self.log.info(
                f"Successfully updated transcription for ID {response.request.body.get('id')}"
            )
        else:
            self.log.error(
                f"Failed to update transcription: {resp_json.get('message')}"
            )
