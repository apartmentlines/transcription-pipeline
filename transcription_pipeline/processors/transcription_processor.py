from download_pipeline_processor.processors.base_processor import BaseProcessor
from download_pipeline_processor.file_data import FileData


class TranscriptionProcessor(BaseProcessor):

    def process(self, file_data: FileData) -> dict:
        # TODO: Implement transcription logic.
        self.log.info(f"Transcribing {file_data.name}, {file_data.record_name}")
        result = {
            "id": file_data.id,
            "success": True,
            "transcription": "This is the transcribed text",
            "metadata": {
                "total_words": 5,
                "predicted_accuracy": 0.95,
            },
        }
        return result
