from download_pipeline_processor.processors.base_processor import BaseProcessor
from download_pipeline_processor.file_data import FileData
from whisperx.utils import get_writer
from ..transcriber import Transcriber


class TranscriptionProcessor(BaseProcessor):
    """Processor for handling audio file transcription tasks.

    Integrates with the pipeline framework to process audio files through
    the Transcriber class, handling the transcription workflow and error reporting.
    """

    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.transcriber = Transcriber(debug=debug)

    def _format_transcription(self, result: dict) -> str:
        """Format transcription result into SRT string using WhisperX writer.

        :param result: Dictionary containing transcription results
        :return: Formatted SRT string
        """
        writer = get_writer("srt", None)
        return writer(result, "dummy")

    def process(self, file_data: FileData) -> dict:
        """Process an audio file for transcription.

        :param file_data: FileData object containing the audio file information
        :return: Dictionary containing transcription results or error information
        """
        self.log.info(f"Transcribing {file_data.name}, {file_data.record_name}")
        try:
            result = self.transcriber.transcribe(file_data.path)
            self.log.debug(f"Transcription successful for {file_data.id}")
            self.log.debug(f"Language detected: {result.get('language')}")
            self.log.debug(f"Number of segments: {len(result.get('segments', []))}")
            srt_content = self._format_transcription(result)
            return {
                "id": file_data.id,
                "success": True,
                "transcription": srt_content,
                "metadata": {
                    "language": result.get("language"),
                    "segments": len(result.get("segments", [])),
                },
            }
        except Exception as e:
            self.log.error(f"Transcription failed: {str(e)}")
            self.log.debug(f"Full error details: {repr(e)}")
            return {"id": file_data.id, "success": False, "error": str(e)}
