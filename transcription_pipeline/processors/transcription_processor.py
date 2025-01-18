from download_pipeline_processor.processors.base_processor import BaseProcessor
from download_pipeline_processor.file_data import FileData
from whisperx.utils import get_writer
from ..transcriber import Transcriber
import tempfile
import os
from transcription_pipeline.constants import (
    INITIAL_PROMPT,
)


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
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".srt", delete=True) as tmp:
            options = {
                "max_line_width": None,
                "max_line_count": 1,
                "highlight_words": False,
            }
            writer = get_writer("srt", os.path.dirname(tmp.name))
            writer(
                result,
                os.path.basename(tmp.name),  # pyright: ignore[reportArgumentType]
                options,
            )
            tmp.seek(0)
            return tmp.read()

    def process(self, file_data: FileData) -> dict:
        """Process an audio file for transcription.

        :param file_data: FileData object containing the audio file information
        :return: Dictionary containing transcription results or error information
        """
        self.log.info(f"Transcribing {file_data.name}, {file_data.record_name}")
        try:
            initial_prompt = INITIAL_PROMPT % file_data.record_name
            result = self.transcriber.transcribe(file_data.local_path, initial_prompt)
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
                    "total_words": result.get("total_words"),
                    "total_duration": result.get("total_duration"),
                    "speaking_duration": result.get("speaking_duration"),
                    "average_word_confidence": result.get("average_word_confidence"),
                },
            }
        except Exception as e:
            self.log.error(f"Transcription failed: {str(e)}")
            self.log.debug(f"Full error details: {repr(e)}")
            raise
