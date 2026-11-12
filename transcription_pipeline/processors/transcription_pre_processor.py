"""Validate downloaded WAV recordings before inference."""

from __future__ import annotations

from typing import override

from download_pipeline_processor.processors.base_pre_processor import (
    BasePreProcessor,
)
from download_pipeline_processor.file_data import (
    FileData,
)

from transcription_pipeline.audio_file_validator import AudioFileValidator


class TranscriptionPreProcessor(BasePreProcessor):
    audio_file_validator: AudioFileValidator

    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.audio_file_validator = AudioFileValidator(debug=debug)

    @override
    def pre_process(self, file_data: FileData) -> FileData:
        if file_data.local_path is None:
            raise ValueError("Downloaded audio path is missing")
        self.audio_file_validator.validate(file_data.local_path)
        return file_data
