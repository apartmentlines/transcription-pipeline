"""Framework workers sharing inference selection, normalization, and subtitles."""

from __future__ import annotations

from threading import Event
from typing import Any, override

from download_pipeline_processor.file_data import FileData
from download_pipeline_processor.processors.base_processor import BaseProcessor

from transcription_pipeline.constants import (
    DEFAULT_COMMUNITY,
)
from ..backends.base import TranscriptionBackend
from ..backends.factory import BackendFactory
from ..deployment import Deployment
from ..errors import FailureCleanup
from ..results import TranscriptionContext
from ..subtitles import SubtitleRenderer


class TranscriptionProcessor(BaseProcessor):
    """Load one backend per worker and reuse it for successive recordings."""

    renderer: SubtitleRenderer

    def __init__(
        self,
        deployment: Deployment,
        debug: bool = False,
        initialization_failed: Event | None = None,
    ) -> None:
        super().__init__(debug=debug)
        self.transcriber: TranscriptionBackend | None = None
        self.initialization_error: Exception | None = None
        self.renderer = SubtitleRenderer()
        try:
            self.transcriber = BackendFactory.create(deployment, debug)
        except Exception as exc:
            FailureCleanup.release_frames(exc)
            self.initialization_error = exc
            if initialization_failed is not None:
                initialization_failed.set()
            self.log.error(f"Worker model initialization failed: {exc}")

    @classmethod
    def bind(
        cls, deployment: Deployment, initialization_failed: Event | None = None
    ) -> type[BaseProcessor]:
        """Adapt immutable configuration to the framework's debug-only constructor."""

        class ConfiguredTranscriptionProcessor(cls):
            """A worker class bound to one pipeline's resolved deployment."""

            def __init__(self, debug: bool = False) -> None:
                super().__init__(
                    deployment=deployment,
                    debug=debug,
                    initialization_failed=initialization_failed,
                )

        return ConfiguredTranscriptionProcessor

    @override
    def process(self, file_data: FileData) -> dict[str, Any]:
        """Return the existing external result contract without temporary files."""
        if self.initialization_error is not None:
            raise self.initialization_error
        if self.transcriber is None:
            raise RuntimeError("Worker has no initialized transcription backend")
        if file_data.local_path is None:
            raise ValueError("Downloaded audio path is missing")
        try:
            result = self.transcriber.transcribe(
                file_data.local_path,
                TranscriptionContext(
                    community=file_data.record_name or DEFAULT_COMMUNITY
                ),
            )
        except Exception as exc:
            FailureCleanup.release_frames(exc)
            raise
        cues = self.renderer.cues(result)
        return {
            "id": file_data.id,
            "success": True,
            "transcription": self.renderer.serialize(cues),
            "metadata": result.metadata(len(cues)),
        }
