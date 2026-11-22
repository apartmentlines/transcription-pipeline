"""Framework workers sharing inference selection, normalization, and subtitles."""

from __future__ import annotations

from threading import Event
from time import monotonic
from typing import Any, override

from download_pipeline_processor.file_data import FileData
from download_pipeline_processor.processors.base_processor import BaseProcessor

from transcription_pipeline.result_metadata import ResultMetadata
from transcription_pipeline.constants import (
    DEFAULT_COMMUNITY,
)
from ..backends.base import TranscriptionBackend
from ..backends.factory import BackendFactory, BackendSession
from ..deployment import Deployment
from ..errors import FailureCleanup
from ..results import TranscriptionContext
from ..observability import RunMetrics
from ..subtitles import SubtitleRenderer


class TranscriptionProcessor(BaseProcessor):
    """Use the run's shared Qwen engine or a worker's reusable WhisperX backend."""

    renderer: SubtitleRenderer
    metrics: RunMetrics | None

    def __init__(
        self,
        deployment: Deployment,
        debug: bool = False,
        initialization_failed: Event | None = None,
        backend_session: BackendSession | None = None,
    ) -> None:
        super().__init__(debug=debug)
        self.transcriber: TranscriptionBackend | None = None
        self.initialization_error: Exception | None = None
        self.renderer = SubtitleRenderer()
        self.metrics = backend_session.metrics if backend_session is not None else None
        try:
            self.transcriber = (
                backend_session.worker()
                if backend_session is not None
                else BackendFactory.create(deployment, debug)
            )
        except Exception as exc:
            FailureCleanup.release_frames(exc)
            self.initialization_error = exc
            if initialization_failed is not None:
                initialization_failed.set()
            self.log.error(f"Worker model initialization failed: {exc}")

    @classmethod
    def bind(
        cls,
        deployment: Deployment,
        initialization_failed: Event | None = None,
        backend_session: BackendSession | None = None,
    ) -> type[BaseProcessor]:
        """Adapt immutable configuration to the framework's debug-only constructor."""

        class ConfiguredTranscriptionProcessor(cls):
            """A worker class bound to one pipeline's resolved deployment."""

            def __init__(self, debug: bool = False) -> None:
                super().__init__(
                    deployment=deployment,
                    debug=debug,
                    initialization_failed=initialization_failed,
                    backend_session=backend_session,
                )

        return ConfiguredTranscriptionProcessor

    @override
    def process(self, file_data: FileData) -> dict[str, Any]:
        """Measure completed recording calls without counting initialization as inference."""
        started = monotonic()
        success = False
        audio_seconds = 0.0
        if self.metrics is not None:
            self.metrics.processing_started()
        try:
            result = self._process(file_data)
            success = True
            audio_seconds = result["metadata"]["audio_duration"]
            return result
        finally:
            if self.metrics is not None:
                self.metrics.processing_finished(success, audio_seconds)
            self.log.info(
                "Recording id=%s success=%s audio_seconds=%.3f elapsed_seconds=%.3f",
                file_data.id,
                success,
                audio_seconds,
                monotonic() - started,
            )

    def _process(self, file_data: FileData) -> dict[str, Any]:
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
            "metadata": ResultMetadata(result).export(len(cues)),
        }
