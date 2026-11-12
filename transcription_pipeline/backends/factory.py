"""Construct only the selected backend with immutable worker configuration."""

from __future__ import annotations

from ..deployment import Deployment
from .base import TranscriptionBackend


class BackendFactory:
    """Keep optional inference imports out of application startup and help."""

    @staticmethod
    def create(deployment: Deployment, debug: bool = False) -> TranscriptionBackend:
        """Return one reusable backend owned by its calling worker."""
        if deployment.backend == "whisperx":
            from .whisperx import WhisperXTranscriber

            return WhisperXTranscriber(deployment, debug)
        from .qwen3 import Qwen3Transcriber

        return Qwen3Transcriber(deployment, debug)
