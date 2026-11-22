"""Construct only the selected backend with immutable worker configuration."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, cast, final

if TYPE_CHECKING:
    from .qwen3 import Qwen3Transcriber
    from .qwen3_batching import SharedQwenTranscriber

from ..deployment import Deployment
from ..observability import RunMetrics
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


@final
class BackendSession:
    """Own one Qwen runtime per pipeline run; WhisperX remains worker-owned."""

    def __init__(
        self,
        deployment: Deployment,
        debug: bool = False,
        metrics: RunMetrics | None = None,
    ) -> None:
        self.deployment = deployment
        self.debug = debug
        self.metrics = metrics
        self.lock = Lock()
        self.shared: SharedQwenTranscriber | None = None
        self.failure: Exception | None = None
        self.closed = False

    def worker(self) -> TranscriptionBackend:
        """Initialize shared models once even when worker threads start together."""
        if self.deployment.backend == "whisperx":
            backend = BackendFactory.create(self.deployment, self.debug)
            if self.metrics is not None:
                self.metrics.worker_ready()
            return backend
        from .qwen3_batching import SharedQwenTranscriber

        with self.lock:
            if self.closed:
                raise RuntimeError("Backend session is closed")
            if self.failure is not None:
                raise self.failure
            if self.shared is None:
                try:
                    backend = cast(
                        "Qwen3Transcriber",
                        BackendFactory.create(self.deployment, self.debug),
                    )
                    self.shared = SharedQwenTranscriber(backend)
                except Exception as exc:
                    self.failure = exc
                    raise
            if self.metrics is not None:
                self.metrics.worker_ready()
            return self.shared

    def close(self) -> None:
        """Finish the shared engine's work before releasing its model resources."""
        with self.lock:
            self.closed = True
            shared = self.shared
        if shared is not None:
            shared.close()
