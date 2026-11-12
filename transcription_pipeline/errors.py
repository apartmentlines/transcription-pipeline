"""Backend-independent failure classification for pipeline post-processing."""

from __future__ import annotations

import traceback


class ConfigurationError(ValueError):
    """An installation or invocation cannot satisfy its configuration."""


class TranscriptionError(Exception):
    """A recording failed inference or output validation.

    :param exception: Original failure, without recording contents.
    :param code: Machine-readable diagnostic category.
    :param transient: Whether retrying the recording may succeed.
    """

    original_error: str
    code: str
    transient: bool | None

    def __init__(
        self,
        exception: Exception | str,
        code: str = "transcription",
        transient: bool | None = None,
    ) -> None:
        self.original_error = str(exception)
        self.code = code
        self.transient = transient
        super().__init__(f"Transcription error [{code}]: {self.original_error}")

    def is_transient_error(self) -> bool:
        """Return whether the external service should leave the job retryable."""
        if self.transient is not None:
            return self.transient
        message = self.original_error.lower()
        return any(
            phrase in message for phrase in ("cuda", "cublas", "cudnn", "out of memory")
        )


class FailureCleanup:
    """Release inference locals retained by traceback chains across queued jobs."""

    @staticmethod
    def release_frames(exception: BaseException) -> None:
        """Keep exception identity and stack locations while releasing tensor references."""
        current: BaseException | None = exception
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            traceback.clear_frames(current.__traceback__)
            current = current.__cause__ or current.__context__
