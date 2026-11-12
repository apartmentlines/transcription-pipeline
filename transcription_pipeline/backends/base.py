"""Contract implemented by each worker-owned inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..results import TranscriptionContext, TranscriptionResult


class TranscriptionBackend(Protocol):
    """Reusable inference models producing data without writing output files."""

    def transcribe(
        self, input_file: str | Path, context: TranscriptionContext
    ) -> TranscriptionResult:
        """Transcribe and align a complete recording."""
        ...
