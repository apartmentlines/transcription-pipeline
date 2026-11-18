"""Backend-neutral native timing and exact transcript text spans."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from transcription_pipeline.constants import (
    DEFAULT_COMMUNITY,
    DEFAULT_TERMINOLOGY,
    TIMESTAMP_TOLERANCE,
    VALID_LANGUAGES,
)
from .errors import TranscriptionError


@dataclass(frozen=True)
class TranscriptionContext:
    """Recording-specific terminology passed to an inference backend."""

    community: str = DEFAULT_COMMUNITY
    terminology: str = DEFAULT_TERMINOLOGY


@dataclass(frozen=True)
class AlignmentToken:
    """One native alignment item, including missing timestamps and confidence."""

    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class TimedWord:
    """A native alignment item mapped to original text without retiming it."""

    char_start: int
    char_end: int
    start: float | None
    end: float | None
    speaker: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class SourceSegment:
    """A backend segment or audio chunk, retaining its original boundaries."""

    char_start: int
    char_end: int
    start: float
    end: float
    language: str
    sentence: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    """Native observations shared by subtitles and application metadata."""

    text: str
    words: tuple[TimedWord, ...]
    segments: tuple[SourceSegment, ...]
    audio_duration: float
    detected_languages: tuple[str, ...]
    backend: str
    asr_model: str
    alignment_model: str
    diagnostics: dict[str, Any] = field(default_factory=dict[str, Any])

    def validate(self) -> None:
        """Check text coverage and native timing without repairing observations."""
        if not self.text.strip() or not self.words or not self.segments:
            raise TranscriptionError("Empty transcription", code="empty")
        if not math.isfinite(self.audio_duration) or self.audio_duration <= 0:
            raise TranscriptionError("Invalid audio duration", code="timing")
        for items in (self.words, self.segments):
            cursor, previous_start = 0, 0.0
            for item in items:
                if not cursor <= item.char_start < item.char_end <= len(self.text):
                    raise TranscriptionError(
                        "Invalid text span ordering", code="alignment"
                    )
                if self.text[cursor : item.char_start].strip():
                    raise TranscriptionError(
                        "Unowned transcript text", code="alignment"
                    )
                for bound in (item.start, item.end):
                    if bound is not None:
                        self.validate_alignment_interval(
                            bound, bound, self.audio_duration
                        )
                if item.start is not None:
                    if item.start < previous_start:
                        raise TranscriptionError(
                            "Reordered timestamps", code="alignment"
                        )
                    previous_start = item.start
                    if item.end is not None:
                        self.validate_alignment_interval(
                            item.start, item.end, self.audio_duration
                        )
                cursor = item.char_end
            if self.text[cursor:].strip():
                raise TranscriptionError("Unowned transcript tail", code="alignment")
        if not any(
            word.start is not None and word.end is not None and word.start < word.end
            for word in self.words
        ) and not any(
            segment.sentence and segment.start < segment.end
            for segment in self.segments
        ):
            raise TranscriptionError("No usable alignment timing", code="alignment")
        if not self.detected_languages or any(
            language not in VALID_LANGUAGES for language in self.detected_languages
        ):
            raise TranscriptionError("Unsupported language", code="language")
        for word in self.words:
            if word.score is not None and not (
                math.isfinite(word.score) and 0 <= word.score <= 1
            ):
                raise TranscriptionError("Invalid word confidence", code="alignment")

    @staticmethod
    def validate_alignment_interval(start: float, end: float, duration: float) -> None:
        """Accept finite native points as well as positive time intervals."""
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= end <= duration + TIMESTAMP_TOLERANCE
        ):
            raise TranscriptionError(
                f"Invalid alignment timestamp interval: start={start}, end={end}, duration={duration}",
                code="timing",
            )


class ResultBuilder:
    """Map ordered native tokens to original text; never guess matches or times."""

    text: str

    def __init__(self) -> None:
        self.text = ""
        self.words: list[TimedWord] = []
        self.segments: list[SourceSegment] = []

    @staticmethod
    def canonical(text: str) -> str:
        """Ignore only case, Unicode compatibility forms, and punctuation."""
        return "".join(
            char
            for source_char in text
            for char in unicodedata.normalize("NFKC", source_char).casefold()
            if char.isalnum()
        )

    def append(
        self,
        text: str,
        tokens: list[AlignmentToken],
        start: float,
        end: float,
        language: str,
        sentence: bool = False,
    ) -> None:
        """Preserve a native segment and project its tokens onto exact source text.

        Token boundaries may split contractions or omit punctuation. The complete
        letter/digit stream must match in order; missing or changed words fail
        explicitly. Missing timestamps remain missing. A native sentence with no
        word alignment retains its text and segment timing for display fallback.
        """
        text = text.strip()
        if not text:
            return
        if not tokens and sentence:
            tokens = [AlignmentToken(text)]
        target = self.canonical(text)
        if not tokens or target != "".join(
            self.canonical(token.text) for token in tokens
        ):
            raise TranscriptionError("ASR/alignment text discrepancy", code="alignment")
        positions = [
            index for index, char in enumerate(text) for _ in self.canonical(char)
        ]
        if self.text:
            self.text += " "
        offset = len(self.text)
        self.text += text
        cursor, text_cursor = 0, 0
        for index, token in enumerate(tokens):
            length = len(self.canonical(token.text))
            cursor += length
            if index == len(tokens) - 1:
                stop = len(text)
            elif length:
                stop = positions[cursor - 1] + 1
                while (
                    stop < len(text)
                    and not text[stop].isalnum()
                    and not text[stop].isspace()
                ):
                    stop += 1
            else:
                stop = text.find(token.text, text_cursor) + len(token.text)
            if stop <= text_cursor:
                raise TranscriptionError("Unmapped alignment token", code="alignment")
            self.words.append(
                TimedWord(
                    offset + text_cursor,
                    offset + stop,
                    float(token.start) if token.start is not None else None,
                    float(token.end) if token.end is not None else None,
                    token.speaker,
                    float(token.score) if token.score is not None else None,
                )
            )
            text_cursor = stop
            while text_cursor < len(text) and text[text_cursor].isspace():
                text_cursor += 1
        self.segments.append(
            SourceSegment(
                offset, len(self.text), float(start), float(end), language, sentence
            )
        )

    def build(
        self,
        audio_duration: float,
        backend: str,
        asr_model: str,
        alignment_model: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        """Validate the assembled result once at the backend boundary."""
        result = TranscriptionResult(
            self.text,
            tuple(self.words),
            tuple(self.segments),
            float(audio_duration),
            tuple(dict.fromkeys(segment.language for segment in self.segments)),
            backend,
            asr_model,
            alignment_model,
            dict(diagnostics or {}),
        )
        result.validate()
        return result
