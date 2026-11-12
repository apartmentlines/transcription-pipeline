"""Canonical text spans, timing reconciliation, and external metadata."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Literal

from transcription_pipeline.constants import (
    ALIGNMENT_MIN_CHARACTER_COVERAGE,
    DEFAULT_COMMUNITY,
    DEFAULT_TERMINOLOGY,
    SOURCE_SPEECH_PAUSE_SECONDS,
    SUSPICIOUS_GAP_SECONDS,
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
class TimedWord:
    """One complete display token referencing authoritative text."""

    char_start: int
    char_end: int
    start: float
    end: float
    provenance: Literal["aligned", "inferred"] = "aligned"
    speaker: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class SourceSegment:
    """A source speech interval, independent of subtitle formatting."""

    char_start: int
    char_end: int
    start: float
    end: float
    language: str
    speaker: str | None = None


@dataclass(frozen=True)
class AlignmentToken:
    """Backend alignment text used only for matching, never for rendering."""

    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    score: float | None = None
    provenance: Literal["aligned", "inferred"] = "aligned"


@dataclass(frozen=True)
class TranscriptionResult:
    """Backend-neutral transcription and its source provenance."""

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
        """Reject empty, reordered, missing, or out-of-range transcript spans."""
        if not self.text.strip() or not self.words or not self.segments:
            raise TranscriptionError("Empty transcription", code="empty")
        if not math.isfinite(self.audio_duration) or self.audio_duration <= 0:
            raise TranscriptionError("Invalid audio duration", code="timing")
        cursor = 0
        previous_start = 0.0
        for word in self.words:
            if not cursor <= word.char_start < word.char_end <= len(self.text):
                raise TranscriptionError("Invalid text span ordering", code="alignment")
            if self.text[cursor : word.char_start].strip():
                raise TranscriptionError("Unowned transcript text", code="alignment")
            self.validate_interval(word.start, word.end, self.audio_duration)
            if word.start < previous_start:
                raise TranscriptionError("Reordered word timestamps", code="alignment")
            if word.score is not None and not (
                math.isfinite(word.score) and 0 <= word.score <= 1
            ):
                raise TranscriptionError("Invalid word confidence", code="alignment")
            previous_start = word.start
            cursor = word.char_end
        if self.text[cursor:].strip():
            raise TranscriptionError("Unowned transcript tail", code="alignment")
        previous_start = 0.0
        cursor = 0
        for segment in self.segments:
            if not cursor <= segment.char_start < segment.char_end <= len(self.text):
                raise TranscriptionError("Invalid source text spans", code="alignment")
            if self.text[cursor : segment.char_start].strip():
                raise TranscriptionError("Unowned source text", code="alignment")
            cursor = segment.char_end
            self.validate_interval(segment.start, segment.end, self.audio_duration)
            if segment.start < previous_start:
                raise TranscriptionError("Reordered source segments", code="alignment")
            previous_start = segment.start
        if self.text[cursor:].strip():
            raise TranscriptionError("Unowned source text tail", code="alignment")
        if not self.detected_languages or any(
            language not in VALID_LANGUAGES for language in self.detected_languages
        ):
            raise TranscriptionError("Unsupported language", code="language")

    @staticmethod
    def validate_interval(start: float, end: float, duration: float) -> None:
        """Validate a positive finite interval within the original audio."""
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start < end <= duration + TIMESTAMP_TOLERANCE
        ):
            raise TranscriptionError(
                "Invalid alignment timestamp interval", code="timing"
            )

    def metadata(self, subtitle_cue_count: int) -> dict[str, Any]:
        """Map results to the recursively merged application metadata contract.

        Languages are a scalar CSV value so retranscription replaces the whole
        value. Primary language is the first nonempty source segment's detection.
        """
        self.validate()
        speaking_duration = 0.0
        interval_end = 0.0
        for segment in self.segments:
            speaking_duration += max(
                0.0, segment.end - max(segment.start, interval_end)
            )
            interval_end = max(interval_end, segment.end)
        scores = [word.score for word in self.words if word.score is not None]
        return {
            "language": self.detected_languages[0],
            "detected_languages": ",".join(self.detected_languages),
            "backend": self.backend,
            "asr_model": self.asr_model,
            "alignment_model": self.alignment_model,
            "segments": len(self.segments),
            "total_words": len(self.words),
            "total_duration": max(segment.end for segment in self.segments),
            "audio_duration": self.audio_duration,
            "speaking_duration": speaking_duration,
            "subtitle_cue_count": subtitle_cue_count,
            "average_word_confidence": (
                round(sum(scores) / len(scores), 4) if scores else None
            ),
            "unaligned_word_count": sum(
                word.provenance == "inferred" for word in self.words
            ),
            "alignment_warnings": int(self.diagnostics.get("alignment_warnings", 0)),
        }


class TextReconciler:
    """Match alignment character streams to lossless display spans.

    Matching ignores punctuation and case, allowing contractions, apartment
    numbers, and abbreviations to use different alignment token boundaries.
    Unmatched display tokens receive bounded inferred times. A transcript with
    no reliable anchors or less than 70 percent character coverage fails.
    """

    @staticmethod
    def canonical(text: str) -> str:
        """Return case-folded letters and digits for alignment comparison."""
        return "".join(
            character
            for character in unicodedata.normalize("NFKC", text).casefold()
            if character.isalnum()
        )

    @classmethod
    def reconcile(
        cls,
        text: str,
        tokens: list[AlignmentToken],
        start: float,
        end: float,
        speaker: str | None = None,
    ) -> tuple[TimedWord, ...]:
        """Produce complete timed display spans within source bounds.

        :raises TranscriptionError: If text or timing cannot be reconciled safely.
        """
        TranscriptionResult.validate_interval(start, end, end)
        spans = list(re.finditer(r"(?:[^\w\s]+\s+)*\S+(?:\s+[^\w\s]+)*(?=\s|$)", text))
        if not spans:
            raise TranscriptionError("Empty source segment", code="empty")
        target = ""
        owners: list[int] = []
        for index, span in enumerate(spans):
            canonical = cls.canonical(span.group())
            target += canonical
            owners.extend([index] * len(canonical))
        aligned = ""
        token_owners: list[int] = []
        for index, token in enumerate(tokens):
            for bound in (token.start, token.end):
                if bound is not None and not (
                    math.isfinite(bound) and start <= bound <= end + TIMESTAMP_TOLERANCE
                ):
                    raise TranscriptionError(
                        "Invalid alignment timestamp", code="timing"
                    )
            if token.start is not None and token.end is not None:
                TranscriptionResult.validate_interval(token.start, token.end, end)
                if token.start < start:
                    raise TranscriptionError(
                        "Alignment precedes source interval", code="timing"
                    )
            canonical = cls.canonical(token.text)
            aligned += canonical
            token_owners.extend([index] * len(canonical))
        matches: list[set[int]] = [set() for _ in spans]
        matched = 0
        word_coverage = [0 for _ in spans]
        for block in SequenceMatcher(
            None, target, aligned, autojunk=False
        ).get_matching_blocks():
            matched += block.size
            for offset in range(block.size):
                owner = owners[block.a + offset]
                matches[owner].add(token_owners[block.b + offset])
                word_coverage[owner] += 1
        if (
            not target
            or not aligned
            or min(matched / len(target), matched / len(aligned))
            < ALIGNMENT_MIN_CHARACTER_COVERAGE
        ):
            raise TranscriptionError("ASR/alignment text discrepancy", code="alignment")
        words: list[TimedWord | None] = []
        for span, indices, coverage in zip(spans, matches, word_coverage):
            anchors = [
                tokens[index]
                for index in sorted(indices)
                if tokens[index].start is not None and tokens[index].end is not None
            ]
            if not anchors:
                words.append(None)
                continue
            starts = [
                float(token.start) for token in anchors if token.start is not None
            ]
            ends = [float(token.end) for token in anchors if token.end is not None]
            scores = [token.score for token in anchors if token.score is not None]
            words.append(
                TimedWord(
                    span.start(),
                    span.end(),
                    min(starts),
                    max(ends),
                    provenance=(
                        "aligned"
                        if coverage == len(cls.canonical(span.group()))
                        and all(token.provenance == "aligned" for token in anchors)
                        else "inferred"
                    ),
                    speaker=anchors[0].speaker or speaker,
                    score=sum(scores) / len(scores) if scores else None,
                )
            )
        if not any(word is not None for word in words):
            raise TranscriptionError(
                "Nonempty transcript has no timed alignment anchors", code="alignment"
            )
        cls._infer_missing(words, spans, start, end, speaker)
        complete = tuple(word for word in words if word is not None)
        for previous, current in zip(complete, complete[1:]):
            if current.start < previous.start:
                raise TranscriptionError(
                    "Reordered alignment anchors", code="alignment"
                )
        return complete

    @staticmethod
    def _infer_missing(
        words: list[TimedWord | None],
        spans: list[re.Match[str]],
        start: float,
        end: float,
        speaker: str | None,
    ) -> None:
        """Interpolate only within adjacent anchors or source segment bounds."""
        index = 0
        while index < len(words):
            if words[index] is not None:
                index += 1
                continue
            stop = index + 1
            while stop < len(words) and words[stop] is None:
                stop += 1
            previous = words[index - 1] if index else None
            following = words[stop] if stop < len(words) else None
            left = previous.end if previous else start
            right = following.start if following else end
            if right <= left:
                raise TranscriptionError(
                    "No interval for unaligned words", code="alignment"
                )
            step = (right - left) / (stop - index)
            for offset, word_index in enumerate(range(index, stop)):
                span = spans[word_index]
                words[word_index] = TimedWord(
                    span.start(),
                    span.end(),
                    left + step * offset,
                    left + step * (offset + 1),
                    "inferred",
                    speaker,
                )
            index = stop


class ResultBuilder:
    """Assemble source segments with explicit whitespace and absolute offsets."""

    text: str

    def __init__(self) -> None:
        self.text = ""
        self.words: list[TimedWord] = []
        self.segments: list[SourceSegment] = []
        self.languages: list[str] = []

    def append(
        self,
        text: str,
        tokens: list[AlignmentToken],
        start: float,
        end: float,
        language: str,
        speaker: str | None = None,
        derive_speech_bounds: bool = False,
    ) -> None:
        """Reconcile one source segment and preserve its original time origin."""
        text = text.strip()
        if not text:
            return
        if language not in VALID_LANGUAGES:
            raise TranscriptionError(
                f"Unsupported language: {language}", code="language"
            )
        words = TextReconciler.reconcile(text, tokens, start, end, speaker)
        if self.text:
            self.text += " "
        offset = len(self.text)
        self.text += text
        self.words.extend(
            replace(
                word,
                char_start=word.char_start + offset,
                char_end=word.char_end + offset,
            )
            for word in words
        )
        if derive_speech_bounds:
            groups: list[list[TimedWord]] = []
            group: list[TimedWord] = []
            for word in words:
                if group and word.start - group[-1].end >= SOURCE_SPEECH_PAUSE_SECONDS:
                    groups.append(group)
                    group = []
                group.append(word)
            if group:
                groups.append(group)
            self.segments.extend(
                SourceSegment(
                    offset + group[0].char_start,
                    offset + group[-1].char_end,
                    group[0].start,
                    max(word.end for word in group),
                    language,
                    speaker,
                )
                for group in groups
            )
        else:
            self.segments.append(
                SourceSegment(offset, len(self.text), start, end, language, speaker)
            )
        if language not in self.languages:
            self.languages.append(language)

    def build(
        self,
        audio_duration: float,
        backend: str,
        asr_model: str,
        alignment_model: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        """Return a validated immutable transcript with diagnostic counts."""
        details = dict(diagnostics or {})
        details["alignment_warnings"] = sum(
            word.provenance == "inferred" for word in self.words
        )
        details["unaligned_word_count"] = details["alignment_warnings"]
        details["last_aligned_word_end"] = max(
            (word.end for word in self.words if word.provenance == "aligned"),
            default=None,
        )
        details["suspicious_gap_count"] = sum(
            right.start - left.end > SUSPICIOUS_GAP_SECONDS
            for left, right in zip(self.words, self.words[1:])
        )
        result = TranscriptionResult(
            self.text,
            tuple(self.words),
            tuple(self.segments),
            audio_duration,
            tuple(self.languages),
            backend,
            asr_model,
            alignment_model,
            details,
        )
        result.validate()
        return result
