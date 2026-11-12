"""Punkt-assisted cue grouping and lossless pysubs2 serialization."""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from transcription_pipeline.constants import (
    LANGUAGE_NAMES,
    MILLISECONDS_PER_SECOND,
    SUBTITLE_MAX_CHARACTERS,
    SUBTITLE_MAX_DURATION,
    SUBTITLE_MAX_LINES,
    SUBTITLE_MIN_DURATION,
    SUBTITLE_PAUSE_BREAK,
    SUBTITLE_RENDER_SPEAKERS,
    SUBTITLE_TOKENIZER_CACHE_SIZE,
    TIMESTAMP_TOLERANCE,
)
from ..deployment import ManifestStore
from ..errors import ConfigurationError, TranscriptionError
from ..results import TimedWord, TranscriptionResult


@dataclass(frozen=True)
class SubtitleConfig:
    """Soft readability targets; complete text and valid timing take precedence."""

    max_characters: int = SUBTITLE_MAX_CHARACTERS
    max_lines: int = SUBTITLE_MAX_LINES
    min_duration: float = SUBTITLE_MIN_DURATION
    max_duration: float = SUBTITLE_MAX_DURATION
    pause_break: float = SUBTITLE_PAUSE_BREAK
    render_speakers: bool = SUBTITLE_RENDER_SPEAKERS

    def __post_init__(self) -> None:
        if (
            self.max_lines <= 0
            or self.max_characters <= 0
            or not 0 < self.min_duration <= self.max_duration
        ):
            raise ValueError("Invalid subtitle readability targets")
        if not math.isfinite(self.max_duration) or not 0 < self.pause_break < math.inf:
            raise ValueError("Invalid subtitle duration or pause target")


@dataclass(frozen=True)
class Cue:
    """An ordered plain-text cue with quantized millisecond timestamps."""

    start: int
    end: int
    text: str


class SubtitleRenderer:
    """Generate identical SRT/VTT for every backend and entry point."""

    config: SubtitleConfig

    def __init__(self, config: SubtitleConfig | None = None) -> None:
        self.config = config or SubtitleConfig()

    @staticmethod
    @lru_cache(maxsize=SUBTITLE_TOKENIZER_CACHE_SIZE)
    def _tokenizer(language: str, data_root: str):
        """Load pinned local Punkt parameters without network access."""
        from nltk import data
        from nltk.tokenize.punkt import PunktSentenceTokenizer, load_punkt_params

        name = LANGUAGE_NAMES[language]
        path = Path(data_root) / "tokenizers" / "punkt_tab" / name
        try:
            if data_root not in data.path:
                data.path.insert(0, data_root)
            pointer = data.find(f"tokenizers/punkt_tab/{name}/", paths=[data_root])
            return PunktSentenceTokenizer(load_punkt_params(pointer))
        except (OSError, ValueError, LookupError) as exc:
            raise ConfigurationError(
                f"Missing or invalid Punkt data: {path}; rerun seed-models"
            ) from exc

    def _sentence_ends(self, result: TranscriptionResult) -> set[int]:
        """Map each source language's sentence spans into the complete transcript."""
        endings: set[int] = set()
        for segment in result.segments:
            tokenizer = self._tokenizer(
                segment.language, str(ManifestStore.roots()["nltk_data"])
            )
            text = result.text[segment.char_start : segment.char_end]
            endings.update(
                segment.char_start + end for _, end in tokenizer.span_tokenize(text)
            )
        return endings

    @staticmethod
    def _whitespace(text: str) -> str:
        """Normalize only whitespace for one-line subtitle display."""
        return " ".join(text.split())

    def _should_break(
        self,
        result: TranscriptionResult,
        group: list[TimedWord],
        word: TimedWord,
        sentence_ends: set[int],
    ) -> bool:
        """Combine source sentence candidates, pauses, and soft layout limits."""
        first, previous = group[0], group[-1]
        duration = previous.end - first.start
        if (
            word.speaker != previous.speaker
            or word.start - previous.end >= self.config.pause_break
        ):
            return True
        if previous.char_end in sentence_ends and duration >= self.config.min_duration:
            return True
        text = self._whitespace(result.text[first.char_start : word.char_end])
        return duration >= self.config.min_duration and (
            len(text) > self.config.max_characters * self.config.max_lines
            or word.end - first.start > self.config.max_duration
        )

    def cues(self, result: TranscriptionResult) -> tuple[Cue, ...]:
        """Construct lossless, nonoverlapping cues and validate after rounding."""
        result.validate()
        sentence_ends = self._sentence_ends(result)
        groups: list[list[TimedWord]] = []
        group: list[TimedWord] = []
        for word in result.words:
            if group and self._should_break(result, group, word, sentence_ends):
                groups.append(group)
                group = []
            group.append(word)
        if group:
            groups.append(group)
        covered = " ".join(
            self._whitespace(result.text[words[0].char_start : words[-1].char_end])
            for words in groups
        )
        if covered != self._whitespace(result.text):
            raise TranscriptionError(
                "Subtitle text coverage mismatch", code="formatting"
            )
        cues: list[Cue] = []
        maximum = math.floor(
            result.audio_duration * MILLISECONDS_PER_SECOND + TIMESTAMP_TOLERANCE
        )
        for words in groups:
            text = self._whitespace(
                result.text[words[0].char_start : words[-1].char_end]
            )
            if self.config.render_speakers and words[0].speaker:
                text = f"[{words[0].speaker}] {text}"
            start = round(words[0].start * MILLISECONDS_PER_SECOND)
            end = min(
                round(max(word.end for word in words) * MILLISECONDS_PER_SECOND),
                maximum,
            )
            if cues and (start < cues[-1].end or end <= start):
                previous = cues[-1]
                cues[-1] = replace(
                    previous, end=max(previous.end, end), text=f"{previous.text} {text}"
                )
            else:
                if end <= start:
                    end = min(start + 1, maximum)
                cues.append(Cue(start, end, text))
        for index, cue in enumerate(cues):
            if not 0 <= cue.start < cue.end <= maximum:
                raise TranscriptionError(
                    "Cue collapses at millisecond precision", code="formatting"
                )
            if index and cue.start < cues[index - 1].end:
                raise TranscriptionError("Overlapping subtitle cues", code="formatting")
        return tuple(replace(cue, text=self._wrap(cue.text)) for cue in cues)

    def _wrap(self, text: str) -> str:
        """Use standard word wrapping while treating line-count targets as soft."""
        if self.config.max_lines == 1:
            return text
        lines = textwrap.wrap(
            text,
            width=self.config.max_characters,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if len(lines) > self.config.max_lines:
            lines = lines[: self.config.max_lines - 1] + [
                " ".join(lines[self.config.max_lines - 1 :])
            ]
        return "\n".join(lines)

    @staticmethod
    def serialize(cues: tuple[Cue, ...], output_format: str = "srt") -> str:
        """Delegate syntax to pysubs2 while shielding literal text from ASS parsing.

        Plain placeholders travel through the serializer and are replaced in a
        single pass. This preserves literal braces, markup, and backslash-N/H
        sequences that pysubs2 would otherwise interpret as styling or spacing.
        """
        import pysubs2

        if output_format not in ("srt", "vtt"):
            raise ConfigurationError("Supported subtitle formats: srt, vtt")
        if not cues:
            raise TranscriptionError("No subtitle cues", code="empty")
        subtitles = pysubs2.SSAFile()
        replacements: dict[str, str] = {}
        for index, cue in enumerate(cues):
            if not 0 <= cue.start < cue.end or not cue.text.strip():
                raise TranscriptionError("Invalid subtitle cue", code="formatting")
            if index and cue.start < cues[index - 1].end:
                raise TranscriptionError("Overlapping subtitle cues", code="formatting")
            marker = f"TRANSCRIPTIONCUE{index:08d}PLACEHOLDER"
            replacements[marker] = cue.text
            subtitles.events.append(
                pysubs2.SSAEvent(start=cue.start, end=cue.end, text=marker)
            )
        rendered = subtitles.to_string(
            output_format, keep_ssa_tags=True, apply_styles=False
        )
        return re.sub(
            r"TRANSCRIPTIONCUE\d{8}PLACEHOLDER",
            lambda match: replacements[match.group()],
            rendered,
        )

    def render(self, result: TranscriptionResult, output_format: str = "srt") -> str:
        """Return subtitle contents without intermediate files."""
        return self.serialize(self.cues(result), output_format)
