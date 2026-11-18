"""Punkt-assisted cue grouping and lossless pysubs2 serialization."""

from __future__ import annotations

import math
import logging
import re
import textwrap
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from transcription_pipeline.constants import (
    LANGUAGE_NAMES,
    MILLISECONDS_PER_SECOND,
    SUBTITLE_MAX_CHARACTERS,
    SUBTITLE_MAX_LINES,
    SUBTITLE_PAUSE_BREAK,
    SUBTITLE_RENDER_SPEAKERS,
    SUBTITLE_SHORT_CUE_MILLISECONDS,
    SUBTITLE_TOKENIZER_CACHE_SIZE,
    TIMESTAMP_TOLERANCE,
)
from ..deployment import ManifestStore
from ..errors import ConfigurationError, TranscriptionError
from ..results import SourceSegment, TimedWord, TranscriptionResult


@dataclass(frozen=True)
class SubtitleConfig:
    """Line wrapping and pause boundaries; cue length never overrides a turn."""

    max_characters: int = SUBTITLE_MAX_CHARACTERS
    max_lines: int = SUBTITLE_MAX_LINES
    pause_break: float = SUBTITLE_PAUSE_BREAK
    render_speakers: bool = SUBTITLE_RENDER_SPEAKERS

    def __post_init__(self) -> None:
        if self.max_lines <= 0 or self.max_characters <= 0:
            raise ValueError("Invalid subtitle line wrapping targets")
        if not 0 < self.pause_break < math.inf:
            raise ValueError("Invalid subtitle pause target")


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
            endings.add(segment.char_end)
            if segment.sentence:
                continue
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
        group: list[TimedWord],
        word: TimedWord,
        sentence_ends: set[int],
        native_sentence: bool,
    ) -> bool:
        """Honor supplied speakers; otherwise preserve sentences and source boundaries.

        Unlabelled sentences remain unlabelled. Neither punctuation nor pauses
        are used to assign speaker identities or assume alternating turns.
        """
        previous = group[-1]
        if word.speaker != previous.speaker:
            return True
        if word.speaker is not None:
            return False
        return previous.char_end in sentence_ends or (
            not native_sentence
            and word.start is not None
            and previous.end is not None
            and word.start - previous.end >= self.config.pause_break
        )

    def cues(self, result: TranscriptionResult) -> tuple[Cue, ...]:
        """Construct lossless, nonoverlapping cues and validate after rounding."""
        result.validate()
        sentence_ends = self._sentence_ends(result)
        groups: list[list[TimedWord]] = []
        group: list[TimedWord] = []
        segments = iter(result.segments)
        segment = next(segments)
        sources: dict[int, SourceSegment] = {}
        for word in result.words:
            while word.char_start >= segment.char_end:
                segment = next(segments)
            sources[word.char_start] = segment
            if group and self._should_break(
                group, word, sentence_ends, segment.sentence
            ):
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
        for index, words in enumerate(groups):
            text = self._whitespace(
                result.text[words[0].char_start : words[-1].char_end]
            )
            if self.config.render_speakers and words[0].speaker:
                text = f"[{words[0].speaker}] {text}"
            first, last = words[0], words[-1]
            first_source, last_source = (
                sources[first.char_start],
                sources[last.char_start],
            )
            start_time = first.start
            if first_source.sentence and first.char_start == first_source.char_start:
                start_time = first_source.start
            if start_time is None:
                previous_end = groups[index - 1][-1].end if index else None
                start_time = (
                    max(first_source.start, previous_end)
                    if previous_end is not None
                    else first_source.start
                )
            end_time = last.end
            if last_source.sentence and last.char_end == last_source.char_end:
                end_time = last_source.end
            if end_time is None:
                following_start = (
                    groups[index + 1][0].start if index + 1 < len(groups) else None
                )
                end_time = (
                    min(last_source.end, following_start)
                    if following_start is not None
                    else last_source.end
                )
            end_time = max(
                [end_time] + [word.end for word in words if word.end is not None]
            )
            start = round(start_time * MILLISECONDS_PER_SECOND)
            end = min(round(end_time * MILLISECONDS_PER_SECOND), maximum)
            cues.append(Cue(start, end, text))
        cues = self._fit_cue_times(cues, maximum)
        short = sum(
            cue.end - cue.start < SUBTITLE_SHORT_CUE_MILLISECONDS for cue in cues
        )
        if short:
            logging.getLogger(__name__).warning(
                "%d subtitle cues are shorter than %d ms; review source alignment",
                short,
                SUBTITLE_SHORT_CUE_MILLISECONDS,
            )
        for index, cue in enumerate(cues):
            if not 0 <= cue.start < cue.end <= maximum:
                raise TranscriptionError(
                    "Cue collapses at millisecond precision", code="formatting"
                )
            if index and cue.start < cues[index - 1].end:
                raise TranscriptionError("Overlapping subtitle cues", code="formatting")
        return tuple(replace(cue, text=self._wrap(cue.text)) for cue in cues)

    @staticmethod
    def _fit_cue_times(cues: list[Cue], maximum: int) -> list[Cue]:
        """Keep every boundary while fitting positive, nonoverlapping display times.

        A collapsed cue gets up to one second of adjacent free time. If none
        exists, reserve a millisecond without merging its text into another
        sentence or speaker. Overlapping ends are clipped at the next cue's
        start. These display adjustments never change native word timestamps.
        """
        if maximum < len(cues):
            raise TranscriptionError(
                "Insufficient time for subtitle cues", code="formatting"
            )
        adjusted = []
        for index, cue in enumerate(cues):
            start, end = cue.start, cue.end
            if end <= start:
                previous_end = cues[index - 1].end if index else 0
                following_start = (
                    cues[index + 1].start if index + 1 < len(cues) else maximum
                )
                if following_start > start:
                    end = min(following_start, start + MILLISECONDS_PER_SECOND)
                elif previous_end < start:
                    start = max(previous_end, start - MILLISECONDS_PER_SECOND)
            earliest = adjusted[-1].start + 1 if adjusted else 0
            latest = maximum - (len(cues) - index)
            adjusted.append(
                replace(cue, start=max(earliest, min(start, latest)), end=end)
            )
        return [
            replace(
                cue,
                end=min(
                    max(cue.end, cue.start + 1),
                    adjusted[index + 1].start if index + 1 < len(adjusted) else maximum,
                ),
            )
            for index, cue in enumerate(adjusted)
        ]

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
