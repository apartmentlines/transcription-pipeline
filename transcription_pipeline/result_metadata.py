"""Application metadata derived from native results, independent of subtitles."""

from __future__ import annotations

from typing import Any

from .constants import (
    SOURCE_SPEECH_PAUSE_SECONDS,
    SUSPICIOUS_GAP_SECONDS,
    SUSPICIOUS_WORD_SECONDS,
)

from .results import TranscriptionResult


class ResultMetadata:
    """Keep legacy metadata fields without changing model segments or word timing."""

    result: TranscriptionResult

    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result

    def diagnostics(self) -> dict[str, Any]:
        """Count missing and point observations without treating confidence as timing."""
        words = self.result.words
        missing = sum(
            len(self.result.text[word.char_start : word.char_end].split())
            for word in words
            if word.start is None or word.end is None
        )
        long_words = sum(
            1
            for word in words
            if word.start is not None
            and word.end is not None
            and word.end - word.start > SUSPICIOUS_WORD_SECONDS
            and len(self.result.text[word.char_start : word.char_end].split()) == 1
        )
        clamped = sum(
            len(chunk.get("alignment_endpoint_clamps", []))
            for chunk in self.result.diagnostics.get("chunks", [])
        )
        return {
            **self.result.diagnostics,
            "alignment_warnings": missing + long_words + clamped,
            "suspicious_word_duration_count": long_words,
            "alignment_endpoint_clamp_count": clamped,
            "unaligned_word_count": missing,
            "zero_duration_word_count": sum(
                word.start is not None and word.start == word.end for word in words
            ),
            "last_aligned_word_end": max(
                (word.end for word in words if word.end is not None), default=None
            ),
            "suspicious_gap_count": sum(
                left.end is not None
                and right.start is not None
                and right.start - left.end > SUSPICIOUS_GAP_SECONDS
                for left, right in zip(words, words[1:])
            ),
        }

    def speech_intervals(self) -> list[tuple[float, float]]:
        """Use native sentence ranges, or estimate speech spans from Qwen words."""
        if self.result.backend != "qwen3":
            return [(segment.start, segment.end) for segment in self.result.segments]
        intervals: list[tuple[float, float]] = []
        for word in self.result.words:
            if word.start is None or word.end is None:
                continue
            if (
                intervals
                and word.start - intervals[-1][1] < SOURCE_SPEECH_PAUSE_SECONDS
            ):
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], word.end))
            else:
                intervals.append((word.start, word.end))
        return intervals

    def export(self, subtitle_cue_count: int) -> dict[str, Any]:
        """Map native results to the existing recursively merged application contract."""
        result = self.result
        intervals = self.speech_intervals()
        speaking_duration, interval_end = 0.0, 0.0
        for start, end in intervals:
            speaking_duration += max(0.0, end - max(start, interval_end))
            interval_end = max(interval_end, end)
        scores = [word.score for word in result.words if word.score is not None]
        diagnostics = self.diagnostics()
        missing = diagnostics["unaligned_word_count"]
        return {
            "language": result.detected_languages[0],
            "detected_languages": ",".join(result.detected_languages),
            "backend": result.backend,
            "asr_model": result.asr_model,
            "alignment_model": result.alignment_model,
            "segments": len(intervals),
            "total_words": len(result.text.split()),
            "total_duration": max((end for _, end in intervals), default=0.0),
            "audio_duration": result.audio_duration,
            "speaking_duration": speaking_duration,
            "subtitle_cue_count": subtitle_cue_count,
            "average_word_confidence": (
                round(sum(scores) / len(scores), 4) if scores else None
            ),
            "unaligned_word_count": missing,
            "alignment_warnings": diagnostics["alignment_warnings"],
        }
