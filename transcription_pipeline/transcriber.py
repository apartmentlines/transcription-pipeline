"""Standalone manifest-selected transcription with shared subtitle output."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, cast

from transcription_pipeline.constants import (
    DEFAULT_COMMUNITY,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_PRECISION,
    DEFAULT_TERMINOLOGY,
)
from .backends.factory import BackendFactory
from .deployment import ManifestStore
from .results import TranscriptionContext
from .subtitles import SubtitleRenderer


class TranscriptionCommand:
    """Run a local recording through the same backend and renderer as workers."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Define output and deployment options before loading any models."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("audio", type=Path, help="Input audio recording")
        parser.add_argument(
            "--backend",
            choices=("whisperx", "qwen3"),
            help="Assert the manifest backend",
        )
        parser.add_argument(
            "--model-manifest", type=Path, help="Override the deployment manifest path"
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("."),
            help="Output directory (default: %(default)s)",
        )
        parser.add_argument(
            "--output-format",
            choices=("srt", "vtt"),
            default="srt",
            help="Subtitle format (default: %(default)s)",
        )
        parser.add_argument(
            "--community",
            default=DEFAULT_COMMUNITY,
            help="Recording community (default: %(default)s)",
        )
        parser.add_argument(
            "--terminology",
            default=DEFAULT_TERMINOLOGY,
            help="Additional recording terminology",
        )
        parser.add_argument(
            "--device",
            default=DEFAULT_INFERENCE_DEVICE,
            help="CUDA device (default: %(default)s)",
        )
        parser.add_argument(
            "--precision",
            choices=("bfloat16", "float16"),
            default=DEFAULT_INFERENCE_PRECISION,
            help="Qwen precision (default: %(default)s)",
        )
        parser.add_argument(
            "--debug-output",
            action="store_true",
            help="Also save raw text and diagnostics beside subtitles",
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        return parser

    @staticmethod
    def run(args: argparse.Namespace) -> Path:
        """Validate the deployment, transcribe, and write UTF-8 artifacts."""
        deployment = ManifestStore.load(
            args.model_manifest,
            args.backend,
            device=args.device,
            precision=args.precision,
        )
        backend = BackendFactory.create(deployment, args.debug)
        result = backend.transcribe(
            args.audio, TranscriptionContext(args.community, args.terminology)
        )
        renderer = SubtitleRenderer()
        contents = renderer.render(
            result, cast(Literal["srt", "vtt"], args.output_format)
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            args.output_dir / args.audio.with_suffix(f".{args.output_format}").name
        )
        destination.write_text(contents, encoding="utf-8")
        if args.debug_output:
            destination.with_suffix(".txt").write_text(result.text, encoding="utf-8")
            destination.with_suffix(".diagnostics.json").write_text(
                json.dumps(result.diagnostics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return destination


def main() -> None:
    """Execute the CLI and return an actionable nonzero failure."""
    parser = TranscriptionCommand.parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        destination = TranscriptionCommand.run(args)
    except Exception as exc:
        parser.exit(1, f"Transcription failed: {exc}\n")
    else:
        logging.getLogger(__name__).info("Saved subtitles to %s", destination)


if __name__ == "__main__":
    main()
