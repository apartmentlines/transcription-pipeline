"""Standalone manifest-selected transcription with shared subtitle output."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from threading import local
from typing import Literal, cast

from transcription_pipeline.constants import (
    DEFAULT_COMMUNITY,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_PRECISION,
    DEFAULT_TERMINOLOGY,
)
from .backends.factory import BackendSession
from .deployment import ManifestStore
from .results import TranscriptionContext
from .result_metadata import ResultMetadata
from .subtitles import SubtitleRenderer
from .backends.base import TranscriptionBackend
from .errors import ConfigurationError, TranscriptionError
from .utils import positive_int


class TranscriptionCommand:
    """Run a local recording through the same backend and renderer as workers."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Define output and deployment options before loading any models."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "audio", type=Path, nargs="+", help="Input audio recordings"
        )
        parser.add_argument(
            "--processing-limit",
            type=positive_int,
            help="Concurrent recordings (environment or backend default)",
        )
        parser.add_argument(
            "--qwen-batch-size",
            type=positive_int,
            help="Maximum active Qwen chunks (environment or default: 16)",
        )
        parser.add_argument(
            "--qwen-chunk-seconds",
            type=positive_int,
            help="Qwen chunk duration, 1–90 seconds (environment or default: 30)",
        )
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
    def run(args: argparse.Namespace) -> list[Path]:
        """Validate the deployment, transcribe, and write UTF-8 artifacts."""
        deployment = ManifestStore.load(
            args.model_manifest,
            args.backend,
            device=args.device,
            precision=args.precision,
            qwen_batch_size=args.qwen_batch_size,
            qwen_chunk_seconds=args.qwen_chunk_seconds,
        )
        deployment = replace(
            deployment, worker_count=deployment.processing_limit(args.processing_limit)
        )
        names = [
            source.with_suffix(f".{args.output_format}").name for source in args.audio
        ]
        if len(set(names)) != len(names):
            raise ConfigurationError(
                "Input recordings would overwrite the same output file"
            )
        session = BackendSession(deployment, args.debug)
        workers = local()

        def process(source: Path) -> Path:
            if not hasattr(workers, "backend"):
                workers.backend = session.worker()
            return TranscriptionCommand._write(source, workers.backend, args)

        destinations = []
        failures = []
        try:
            with ThreadPoolExecutor(
                max_workers=min(deployment.worker_count, len(args.audio))
            ) as executor:
                futures = {
                    executor.submit(process, source): source for source in args.audio
                }
                for future in as_completed(futures):
                    try:
                        destinations.append(future.result())
                    except Exception as exc:
                        logging.getLogger(__name__).error(
                            "Transcription failed for %s: %s", futures[future], exc
                        )
                        failures.append(exc)
        finally:
            session.close()
        if failures:
            if len(args.audio) == 1:
                raise failures[0]
            raise TranscriptionError(
                f"{len(failures)} of {len(args.audio)} recordings failed",
                code="inference",
            )
        return sorted(destinations)

    @staticmethod
    def _write(
        source: Path, backend: TranscriptionBackend, args: argparse.Namespace
    ) -> Path:
        """Write each completed recording independently of failures in other jobs."""
        result = backend.transcribe(
            source, TranscriptionContext(args.community, args.terminology)
        )
        renderer = SubtitleRenderer()
        contents = renderer.render(
            result, cast(Literal["srt", "vtt"], args.output_format)
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            args.output_dir / source.with_suffix(f".{args.output_format}").name
        )
        destination.write_text(contents, encoding="utf-8")
        if args.debug_output:
            destination.with_suffix(".txt").write_text(result.text, encoding="utf-8")
            destination.with_suffix(".diagnostics.json").write_text(
                json.dumps(
                    ResultMetadata(result).diagnostics(), ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        return destination


def main() -> None:
    """Execute the CLI and return an actionable nonzero failure."""
    parser = TranscriptionCommand.parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        destinations = TranscriptionCommand.run(args)
    except Exception as exc:
        parser.exit(1, f"Transcription failed: {exc}\n")
    else:
        for destination in destinations:
            logging.getLogger(__name__).info("Saved subtitles to %s", destination)


if __name__ == "__main__":
    main()
