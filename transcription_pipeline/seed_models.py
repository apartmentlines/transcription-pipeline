"""Bootstrap an offline deployment manifest and its pinned model assets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


class ModelSeedingCommand:
    """Select a seeder without importing the other backend's dependencies."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Seeding selects what to install before any manifest exists."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--backend",
            required=True,
            choices=("whisperx", "qwen3"),
            help="Backend to install",
        )
        parser.add_argument(
            "--model-manifest",
            type=Path,
            help="Manifest destination; defaults to the Hugging Face application directory",
        )
        parser.add_argument(
            "--whisperx-diarization",
            action="store_true",
            help="Also seed gated WhisperX diarization models using HF_TOKEN",
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        return parser

    @staticmethod
    def run(args: argparse.Namespace) -> Path:
        """Download the selected immutable model set and publish its manifest."""
        if args.backend == "whisperx":
            from .backends.whisperx_seed import WhisperXModelSeeder

            return WhisperXModelSeeder().seed(
                args.model_manifest, args.whisperx_diarization
            )
        if args.whisperx_diarization:
            raise ValueError("--whisperx-diarization requires --backend whisperx")
        from .backends.qwen3_seed import Qwen3ModelSeeder

        return Qwen3ModelSeeder().seed(args.model_manifest)


def main() -> None:
    """Run explicit build-time seeding and surface failures with a nonzero exit."""
    parser = ModelSeedingCommand.parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        ModelSeedingCommand.run(args)
    except Exception as exc:
        parser.exit(1, f"Model installation failed: {exc}\n")


if __name__ == "__main__":
    main()
