#!/usr/bin/env python3
"""
Model seeding utility for transcription pipeline.

This script pre-downloads and initializes WhisperX models for later use in the
transcription pipeline. It downloads the main Whisper model, alignment models for
supported languages, and optionally a diarization model.
"""

import argparse
import logging
import os
import sys

import whisperx

from transcription_pipeline.constants import (
    DEFAULT_WHISPER_MODEL,
    VALID_LANGUAGES,
)

from download_pipeline_processor.logger import (
    Logger,
)


class ModelSeeder:
    """
    Downloads and initializes models for the transcription pipeline.

    This class handles downloading and initializing WhisperX models, including
    the main transcription model, alignment models for supported languages,
    and optionally a diarization model.

    :param whisper_model_name: Name of the WhisperX model to use
    :param languages: List of language codes to download alignment models for
    :param device: Device to use for model computation (cuda, cpu)
    :param compute_type: Model computation type (float16, float32, etc.)
    :param hf_token: Hugging Face authentication token for model access
    :param download_diarization: Whether to download the diarization model
    :param debug: Enable debug logging
    """
    def __init__(
        self,
        whisper_model_name: str = DEFAULT_WHISPER_MODEL,
        languages: list[str] = VALID_LANGUAGES,
        device: str = "cuda",
        compute_type: str = "float16",
        hf_token: str | None = None,
        download_diarization: bool = True,
        debug: bool = False,
    ) -> None:
        self.whisper_model_name: str = whisper_model_name
        self.languages: list[str] = languages
        self.device: str = device
        self.compute_type: str = compute_type
        self.hf_token: str | None = hf_token
        self.download_diarization: bool = download_diarization

        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)

    def download_whisper_model(self) -> None:
        """Download and initialize the Whisper model."""
        self.log.info(f"Downloading Whisper model: {self.whisper_model_name}")
        try:
            _ = whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type=self.compute_type
            )
            self.log.info(f"Successfully loaded Whisper model: {self.whisper_model_name}")
        except Exception as e:
            self.log.error(f"Failed to download Whisper model: {e}")
            raise

    def download_alignment_models(self) -> None:
        """Download alignment models for all specified languages."""
        for language in self.languages:
            self.log.info(f"Downloading alignment model for language: {language}")
            try:
                _, _ = whisperx.load_align_model(
                    language_code=language,
                    device=self.device
                )
                self.log.info(f"Successfully loaded alignment model for {language}")
            except Exception as e:
                self.log.error(f"Failed to download alignment model for {language}: {e}")
                raise

    def download_diarization_model(self) -> None:
        """Download diarization model if requested."""
        if not self.download_diarization:
            self.log.info("Skipping diarization model download as requested")
            return

        if not self.hf_token:
            self.log.warning("No Hugging Face token provided, skipping diarization model")
            return

        self.log.info("Downloading diarization model")
        try:
            _ = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            self.log.info("Successfully loaded diarization model")
        except Exception as e:
            self.log.error(f"Failed to download diarization model: {e}")
            raise

    def download_all_models(self) -> None:
        """Download all required models."""
        try:
            self.download_whisper_model()
            self.download_alignment_models()
            self.download_diarization_model()
            self.log.info("All models downloaded and initialized successfully")
        except Exception as e:
            self.log.error(f"Model download process failed: {e}")
            sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the model seeder.

    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Download and initialize models for the transcription pipeline"
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="WhisperX model to download (default: %(default)s)"
    )

    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(VALID_LANGUAGES),
        help="Comma-separated list of language codes to download alignment models for (default: %(default)s)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: %(default)s)"
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        choices=["float16", "float32", "int8"],
        help="Computation type for models (default: %(default)s)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="Hugging Face token for accessing models (default: HUGGINGFACEHUB_API_TOKEN env var)"
    )

    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip downloading the diarization model"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the model seeder script."""
    args = parse_arguments()
    languages = args.languages.split(",") if args.languages else VALID_LANGUAGES
    seeder = ModelSeeder(
        whisper_model_name=args.whisper_model,
        languages=languages,
        device=args.device,
        compute_type=args.compute_type,
        hf_token=args.hf_token,
        download_diarization=not args.skip_diarization,
        debug=args.debug,
    )

    seeder.download_all_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
