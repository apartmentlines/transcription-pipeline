#!/usr/bin/env python3

import sys
import wave
import argparse
from pathlib import Path
from typing import Union, Any

from download_pipeline_processor.logger import Logger
from .constants import (
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
)


class AudioFileLengthError(Exception):
    """Custom exception for audio file length validation errors."""

    def __init__(self, exception: Union[str, Exception]) -> None:
        super().__init__(f"Audio file length error: {str(exception)}")


class AudioFileValidator:
    """Audio file validation class for WAV files.

    Validates audio files for proper format and duration constraints.
    Provides detailed logging of the validation process.

    :param file_path: Path to the audio file to validate
    :param min_duration: Minimum allowed duration in seconds
    :param max_duration: Maximum allowed duration in seconds
    :param debug: Enable debug logging
    :raises FileNotFoundError: If the audio file doesn't exist
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        min_duration: float = MIN_AUDIO_DURATION,
        max_duration: float = MAX_AUDIO_DURATION,
        debug: bool = False,
        wave_module: Any = wave,
    ) -> None:
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.file_path = self._validate_file_path(file_path)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.wave_module = wave_module

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and convert file path to Path object.

        :param file_path: Path to validate
        :return: Validated Path object
        :raises FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            self.log.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        self.log.debug(f"Validated file path: {path}")
        return path

    def _open_wave_file(self) -> Any:
        """Open and return wave file handle.

        :return: Wave file handle
        :raises AudioFileLengthError: If file is invalid or unreadable
        """
        try:
            return self.wave_module.open(str(self.file_path), "r")
        except self.wave_module.Error as e:
            self.log.error(f"Failed to read WAV file: {e}")
            raise AudioFileLengthError(e)
        except Exception as e:
            self.log.error(f"Unexpected error reading WAV file: {e}")
            raise AudioFileLengthError(e)

    def _calculate_duration(self, wav_file: Any) -> float:
        """Calculate duration from wave file parameters.

        :param wav_file: Wave file handle
        :return: Duration in seconds
        """
        return wav_file.getnframes() / wav_file.getframerate()

    def get_duration(self) -> float:
        """Calculate duration of WAV file in seconds.

        :return: Duration in seconds
        :raises AudioFileLengthError: If file is invalid or unreadable
        """
        self.log.debug(f"Getting duration for {self.file_path}")
        with self._open_wave_file() as wav:
            duration = self._calculate_duration(wav)
            self.log.debug(f"Calculated duration: {duration:.2f}s")
            return duration

    def validate(self) -> None:
        """Validate the audio file meets all requirements.

        :raises AudioFileLengthError: If file duration is outside allowed range
        """
        self.log.debug(f"Validating audio file: {self.file_path}")
        duration = self.get_duration()
        if duration < self.min_duration:
            msg = (
                f"Audio file too short: {duration:.2f}s (minimum: {self.min_duration}s)"
            )
            self.log.error(msg)
            raise AudioFileLengthError(msg)
        if duration > self.max_duration:
            msg = (
                f"Audio file too long: {duration:.2f}s (maximum: {self.max_duration}s)"
            )
            self.log.error(msg)
            raise AudioFileLengthError(msg)
        self.log.debug(f"Audio file validation successful, duration: {duration:.2f}s")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Validate WAV file duration")
    parser.add_argument("file", help="Path to WAV file to validate")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_AUDIO_DURATION,
        help=f"Minimum allowed duration in seconds (default: {MIN_AUDIO_DURATION})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_AUDIO_DURATION,
        help=f"Maximum allowed duration in seconds (default: {MAX_AUDIO_DURATION})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the validation script.

    Parses arguments, validates the audio file, and handles any errors.
    Exits with status code 1 if validation fails.
    """
    args = parse_arguments()
    try:
        validator = AudioFileValidator(
            args.file,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            debug=args.debug,
        )
        validator.validate()
    except (FileNotFoundError, AudioFileLengthError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
