#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import whisperx
from whisperx.utils import get_writer
import torch
import os
import sys

from download_pipeline_processor.logger import Logger
from .constants import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_NUM_SPEAKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_BATCH_SIZE,
)


class Transcriber:
    """Audio transcription class using WhisperX for speech-to-text conversion.

    Handles audio file transcription with optional speaker diarization. Supports
    multiple output formats and provides detailed logging of the transcription process.

    :param whisper_model_name: Name of the WhisperX model to use for transcription
    :param diarization_model_name: Optional name of model for speaker diarization
    :param debug: Enable debug logging
    """

    def __init__(
        self,
        whisper_model_name: str = DEFAULT_WHISPER_MODEL,
        diarization_model_name: Optional[str] = None,
        debug: bool = False,
    ):
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.whisper_model_name = whisper_model_name
        self.diarization_model_name = diarization_model_name
        self._initialize_whisper_model()
        self._initialize_diarization_model()

    def _initialize_whisper_model(self) -> None:
        """Initialize the WhisperX model for transcription.

        Loads the specified WhisperX model using the configured device and compute type.
        Sets up the model for subsequent transcription operations.
        """
        self.log.debug(
            f"Initializing with device={self.device}, compute_type={self.compute_type}"
        )
        self.log.info(f"Loading WhisperX model: {self.whisper_model_name}")
        self.model = whisperx.load_model(
            self.whisper_model_name, self.device, compute_type=self.compute_type
        )
        self.log.debug("WhisperX model loaded successfully")

    def _initialize_diarization_model(self) -> None:
        """Initialize the speaker diarization model if configured.

        Loads the specified diarization model if a model name was provided.
        Uses HuggingFace authentication token if available in environment.
        """
        if self.diarization_model_name:
            self.log.info(f"Loading diarization model: {self.diarization_model_name}")
            self.diarization_model = whisperx.DiarizationPipeline(
                model_name=self.diarization_model_name,
                use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                device=self.device,
            )
            self.log.debug("Diarization model loaded successfully")
        else:
            self.log.info("No diarization model provided, skipped loading")
            self.diarization_model = None

    def _validate_input_file(self, input_file: Union[str, Path]) -> Path:
        """Validate that the input file exists and is accessible.

        :param input_file: Path to the audio file
        :return: Validated Path object
        :raises FileNotFoundError: If the file does not exist
        """
        path = Path(input_file)
        if not path.exists():
            self.log.error(f"Input file not found: {path}")
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    def _load_audio(self, input_file: Union[str, Path]) -> np.ndarray:
        """Load audio file into memory using WhisperX.

        :param input_file: Path to the audio file
        :return: Numpy array containing the audio data
        """
        self.log.debug(f"Loading audio file: {input_file}")
        return whisperx.load_audio(input_file)

    def _perform_base_transcription(self, audio: np.ndarray) -> Dict[str, Any]:
        """Perform initial transcription of audio using WhisperX.

        :param audio: Numpy array containing audio data
        :return: Dictionary containing transcription results and metadata
        """
        self.log.info("Performing base transcription")
        self.log.debug(f"Using batch size: {DEFAULT_BATCH_SIZE}")
        result = self.model.transcribe(audio, batch_size=DEFAULT_BATCH_SIZE)
        self.log.debug(
            f"Base transcription complete with {len(result['segments'])} segments"
        )
        return result

    def _align_transcription(
        self, segments: list, audio: np.ndarray, language: str
    ) -> Dict[str, Any]:
        """Align transcription segments with audio timing.

        :param segments: List of transcription segments
        :param audio: Numpy array containing audio data
        :param language: Detected language code
        :return: Dictionary containing aligned transcription results
        """
        self.log.info("Aligning transcription with audio")
        self.log.debug(f"Loading alignment model for language: {language}")
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        self.log.debug("Performing alignment")
        aligned_result = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        self.log.debug(
            f"Alignment complete with {len(aligned_result['segments'])} segments"
        )
        aligned_result["language"] = language
        return aligned_result

    def _perform_diarization(
        self, audio: np.ndarray, num_speakers: int
    ) -> Optional[Dict[str, Any]]:
        """Perform speaker diarization on the audio.

        :param audio: Numpy array containing audio data
        :param num_speakers: Expected number of speakers in the audio
        :return: Dictionary containing diarization results or None if diarization fails
        """
        self.log.info(f"Performing diarization with {num_speakers} speakers")
        return self.diarization_model(audio, num_speakers=num_speakers)

    def _assign_speakers(
        self, diarization_segments: Dict[str, Any], aligned_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign speaker labels to transcription segments.

        :param diarization_segments: Dictionary containing speaker diarization results
        :param aligned_result: Dictionary containing aligned transcription results
        :return: Dictionary containing transcription with speaker labels assigned
        """
        self.log.info("Assigning speaker labels")
        self.log.debug(
            f"Processing {len(diarization_segments['segments'])} diarization segments"
        )
        result = whisperx.assign_word_speakers(diarization_segments, aligned_result)
        self.log.debug(
            f"Speaker assignment complete with {len(result['segments'])} labeled segments"
        )
        result["language"] = aligned_result["language"]
        return result

    def _handle_diarization(
        self, audio: np.ndarray, aligned_result: Dict[str, Any], num_speakers: int
    ) -> Dict[str, Any]:
        """Handle the complete diarization workflow.

        :param audio: Numpy array containing audio data
        :param aligned_result: Dictionary containing aligned transcription results
        :param num_speakers: Expected number of speakers
        :return: Dictionary containing transcription with speaker information
        """
        if not self.diarization_model:
            self.log.debug("Skipping diarization - no diarization model loaded")
            return aligned_result
        diarization_segments = self._perform_diarization(audio, num_speakers)
        if diarization_segments:
            return self._assign_speakers(diarization_segments, aligned_result)
        return aligned_result

    def _save_output(
        self,
        result: Dict[str, Any],
        input_file: Union[str, Path],
        output_dir: Optional[Path],
        output_format: str,
    ) -> None:
        """Save transcription results to file.

        :param result: Dictionary containing transcription results
        :param input_file: Path to the original input audio file
        :param output_dir: Directory to save output files
        :param output_format: Format for output files (e.g., 'srt', 'vtt')
        """
        if not output_dir:
            self.log.debug("No output directory specified, skipping file output")
            return
        self.log.debug(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log.debug(f"Initializing {output_format} writer")
        writer = get_writer(output_format, str(output_dir))
        output_path = (
            output_dir / Path(input_file).with_suffix(f".{output_format}").name
        )
        self.log.debug(f"Writing output to: {output_path}")
        writer(
            result,
            input_file,
            {"max_line_width": None, "max_line_count": None, "highlight_words": False},
        )
        self.log.info(f"Output saved to {output_dir}")

    def transcribe(
        self,
        input_file: Union[str, Path],
        num_speakers: int = DEFAULT_NUM_SPEAKERS,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> Dict[str, Any]:
        """Transcribe an audio file with optional speaker diarization.

        Main entry point for transcription. Processes the audio file through
        multiple stages: transcription, alignment, and optional speaker diarization.

        :param input_file: Path to the audio file to transcribe
        :param num_speakers: Number of speakers for diarization
        :param output_dir: Directory to save output files
        :param output_format: Format for output files (e.g., 'srt', 'vtt')
        :return: Dictionary containing complete transcription results
        :raises FileNotFoundError: If input file doesn't exist
        :raises Exception: For any transcription-related errors
        """
        self.log.info(f"Starting transcription of {input_file}")
        try:
            validated_path = self._validate_input_file(input_file)
            audio = self._load_audio(validated_path)
            result = self._perform_base_transcription(audio)
            aligned_result = self._align_transcription(
                result["segments"], audio, result["language"]
            )
            final_result = self._handle_diarization(audio, aligned_result, num_speakers)
            self._save_output(final_result, input_file, output_dir, output_format)
            return final_result
        except Exception as e:
            self.log.error(f"Transcription failed: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the transcription script.

    :return: Namespace containing parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Audio transcription tool using WhisperX"
    )
    parser.add_argument("input_file", help="Path to audio file to transcribe")
    parser.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL, help="WhisperX model to use"
    )
    parser.add_argument(
        "--diarization-model",
        help="Diarization model to use. If provided, enables diarization",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=DEFAULT_NUM_SPEAKERS,
        help="Number of speakers when diarization is enabled",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files",
    )
    parser.add_argument(
        "--output-format",
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format (e.g. srt, vtt)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the transcription script.

    Parses arguments, initializes the transcriber, and handles the transcription process.
    Exits with status code 1 if an error occurs during transcription.
    """
    args = parse_arguments()
    transcriber = Transcriber(
        whisper_model_name=args.whisper_model,
        diarization_model_name=args.diarization_model,
        debug=args.debug,
    )
    try:
        transcriber.transcribe(
            args.input_file,
            num_speakers=args.num_speakers,
            output_dir=args.output_dir,
            output_format=args.output_format,
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
