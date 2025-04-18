#!/usr/bin/env python3

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, TYPE_CHECKING
from whisperx.utils import get_writer
from download_pipeline_processor.logger import (
    Logger,
)
from .constants import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_NUM_SPEAKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_BATCH_SIZE,
    VALID_LANGUAGES,
    INITIAL_PROMPT,
)

if TYPE_CHECKING:
    import numpy as np
    import whisperx  # noqa : F401
    import torch  # noqa : F401


def _import_dependencies():
    """Lazily import heavy dependencies only when needed."""
    global np, whisperx, torch
    import numpy as np
    import whisperx
    import torch

    return np, whisperx, torch


class TranscriptionError(Exception):
    """Custom exception for errors that occur during transcription."""

    TRANSIENT_ERROR_PHRASES = ["CUDA failed", "cuBLAS failed"]

    def __init__(self, exception):
        self.original_error = str(exception)
        super().__init__(f"Transcription error: {self.original_error}")

    def is_transient_error(self) -> bool:
        """Check if the error is GPU-related.

        Returns:
            bool: True if the error message contains GPU-related phrases, False otherwise.
        """
        return any(
            phrase in self.original_error for phrase in self.TRANSIENT_ERROR_PHRASES
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
        whisperx_module: Any = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        self.log = Logger(self.__class__.__name__, debug=debug)

        if whisperx_module is None:
            _, self.whisperx, torch = _import_dependencies()  # noqa : F401
        else:
            self.whisperx = whisperx_module
            _, _, torch = _import_dependencies()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or (
            "float16" if torch.cuda.is_available() else "int8"
        )
        self.whisper_model_name = whisper_model_name
        self.diarization_model_name = diarization_model_name
        self.auth_token = auth_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._initialize_whisper_model()
        self.alignment_models = {}
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
        self.model = self.whisperx.load_model(
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
            self.diarization_model = self.whisperx.DiarizationPipeline(
                model_name=self.diarization_model_name,
                use_auth_token=self.auth_token,
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

    def _load_audio(self, input_file: Union[str, Path]) -> "np.ndarray":
        """Load audio file into memory using WhisperX.

        :param input_file: Path to the audio file
        :return: Numpy array containing the audio data
        """
        self.log.debug(f"Loading audio file: {input_file}")
        try:
            _, _, _ = _import_dependencies()
            return self.whisperx.load_audio(input_file)
        except Exception as e:
            self.log.error(f"Failed to load audio file: {str(e)}")
            raise TranscriptionError(e) from e

    def _perform_base_transcription(
        self, audio: "np.ndarray", initial_prompt: str
    ) -> Dict[str, Any]:
        """Perform initial transcription of audio using WhisperX.

        :param audio: Numpy array containing audio data
        :return: Dictionary containing transcription results and metadata
        """
        _, _, torch = _import_dependencies()  # noqa : F401
        self.log.info("Performing base transcription")
        self.log.debug(f"Using batch size: {DEFAULT_BATCH_SIZE}")
        self.log.debug(f"Initial prompt: {initial_prompt}")
        try:
            # WhisperX doesn't allow passing the prompt to transcribe(), so hack
            # the options directly.
            self.model.options.initial_prompt = initial_prompt
            with torch.inference_mode():
                result = self.model.transcribe(
                    audio,
                    batch_size=DEFAULT_BATCH_SIZE,
                )
            result = self._move_result_tensors_to_cpu(result)
            self.log.debug(
                f"Base transcription complete with {len(result['segments'])} segments"
            )
            return result
        except Exception as e:
            self.log.error(f"Failed to perform base transcription: {str(e)}")
            raise TranscriptionError(e) from e

    def load_alignment_model(self, language: str) -> Tuple[Any, dict]:
        """Load alignment model for a specific language.

        :param language: Detected language code
        :return: Tuple containing alignment model and metadata
        """
        if language not in self.alignment_models:
            self.log.info(f"Loading alignment model for language: {language}")
            model_a, metadata = self.whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self.alignment_models[language] = (model_a, metadata)
        else:
            self.log.debug(f"Loading cached alignment model for language: {language}")
            model_a, metadata = self.alignment_models[language]
        return model_a, metadata

    def _align_transcription(
        self, segments: list, audio: "np.ndarray", language: str
    ) -> Dict[str, Any]:
        """Align transcription segments with audio timing.

        :param segments: List of transcription segments
        :param audio: Numpy array containing audio data
        :param language: Detected language code
        :return: Dictionary containing aligned transcription results
        """
        _, _, torch = _import_dependencies()  # noqa : F401
        self.log.info("Aligning transcription with audio")
        model_a, metadata = self.load_alignment_model(language)
        self.log.debug("Performing alignment")
        try:
            with torch.inference_mode():
                aligned_result = self.whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=True,
                )
            aligned_result = self._move_result_tensors_to_cpu(aligned_result)
        except Exception as e:
            self.log.error(f"Failed to perform alignment: {str(e)}")
            raise TranscriptionError(e) from e
        finally:
            del model_a
            del metadata
        self.log.debug(
            f"Alignment complete with {len(aligned_result['segments'])} segments"
        )
        aligned_result["language"] = language
        return aligned_result

    def _perform_diarization(
        self, audio: "np.ndarray", num_speakers: int
    ) -> Optional[Dict[str, Any]]:
        """Perform speaker diarization on the audio.

        :param audio: Numpy array containing audio data
        :param num_speakers: Expected number of speakers in the audio
        :return: Dictionary containing diarization results or None if diarization fails
        """
        self.log.info(f"Performing diarization with {num_speakers} speakers")
        try:
            if self.diarization_model:
                _, _, torch = _import_dependencies()  # noqa : F401
                with torch.inference_mode():
                    diarization_result = self.diarization_model(
                        audio, num_speakers=num_speakers
                    )
                diarization_result = self._move_result_tensors_to_cpu(
                    diarization_result
                )
                return diarization_result
        except Exception as e:
            self.log.error(f"Failed to perform diarization: {str(e)}")
            raise TranscriptionError(e) from e

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
        try:
            result = self.whisperx.assign_word_speakers(
                diarization_segments, aligned_result
            )
        except Exception as e:
            self.log.error(f"Failed to assign speaker labels: {str(e)}")
            raise TranscriptionError(e) from e
        self.log.debug(
            f"Speaker assignment complete with {len(result['segments'])} labeled segments"
        )
        result["language"] = aligned_result["language"]
        return result

    def _handle_diarization(
        self, audio: "np.ndarray", aligned_result: Dict[str, Any], num_speakers: int
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
        try:
            writer(
                result,
                input_file,  # pyright: ignore[reportArgumentType]
                {
                    "max_line_width": None,
                    "max_line_count": None,
                    "highlight_words": False,
                },
            )
        except Exception as e:
            self.log.error(f"Failed to save output: {str(e)}")
            raise TranscriptionError(e) from e
        self.log.info(f"Output saved to {output_dir}")

    def _validate_language(self, detected_language: str) -> None:
        """Validate that the detected language is supported for alignment.

        :param detected_language: Language code detected by WhisperX
        :raises TranscriptionError: If the language is not supported
        """
        if detected_language not in VALID_LANGUAGES:
            self.log.error(f"Unsupported language detected: {detected_language}")
            raise TranscriptionError(
                f"Language '{detected_language}' is not supported. Supported languages are: {', '.join(VALID_LANGUAGES)}"
            )

    def _extract_transcription_metadata(self, final_result: Dict[str, Any]) -> None:
        """Extract and add metadata to the transcription result.

        :param final_result: The final transcription result to augment with metadata
        :param base_result: The initial transcription result containing base metadata
        """
        segments = final_result.get("segments", [])
        word_segments = final_result.get("word_segments", [])
        final_result["total_words"] = sum(len(seg.get("words", [])) for seg in segments)
        final_duration = 0
        for segment in reversed(segments):
            if "end" in segment:
                final_duration = segment["end"]
                break
        final_result["total_duration"] = final_duration
        final_result["speaking_duration"] = sum(
            segment.get("end", 0) - segment.get("start", 0)
            for segment in segments
            if "start" in segment and "end" in segment
        )
        if word_segments:
            valid_scores = [seg["score"] for seg in word_segments if "score" in seg]
            if valid_scores:  # Only calculate if we have any valid scores
                avg_confidence = sum(valid_scores) / len(valid_scores)
                final_result["average_word_confidence"] = round(
                    float(avg_confidence), 4
                )

    def _move_result_tensors_to_cpu(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Recursively move all tensors in the result dict to CPU
        _, _, torch = _import_dependencies()  # noqa : F401
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu()
            elif isinstance(value, list):
                result[key] = [self._move_tensors_in_item(v) for v in value]
            elif isinstance(value, dict):
                result[key] = self._move_result_tensors_to_cpu(value)
        return result

    def _move_tensors_in_item(self, item: Any) -> Any:
        _, _, torch = _import_dependencies()  # noqa : F401
        if isinstance(item, torch.Tensor):
            return item.cpu()
        elif isinstance(item, list):
            return [self._move_tensors_in_item(i) for i in item]
        elif isinstance(item, dict):
            return self._move_result_tensors_to_cpu(item)
        else:
            return item

    def transcribe(
        self,
        input_file: Union[str, Path],
        initial_prompt: str,
        num_speakers: int = DEFAULT_NUM_SPEAKERS,
        output_dir: Path | None = DEFAULT_OUTPUT_DIR,
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
        _, _, torch = _import_dependencies()  # noqa : F401
        try:
            validated_path = self._validate_input_file(input_file)
            self.log.debug(
                f"Memory before processing {validated_path}: "
                f"{torch.cuda.memory_allocated() / 1e6} MB allocated, "
                f"{torch.cuda.memory_reserved() / 1e6} MB reserved"
            )
            audio = self._load_audio(validated_path)
            result = self._perform_base_transcription(audio, initial_prompt)
            self._validate_language(result["language"])
            aligned_result = self._align_transcription(
                result["segments"], audio, result["language"]
            )
            final_result = self._handle_diarization(audio, aligned_result, num_speakers)
            augmented_result = deepcopy(final_result)
            self._extract_transcription_metadata(augmented_result)
            self._save_output(final_result, input_file, output_dir, output_format)
            del audio
            del result
            del aligned_result
            del final_result
            self.log.debug(
                f"Memory after processing {validated_path}: "
                f"{torch.cuda.memory_allocated() / 1e6} MB allocated, "
                f"{torch.cuda.memory_reserved() / 1e6} MB reserved"
            )
            return augmented_result
        except Exception as e:
            self.log.error(f"Transcription failed: {str(e)}")
            self.log.debug(f"Full error details: {repr(e)}")
            raise
        finally:
            torch.cuda.empty_cache()


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
        initial_prompt = INITIAL_PROMPT % "Abbie Lake Apartments"
        transcriber.transcribe(
            args.input_file,
            initial_prompt,
            num_speakers=args.num_speakers,
            output_dir=args.output_dir,
            output_format=args.output_format,
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
