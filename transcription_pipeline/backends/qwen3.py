"""Native Qwen3 ASR and forced alignment with complete audio coverage."""

from __future__ import annotations

import importlib
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from transcription_pipeline.constants import (
    MAX_AUDIO_DURATION,
    MILLISECONDS_PER_SECOND,
    MIN_AUDIO_DURATION,
    QWEN_CHUNK_SEARCH_SECONDS,
    QWEN_CONTEXT,
    QWEN_ENERGY_FRAME_MILLISECONDS,
    QWEN_GENERATION_BUDGET,
    QWEN_LANGUAGE_CODES,
    QWEN_MAX_CHUNK_SECONDS,
    QWEN_SAMPLE_RATE,
)
from ..deployment import Deployment
from ..errors import ConfigurationError, TranscriptionError
from ..results import (
    AlignmentToken,
    ResultBuilder,
    TranscriptionContext,
    TranscriptionResult,
)


@dataclass(frozen=True)
class AudioChunk:
    """Half-open source sample range; no samples are removed or duplicated."""

    start: int
    end: int


class AudioChunker:
    """Decode once and select deterministic low-energy contiguous boundaries."""

    @staticmethod
    def decode(path: str | Path) -> NDArray[np.float32]:
        """Use FFmpeg's lossless in-memory mono 16 kHz PCM output."""
        source = Path(path)
        if not source.is_file():
            raise ConfigurationError(f"Audio file does not exist: {source}")
        try:
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    str(QWEN_SAMPLE_RATE),
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TranscriptionError("Audio decoding failed", code="audio") from exc
        waveform = np.frombuffer(process.stdout, dtype=np.float32)
        duration = len(waveform) / QWEN_SAMPLE_RATE
        if not MIN_AUDIO_DURATION <= duration <= MAX_AUDIO_DURATION:
            raise TranscriptionError(
                f"Audio duration must be {MIN_AUDIO_DURATION}–{MAX_AUDIO_DURATION} seconds",
                code="audio",
            )
        if not np.all(np.isfinite(waveform)):
            raise TranscriptionError("Nonfinite audio samples", code="audio")
        return waveform

    @staticmethod
    def ranges(waveform: NDArray[np.float32]) -> tuple[AudioChunk, ...]:
        """Choose a 20 ms energy minimum in each final five-second search window."""
        maximum = QWEN_MAX_CHUNK_SECONDS * QWEN_SAMPLE_RATE
        search = QWEN_CHUNK_SEARCH_SECONDS * QWEN_SAMPLE_RATE
        frame = (
            QWEN_SAMPLE_RATE * QWEN_ENERGY_FRAME_MILLISECONDS // MILLISECONDS_PER_SECOND
        )
        chunks: list[AudioChunk] = []
        start = 0
        while start < len(waveform):
            end = min(start + maximum, len(waveform))
            if end < len(waveform):
                window = (
                    waveform[end - search : end].astype(np.float64).reshape(-1, frame)
                )
                energy = np.mean(window * window, axis=1)
                index = len(energy) - 1 - int(np.argmin(energy[::-1]))
                end = end - search + index * frame + frame // 2
            chunks.append(AudioChunk(start, end))
            start = end
        return tuple(chunks)


class Qwen3Transcriber:
    """One reusable ASR/aligner pair per pipeline worker.

    :param deployment: Validated local model paths, device, and precision.
    :param debug: Enable adapter diagnostics without logging transcript text.
    """

    deployment: Deployment
    log: logging.Logger
    torch: ModuleType
    asr_processor: Any
    aligner_processor: Any
    asr_model: Any
    aligner_model: Any

    def __init__(self, deployment: Deployment, debug: bool = False) -> None:
        self.deployment = deployment
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if debug else logging.INFO)
        self.torch, transformers = self._dependencies()
        if not self.torch.cuda.is_available():
            raise ConfigurationError("Qwen3 requires an NVIDIA CUDA GPU")
        if (
            deployment.precision == "bfloat16"
            or deployment.qwen_attention == "flash_attention_2"
        ):
            major, _ = self.torch.cuda.get_device_capability(deployment.device)
            if major < 8:
                raise ConfigurationError(
                    "BF16 and FlashAttention 2 require Ampere or newer NVIDIA hardware"
                )
        if deployment.qwen_attention == "flash_attention_2":
            try:
                importlib.import_module("flash_attn")
            except (ImportError, OSError, RuntimeError) as exc:
                raise ConfigurationError(
                    "FlashAttention 2 could not load; install the pinned optional "
                    + "wheel matching Python 3.12 and Torch 2.8 CUDA 12, or set "
                    + "TRANSCRIPTION_QWEN_ATTENTION=sdpa"
                ) from exc
        self.log.info("Qwen attention implementation: %s", deployment.qwen_attention)
        dtype = getattr(self.torch, deployment.precision)
        options = {
            "dtype": dtype,
            "device_map": deployment.device,
            "attn_implementation": deployment.qwen_attention,
            "local_files_only": True,
        }
        try:
            asr_path = str(deployment.model("asr").path)
            aligner_path = str(deployment.model("aligner").path)
            self.asr_processor = transformers.AutoProcessor.from_pretrained(
                asr_path, local_files_only=True
            )
            self.aligner_processor = transformers.AutoProcessor.from_pretrained(
                aligner_path, local_files_only=True
            )
            self.asr_model = transformers.AutoModelForMultimodalLM.from_pretrained(
                asr_path, **options
            ).eval()
            self.aligner_model = (
                transformers.AutoModelForTokenClassification.from_pretrained(
                    aligner_path, **options
                ).eval()
            )
        except Exception as exc:
            self._classify_cuda_error(exc)
            raise ConfigurationError(
                f"Qwen3 local model initialization failed: {exc}"
            ) from exc

    def _classify_cuda_error(self, exception: Exception) -> None:
        """Distinguish device failures from deterministic model configuration errors."""
        message = str(exception).lower()
        if "out of memory" in message:
            raise TranscriptionError(
                f"CUDA out of memory with {self.deployment.worker_count} configured workers; "
                + "lower --processing-limit",
                code="cuda_oom",
                transient=True,
            ) from exception
        if any(name in message for name in ("cuda", "cublas", "cudnn")):
            raise TranscriptionError(
                exception, code="cuda", transient=True
            ) from exception

    @staticmethod
    def _dependencies() -> tuple[ModuleType, ModuleType]:
        """Import the selected inference stack only inside worker construction."""
        return importlib.import_module("torch"), importlib.import_module("transformers")

    @staticmethod
    def parse_transcript(raw: str) -> tuple[str | None, str, str]:
        """Parse the model protocol without repetition cleanup or translation."""
        raw = raw.strip()
        for marker in ("<|im_end|>", "<|endoftext|>"):
            raw = raw.removesuffix(marker).strip()
        if not raw:
            return None, "", ""
        if "<asr_text>" not in raw:
            raise TranscriptionError(
                "ASR response lacks language/text delimiter", code="protocol"
            )
        header, text = raw.split("<asr_text>", 1)
        detected = header.removeprefix("language ").strip()
        language = QWEN_LANGUAGE_CODES.get(detected.lower())
        text = text.strip()
        if language is None and text:
            raise TranscriptionError(
                "Unsupported or unknown detected language", code="language"
            )
        return language, text, detected

    @staticmethod
    def check_termination(
        token_ids: list[int], eos_ids: int | list[int] | None
    ) -> None:
        """Accept natural final EOS, including at the generation-budget boundary."""
        eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        if not token_ids or token_ids[-1] not in eos:
            raise TranscriptionError(
                f"ASR generation ended without EOS after {len(token_ids)} tokens "
                + f"(budget {QWEN_GENERATION_BUDGET}); partial transcription rejected",
                code="generation_incomplete",
            )

    @staticmethod
    def _validate_features(inputs: Any, processor: Any, sample_count: int) -> None:
        """Reject processor truncation using the complete audio feature mask."""
        mask = inputs.get("input_features_mask")
        if mask is None:
            raise TranscriptionError(
                "Processor omitted audio coverage mask", code="audio_coverage"
            )
        hop = int(processor.feature_extractor.hop_length)
        frames = int(mask.sum().item())
        if frames < math.floor(sample_count / hop):
            raise TranscriptionError(
                "Processor truncated audio features", code="audio_coverage"
            )

    def _asr(
        self, audio: NDArray[np.float32], prompt: str
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Generate one chunk, validating coverage, context capacity, and termination."""
        inputs = self.asr_processor.apply_transcription_request(
            audio=audio,
            language=None,
            prompt=prompt,
            processor_kwargs={
                "audio_kwargs": {"sampling_rate": QWEN_SAMPLE_RATE, "truncation": False}
            },
        )
        self._validate_features(inputs, self.asr_processor, len(audio))
        prompt_length = inputs["input_ids"].shape[1]
        capacity = self.asr_model.config.text_config.max_position_embeddings
        if prompt_length + QWEN_GENERATION_BUDGET > capacity:
            raise TranscriptionError(
                "Audio and prompt exceed model context capacity", code="context"
            )
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with self.torch.inference_mode():
            output = self.asr_model.generate(
                **inputs,
                max_new_tokens=QWEN_GENERATION_BUDGET,
                do_sample=False,
                use_cache=True,
                logits_to_keep=1,
                return_dict_in_generate=False,
                output_scores=False,
                forced_eos_token_id=None,
            )
        suffix = output[0, prompt_length:].detach().cpu().tolist()
        eos_ids = self.asr_model.generation_config.eos_token_id
        self.check_termination(suffix, eos_ids)
        raw = self.asr_processor.decode(
            suffix[:-1],
            return_format="raw",
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        language, text, detected = self.parse_transcript(raw)
        diagnostics = {
            "generated_tokens": len(suffix),
            "normal_eos": True,
            "detected_language": detected,
            "suspected_repetition": bool(
                re.search(r"\b(\w+)(?:\s+\1){4,}\b", text, re.IGNORECASE)
            ),
        }
        return language, text, diagnostics

    def _align(
        self, audio: NDArray[np.float32], text: str, language: str, offset: float
    ) -> list[AlignmentToken]:
        """Align the identical chunk and apply its original start offset once."""
        inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=audio,
            transcript=text,
            language=language,
            processor_kwargs={
                "audio_kwargs": {"sampling_rate": QWEN_SAMPLE_RATE, "truncation": False}
            },
        )
        self._validate_features(inputs, self.aligner_processor, len(audio))
        inputs = inputs.to(self.aligner_model.device, self.aligner_model.dtype)
        with self.torch.inference_mode():
            output = self.aligner_model(**inputs, use_cache=False)
        timings = self.aligner_processor.decode_forced_alignment(
            logits=output.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.aligner_model.config.timestamp_token_id,
        )[0]
        tokens = []
        duration = len(audio) / QWEN_SAMPLE_RATE
        for item in timings:
            start, end = item.get("start_time"), item.get("end_time")
            if start is not None and end is not None:
                TranscriptionResult.validate_interval(
                    float(start), float(end), duration
                )
            tokens.append(
                AlignmentToken(
                    str(item["text"]),
                    float(start) + offset if start is not None else None,
                    float(end) + offset if end is not None else None,
                )
            )
        return tokens

    def transcribe(
        self, input_file: str | Path, context: TranscriptionContext
    ) -> TranscriptionResult:
        """Transcribe every source sample exactly once, keeping chunk-local tensors local."""
        waveform = AudioChunker.decode(input_file)
        prompt = f"{QWEN_CONTEXT} Community: {context.community}. {context.terminology}".strip()
        builder = ResultBuilder()
        diagnostics: list[dict[str, Any]] = []
        try:
            for chunk in AudioChunker.ranges(waveform):
                audio = waveform[chunk.start : chunk.end]
                language, text, details = self._asr(audio, prompt)
                details.update({"start_sample": chunk.start, "end_sample": chunk.end})
                diagnostics.append(details)
                if not text:
                    continue
                if language is None:
                    raise TranscriptionError(
                        "Nonempty text has no language", code="language"
                    )
                offset = chunk.start / QWEN_SAMPLE_RATE
                tokens = self._align(audio, text, language, offset)
                starts = [token.start for token in tokens if token.start is not None]
                ends = [token.end for token in tokens if token.end is not None]
                if not starts or not ends:
                    raise TranscriptionError(
                        "Nonempty ASR text has no timestamps", code="alignment"
                    )
                builder.append(
                    text,
                    tokens,
                    offset,
                    chunk.end / QWEN_SAMPLE_RATE,
                    language,
                    derive_speech_bounds=True,
                )
            return builder.build(
                len(waveform) / QWEN_SAMPLE_RATE,
                "qwen3",
                self.deployment.model("asr").repo_id,
                self.deployment.model("aligner").repo_id,
                {"chunks": diagnostics},
            )
        except (TranscriptionError, ConfigurationError):
            raise
        except Exception as exc:
            self._classify_cuda_error(exc)
            raise TranscriptionError(exc, code="inference") from exc
