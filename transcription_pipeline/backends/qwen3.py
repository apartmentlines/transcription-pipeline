"""Batched vLLM ASR and native Qwen alignment with complete audio coverage."""

from __future__ import annotations

import importlib
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from time import monotonic
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
    QWEN_KV_CACHE_BYTES,
    QWEN_LANGUAGE_CODES,
    QWEN_MAX_CHUNK_SECONDS,
    QWEN_SAMPLE_RATE,
    TIMESTAMP_TOLERANCE,
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
    def ranges(
        waveform: NDArray[np.float32], chunk_seconds: int = QWEN_MAX_CHUNK_SECONDS
    ) -> tuple[AudioChunk, ...]:
        """Choose a 20 ms energy minimum in each final five-second search window."""
        maximum = chunk_seconds * QWEN_SAMPLE_RATE
        frame = (
            QWEN_SAMPLE_RATE * QWEN_ENERGY_FRAME_MILLISECONDS // MILLISECONDS_PER_SECOND
        )
        search = min(QWEN_CHUNK_SEARCH_SECONDS * QWEN_SAMPLE_RATE, maximum - frame)
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
    """One vLLM ASR engine and native aligner, called by a single batch owner.

    :param deployment: Validated local model paths, device, and precision.
    :param debug: Enable adapter diagnostics without logging transcript text.
    """

    deployment: Deployment
    log: logging.Logger
    torch: ModuleType
    asr_processor: Any
    aligner_processor: Any
    llm: Any
    sampling: Any
    aligner_model: Any

    def __init__(self, deployment: Deployment, debug: bool = False) -> None:
        self.deployment = deployment
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if debug else logging.INFO)
        started = phase_started = monotonic()
        self.log.info("Qwen initialization started: importing torch and transformers")
        self.torch, transformers = self._dependencies()
        self.log.info(
            "Qwen dependency imports completed: seconds=%.3f",
            monotonic() - phase_started,
        )
        phase_started = monotonic()
        if not self.torch.cuda.is_available():
            raise ConfigurationError("Qwen3 requires an NVIDIA CUDA GPU")
        self.torch.cuda.set_device(deployment.device)
        major, _ = self.torch.cuda.get_device_capability(deployment.device)
        if major < 8:
            raise ConfigurationError(
                "Qwen vLLM requires Ampere or newer NVIDIA hardware"
            )
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        vllm = importlib.import_module("vllm")
        self.log.info(
            "Qwen CUDA setup and vLLM import completed: seconds=%.3f",
            monotonic() - phase_started,
        )
        dtype = getattr(self.torch, deployment.precision)
        options = {
            "dtype": dtype,
            "device_map": deployment.device,
            "attn_implementation": "sdpa",
            "local_files_only": True,
        }
        try:
            phase_started = monotonic()
            asr_path = str(deployment.model("asr").path)
            aligner_path = str(deployment.model("aligner").path)
            self.asr_processor = transformers.AutoProcessor.from_pretrained(
                asr_path, local_files_only=True
            )
            self.aligner_processor = transformers.AutoProcessor.from_pretrained(
                aligner_path, local_files_only=True
            )
            self.log.info(
                "Qwen processors loaded: seconds=%.3f", monotonic() - phase_started
            )
            phase_started = monotonic()
            self.log.info("Qwen vLLM engine initialization started: engine_count=1")
            self.llm = vllm.LLM(
                model=asr_path,
                dtype=deployment.precision,
                attention_config={"backend": "FLASH_ATTN"},
                device_ids=[int(deployment.device.partition(":")[2] or 0)],
                max_model_len=4096 if deployment.qwen_chunk_seconds <= 60 else 8192,
                max_num_seqs=deployment.qwen_batch_size,
                max_num_batched_tokens=4096,
                kv_cache_memory_bytes=QWEN_KV_CACHE_BYTES,
                enable_prefix_caching=False,
                limit_mm_per_prompt={"audio": 1},
                mm_processor_kwargs={
                    "sampling_rate": QWEN_SAMPLE_RATE,
                    "truncation": False,
                },
                seed=0,
            )
            self.log.info(
                "Qwen vLLM engine initialized: seconds=%.3f",
                monotonic() - phase_started,
            )
            phase_started = monotonic()
            self.sampling = vllm.SamplingParams(
                temperature=0,
                max_tokens=QWEN_GENERATION_BUDGET,
                skip_special_tokens=False,
            )
            self.aligner_model = (
                transformers.AutoModelForTokenClassification.from_pretrained(
                    aligner_path, **options
                ).eval()
            )
            self.log.info(
                "Qwen shared engine ready: active_chunks=%d chunk_seconds=%d aligner_load_seconds=%.3f total_initialization_seconds=%.3f",
                deployment.qwen_batch_size,
                deployment.qwen_chunk_seconds,
                monotonic() - phase_started,
                monotonic() - started,
            )
        except Exception as exc:
            self.close()
            self._classify_cuda_error(exc)
            raise ConfigurationError(
                f"Qwen3 local model initialization failed: {exc}"
            ) from exc

    def _classify_cuda_error(self, exception: Exception) -> None:
        """Distinguish device failures from deterministic model configuration errors."""
        message = str(exception).lower()
        if "out of memory" in message:
            raise TranscriptionError(
                "CUDA out of memory in the shared Qwen engine",
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

    def _prompt(self, context: TranscriptionContext) -> str:
        """Keep each recording's terminology and community in its own request."""
        prompt = f"{QWEN_CONTEXT} Community: {context.community}. {context.terminology}".strip()
        return self.asr_processor.apply_chat_template(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, requests: list[dict[str, Any]]) -> list[Any]:
        """Submit independent audio requests in source order to vLLM's scheduler."""
        responses = self.llm.generate(requests, self.sampling, use_tqdm=False)
        if len(responses) != len(requests):
            raise TranscriptionError("ASR response count mismatch", code="protocol")
        return [response.outputs[0] for response in responses]

    @staticmethod
    def _request(audio: NDArray[np.float32], prompt: str) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio, QWEN_SAMPLE_RATE)},
        }

    def close(self) -> None:
        """Release the engine process and aligner after all submitted work finishes."""
        llm = getattr(self, "llm", None)
        if llm is not None:
            started = monotonic()
            self.log.info("Stopping Qwen shared engine")
            llm.llm_engine.engine_core.shutdown()
            del self.llm
            self.log.info(
                "Qwen engine shutdown completed: seconds=%.3f", monotonic() - started
            )
        if hasattr(self, "aligner_model"):
            del self.aligner_model

    def _align(
        self,
        audio: NDArray[np.float32],
        text: str,
        language: str,
        offset: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[AlignmentToken]:
        """Align the chunk and apply its offset once.

        Clip and report end quantization within one native timestamp bin.
        Larger overshoots and other invalid intervals remain errors.
        """
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
        resolution = (
            self.aligner_processor.timestamp_segment_time / MILLISECONDS_PER_SECOND
        )
        clipped = []
        for index, item in enumerate(timings):
            start, end = item.get("start_time"), item.get("end_time")
            if start is not None and end is not None:
                if (
                    0 <= start <= end
                    and duration < end <= duration + resolution + TIMESTAMP_TOLERANCE
                ):
                    clipped.append(
                        {
                            "word_index": index,
                            "start": float(start),
                            "end": float(end),
                            "duration": duration,
                        }
                    )
                    start, end = min(start, duration), duration
                TranscriptionResult.validate_alignment_interval(
                    float(start), float(end), duration
                )
            tokens.append(
                AlignmentToken(
                    str(item["text"]),
                    float(start) + offset if start is not None else None,
                    float(end) + offset if end is not None else None,
                )
            )
        if clipped:
            if diagnostics is not None:
                diagnostics["alignment_endpoint_clamps"] = clipped
            self.log.warning(
                "Clipped %d alignment endpoints within one timestamp bin of the chunk end",
                len(clipped),
            )
        return tokens

    def transcribe(
        self, input_file: str | Path, context: TranscriptionContext
    ) -> TranscriptionResult:
        """Use the same batched inference path for a standalone recording."""
        result = self.transcribe_batch([(input_file, context)])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def transcribe_batch(
        self, recordings: list[tuple[str | Path, TranscriptionContext]]
    ) -> list[TranscriptionResult | Exception]:
        """Keep preparation and alignment failures local to their recording."""
        prepared = []
        requests: list[dict[str, Any]] = []
        results: list[TranscriptionResult | Exception] = [
            TranscriptionError("Recording was not processed", code="inference")
            for _ in recordings
        ]
        for index, (source, context) in enumerate(recordings):
            try:
                waveform = AudioChunker.decode(source)
                chunks = AudioChunker.ranges(
                    waveform, self.deployment.qwen_chunk_seconds
                )
                prompt = self._prompt(context)
                start = len(requests)
                requests.extend(
                    self._request(waveform[chunk.start : chunk.end], prompt)
                    for chunk in chunks
                )
                prepared.append((index, waveform, chunks, prompt, start, len(requests)))
            except Exception as exc:
                results[index] = exc
        if not requests:
            return results
        try:
            responses = self._generate(requests)
        except Exception as exc:
            self._classify_cuda_error(exc)
            raise TranscriptionError(exc, code="inference") from exc
        for index, waveform, chunks, prompt, start, end in prepared:
            try:
                results[index] = self._finish_recording(
                    waveform, chunks, prompt, responses[start:end]
                )
            except Exception as exc:
                if isinstance(exc, (TranscriptionError, ConfigurationError)):
                    results[index] = exc
                else:
                    try:
                        self._classify_cuda_error(exc)
                    except TranscriptionError as classified:
                        results[index] = classified
                    else:
                        results[index] = TranscriptionError(exc, code="inference")
        return results

    def _finish_recording(
        self,
        waveform: NDArray[np.float32],
        chunks: tuple[AudioChunk, ...],
        prompt: str,
        responses: list[Any],
    ) -> TranscriptionResult:
        """Retry disjoint windows once, then align only the accepted transcript."""
        builder = ResultBuilder()
        diagnostics: list[dict[str, Any]] = []
        index = 0
        while index < len(chunks):
            end_index = index + 1
            response = responses[index]
            retry = response.finish_reason == "length" or (
                end_index < len(chunks)
                and responses[end_index].finish_reason == "length"
            )
            if retry:
                end_index = min(index + 2, len(chunks))
            chunk = AudioChunk(chunks[index].start, chunks[end_index - 1].end)
            audio = waveform[chunk.start : chunk.end]
            details: dict[str, Any] = {
                "start_sample": chunk.start,
                "end_sample": chunk.end,
            }
            if retry:
                details["recognition_retry"] = {
                    "reason": "generation_incomplete",
                    "original_chunks": [
                        {
                            "start_sample": original.start,
                            "end_sample": original.end,
                            "generated_tokens": len(output.token_ids),
                            "finish_reason": output.finish_reason,
                        }
                        for original, output in zip(
                            chunks[index:end_index],
                            responses[index:end_index],
                            strict=True,
                        )
                    ],
                }
                self.log.warning(
                    "Retrying incomplete ASR with audio window %.2f–%.2fs",
                    chunk.start / QWEN_SAMPLE_RATE,
                    chunk.end / QWEN_SAMPLE_RATE,
                )
                response = self._generate([self._request(audio, prompt)])[0]
            if response.finish_reason != "stop":
                raise TranscriptionError(
                    "ASR generation did not complete within its token budget"
                    + (" after one retry" if retry else ""),
                    code="generation_incomplete",
                )
            language, text, detected = self.parse_transcript(response.text)
            details.update(
                generated_tokens=len(response.token_ids),
                normal_eos=True,
                detected_language=detected,
                suspected_repetition=bool(
                    re.search(r"\b(\w+)(?:\s+\1){4,}\b", text, re.IGNORECASE)
                ),
            )
            diagnostics.append(details)
            index = end_index
            if not text:
                continue
            if language is None:
                raise TranscriptionError(
                    "Nonempty text has no language", code="language"
                )
            offset = chunk.start / QWEN_SAMPLE_RATE
            tokens = self._align(audio, text, language, offset, details)
            if not any(
                token.start is not None and token.end is not None for token in tokens
            ):
                raise TranscriptionError(
                    "Nonempty ASR text has no timestamps", code="alignment"
                )
            builder.append(text, tokens, offset, chunk.end / QWEN_SAMPLE_RATE, language)
        return builder.build(
            len(waveform) / QWEN_SAMPLE_RATE,
            "qwen3",
            self.deployment.model("asr").repo_id,
            self.deployment.model("aligner").repo_id,
            {"chunks": diagnostics},
        )
