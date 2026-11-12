"""Native Qwen interfaces and complete sample coverage without model downloads."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import os
from unittest.mock import MagicMock, Mock
from typing import Any

import numpy as np
import pytest

from transcription_pipeline.constants import (
    QWEN_GENERATION_BUDGET,
    QWEN_SAMPLE_RATE,
)
from transcription_pipeline.deployment import Deployment
from transcription_pipeline.backends.qwen3 import AudioChunker, Qwen3Transcriber
from transcription_pipeline.deployment import ModelAsset
from transcription_pipeline.errors import ConfigurationError, TranscriptionError
from transcription_pipeline.results import AlignmentToken, TranscriptionContext


@pytest.fixture
def qwen(deployment: Deployment, monkeypatch: pytest.MonkeyPatch):
    config = replace(
        deployment,
        backend="qwen3",
        models=tuple(
            ModelAsset(
                role, f"Qwen/{role}", "a" * 40, deployment.manifest_path.parent / role
            )
            for role in ("asr", "aligner")
        ),
    )
    torch = Mock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_capability.return_value = (8, 6)
    torch.inference_mode.side_effect = nullcontext
    transformers = Mock()
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    monkeypatch.setattr(
        Qwen3Transcriber, "_dependencies", staticmethod(lambda: (torch, transformers))
    )
    backend = Qwen3Transcriber(config)
    backend.asr_model.config.text_config.max_position_embeddings = 32768
    backend.asr_model.generation_config.eos_token_id = [99, 100]
    return backend, torch, transformers


def test_models_load_locally_once_with_native_defaults(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, torch, transformers = qwen
    for loader in (
        transformers.AutoModelForMultimodalLM,
        transformers.AutoModelForTokenClassification,
    ):
        assert loader.from_pretrained.call_count == 1
        kwargs = loader.from_pretrained.call_args.kwargs
        assert kwargs == {
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
            "attn_implementation": "sdpa",
            "local_files_only": True,
        }
    assert transformers.AutoProcessor.from_pretrained.call_count == 2
    assert (
        backend.asr_model
        is transformers.AutoModelForMultimodalLM.from_pretrained.return_value.eval.return_value
    )


@pytest.mark.parametrize(
    "capability",
    [
        pytest.param((8, 6), id="rtx-3090-a6000-a40"),
        pytest.param((8, 9), id="rtx-4090"),
        pytest.param((9, 0), id="hopper"),
        pytest.param((10, 0), id="blackwell-datacenter"),
        pytest.param((12, 0), id="rtx-pro-4000-4500-6000-blackwell-including-mig"),
    ],
)
@pytest.mark.parametrize("precision", ["bfloat16", "float16"])
def test_flash_attention_loads_both_models_on_ampere_and_newer(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    precision: str,
) -> None:
    backend, torch, transformers = qwen
    torch.cuda.get_device_capability.return_value = capability
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    optional_import = Mock()
    monkeypatch.setattr(
        "transcription_pipeline.backends.qwen3.importlib.import_module", optional_import
    )
    Qwen3Transcriber(
        replace(
            backend.deployment,
            precision=precision,
            qwen_attention="flash_attention_2",
        )
    )
    optional_import.assert_called_once_with("flash_attn")
    for loader in (
        transformers.AutoModelForMultimodalLM,
        transformers.AutoModelForTokenClassification,
    ):
        kwargs = loader.from_pretrained.call_args.kwargs
        assert kwargs["attn_implementation"] == "flash_attention_2"
        assert kwargs["dtype"] is getattr(torch, precision)


@pytest.mark.parametrize("attention", ["sdpa", "flash_attention_2"])
@pytest.mark.parametrize(
    "visible_devices,device",
    [
        pytest.param("MIG-fixture-uuid", "cuda:0", id="isolated-mig-device"),
        pytest.param("GPU-first,GPU-second", "cuda:1", id="selected-second-device"),
    ],
)
def test_attention_respects_visible_cuda_device_selection(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    attention: str,
    visible_devices: str,
    device: str,
) -> None:
    """Keep CUDA's logical ordinals, including the assigned MIG slice, intact."""
    backend, torch, transformers = qwen
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    torch.cuda.get_device_capability.reset_mock()
    torch.cuda.get_device_capability.return_value = (12, 0)
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    monkeypatch.setattr(
        "transcription_pipeline.backends.qwen3.importlib.import_module", Mock()
    )
    Qwen3Transcriber(
        replace(backend.deployment, device=device, qwen_attention=attention)
    )
    torch.cuda.get_device_capability.assert_called_once_with(device)
    for loader in (
        transformers.AutoModelForMultimodalLM,
        transformers.AutoModelForTokenClassification,
    ):
        assert loader.from_pretrained.call_args.kwargs["device_map"] == device
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible_devices


def test_flash_attention_rejects_older_gpu_even_with_fp16(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, torch, transformers = qwen
    torch.cuda.get_device_capability.return_value = (7, 5)
    transformers.AutoModelForMultimodalLM.from_pretrained.reset_mock()
    with pytest.raises(ConfigurationError, match="FlashAttention 2 require Ampere"):
        Qwen3Transcriber(
            replace(
                backend.deployment,
                precision="float16",
                qwen_attention="flash_attention_2",
            )
        )
    transformers.AutoModelForMultimodalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    "error", [ImportError("missing"), OSError("undefined symbol"), RuntimeError("ABI")]
)
def test_broken_flash_attention_fails_with_actionable_error(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    backend, _, transformers = qwen
    transformers.AutoModelForMultimodalLM.from_pretrained.reset_mock()
    monkeypatch.setattr(
        "transcription_pipeline.backends.qwen3.importlib.import_module",
        Mock(side_effect=error),
    )
    with pytest.raises(ConfigurationError, match="TRANSCRIPTION_QWEN_ATTENTION=sdpa"):
        Qwen3Transcriber(
            replace(backend.deployment, qwen_attention="flash_attention_2")
        )
    transformers.AutoModelForMultimodalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    "available,capability,message", [(False, (8, 6), "CUDA"), (True, (7, 0), "BF16")]
)
def test_device_requirements(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
    available: bool,
    capability: tuple[int, int],
    message: str,
) -> None:
    backend, torch, _ = qwen
    torch.cuda.is_available.return_value = available
    torch.cuda.get_device_capability.return_value = capability
    with pytest.raises(ConfigurationError, match=message):
        Qwen3Transcriber(backend.deployment)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "language English<asr_text>Hello, hello hello hello hello!",
            ("en", "Hello, hello hello hello hello!", "English"),
        ),
        (
            "language Spanish<asr_text>¿Dónde está la llave?",
            ("es", "¿Dónde está la llave?", "Spanish"),
        ),
        ("language None<asr_text>", (None, "", "None")),
        ("", (None, "", "")),
    ],
)
def test_raw_protocol_preserves_spoken_text(raw: str, expected: Any) -> None:
    assert Qwen3Transcriber.parse_transcript(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "language French<asr_text>Bonjour",
        "language None<asr_text>Hello",
        "Hello without header",
    ],
)
def test_unknown_language_and_protocol_fail(raw: str) -> None:
    with pytest.raises(TranscriptionError):
        Qwen3Transcriber.parse_transcript(raw)


@pytest.mark.parametrize(
    "tokens", [[7, 99], [7, 100], [7] * (QWEN_GENERATION_BUDGET - 1) + [99]]
)
def test_natural_eos_is_accepted_including_final_budget_token(
    tokens: list[int],
) -> None:
    Qwen3Transcriber.check_termination(tokens, [99, 100])


@pytest.mark.parametrize("tokens", [[], [7], [7] * QWEN_GENERATION_BUDGET])
def test_incomplete_generation_is_rejected(tokens: list[int]) -> None:
    with pytest.raises(TranscriptionError, match="without EOS"):
        Qwen3Transcriber.check_termination(tokens, [99, 100])


@pytest.mark.parametrize("seconds", [0.1, 5, 180, 180.01, 360, 599.999, 600])
def test_chunker_covers_every_sample_once(seconds: float) -> None:
    waveform = np.zeros(int(seconds * QWEN_SAMPLE_RATE), dtype=np.float32)
    ranges = AudioChunker.ranges(waveform)
    assert ranges[0].start == 0
    assert ranges[-1].end == len(waveform)
    assert all(0 < part.end - part.start <= 180 * QWEN_SAMPLE_RATE for part in ranges)
    assert all(left.end == right.start for left, right in zip(ranges, ranges[1:]))
    assert sum(part.end - part.start for part in ranges) == len(waveform)
    assert AudioChunker.ranges(waveform) == ranges


def test_chunker_selects_low_energy_without_discarding_audio() -> None:
    waveform = np.ones(181 * QWEN_SAMPLE_RATE, dtype=np.float32)
    waveform[177 * QWEN_SAMPLE_RATE : 177 * QWEN_SAMPLE_RATE + 320] = 0
    ranges = AudioChunker.ranges(waveform)
    assert ranges[0].end == 177 * QWEN_SAMPLE_RATE + 160
    np.testing.assert_array_equal(
        np.concatenate([waveform[c.start : c.end] for c in ranges]), waveform
    )


def test_generation_excludes_prompt_checks_coverage_and_preserves_raw_text(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, _, _ = qwen
    inputs = MagicMock()
    inputs.__getitem__.return_value.shape = (1, 10)
    inputs.get.return_value.sum.return_value.item.return_value = 500
    inputs.to.return_value = inputs
    backend.asr_processor.feature_extractor.hop_length = 160
    backend.asr_processor.apply_transcription_request.return_value = inputs
    generated = MagicMock()
    generated.__getitem__.return_value.detach.return_value.cpu.return_value.tolist.return_value = [
        5,
        99,
    ]
    backend.asr_model.generate.return_value = generated
    backend.asr_processor.decode.return_value = "language English<asr_text>Hi!"
    language, text, details = backend._asr(np.zeros(80000, dtype=np.float32), "context")
    assert (language, text) == ("en", "Hi!")
    generated.__getitem__.assert_called_once_with((0, slice(10, None)))
    assert details["normal_eos"] is True
    assert backend.asr_model.generate.call_args.kwargs["forced_eos_token_id"] is None
    assert backend.asr_model.generate.call_args.kwargs["use_cache"] is True
    assert (
        backend.asr_processor.apply_transcription_request.call_args.kwargs["language"]
        is None
    )
    assert (
        backend.asr_processor.apply_transcription_request.call_args.kwargs[
            "processor_kwargs"
        ]["audio_kwargs"]["truncation"]
        is False
    )
    backend.asr_processor.decode.assert_called_once_with(
        [5],
        return_format="raw",
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def test_feature_truncation_is_rejected() -> None:
    inputs, processor = Mock(), Mock()
    inputs.get.return_value.sum.return_value.item.return_value = 100
    processor.feature_extractor.hop_length = 160
    with pytest.raises(TranscriptionError, match="truncated audio"):
        Qwen3Transcriber._validate_features(inputs, processor, 80000)


def test_mixed_language_chunk_offsets_and_empty_chunk_coverage(
    qwen: tuple[Qwen3Transcriber, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, transformers = qwen
    waveform = np.zeros(400 * QWEN_SAMPLE_RATE, dtype=np.float32)
    chunks = AudioChunker.ranges(waveform)
    monkeypatch.setattr(AudioChunker, "decode", Mock(return_value=waveform))
    backend._asr = Mock(
        side_effect=[("en", "Hello", {}), (None, "", {}), ("es", "¡Hola!", {})]
    )

    def align(
        _audio: Any, text: str, _language: str, offset: float
    ) -> list[AlignmentToken]:
        return [AlignmentToken(text, offset + 1, offset + 2)]

    backend._align = Mock(side_effect=align)
    result = backend.transcribe("unused.wav", TranscriptionContext())
    assert result.text == "Hello ¡Hola!"
    assert result.detected_languages == ("en", "es")
    assert result.words[1].start == chunks[2].start / QWEN_SAMPLE_RATE + 1
    assert len(result.diagnostics["chunks"]) == 3
    assert backend._align.call_count == 2
    assert transformers.AutoModelForMultimodalLM.from_pretrained.call_count == 1
    assert result.metadata(2)["average_word_confidence"] is None


def test_entirely_empty_recording_fails(
    qwen: tuple[Qwen3Transcriber, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _ = qwen
    monkeypatch.setattr(
        AudioChunker,
        "decode",
        Mock(return_value=np.zeros(80000, dtype=np.float32)),
    )
    backend._asr = Mock(return_value=(None, "", {}))
    with pytest.raises(TranscriptionError, match="Empty transcription"):
        backend.transcribe("unused.wav", TranscriptionContext())


def test_aligner_api_and_timestamp_offset(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, _, _ = qwen
    inputs = MagicMock()
    inputs.get.return_value.sum.return_value.item.return_value = 500
    inputs.to.return_value = inputs
    backend.aligner_processor.feature_extractor.hop_length = 160
    backend.aligner_processor.prepare_forced_aligner_inputs.return_value = (
        inputs,
        [["hola"]],
    )
    backend.aligner_processor.decode_forced_alignment.return_value = [
        [{"text": "hola", "start_time": 1.25, "end_time": 2.5}]
    ]
    tokens = backend._align(np.zeros(80000, dtype=np.float32), "¡Hola!", "es", 179.99)
    assert backend.aligner_model.call_args.kwargs["use_cache"] is False
    assert tokens[0].start == pytest.approx(181.24)
    assert tokens[0].end == pytest.approx(182.49)
    kwargs = backend.aligner_processor.prepare_forced_aligner_inputs.call_args.kwargs
    assert kwargs["transcript"] == "¡Hola!"
    assert kwargs["language"] == "es"
    assert kwargs["processor_kwargs"]["audio_kwargs"] == {
        "sampling_rate": 16000,
        "truncation": False,
    }
    decode = backend.aligner_processor.decode_forced_alignment.call_args.kwargs
    assert decode["word_lists"] == [["hola"]]
    assert (
        decode["timestamp_token_id"] is backend.aligner_model.config.timestamp_token_id
    )


def test_qwen_reuses_pair_across_recordings(
    qwen: tuple[Qwen3Transcriber, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, transformers = qwen
    monkeypatch.setattr(
        AudioChunker,
        "decode",
        Mock(return_value=np.zeros(80000, dtype=np.float32)),
    )
    backend._asr = Mock(return_value=("en", "Hello", {}))
    backend._align = Mock(return_value=[AlignmentToken("hello", 1, 2)])
    first = backend.transcribe("first.wav", TranscriptionContext())
    second = backend.transcribe("second.wav", TranscriptionContext())
    assert first.text == second.text == "Hello"
    assert backend._asr.call_count == 2
    assert transformers.AutoModelForMultimodalLM.from_pretrained.call_count == 1
    assert transformers.AutoModelForTokenClassification.from_pretrained.call_count == 1


def test_fp16_is_an_explicit_configuration(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, torch, transformers = qwen
    torch.cuda.get_device_capability.return_value = (7, 0)
    transformers.AutoProcessor.from_pretrained.side_effect = [Mock(), Mock()]
    Qwen3Transcriber(replace(backend.deployment, precision="float16"))
    assert (
        transformers.AutoModelForMultimodalLM.from_pretrained.call_args.kwargs["dtype"]
        is torch.float16
    )


def test_oom_reports_actual_worker_count(
    qwen: tuple[Qwen3Transcriber, Mock, Mock],
) -> None:
    backend, _, _ = qwen
    backend.deployment = replace(backend.deployment, worker_count=3)
    with pytest.raises(
        TranscriptionError, match="3 configured workers; lower --processing-limit"
    ) as caught:
        backend._classify_cuda_error(RuntimeError("CUDA out of memory"))
    assert caught.value.code == "cuda_oom"
    assert caught.value.is_transient_error()
