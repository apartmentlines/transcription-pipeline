"""Opt-in real-recording checks; ordinary development requires no NVIDIA GPU."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from transcription_pipeline.backends.factory import BackendFactory
from transcription_pipeline.deployment import ManifestStore
from transcription_pipeline.errors import ConfigurationError
from transcription_pipeline.results import TranscriptionContext
from transcription_pipeline.subtitles import SubtitleRenderer

pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSCRIPTION_GPU_TESTS") != "1",
    reason="GPU inference checks are opt-in",
)


@pytest.fixture(scope="module")
def gpu_backend():
    """Skip clearly when CUDA or prepared model assets are unavailable."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("An NVIDIA CUDA GPU is required")
    try:
        deployment = ManifestStore.load()
    except ConfigurationError as exc:
        pytest.skip(f"Prepared model assets are unavailable: {exc}")
    return deployment, BackendFactory.create(deployment), torch


@pytest.mark.parametrize(
    "filename,language",
    [
        ("english.wav", "en"),
        ("spanish.wav", "es"),
        ("maintenance_en.wav", "en"),
        ("maintenance_es.wav", "es"),
        ("mixed.wav", None),
        ("long_silence.wav", None),
        ("over_180_seconds.wav", None),
        ("near_600_seconds.wav", None),
        ("chunk_boundaries.wav", None),
        ("whisperx_omission.wav", None),
    ],
)
def test_real_recording(
    gpu_backend: Any, filename: str, language: str | None, tmp_path: Path
) -> None:
    """Save inspectable subtitles and actual single-worker timing/memory observations."""
    directory = os.environ.get("TRANSCRIPTION_GPU_TEST_AUDIO_DIR")
    if not directory or not (Path(directory) / filename).is_file():
        pytest.skip(f"Recording fixture unavailable: {filename}")
    deployment, backend, torch = gpu_backend
    torch.cuda.reset_peak_memory_stats(deployment.device)
    torch.cuda.synchronize(deployment.device)
    started = time.perf_counter()
    result = backend.transcribe(Path(directory) / filename, TranscriptionContext())
    torch.cuda.synchronize(deployment.device)
    elapsed = time.perf_counter() - started
    rendered = SubtitleRenderer().render(result)
    if language is not None:
        assert language in result.detected_languages
    assert result.text and rendered
    parsed = importlib.import_module("pysubs2").SSAFile.from_string(
        rendered, format_="srt"
    )
    assert len(parsed) > 0
    target = Path(os.environ.get("TRANSCRIPTION_GPU_TEST_OUTPUT_DIR", str(tmp_path)))
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{filename}.srt").write_text(rendered, encoding="utf-8")
    report = {
        "backend": deployment.backend,
        "gpu": torch.cuda.get_device_name(deployment.device),
        "device": deployment.device,
        "compute_capability": torch.cuda.get_device_capability(deployment.device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(
            deployment.device
        ).total_memory,
        "precision": deployment.precision,
        "qwen_attention": deployment.qwen_attention,
        "workers": 1,
        "audio_seconds": result.audio_duration,
        "processing_seconds": elapsed,
        "real_time_factor": elapsed / result.audio_duration,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(deployment.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(deployment.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "metadata": result.metadata(len(parsed)),
    }
    (target / f"{filename}.metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
