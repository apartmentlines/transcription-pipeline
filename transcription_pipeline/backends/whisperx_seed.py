"""Bake pinned WhisperX, alignment, VAD, and optional diarization assets."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from transcription_pipeline.constants import (
    WHISPERX_ALIGNMENT,
    WHISPERX_ASR,
    WHISPERX_DIARIZATION,
    WHISPERX_VAD_REPO_ID,
)
from ..deployment import ManifestStore
from ..errors import ConfigurationError
from ..model_seeding import ModelInstaller


class WhisperXModelSeeder:
    """Install weights without constructing inference models or requiring CUDA."""

    def seed(
        self, manifest_path: Path | None = None, diarization: bool = False
    ) -> Path:
        """Install only this backend and publish a validated asset inventory."""
        models = {"asr": ModelInstaller.snapshot(*WHISPERX_ASR)}
        torch_root = ManifestStore.roots()["torch_home"]
        for language, (bundle, filename, checksum) in WHISPERX_ALIGNMENT.items():
            relative = f"hub/checkpoints/{filename}"
            ModelInstaller.download_file(
                f"https://download.pytorch.org/torchaudio/models/{filename}",
                torch_root / relative,
                checksum,
            )
            models[f"aligner_{language}"] = {
                "repo_id": bundle,
                "revision": checksum,
                "root": "torch_home",
                "path": relative,
            }
        specification = importlib.util.find_spec("whisperx")
        if specification is None or specification.origin is None:
            raise ConfigurationError(
                "Install the WhisperX dependency set before seeding"
            )
        packaged_vad = Path(specification.origin).parent / "assets/pytorch_model.bin"
        if not packaged_vad.is_file():
            raise ConfigurationError(
                f"WhisperX package lacks its VAD model: {packaged_vad}"
            )
        vad_path = torch_root / "whisperx/vad.bin"
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packaged_vad, vad_path)
        models["vad"] = {
            "repo_id": WHISPERX_VAD_REPO_ID,
            "revision": ModelInstaller.checksum(vad_path),
            "root": "torch_home",
            "path": "whisperx/vad.bin",
        }
        if diarization:
            models.update(
                {
                    role: ModelInstaller.snapshot(repo, revision)
                    for role, (repo, revision) in WHISPERX_DIARIZATION.items()
                }
            )
        return ModelInstaller.publish("whisperx", models, [], manifest_path)
