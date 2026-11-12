"""Install native Qwen snapshots without loading GPU weights."""

from __future__ import annotations

from pathlib import Path

from transcription_pipeline.constants import (
    QWEN_MODELS,
)
from ..model_seeding import ModelInstaller


class Qwen3ModelSeeder:
    """Bake complete snapshots and shared language assets into one installation."""

    def seed(self, manifest_path: Path | None = None) -> Path:
        """Download pinned local assets, then atomically advertise their backend."""
        models = {
            role: ModelInstaller.snapshot(repo, revision)
            for role, (repo, revision) in QWEN_MODELS.items()
        }
        return ModelInstaller.publish("qwen3", models, [], manifest_path)
