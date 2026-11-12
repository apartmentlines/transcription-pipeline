"""Reproducible shared model and sentence-tokenizer asset installation."""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from transcription_pipeline.constants import (
    LANGUAGE_NAMES,
    MANIFEST_SCHEMA_VERSION,
    MODEL_DOWNLOAD_TIMEOUT,
    PUNKT_PARAMETER_FILES,
    PUNKT_REVISION,
    PUNKT_SHA256,
)
from .deployment import ManifestStore
from .errors import ConfigurationError


class ModelInstaller:
    """Download immutable assets only during explicitly requested seeding."""

    @staticmethod
    def snapshot(repo_id: str, revision: str) -> dict[str, str]:
        """Install a complete immutable Hub snapshot and return its manifest entry."""
        from huggingface_hub import snapshot_download

        root = ManifestStore.roots()["hf_hub_cache"]
        path = Path(snapshot_download(repo_id, revision=revision, cache_dir=root))
        return {
            "repo_id": repo_id,
            "revision": revision,
            "root": "hf_hub_cache",
            "path": str(path.relative_to(root)),
        }

    @staticmethod
    def checksum(path: Path) -> str:
        """Hash large files without loading weights into memory."""
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def download_file(cls, url: str, destination: Path, checksum: str) -> Path:
        """Atomically install and verify an asset with a pinned SHA-256."""
        if destination.is_file() and cls.checksum(destination) == checksum:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(
                    url, timeout=MODEL_DOWNLOAD_TIMEOUT
                ) as response:
                    shutil.copyfileobj(response, stream)
            if cls.checksum(temporary) != checksum:
                raise ConfigurationError(
                    f"Downloaded asset checksum mismatch: {destination.name}"
                )
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def punkt() -> list[dict[str, str]]:
        """Install only the English/Spanish parameters from the pinned NLTK artifact."""
        url = f"https://raw.githubusercontent.com/nltk/nltk_data/{PUNKT_REVISION}/packages/tokenizers/punkt_tab.zip"
        with urllib.request.urlopen(url, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != PUNKT_SHA256:
            raise ConfigurationError("Punkt download checksum mismatch")
        root = ManifestStore.roots()["nltk_data"]
        assets = []
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            for language in LANGUAGE_NAMES.values():
                relative = f"tokenizers/punkt_tab/{language}"
                directory = root / relative
                for parent in (root, root / "tokenizers", directory.parent, directory):
                    parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                    parent.chmod(parent.stat().st_mode & ~0o022)
                for name in PUNKT_PARAMETER_FILES:
                    destination = directory / name
                    destination.write_bytes(
                        archive.read(f"punkt_tab/{language}/{name}")
                    )
                    destination.chmod(destination.stat().st_mode & ~0o022)
                assets.append(
                    {
                        "root": "nltk_data",
                        "path": relative,
                        "revision": PUNKT_REVISION,
                        "archive_sha256": PUNKT_SHA256,
                    }
                )
        return assets

    @classmethod
    def publish(
        cls,
        backend: str,
        models: dict[str, Any],
        assets: list[dict[str, str]],
        path: Path | None,
    ) -> Path:
        """Validate installed files and publish a complete deployment atomically."""
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "backend": backend,
            "runtime": "transformers" if backend == "qwen3" else "whisperx",
            "models": models,
            "assets": assets + cls.punkt(),
        }
        destination = ManifestStore.write(manifest, path)
        logging.getLogger(__name__).info(
            "Installed %s manifest at %s", backend, destination
        )
        return destination
