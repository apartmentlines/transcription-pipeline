"""Resolve and validate an offline, single-backend model installation."""

from __future__ import annotations


import importlib.util
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from transcription_pipeline.constants import (
    BACKENDS,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_PRECISION,
    DEFAULT_QWEN_ATTENTION,
    LANGUAGE_NAMES,
    MANIFEST_SCHEMA_VERSION,
    PROCESSING_LIMITS,
    PUNKT_PARAMETER_FILES,
    QWEN_ATTENTION_IMPLEMENTATIONS,
    WHISPERX_BATCH_SIZE,
    WHISPERX_NUM_SPEAKERS,
)
from .errors import ConfigurationError

BackendName = Literal["whisperx", "qwen3"]
QwenAttention = Literal["sdpa", "flash_attention_2"]


@dataclass(frozen=True)
class ModelAsset:
    """An immutable model identity and its resolved local asset."""

    role: str
    repo_id: str
    revision: str
    path: Path


@dataclass(frozen=True)
class Deployment:
    """Validated configuration passed unchanged to every worker."""

    manifest_path: Path
    backend: BackendName
    models: tuple[ModelAsset, ...]
    device: str = DEFAULT_INFERENCE_DEVICE
    precision: str = DEFAULT_INFERENCE_PRECISION
    whisperx_batch_size: int = WHISPERX_BATCH_SIZE
    whisperx_num_speakers: int = WHISPERX_NUM_SPEAKERS
    worker_count: int = 1
    qwen_attention: QwenAttention = DEFAULT_QWEN_ATTENTION

    def __post_init__(self) -> None:
        """Reject unsupported attention implementations before worker creation."""
        if self.qwen_attention not in QWEN_ATTENTION_IMPLEMENTATIONS:
            raise ConfigurationError(
                "TRANSCRIPTION_QWEN_ATTENTION must be sdpa or flash_attention_2"
            )

    def description(self) -> str:
        """Describe reproducible model identities without recording contents."""
        models = ", ".join(
            f"{model.role}={model.repo_id}@{model.revision}" for model in self.models
        )
        attention = (
            f" attention={self.qwen_attention}" if self.backend == "qwen3" else ""
        )
        return (
            f"manifest={self.manifest_path} backend={self.backend}{attention} {models}"
        )

    def model(self, role: str) -> ModelAsset:
        """Return a required model or report an installation error."""
        for model in self.models:
            if model.role == role:
                return model
        raise ConfigurationError(f"Manifest is missing model role {role!r}")

    def processing_limit(self, override: int | None = None) -> int:
        """Resolve concurrency once, with explicit positive overrides."""
        if override is None:
            return PROCESSING_LIMITS[self.backend]
        if type(override) is not int or override <= 0:
            raise ConfigurationError("processing_limit must be a positive integer")
        return override

    def assert_backend(self, expected: object) -> None:
        """Validate an explicitly supplied deployment assertion."""
        if not isinstance(expected, str) or expected not in BACKENDS:
            raise ConfigurationError(f"backend must be one of {', '.join(BACKENDS)}")
        if expected != self.backend:
            raise ConfigurationError(
                f"Requested backend {expected!r}, but {self.manifest_path} "
                + f"installs {self.backend!r}; select the matching image or manifest"
            )


class ManifestStore:
    """Discover, verify, and atomically publish deployment manifests."""

    @staticmethod
    def default_path() -> Path:
        """Use Hugging Face's configured application-data root."""
        from huggingface_hub.constants import HF_HOME

        return Path(HF_HOME) / "transcription-pipeline" / "manifest.json"

    @classmethod
    def resolve_path(cls, explicit: str | Path | None = None) -> Path:
        """Resolve CLI, environment, then the conventional default."""
        value = explicit or os.environ.get("TRANSCRIPTION_MODEL_MANIFEST")
        return Path(value).expanduser().absolute() if value else cls.default_path()

    @staticmethod
    def roots() -> dict[str, Path]:
        """Keep model, Torch, and language-data caches independently relocatable."""
        from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        nltk_root = os.environ.get("NLTK_DATA", str(Path(HF_HOME) / "nltk"))
        return {
            "hf_hub_cache": Path(HF_HUB_CACHE),
            "torch_home": Path(os.environ.get("TORCH_HOME", cache_home / "torch")),
            "nltk_data": Path(nltk_root.split(os.pathsep)[0]),
        }

    @classmethod
    def asset_path(cls, entry: dict[str, Any]) -> Path:
        """Resolve a root-relative asset and reject malformed or missing files."""
        roots = cls.roots()
        root_name, relative = entry.get("root"), entry.get("path")
        if not isinstance(root_name, str) or root_name not in roots:
            raise ConfigurationError(f"Unknown manifest asset root: {root_name!r}")
        if not isinstance(relative, str) or not relative:
            raise ConfigurationError("Manifest asset path must be a nonempty string")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigurationError(
                f"Manifest asset must be root-relative: {relative}"
            )
        resolved = roots[root_name] / path
        if not resolved.exists():
            raise ConfigurationError(
                f"Missing model asset: {resolved}; rerun seed-models"
            )
        if resolved.is_dir():
            files = list(resolved.rglob("*"))
            if not files or any(p.is_symlink() and not p.exists() for p in files):
                raise ConfigurationError(f"Incomplete model asset: {resolved}")
        return resolved

    @staticmethod
    def validate_dependencies(backend: BackendName) -> None:
        """Check packages without importing either heavy inference implementation."""
        required = ["torch", "numpy", "nltk", "pysubs2"]
        required += (
            ["whisperx"] if backend == "whisperx" else ["transformers", "accelerate"]
        )
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise ConfigurationError(
                f"Missing {backend} dependencies: {', '.join(missing)}; "
                + f"install transcription-pipeline[{backend}] or use its image"
            )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        backend: str | None = None,
        *,
        device: str | None = None,
        precision: str | None = None,
    ) -> Deployment:
        """Read and validate one manifest before work retrieval or GPU loading."""
        manifest_path = cls.resolve_path(path)
        device = device or os.environ.get(
            "TRANSCRIPTION_DEVICE", DEFAULT_INFERENCE_DEVICE
        )
        precision = precision or os.environ.get(
            "TRANSCRIPTION_PRECISION", DEFAULT_INFERENCE_PRECISION
        )
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            deployment = cls.validate(data, manifest_path, device, precision)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ConfigurationError(
                f"Invalid model manifest {manifest_path}: {exc}. "
                + "Prepare this installation with seed-models --backend BACKEND"
            ) from exc
        assertion = (
            backend if backend is not None else os.environ.get("TRANSCRIPTION_BACKEND")
        )
        if assertion is not None:
            deployment.assert_backend(assertion)
        if deployment.backend == "qwen3":
            deployment = replace(
                deployment,
                qwen_attention=cast(
                    QwenAttention,
                    os.environ.get(
                        "TRANSCRIPTION_QWEN_ATTENTION", DEFAULT_QWEN_ATTENTION
                    ),
                ),
            )
            if (
                deployment.qwen_attention == "flash_attention_2"
                and importlib.util.find_spec("flash_attn") is None
            ):
                raise ConfigurationError(
                    "FlashAttention 2 was selected but flash-attn is missing; use the "
                    + "Qwen runtime-flash-attention image target or install "
                    + "requirements-qwen3-flash-attention.txt. "
                    + "Set TRANSCRIPTION_QWEN_ATTENTION=sdpa to use PyTorch attention"
                )
        cls.validate_dependencies(deployment.backend)
        logging.getLogger(__name__).info(
            "Deployment %s: backend=%s models=%s",
            manifest_path,
            deployment.backend,
            ", ".join(f"{m.repo_id}@{m.revision}" for m in deployment.models),
        )
        return deployment

    @classmethod
    def validate(
        cls,
        data: object,
        manifest_path: Path,
        device: str = DEFAULT_INFERENCE_DEVICE,
        precision: str = DEFAULT_INFERENCE_PRECISION,
    ) -> Deployment:
        """Validate schema, required roles, immutable revisions, and asset inventory."""
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise ConfigurationError("Manifest must be an object with schema_version=1")
        if (
            data["schema_version"] != MANIFEST_SCHEMA_VERSION
            or data.get("backend") not in BACKENDS
        ):
            raise ConfigurationError("Unsupported manifest schema or backend")
        backend = cast(BackendName, data["backend"])
        runtime = "transformers" if backend == "qwen3" else "whisperx"
        if data.get("runtime") != runtime:
            raise ConfigurationError(f"Expected manifest runtime {runtime}")
        if precision not in ("bfloat16", "float16"):
            raise ConfigurationError("precision must be bfloat16 or float16")
        if not re.fullmatch(r"cuda(?::\d+)?", device):
            raise ConfigurationError("Inference device must be cuda or cuda:N")
        models = data.get("models")
        if not isinstance(models, dict):
            raise ConfigurationError("Manifest models must be an object")
        required = (
            {"asr", "aligner"}
            if backend == "qwen3"
            else {"asr", "aligner_en", "aligner_es", "vad"}
        )
        if backend == "whisperx" and any(
            role.startswith("diarizer") for role in models
        ):
            required |= {"diarizer", "diarizer_segmentation", "diarizer_embedding"}
        if not required.issubset(models):
            raise ConfigurationError(
                f"Missing required model roles: {sorted(required - models.keys())}"
            )
        resolved = []
        for role, entry in models.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("repo_id"), str):
                raise ConfigurationError(f"Invalid model identity: {role}")
            revision = entry.get("revision", "")
            if not isinstance(revision, str) or not re.fullmatch(
                r"[a-f0-9]{40,64}", revision
            ):
                raise ConfigurationError(
                    f"Model {role} requires an immutable revision or checksum"
                )
            asset = cls.asset_path(entry)
            cls.validate_model_files(backend, role, asset)
            resolved.append(ModelAsset(role, entry["repo_id"], revision, asset))
        cls.validate_language_assets(data.get("assets"))
        return Deployment(manifest_path, backend, tuple(resolved), device, precision)

    @classmethod
    def validate_language_assets(cls, assets: object) -> None:
        """Require both complete local Punkt parameter sets."""
        if not isinstance(assets, list):
            raise ConfigurationError("Manifest assets must be a list")
        asset_paths = set()
        for asset in assets:
            if not isinstance(asset, dict):
                raise ConfigurationError("Invalid manifest asset")
            asset_directory = cls.asset_path(asset)
            if asset.get("root") == "nltk_data":
                for filename in PUNKT_PARAMETER_FILES:
                    if not (asset_directory / filename).is_file():
                        raise ConfigurationError(
                            f"Incomplete Punkt asset: {asset_directory / filename}"
                        )
            asset_paths.add((asset.get("root"), asset.get("path")))
        for language in LANGUAGE_NAMES.values():
            if ("nltk_data", f"tokenizers/punkt_tab/{language}") not in asset_paths:
                raise ConfigurationError(f"Missing Punkt inventory for {language}")

    @staticmethod
    def validate_model_files(backend: BackendName, role: str, path: Path) -> None:
        """Require the pinned runtime's processor, tokenizer, and weight files."""
        required: tuple[str, ...] = ()
        if backend == "qwen3":
            required = (
                "config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
            if role == "asr":
                required += ("generation_config.json",)
        elif role == "asr":
            required = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
        elif role == "diarizer":
            required = ("config.yaml",)
        elif role.startswith("diarizer_"):
            required = ("pytorch_model.bin",)
        elif not path.is_file() or path.stat().st_size == 0:
            raise ConfigurationError(f"Missing or empty WhisperX model: {path}")
        for filename in required:
            asset = path / filename
            if not asset.is_file() or asset.stat().st_size == 0:
                raise ConfigurationError(f"Missing or empty model file: {asset}")

    @classmethod
    def write(cls, data: dict[str, Any], path: str | Path | None = None) -> Path:
        """Publish complete installations using an atomic same-directory rename."""
        destination = cls.resolve_path(path)
        cls.validate(data, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination
