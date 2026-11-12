"""Deployment discovery, relocatable cache roots, and configuration failures."""

from __future__ import annotations

from unittest.mock import Mock
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from transcription_pipeline.deployment import Deployment
from transcription_pipeline.deployment import ManifestStore
from transcription_pipeline.errors import ConfigurationError


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    roots = {key: tmp_path / key for key in ("hf_hub_cache", "torch_home", "nltk_data")}
    monkeypatch.setattr(ManifestStore, "roots", staticmethod(lambda: roots))
    monkeypatch.setattr(ManifestStore, "validate_dependencies", Mock(return_value=None))
    data: dict[str, Any] = {
        "schema_version": 1,
        "backend": "qwen3",
        "runtime": "transformers",
        "models": {},
        "assets": [],
    }
    for role in ("asr", "aligner"):
        path = roots["hf_hub_cache"] / role
        path.mkdir(parents=True)
        for filename in (
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ):
            (path / filename).write_text("{}")
        data["models"][role] = {
            "repo_id": f"Qwen/{role}",
            "revision": "a" * 40,
            "root": "hf_hub_cache",
            "path": role,
        }
    for language in ("english", "spanish"):
        path = f"tokenizers/punkt_tab/{language}"
        directory = roots["nltk_data"] / path
        directory.mkdir(parents=True)
        for filename in (
            "abbrev_types.txt",
            "collocations.tab",
            "ortho_context.tab",
            "sent_starters.txt",
        ):
            (directory / filename).write_text("fixture")
        data["assets"].append({"root": "nltk_data", "path": path})
    destination = tmp_path / "manifest.json"
    ManifestStore.write(data, destination)
    return destination


def test_auto_discovery_and_default_limit(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIPTION_MODEL_MANIFEST", str(manifest))
    deployment = ManifestStore.load()
    assert deployment.backend == "qwen3"
    assert deployment.processing_limit() == 3
    assert deployment.processing_limit(2) == 2
    assert deployment.qwen_attention == "sdpa"
    assert deployment.model("asr").path.is_dir()


def test_explicit_assertion_overrides_environment(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "whisperx")
    assert ManifestStore.load(manifest, "qwen3").backend == "qwen3"
    with pytest.raises(ConfigurationError, match="Requested backend"):
        ManifestStore.load(manifest)


@pytest.mark.parametrize("backend", [None, "", 1, False, "other", "whisperx"])
def test_invalid_assertion(manifest: Path, backend: object) -> None:
    deployment = ManifestStore.load(manifest)
    with pytest.raises(ConfigurationError):
        deployment.assert_backend(backend)


@pytest.mark.parametrize("value", [0, -1, True, "3"])
def test_invalid_concurrency(manifest: Path, value: Any) -> None:
    with pytest.raises(ConfigurationError):
        ManifestStore.load(manifest).processing_limit(value)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"backend": "other"},
        {"runtime": "vllm"},
        {"models": {}},
        {"assets": []},
    ],
)
def test_malformed_manifest(manifest: Path, change: dict[str, Any]) -> None:
    data = json.loads(manifest.read_text())
    data.update(change)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="seed-models"):
        ManifestStore.load(manifest)


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "missing"])
def test_invalid_asset_path(manifest: Path, path: str) -> None:
    data = json.loads(manifest.read_text())
    data["models"]["asr"]["path"] = path
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError):
        ManifestStore.load(manifest)


def test_atomic_write_preserves_previous_manifest(manifest: Path) -> None:
    original = manifest.read_bytes()
    with pytest.raises(ConfigurationError):
        ManifestStore.write({"schema_version": 2}, manifest)
    assert manifest.read_bytes() == original


def test_manifest_path_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIPTION_MODEL_MANIFEST", str(tmp_path / "env"))
    assert ManifestStore.resolve_path(tmp_path / "cli") == tmp_path / "cli"
    assert ManifestStore.resolve_path() == tmp_path / "env"


def test_huggingface_root_conventions(tmp_path: Path) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HOME", "HF_HUB_CACHE")
    }
    env.update(XDG_CACHE_HOME=str(tmp_path), HF_HUB_CACHE=str(tmp_path / "independent"))
    code = "from transcription_pipeline.deployment import ManifestStore; print(ManifestStore.default_path()); print(ManifestStore.roots()['hf_hub_cache'])"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        str(tmp_path / "huggingface/transcription-pipeline/manifest.json"),
        str(tmp_path / "independent"),
    ]


def test_missing_manifest_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="seed-models --backend"):
        ManifestStore.load(tmp_path / "missing")


def test_missing_runtime_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def find_spec(name: str) -> object | None:
        return None if name == "transformers" else object()

    monkeypatch.setattr(
        "transcription_pipeline.deployment.importlib.util.find_spec", find_spec
    )
    with pytest.raises(
        ConfigurationError, match="Missing qwen3 dependencies: transformers"
    ):
        ManifestStore.validate_dependencies("qwen3")


def test_missing_individual_model_file_fails_before_gpu_initialization(
    manifest: Path,
) -> None:
    model = ManifestStore.load(manifest).model("asr")
    (model.path / "tokenizer.json").unlink()
    with pytest.raises(ConfigurationError, match="tokenizer.json"):
        ManifestStore.load(manifest)


def test_environment_precision_and_device_resolve_once(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cuda:1")
    monkeypatch.setenv("TRANSCRIPTION_PRECISION", "float16")
    deployment = ManifestStore.load(manifest)
    assert deployment.device == "cuda:1"
    assert deployment.precision == "float16"
    explicit = ManifestStore.load(manifest, device="cuda:0", precision="bfloat16")
    assert explicit.device == "cuda:0"
    assert explicit.precision == "bfloat16"


def test_environment_attention_is_resolved_once(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "transcription_pipeline.deployment.importlib.util.find_spec",
        Mock(return_value=object()),
    )
    monkeypatch.setenv("TRANSCRIPTION_QWEN_ATTENTION", "flash_attention_2")
    deployment = ManifestStore.load(manifest)
    monkeypatch.setenv("TRANSCRIPTION_QWEN_ATTENTION", "sdpa")
    assert deployment.qwen_attention == "flash_attention_2"
    assert "attention=flash_attention_2" in deployment.description()
    assert ManifestStore.load(manifest).qwen_attention == "sdpa"


@pytest.mark.parametrize("value", ["", "auto", "flash_attention_3", "typo"])
def test_invalid_attention_is_rejected_before_workers(
    manifest: Path, monkeypatch: pytest.MonkeyPatch, value: Any
):
    monkeypatch.setenv("TRANSCRIPTION_QWEN_ATTENTION", value)
    with pytest.raises(ConfigurationError, match="must be sdpa or flash_attention_2"):
        ManifestStore.load(manifest)


def test_missing_optional_attention_dependency_fails_at_startup(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TRANSCRIPTION_QWEN_ATTENTION", "flash_attention_2")
    monkeypatch.setattr(
        "transcription_pipeline.deployment.importlib.util.find_spec",
        Mock(return_value=None),
    )
    with pytest.raises(ConfigurationError, match="runtime-flash-attention"):
        ManifestStore.load(manifest)


def test_whisperx_ignores_qwen_attention_environment(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TRANSCRIPTION_QWEN_ATTENTION", "flash_attention_2")
    monkeypatch.setattr(Path, "read_text", Mock(return_value="{}"))
    monkeypatch.setattr(ManifestStore, "validate", Mock(return_value=deployment))
    monkeypatch.setattr(ManifestStore, "validate_dependencies", Mock(return_value=None))
    assert ManifestStore.load().qwen_attention == "sdpa"


def test_direct_deployment_rejects_unknown_attention(deployment: Deployment):
    with pytest.raises(ConfigurationError, match="must be sdpa or flash_attention_2"):
        replace(deployment, qwen_attention="typo")


def test_application_imports_without_either_inference_stack() -> None:
    code = "import sys; sys.modules.update(whisperx=None, transformers=None, torch=None); import transcription_pipeline; from transcription_pipeline.main import TranscriptionPipeline; from transcription_pipeline.rest_interface import TranscriptionRestInterface; from transcription_pipeline.processors.transcription_processor import TranscriptionProcessor"
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "module", ["main", "transcriber", "rest_interface", "seed_models"]
)
def test_cli_help_requires_no_manifest_or_models(module: str, tmp_path: Path) -> None:
    env = dict(os.environ, TRANSCRIPTION_MODEL_MANIFEST=str(tmp_path / "missing"))
    result = subprocess.run(
        [sys.executable, "-m", f"transcription_pipeline.{module}", "--help"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--backend" in result.stdout
    assert "--model-manifest" in result.stdout
