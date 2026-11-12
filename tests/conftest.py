"""Lightweight fixtures shared by backend and external-contract tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from download_pipeline_processor.file_data import FileData

from transcription_pipeline.deployment import Deployment, ModelAsset
from transcription_pipeline.results import AlignmentToken, ResultBuilder


@pytest.fixture
def deployment(tmp_path: Path) -> Deployment:
    """A model-free deployment for adapter and orchestration tests."""
    return Deployment(
        tmp_path / "manifest.json",
        "whisperx",
        tuple(
            ModelAsset(role, role, "a" * 40, tmp_path / role)
            for role in ("asr", "aligner_en", "aligner_es", "vad")
        ),
    )


@pytest.fixture
def mock_deployment(deployment: Deployment):
    """Bypass filesystem discovery only in orchestration unit tests."""
    with patch(
        "transcription_pipeline.deployment.ManifestStore.load", return_value=deployment
    ) as mocked:
        yield mocked


@pytest.fixture
def file_data() -> FileData:
    result = FileData(id="123", name="test_file", url="http://example.com/file")
    result.record_name = "Test Apartments"
    result.local_path = Path("/path/to/test_file")
    return result


@pytest.fixture
def mock_transcriber():
    with patch(
        "transcription_pipeline.processors.transcription_processor.BackendFactory.create"
    ) as mocked:
        yield mocked


@pytest.fixture
def sample_transcription_result():
    builder = ResultBuilder()
    builder.append(
        "Hello world",
        [AlignmentToken("Hello", 0, 1), AlignmentToken("world", 1, 2)],
        0,
        2,
        "en",
    )
    return builder.build(5, "whisperx", "whisper", "alignment")


@pytest.fixture
def sample_srt() -> str:
    return "1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n"


@pytest.fixture
def sentence_tokenizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Load real Punkt with small local parameters and no downloaded assets."""
    from nltk import data

    root = tmp_path / "nltk"
    for language in ("english", "spanish"):
        directory = root / "tokenizers" / "punkt_tab" / language
        for parent in (root, root / "tokenizers", directory.parent, directory):
            parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        for name in (
            "abbrev_types.txt",
            "collocations.tab",
            "ortho_context.tab",
            "sent_starters.txt",
        ):
            destination = directory / name
            destination.write_text("", encoding="utf-8")
            destination.chmod(0o644)
    monkeypatch.setenv("NLTK_DATA", str(root))
    monkeypatch.setattr(data, "path", data.path.copy())
