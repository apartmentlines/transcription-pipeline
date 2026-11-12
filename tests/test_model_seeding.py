"""Model seeding isolates backends and publishes only verified local assets."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from transcription_pipeline.constants import (
    QWEN_MODELS,
)
from transcription_pipeline.backends.qwen3_seed import Qwen3ModelSeeder
from transcription_pipeline.errors import ConfigurationError
from transcription_pipeline.model_seeding import ModelInstaller
from transcription_pipeline.seed_models import ModelSeedingCommand
from transcription_pipeline.subtitles import SubtitleRenderer


def test_qwen_seeding_uses_immutable_native_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot, publish = Mock(side_effect=[{"role": "asr"}, {"role": "aligner"}]), Mock()
    monkeypatch.setattr(ModelInstaller, "snapshot", snapshot)
    monkeypatch.setattr(ModelInstaller, "publish", publish)
    destination = tmp_path / "manifest.json"
    Qwen3ModelSeeder().seed(destination)
    assert [call.args for call in snapshot.call_args_list] == list(QWEN_MODELS.values())
    assert publish.call_args.args == (
        "qwen3",
        {"asr": {"role": "asr"}, "aligner": {"role": "aligner"}},
        [],
        destination,
    )
    assert all(len(revision) == 40 for _, revision in QWEN_MODELS.values())


def test_partial_download_never_publishes_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModelInstaller, "snapshot", Mock(side_effect=OSError("download failed"))
    )
    publish = Mock()
    monkeypatch.setattr(ModelInstaller, "publish", publish)
    with pytest.raises(OSError):
        Qwen3ModelSeeder().seed()
    publish.assert_not_called()


def test_invalid_backend_seeding_option_rejected() -> None:
    args = ModelSeedingCommand.parser().parse_args(
        ["--backend", "qwen3", "--whisperx-diarization"]
    )
    with pytest.raises(ValueError, match="requires --backend whisperx"):
        ModelSeedingCommand.run(args)


def test_verified_download_is_atomic_and_cacheable(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"model")
    expected = hashlib.sha256(b"model").hexdigest()
    assert (
        ModelInstaller.download_file(source.as_uri(), destination, expected)
        == destination
    )
    source.unlink()
    assert (
        ModelInstaller.download_file(source.as_uri(), destination, expected)
        == destination
    )


def test_checksum_failure_preserves_previous_file(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"wrong")
    destination.write_bytes(b"previous")
    with pytest.raises(ConfigurationError, match="checksum mismatch"):
        ModelInstaller.download_file(source.as_uri(), destination, "a" * 64)
    assert destination.read_bytes() == b"previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["destination", "source"]


def test_seeded_punkt_is_readable_with_nltk_permission_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nltk import data
    from transcription_pipeline import model_seeding

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for language in ("english", "spanish"):
            for name in (
                "abbrev_types.txt",
                "collocations.tab",
                "ortho_context.tab",
                "sent_starters.txt",
            ):
                archive.writestr(f"punkt_tab/{language}/{name}", "")
    contents = archive_bytes.getvalue()
    monkeypatch.setattr(
        model_seeding, "PUNKT_SHA256", hashlib.sha256(contents).hexdigest()
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", Mock(return_value=io.BytesIO(contents))
    )
    root = tmp_path / "nltk"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    monkeypatch.setenv("NLTK_DATA", str(root))
    monkeypatch.setattr(data, "path", data.path.copy())
    ModelInstaller.punkt()
    for path in (root, *root.rglob("*")):
        assert path.stat().st_mode & 0o022 == 0
    for language in ("en", "es"):
        tokenizer = SubtitleRenderer._tokenizer(language, str(root))
        assert list(tokenizer.span_tokenize("Hello. Goodbye.")) == [(0, 6), (7, 15)]
