"""Protect timing boundaries, readiness and backend log capture."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from transcription_pipeline.observability import ResourceSampler, RunMetrics


def test_overlapping_recordings_do_not_inflate_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two overlapping recordings and a failure share one processing window."""
    now = Mock(return_value=100.0)
    monkeypatch.setattr("transcription_pipeline.observability.monotonic", now)
    monkeypatch.setattr(ResourceSampler, "start", Mock())
    metrics = RunMetrics()
    metrics.start()
    assert not metrics.readiness()["models_ready"]
    now.return_value = 110.0
    metrics.worker_ready()
    metrics.processing_started()
    now.return_value = 111.0
    metrics.processing_started()
    now.return_value = 115.0
    metrics.processing_finished(True, 60.0)
    now.return_value = 120.0
    metrics.processing_finished(True, 40.0)
    metrics.processing_finished(False, 0.0)
    now.return_value = 122.0
    metrics.set_phase("shutting_down")
    assert not metrics.readiness()["models_ready"]
    now.return_value = 125.0
    metrics.finish(False)
    summary = metrics.summary()
    assert summary["startup_seconds"] == 10
    assert summary["processing_wall_seconds"] == 10
    assert summary["drain_seconds"] == 2
    assert summary["shutdown_seconds"] == 3
    assert summary["total_seconds"] == 25
    assert summary["overhead_seconds"] == 15
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert summary["recordings_per_second_processing"] == 0.2
    assert summary["recordings_per_second_total"] == 0.08
    assert summary["audio_realtime_multiple"] == 10
    assert metrics.readiness()["phase"] == "failed"


def test_package_info_logs_reach_console_and_current_run_file_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reconfiguration must not drop INFO messages, duplicate them or retain old files."""
    package = logging.getLogger("transcription_pipeline")
    original_handlers, original_level, original_propagate = (
        package.handlers[:],
        package.level,
        package.propagate,
    )
    package.handlers = []
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    logger = logging.getLogger("transcription_pipeline.backends.qwen3")
    original_child_level = logger.level
    try:
        logger.setLevel(logging.INFO)
        monkeypatch.setenv("DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE", str(first))
        RunMetrics.configure_logging(False)
        logger.info("first ready")
        monkeypatch.setenv("DOWNLOAD_PIPELINE_PROCESSOR_LOG_FILE", str(second))
        RunMetrics.configure_logging(False)
        logger.info("second ready")
        output = capsys.readouterr().err
        assert output.count("first ready") == 1
        assert output.count("second ready") == 1
        assert "second ready" not in first.read_text()
        assert second.read_text().count("second ready") == 1
    finally:
        for handler in package.handlers:
            handler.close()
        package.handlers = original_handlers
        package.setLevel(original_level)
        package.propagate = original_propagate
        logger.setLevel(original_child_level)
