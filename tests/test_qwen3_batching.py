"""Shared model ownership, recording routing, backpressure, and shutdown."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from typing import Any
from unittest.mock import Mock

import pytest

from transcription_pipeline.backends.factory import BackendFactory, BackendSession
from transcription_pipeline.backends.qwen3_batching import SharedQwenTranscriber
from transcription_pipeline.deployment import Deployment
from transcription_pipeline.observability import RunMetrics
from transcription_pipeline.results import TranscriptionContext
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)


def test_simultaneous_workers_share_one_engine_and_receive_their_own_results(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Mock(deployment=replace(deployment, backend="qwen3", worker_count=4))

    def process(
        requests: list[tuple[str, TranscriptionContext]],
    ) -> list[str | ValueError]:
        return [
            ValueError("bad recording") if source == "bad" else context.community
            for source, context in requests
        ]

    backend.transcribe_batch.side_effect = process
    factory = Mock(return_value=backend)
    monkeypatch.setattr(BackendFactory, "create", factory)
    metrics = RunMetrics()
    session = BackendSession(backend.deployment, metrics=metrics)
    worker_class = TranscriptionProcessor.bind(
        backend.deployment, backend_session=session
    )
    monkeypatch.setattr(
        "transcription_pipeline.backends.qwen3_batching.BATCH_WAIT_SECONDS", 5
    )
    barrier = Barrier(4)

    def submit(source: str):
        processor = worker_class()
        assert isinstance(processor, TranscriptionProcessor)
        worker = processor.transcriber
        assert worker is not None
        barrier.wait(timeout=5)
        return worker.transcribe(source, TranscriptionContext(source))

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                name: executor.submit(submit, name)
                for name in ("one", "bad", "three", "four")
            }
            for name, future in futures.items():
                if name == "bad":
                    with pytest.raises(ValueError, match="bad recording"):
                        future.result(timeout=5)
                else:
                    assert future.result(timeout=5) == name
        factory.assert_called_once()
        assert metrics.readiness()["ready_workers"] == 4
        assert metrics.readiness()["models_ready"] is True
        assert backend.transcribe_batch.call_count == 1
    finally:
        session.close()
    backend.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        session.worker()


def test_engine_failure_releases_waiters_and_close_is_idempotent(
    deployment: Deployment,
) -> None:
    backend = Mock(deployment=replace(deployment, worker_count=2))
    backend.transcribe_batch.side_effect = RuntimeError("engine failed")
    shared = SharedQwenTranscriber(backend)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(shared.transcribe, str(i), TranscriptionContext())
                for i in range(2)
            ]
            for future in futures:
                with pytest.raises(RuntimeError, match="engine failed"):
                    future.result(timeout=5)
    finally:
        shared.close()
        shared.close()
    backend.close.assert_called_once()
    assert not shared.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        shared.transcribe("late", TranscriptionContext())


def test_shared_initialization_failure_is_not_retried_per_worker(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Mock(side_effect=RuntimeError("model load failed"))
    monkeypatch.setattr(BackendFactory, "create", factory)
    session = BackendSession(replace(deployment, backend="qwen3"))
    for _ in range(4):
        with pytest.raises(RuntimeError, match="model load failed"):
            session.worker()
    session.close()
    factory.assert_called_once()


def test_close_waits_for_accepted_recording_before_releasing_models(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release, closing = Event(), Event(), Event()
    backend = Mock(deployment=deployment)

    def process(_requests: Any) -> list[str]:
        started.set()
        assert release.wait(timeout=5)
        return ["completed"]

    backend.transcribe_batch.side_effect = process
    shared = SharedQwenTranscriber(backend)
    original_put = shared.queue.put

    def enqueue(item: Any) -> None:
        original_put(item)
        if item is None:
            closing.set()

    monkeypatch.setattr(shared.queue, "put", enqueue)
    with ThreadPoolExecutor(max_workers=2) as executor:
        recording = executor.submit(shared.transcribe, "audio", TranscriptionContext())
        assert started.wait(timeout=5)
        shutdown = executor.submit(shared.close)
        try:
            assert closing.wait(timeout=5)
            assert not shutdown.done()
            backend.close.assert_not_called()
        finally:
            release.set()
        assert recording.result(timeout=5) == "completed"
        shutdown.result(timeout=5)
    backend.close.assert_called_once()


def test_whisperx_session_keeps_independent_worker_models(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [Mock(), Mock()]
    monkeypatch.setattr(BackendFactory, "create", Mock(side_effect=models))
    session = BackendSession(deployment)
    assert session.worker() is models[0]
    assert session.worker() is models[1]
    assert session.shared is None
    session.close()
