"""Bound recording concurrency and serialize access to one Qwen model pair."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Lock, Thread
from time import monotonic
from typing import final

from ..results import TranscriptionContext, TranscriptionResult
from .qwen3 import Qwen3Transcriber

BATCH_WAIT_SECONDS = 0.05


@dataclass(frozen=True)
class RecordingRequest:
    """Keep a caller's context and completion attached to its recording."""

    source: str | Path
    context: TranscriptionContext
    future: Future[TranscriptionResult]


@final
class SharedQwenTranscriber:
    """Serve synchronous recording jobs through bounded native vLLM batches."""

    def __init__(self, backend: Qwen3Transcriber) -> None:
        self.backend = backend
        self.recording_limit = backend.deployment.worker_count
        self.queue: Queue[RecordingRequest | None] = Queue()
        self.slots = BoundedSemaphore(self.recording_limit)
        self.lock = Lock()
        self.closed = False
        self.thread = Thread(target=self._run, name="qwen-batches", daemon=True)
        self.thread.start()

    def transcribe(
        self, input_file: str | Path, context: TranscriptionContext
    ) -> TranscriptionResult:
        """Backpressure callers before loading audio and wait for their own result."""
        with self.slots:
            request = RecordingRequest(input_file, context, Future())
            with self.lock:
                if self.closed:
                    raise RuntimeError("Shared Qwen engine is closed")
                self.queue.put(request)
            return request.future.result()

    def _run(self) -> None:
        """Collect ready recordings with at most 50 ms of batch formation latency."""
        try:
            stopping = False
            while not stopping:
                request = self.queue.get()
                if request is None:
                    break
                batch = [request]
                deadline = monotonic() + BATCH_WAIT_SECONDS
                while len(batch) < self.recording_limit:
                    try:
                        request = self.queue.get(timeout=max(0, deadline - monotonic()))
                    except Empty:
                        break
                    if request is None:
                        stopping = True
                        break
                    batch.append(request)
                try:
                    results = self.backend.transcribe_batch(
                        [(request.source, request.context) for request in batch]
                    )
                    if len(results) != len(batch):
                        raise RuntimeError("Qwen recording result count mismatch")
                except Exception as exc:
                    results = [exc] * len(batch)
                for request, result in zip(batch, results, strict=True):
                    if isinstance(result, Exception):
                        request.future.set_exception(result)
                    else:
                        request.future.set_result(result)
        finally:
            self.backend.close()

    def close(self) -> None:
        """Drain accepted work, reject new submissions, and release the GPU owner."""
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join()
