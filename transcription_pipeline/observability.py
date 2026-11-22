"""Run-local readiness, wall-clock measurements and sampled resource usage."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, final

from download_pipeline_processor.logger import Logger


@final
class RunMetrics:
    """Measure overlapping workers without summing their elapsed times."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.phase = "starting"
        self.models_ready = False
        self.ready_workers = 0
        self.started: float | None = None
        self.ready_at: float | None = None
        self.first_processing: float | None = None
        self.last_processing: float | None = None
        self.shutdown_at: float | None = None
        self.finished: float | None = None
        self.succeeded = 0
        self.failed = 0
        self.audio_seconds = 0.0
        self.resources = ResourceSampler()
        self.log = logging.getLogger(__name__)

    @staticmethod
    def configure_logging(debug: bool) -> None:
        """Route package logs to stdout/stderr and the framework's run log once."""
        logger = logging.getLogger("transcription_pipeline")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        Logger("transcription_pipeline", debug=debug)

    def start(self) -> None:
        """Start before retrieval and dependency/model initialization."""
        self.started = monotonic()
        self.set_phase("initializing")
        self.resources.start()

    def set_phase(self, phase: str) -> None:
        """Publish a lifecycle transition without changing REST status semantics."""
        with self.lock:
            self.phase = phase
            if phase == "shutting_down":
                self.shutdown_at = monotonic()
            if phase in {"shutting_down", "complete", "failed"}:
                self.models_ready = False
        self.log.info("Pipeline phase=%s", phase)

    def worker_ready(self) -> None:
        """Report the first usable backend and count initialized worker handles."""
        with self.lock:
            self.ready_workers += 1
            self.models_ready = True
            if self.ready_at is None:
                self.ready_at = monotonic()
                self.phase = "ready"
            count = self.ready_workers
        self.log.info("Models ready: initialized_recording_workers=%d", count)

    def processing_started(self) -> None:
        """Mark the first call after worker initialization, excluding model load."""
        with self.lock:
            if self.first_processing is None:
                self.first_processing = monotonic()
                self.log.info("Transcription processing started")
            self.phase = "processing"

    def processing_finished(self, success: bool, audio_seconds: float) -> None:
        """Include failures in the window, but count throughput from successes."""
        with self.lock:
            self.last_processing = monotonic()
            self.succeeded += int(success)
            self.failed += int(not success)
            self.audio_seconds += audio_seconds

    def readiness(self) -> dict[str, Any]:
        """Return a consistent snapshot for the REST status endpoint."""
        with self.lock:
            return {
                "phase": self.phase,
                "models_ready": self.models_ready,
                "ready_workers": self.ready_workers,
            }

    def finish(self, success: bool) -> None:
        """Finish after backend teardown and log explicitly bounded measurements."""
        self.resources.close()
        self.finished = monotonic()
        self.set_phase("complete" if success else "failed")
        self.log.info(
            "Pipeline measurements: %s", json.dumps(self.summary(), sort_keys=True)
        )

    def summary(self) -> dict[str, Any]:
        """Return seconds/rates; processing spans first call to last result."""
        end = self.finished if self.finished is not None else monotonic()
        total = end - self.started if self.started is not None else 0.0
        processing = (
            self.last_processing - self.first_processing
            if self.last_processing is not None and self.first_processing is not None
            else 0.0
        )
        return {
            "total_seconds": total,
            "startup_seconds": (
                self.first_processing - self.started
                if self.first_processing is not None and self.started is not None
                else None
            ),
            "model_ready_seconds": (
                self.ready_at - self.started
                if self.ready_at is not None and self.started is not None
                else None
            ),
            "processing_wall_seconds": processing,
            "drain_seconds": (
                self.shutdown_at - self.last_processing
                if self.shutdown_at is not None and self.last_processing is not None
                else None
            ),
            "shutdown_seconds": (
                end - self.shutdown_at if self.shutdown_at is not None else None
            ),
            "overhead_seconds": total - processing,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "successful_audio_seconds": self.audio_seconds,
            "recordings_per_second_processing": (
                self.succeeded / processing if processing else None
            ),
            "recordings_per_second_total": self.succeeded / total if total else None,
            "audio_realtime_multiple": (
                self.audio_seconds / processing if processing else None
            ),
            "sampled_peak_process_tree_rss_mib": self.resources.peak_rss_mib,
            "sampled_peak_device_memory_mib": self.resources.gpu_peaks,
            "resource_sample_interval_seconds": 1,
            "memory_scope": "GPU usage is device-wide; process-tree RSS may double-count shared pages",
            "timing_scope": "pipeline.run through backend teardown; excludes HTTP wait, callback, container provisioning and pod stop",
        }


@final
class ResourceSampler:
    """Sample Linux process-tree RSS and device-wide GPU usage without CUDA imports."""

    def __init__(self) -> None:
        self.stop = Event()
        self.thread: Thread | None = None
        self.peak_rss_mib: float | None = None
        self.gpu_peaks: dict[str, float] = {}
        self.log = logging.getLogger(__name__)

    def start(self) -> None:
        """Gather resources off the processing path once per second."""
        self.thread = Thread(target=self._run, name="resource-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        """Log hardware once; missing metrics never prevent transcription."""
        try:
            model = next(
                (
                    line.partition(":")[2].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            )
            self.log.info(
                "Host CPU=%s count=%s available_cpu_count=%s",
                model,
                os.cpu_count(),
                len(os.sched_getaffinity(0)),
            )
        except (OSError, AttributeError):
            self.log.info("Host CPU count=%s", os.cpu_count())
        first = True
        while not self.stop.is_set():
            try:
                rss = self._process_tree_rss()
                self.peak_rss_mib = max(self.peak_rss_mib or 0.0, rss)
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,name,driver_version,memory.total,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                ).stdout
                if first:
                    self.log.info(
                        "GPU hardware (uuid,name,driver,total_mib,used_mib): %s",
                        output.strip(),
                    )
                for row in output.splitlines():
                    uuid, _name, _driver, _total, used = row.split(", ")
                    self.gpu_peaks[uuid] = max(
                        self.gpu_peaks.get(uuid, 0.0), float(used)
                    )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                if first:
                    self.log.warning("Resource sampling partially unavailable: %s", exc)
            first = False
            self.stop.wait(1)

    @staticmethod
    def _process_tree_rss() -> float:
        """Sum live descendant RSS; shared pages may be counted more than once."""
        processes: dict[int, tuple[int, int]] = {}
        for path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1) for line in path.read_text().splitlines()
                )
                processes[int(path.parent.name)] = (
                    int(fields["PPid"]),
                    int(fields.get("VmRSS", "0 kB").split()[0]),
                )
            except (OSError, KeyError, ValueError):
                continue
        descendants = {os.getpid()}
        while True:
            children = {
                pid for pid, (parent, _) in processes.items() if parent in descendants
            }
            if children <= descendants:
                break
            descendants |= children
        return (
            sum(rss for pid, (_, rss) in processes.items() if pid in descendants) / 1024
        )

    def close(self) -> None:
        """Stop sampling after engine shutdown."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
