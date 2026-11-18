"""Retain outputs or collect new outputs through the production transcriber CLI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import time

from reference_core import Corpus, Record, Records


class Runs:
    """Freeze each run, including missing outputs, without modifying production."""

    @staticmethod
    def retain(
        corpus: Corpus,
        source: Path,
        output: Path,
        backend: str,
        provenance: Record,
        selected: list[str] | None = None,
    ) -> Record:
        """Freeze outputs and missing or failed cases in a new run directory.

        :param corpus: Source audio identities associated with the run.
        :param source: Directory of SRTs named by case identity.
        :param output: New directory for copied artifacts and the run manifest.
        :param backend: Backend that produced the outputs.
        :param provenance: Original run settings and known association limits.
        :param selected: Requested cases, or all corpus cases when omitted.
        :return: Sealed manifest, including cases without output.
        :raises ValueError: If selected identities are unknown or duplicated.
        """
        identities = selected if selected is not None else list(corpus.cases)
        Records.require(
            bool(identities) and set(identities) <= set(corpus.cases),
            "Unknown run selection",
        )
        Records.require(len(set(identities)) == len(identities), "Duplicate run case")
        output.mkdir(parents=True, exist_ok=False)
        artifacts = output / "outputs"
        artifacts.mkdir()
        cases = []
        for identity in identities:
            entry = {
                "case_id": identity,
                "audio_sha256": corpus.cases[identity]["audio"]["sha256"],
                "status": "failed" if provenance.get("returncode", 0) else "missing",
                "srt": None,
            }
            original = source / f"{identity}.srt"
            if original.is_file():
                destination = artifacts / original.name
                shutil.copyfile(original, destination)
                entry["status"] = "success"
                entry["srt"] = {
                    "path": destination.relative_to(output).as_posix(),
                    "sha256": Records.file_digest(destination),
                }
                for suffix in (".txt", ".diagnostics.json"):
                    extra = original.with_suffix(suffix)
                    if extra.exists():
                        shutil.copyfile(extra, artifacts / extra.name)
            cases.append(entry)
        run = Records.seal(
            {
                "schema_version": 1,
                "created_at": Records.now(),
                "corpus_id": corpus.data["corpus_id"],
                "collection_corpus_sha256": corpus.data["content_sha256"],
                "backend": backend,
                "provenance": provenance,
                "cases": cases,
            }
        )
        Records.write(output / "run.json", run)
        return run

    @staticmethod
    def command(
        corpus: Corpus,
        root: Path,
        destination: Path,
        backend: str,
        image: str | None,
        manifest: Path | None,
        processing_limit: int,
        batch_size: int | None,
        chunk_seconds: int | None,
        selected: list[str],
    ) -> list[str]:
        """Construct arguments for the real local or Docker transcriber CLI.

        :param corpus: Reference audio paths and stable case identities.
        :param root: Repository containing those paths.
        :param destination: Writable directory for transcriber outputs.
        :param backend: Qwen or WhisperX backend name.
        :param image: Bundled Docker image, or None for the local executable.
        :param manifest: Optional model manifest beneath the repository root.
        :param processing_limit: Maximum simultaneously processed recordings.
        :param batch_size: Optional Qwen chunk batch limit.
        :param chunk_seconds: Optional Qwen ASR context duration.
        :param selected: Case identities passed to this invocation.
        :return: Arguments suitable for subprocess execution without a shell.
        :raises ValueError: If configuration or selection is invalid.
        """
        Records.require(backend in ("qwen3", "whisperx"), "Unknown backend")
        Records.require(processing_limit > 0, "Processing limit must be positive")
        Records.require(
            set(selected) <= set(corpus.cases) and bool(selected),
            "Unknown selected recording",
        )
        prefix = ["transcriber"]
        output = str(destination.resolve())
        audio_paths = [
            str(Records.within(root, corpus.cases[key]["audio"]["path"]))
            for key in selected
        ]
        manifest_path = str(manifest.resolve()) if manifest else None
        if image:
            prefix = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--ipc",
                "host",
                "--network",
                "none",
                "-v",
                f"{root.resolve()}:/inputs:ro",
                "-v",
                f"{destination.resolve()}:/outputs",
                "-v",
                f"{Records.audio_dir.resolve()}:/reference-audio:ro",
                image,
                "transcriber",
            ]
            output = "/outputs"
            audio_paths = [
                (
                    "/reference-audio/"
                    + Path(host).relative_to(Records.audio_dir.resolve()).as_posix()
                    if Path(host).is_relative_to(Records.audio_dir.resolve())
                    else "/inputs/" + Path(host).relative_to(root.resolve()).as_posix()
                )
                for host in audio_paths
            ]
            if manifest:
                manifest_path = (
                    "/inputs/"
                    + manifest.resolve().relative_to(root.resolve()).as_posix()
                )
        command = prefix + [
            "--backend",
            backend,
            "--output-dir",
            output,
            "--processing-limit",
            str(processing_limit),
            "--debug-output",
        ]
        if manifest_path:
            command.extend(["--model-manifest", manifest_path])
        if batch_size is not None:
            Records.require(
                backend == "qwen3" and batch_size > 0, "Invalid Qwen batch size"
            )
            command.extend(["--qwen-batch-size", str(batch_size)])
        if chunk_seconds is not None:
            Records.require(
                backend == "qwen3" and 1 <= chunk_seconds <= 90,
                "Invalid Qwen chunk duration",
            )
            command.extend(["--qwen-chunk-seconds", str(chunk_seconds)])
        return command + audio_paths

    @classmethod
    def collect(
        cls,
        corpus: Corpus,
        root: Path,
        output: Path,
        backend: str,
        image: str | None,
        manifest: Path | None,
        processing_limit: int,
        batch_size: int | None,
        chunk_seconds: int | None,
        selected: list[str] | None = None,
    ) -> Record:
        """Collect through the production CLI and retain complete or partial work.

        :param corpus: Reference corpus identifying the original audio.
        :param root: Repository containing verified source artifacts.
        :param output: New attempt directory; retries require a distinct path.
        :param backend: Backend selected for the real transcriber.
        :param image: Optional Docker image, resolved to an immutable image ID.
        :param manifest: Optional deployment manifest.
        :param processing_limit: Maximum simultaneously processed recordings.
        :param batch_size: Optional Qwen chunk batch limit.
        :param chunk_seconds: Optional Qwen ASR context duration.
        :param selected: Cases to collect, or all corpus cases when omitted.
        :return: Sealed run with exit status, command, timing and output hashes.
        :raises ValueError: If sources or execution settings fail validation.
        """
        identities = selected if selected is not None else list(corpus.cases)
        corpus.verify_sources(root)
        work = output / "attempt"
        command = cls.command(
            corpus,
            root,
            work,
            backend,
            image,
            manifest,
            processing_limit,
            batch_size,
            chunk_seconds,
            identities,
        )
        image_id = None
        if image:
            image_id = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
            ).strip()
            command[command.index(image)] = image_id
        output.mkdir(parents=True, exist_ok=False)
        work.mkdir()
        provenance: Record = {
            "kind": "production_transcriber_cli",
            "command": command,
            "image_id": image_id,
            "backend": backend,
            "manifest_sha256": Records.file_digest(manifest) if manifest else None,
            "processing_limit": processing_limit,
            "qwen_batch_size": batch_size,
            "qwen_chunk_seconds": chunk_seconds,
            "input_association": "verified_original_audio_hashes",
            "started_at": Records.now(),
        }
        Records.write(output / "request.json", provenance)
        started = time.monotonic()
        with (output / "transcriber.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            ) as process:
                try:
                    returncode = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    returncode = 130
        provenance["returncode"] = returncode
        provenance["wall_seconds_including_startup_shutdown"] = (
            time.monotonic() - started
        )
        provenance["finished_at"] = Records.now()
        return cls.retain(
            corpus, work, output / "retained", backend, provenance, identities
        )
