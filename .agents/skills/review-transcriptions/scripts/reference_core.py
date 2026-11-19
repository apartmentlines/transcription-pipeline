"""Store immutable audio reference revisions and validate their provenance."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

Record = dict[str, Any]
PROJECT = "transcription-pipeline"
SCHEMA_VERSION = 1
PASSAGE_FIELDS = {
    "id",
    "text",
    "start",
    "end",
    "origin_ids",
    "uncertainty",
    "alternatives",
    "speaker_change_before",
    "boundary_basis",
}


class Records:
    """Serialize JSON reproducibly and reject ambiguous or changed records."""

    audio_dir: Path = (
        Path.home() / "git/apartmentlines/testing-references/audio/operator-recordings"
    )
    srt_dir: Path = (
        Path.home() / "git/apartmentlines/testing-references/transcription-pipeline/srt"
    )
    corpus_dir: Path = (
        Path.home()
        / "git/apartmentlines/testing-references/transcription-pipeline/corpus"
    )
    evidence_dir: Path = (
        Path.home()
        / "git/apartmentlines/testing-references/transcription-pipeline/evidence"
    )
    data_dir: Path | None = None

    @classmethod
    def location_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Expose overrides for stable references and generated work."""
        parser.add_argument(
            "--evidence-dir",
            type=Path,
            default=cls.evidence_dir,
            help="Durable evidence directory (default: %(default)s)",
        )
        parser.add_argument(
            "--corpus-dir",
            type=Path,
            default=cls.corpus_dir,
            help="Versioned corpus directory (default: %(default)s)",
        )
        parser.add_argument(
            "--audio-dir",
            type=Path,
            default=cls.audio_dir,
            help="Reference audio directory (default: %(default)s)",
        )
        parser.add_argument(
            "--srt-dir",
            type=Path,
            default=cls.srt_dir,
            help="Reference SRT directory (default: %(default)s)",
        )
        parser.add_argument(
            "--data-dir",
            type=Path,
            default=Path.cwd() / "test-data",
            help="Working data directory (default: %(default)s)",
        )

    @classmethod
    def configure(cls, args: argparse.Namespace) -> None:
        """Apply explicit CLI locations before resolving source artifacts."""
        cls.evidence_dir = args.evidence_dir.expanduser().resolve()
        cls.corpus_dir = args.corpus_dir.expanduser().resolve()
        cls.audio_dir = args.audio_dir.expanduser().resolve()
        cls.srt_dir = args.srt_dir.expanduser().resolve()
        cls.data_dir = args.data_dir.expanduser().resolve()

    @staticmethod
    def now() -> str:
        """Return the current UTC time in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def unique_object(pairs: list[tuple[str, Any]]) -> Record:
        """Decode object pairs, rejecting duplicate JSON keys."""
        result: Record = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    @staticmethod
    def load(path: Path) -> Record:
        """Read a JSON object, rejecting duplicate keys and other root types."""
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=Records.unique_object,
        )
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    @staticmethod
    def encoded(value: Any) -> bytes:
        """Encode canonical UTF-8 JSON with finite numeric values."""
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def digest(value: Any) -> str:
        """Return the SHA-256 of a value's canonical JSON representation."""
        return hashlib.sha256(Records.encoded(value)).hexdigest()

    @staticmethod
    def file_digest(path: Path) -> str:
        """Return the SHA-256 of the original file bytes."""
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    @staticmethod
    def seal(value: Record) -> Record:
        """Copy a record and replace its hash using all other fields."""
        result = deepcopy(value)
        result.pop("content_sha256", None)
        result["content_sha256"] = Records.digest(result)
        return result

    @staticmethod
    def verify(value: Record) -> None:
        """Reject a record whose stored hash no longer matches its contents."""
        if Records.seal(value)["content_sha256"] != value.get("content_sha256"):
            raise ValueError("Content hash mismatch")

    @staticmethod
    def write(path: Path, value: Any) -> None:
        """Create a new artifact without replacing an earlier revision."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as destination:
            destination.write(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )

    @staticmethod
    def within(root: Path, relative: str) -> Path:
        """Resolve an artifact path while rejecting traversal outside its root."""
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected a relative artifact path: {relative}")
        locations = {
            "reference-evidence": Records.evidence_dir,
            "reference-corpus": Records.corpus_dir,
            "reference-audio": Records.audio_dir,
            "reference-srt": Records.srt_dir,
            "test-data": Records.data_dir or root / "test-data",
        }
        if path.parts and path.parts[0] in locations:
            return Records.within(
                locations[path.parts[0]], Path(*path.parts[1:]).as_posix()
            )
        if path.parts and path.parts[0] == "audio-files":
            evidence_index = Records.evidence_dir / "index.json"
            if evidence_index.exists():
                mapped = Records.load(evidence_index).get(relative)
                if mapped is not None:
                    return Records.within(Records.evidence_dir, mapped)
            tail = Path(*path.parts[1:])
            if tail.parts[:2] == ("reference-framework", "corpus"):
                return Records.within(
                    Records.corpus_dir, Path(*tail.parts[2:]).as_posix()
                )
            if len(tail.parts) == 1 and tail.suffix == ".wav":
                return Records.within(Records.audio_dir, tail.as_posix())
            return Records.within(
                Records.data_dir or root / "test-data", tail.as_posix()
            )
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"Artifact escapes its root: {relative}")
        return resolved

    @staticmethod
    def require(condition: bool, message: str) -> None:
        """Raise a validation error with the supplied message when false."""
        if not condition:
            raise ValueError(message)

    @staticmethod
    def index(items: Iterable[Any], key: str = "id") -> dict[str, Record]:
        """Index records by a required, unique, nonempty string identity."""
        result: dict[str, Record] = {}
        for item in items:
            Records.require(isinstance(item, dict), "Expected a record object")
            identity = item.get(key)
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"Missing {key}")
            Records.require(identity not in result, f"Duplicate {key}: {identity}")
            result[identity] = item
        return result


class Corpus:
    """One authoritative revision; SRT and browser data are derived views."""

    def __init__(self, data: Record) -> None:
        """Load and validate one sealed corpus revision."""
        self.data: Record = data
        self.cases: dict[str, Record] = {}
        self.validate()

    @classmethod
    def load(cls, path: Path) -> Corpus:
        """Read and validate a corpus JSON artifact."""
        return cls(Records.load(path))

    @classmethod
    def current(cls, root: Path) -> Corpus:
        """Load the configured current revision and verify its pointer hash."""
        pointer = Records.load(Records.corpus_dir / "current.json")
        corpus = cls.load(Records.within(root, pointer["corpus_path"]))
        Records.require(
            corpus.data["content_sha256"] == pointer["corpus_sha256"],
            "Current corpus pointer hash mismatch",
        )
        return corpus

    @staticmethod
    def passages(
        items: Any, duration: float, allow_empty: bool = False
    ) -> dict[str, Record]:
        """Validate passage fields without imposing a cue grouping policy.

        :param items: Ordered passage records with stable identities.
        :param duration: Original audio duration in seconds.
        :param allow_empty: Permit unfinished blank text in review drafts.
        :return: Validated passages indexed by identity.
        :raises ValueError: If text, navigation intervals or evidence is invalid.
        """
        Records.require(
            isinstance(items, list) and bool(items), "No reference passages"
        )
        indexed = Records.index(items)
        for index, item in enumerate(items):
            identity = item["id"]
            Records.require(
                set(item) == PASSAGE_FIELDS, f"Invalid passage fields: {identity}"
            )
            Records.require(
                bool(re.fullmatch(r"[\w:-]+", identity)), "Invalid passage identity"
            )
            Records.require(
                isinstance(item["text"], str)
                and (bool(item["text"].strip()) or allow_empty),
                f"Empty text: {identity}",
            )
            Records.require(isinstance(item["uncertainty"], str), "Invalid uncertainty")
            for field in ("origin_ids", "alternatives"):
                values = item[field]
                Records.require(
                    isinstance(values, list)
                    and all(isinstance(v, str) for v in values),
                    f"Invalid {field}",
                )
                Records.require(len(values) == len(set(values)), f"Duplicate {field}")
            start, end = item["start"], item["end"]
            Records.require(
                all(type(v) in (int, float) and math.isfinite(v) for v in (start, end))
                and 0 <= start <= end <= duration,
                f"Invalid navigation interval: {identity}",
            )
            Records.require(
                type(item["speaker_change_before"]) is bool, "Invalid speaker change"
            )
            basis = item["boundary_basis"]
            if item["speaker_change_before"]:
                Records.require(
                    index > 0, "First passage cannot follow a speaker change"
                )
                Records.require(
                    isinstance(basis, dict)
                    and basis.get("actor_type") in ("human", "model", "assistant")
                    and bool(basis.get("note")),
                    f"Speaker boundary needs evidence: {identity}",
                )
            else:
                Records.require(basis is None, "Unexpected boundary evidence")
        return indexed

    def validate(self) -> None:
        """Check corpus integrity, provenance fields and explicit review state.

        :raises ValueError: If hashes, identities or record invariants fail.
        """
        data = self.data
        Records.verify(data)
        Records.require(
            data.get("schema_version") == SCHEMA_VERSION, "Unsupported corpus schema"
        )
        Records.require(data.get("project") == PROJECT, "Wrong project")
        Records.require(
            type(data.get("revision")) is int and data["revision"] > 0,
            "Invalid corpus revision",
        )
        Records.require(bool(data.get("corpus_id")), "Missing corpus identity")
        self.cases = Records.index(data["cases"])
        Records.require(bool(self.cases), "Empty corpus")
        for case in self.cases.values():
            Records.verify(case)
            Records.require(
                type(case.get("revision")) is int and case["revision"] > 0,
                "Invalid case revision",
            )
            audio = case["audio"]
            Records.require(
                bool(re.fullmatch(r"[a-f0-9]{64}", audio["sha256"])),
                "Invalid audio hash",
            )
            duration = audio["duration"]
            Records.require(
                type(duration) in (int, float)
                and math.isfinite(duration)
                and duration > 0,
                "Invalid audio duration",
            )
            self.passages(case["passages"], duration)
            evidence = Records.index(case["evidence"])
            for item in evidence.values():
                Records.require(
                    bool(re.fullmatch(r"[a-f0-9]{64}", item["sha256"])),
                    "Invalid evidence hash",
                )
                Records.require(bool(item.get("kind")), "Missing evidence kind")
            review = case["review"]
            Records.require(
                review["status"]
                in ("pending", "complete", "complete_with_uncertainty"),
                "Invalid review state",
            )
            if review["status"] != "pending":
                Records.require(
                    review.get("actor_type") == "human"
                    and bool(review.get("reviewer"))
                    and bool(review.get("feedback_sha256"))
                    and bool(review.get("reviewed_at")),
                    "Human approval needs explicit feedback provenance",
                )
            if review["status"] == "complete":
                Records.require(
                    not any(p["uncertainty"] for p in case["passages"]),
                    "Resolved review contains uncertainty",
                )

    def verify_sources(self, root: Path) -> None:
        """Verify every referenced audio and evidence file against its hash.

        :param root: Repository containing the relative artifact paths.
        :raises ValueError: If an artifact changed or escapes the root.
        :raises OSError: If a referenced artifact cannot be read.
        """
        checked: set[tuple[str, str]] = set()
        for case in self.cases.values():
            for source in [case["audio"], *case["evidence"]]:
                key = (source["path"], source["sha256"])
                if key in checked:
                    continue
                actual = Records.file_digest(Records.within(root, source["path"]))
                Records.require(
                    actual == source["sha256"], f"Source changed: {source['path']}"
                )
                checked.add(key)

    def save(self, path: Path) -> None:
        """Validate and write a new revision without replacing an existing file."""
        self.validate()
        Records.write(path, self.data)

    @staticmethod
    def srt(case: Record) -> str:
        """Derive an SRT view using approximate navigation intervals."""
        import pysubs2

        subtitles = pysubs2.SSAFile()
        for passage in case["passages"]:
            start = round(1000 * passage["start"])
            end = max(start + 1, round(1000 * passage["end"]))
            subtitles.append(
                pysubs2.SSAEvent(start=start, end=end, text=passage["text"])
            )
        return subtitles.to_string("srt")

    @staticmethod
    def markdown(case: Record) -> str:
        """Derive a readable transcript that keeps uncertainty visible."""
        lines = [f"# {case['id']}", "", f"Review: {case['review']['status']}", ""]
        for passage in case["passages"]:
            boundary = " · speaker changes" if passage["speaker_change_before"] else ""
            lines.extend(
                [
                    f"{passage['start']:.1f}s · {passage['id']}{boundary}",
                    "",
                    passage["text"],
                    "",
                ]
            )
            if passage["uncertainty"]:
                lines.extend([f"Unresolved: {passage['uncertainty']}", ""])
        return "\n".join(lines)
