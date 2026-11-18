"""Prepare and collect authorized audio assessments using the existing audio tool."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import logging
from pathlib import Path
import subprocess
import sys

from reference_core import Corpus, Record, Records

LOGGER = logging.getLogger(__name__)

QUESTION = """Listen to the COMPLETE original recording and reconcile the base passages
against the audio. The source audio is authoritative; all transcripts and earlier
judgments can be wrong. Produce minimal explicit proposals, not a fresh rewrite.
Preserve the original language, repeated words, hesitations, grammar and number
format unless the audible words justify a correction. Do not clean up speech.
Do not use timestamps to decide whether a word exists. Do not remove a word
because it has a short or separate SRT cue. Never add hold/music labels to speech.

The ONLY important grouping defect is words from different voices in one
passage. Listen for actual voice changes. A sentence break is not a speaker
change, and consecutive SRT passages often belong to the SAME voice. Two
consecutive acknowledgments can also be the SAME voice. Do not mechanically
alternate voices. No identities or Operator/Caller labels are needed.
Check carefully for a question and its answer within one passage; for example,
'May I have your name please Virginia Moreno' needs two parts IF the audio
contains two voices. Do not leave the question and answer in one part merely
because the source omitted punctuation. Mark uncertain changes as uncertain.

Return one fenced JSON object under Developer follow-up with exactly these keys:
edits: list of {passage_id, original_text, parts, reason, confidence}. Quote the
exact base passage in original_text. parts is a list of {text, uncertainty,
alternatives, speaker_change_before}. Each part contains one voice. More than
one part means split a mixed-speaker passage, not arbitrary sentence formatting.
An empty parts list explicitly proposes deleting hallucinated speech. Such a
deletion needs acoustic evidence, not merely a cleaner or shorter transcript.
uncertainty MUST be a string: empty when resolved, or a specific unresolved
issue. alternatives MUST be a list of strings. speaker_change_before MUST be
true or false. The first part's flag refers to the preceding retained passage;
later parts refer to the preceding part. Spellings that audio cannot establish
must remain uncertain. For minor ambiguity retain the base wording and record
the competing interpretation instead of confidently replacing it.

speaker_changes_before: list of existing base passage IDs that start a different
audible voice from the preceding base passage. Inspect the entire call. Do NOT
include the first passage; it has no preceding voice. Do NOT mark all cues or all
sentences as changes. For changes inside a passage use edits with multiple parts.
uncertain_boundaries: list of {passage_id, reason} for ambiguous voice changes.
insertions: list of {after_passage_id, text, uncertainty, alternatives,
speaker_change_before, reason} for speech missing from ALL base passages.
after_passage_id is an existing ID, or null for insertion before the first.
Never duplicate words present in adjacent base passages. Preserve original
language and repetitions here too. Do not invent timestamps.
summary: brief account of coverage and limitations, including any areas that
need a human listen. Treat all proposals as AI-assisted, never human-approved.

Before returning, check that every edit quotes an actual base passage exactly,
that its explanation covers EVERY changed word, and that no unchanged-word
punctuation/case edit is included merely for formatting. Do not trade missing
speech for a fluent transcript. An empty edit list is acceptable when justified.
"""


class AudioReferences:
    """Reuse native-audio collection and preserve its original audit artifacts."""

    @staticmethod
    def prepare(
        corpus: Corpus,
        root: Path,
        output: Path,
        tool: Path,
        selected: list[str],
    ) -> None:
        """Freeze audio-review requests without making paid model calls.

        :param corpus: Reference revision supplying candidate text and evidence.
        :param root: Repository containing original audio and source artifacts.
        :param output: New request directory.
        :param tool: Existing native-audio analysis CLI to reuse in place.
        :param selected: Case identities included in the authorized review scope.
        :raises ValueError: If sources or selected identities are invalid.
        """
        LOGGER.debug("Preparing %s audio requests in %s", len(selected), output)
        corpus.verify_sources(root)
        Records.require(
            bool(selected) and set(selected) <= set(corpus.cases),
            "Invalid audio selection",
        )
        Records.require(
            len(selected) == len(set(selected)), "Duplicate audio selection"
        )
        output.mkdir(parents=True, exist_ok=False)
        for identity in selected:
            case = corpus.cases[identity]
            folder = output / identity
            folder.mkdir()
            alternatives = {}
            earlier = {}
            for item in case["evidence"]:
                if item["id"] == "seed":
                    continue
                if item["kind"] == "machine_output":
                    alternatives[item["id"]] = Records.within(
                        root, item["path"]
                    ).read_text(encoding="utf-8")
                elif item["id"] == "audio-review-0-1":
                    earlier[item["id"]] = Records.within(root, item["path"]).read_text(
                        encoding="utf-8"
                    )
            metadata = {
                "base_passages": [
                    {key: p[key] for key in ("id", "text", "start", "end")}
                    for p in case["passages"]
                ],
                "alternative_machine_outputs": alternatives,
                "earlier_fallible_audio_assessments": earlier,
                "corpus_sha256": corpus.data["content_sha256"],
                "case_sha256": case["content_sha256"],
            }
            metadata_path = folder / "candidates.json"
            Records.write(metadata_path, metadata)
            manifest = {
                "schema_version": 1,
                "case_id": identity + "--reference-proposals",
                "audio_path": str(Records.within(root, case["audio"]["path"])),
                "metadata_path": str(metadata_path.resolve()),
                "reported_failure": (
                    "Reference preparation: audit an entire recording for precise "
                    "text corrections and mixed-speaker passages. "
                    "No human reference exists."
                ),
                "expected_behavior": (
                    "Preserve all audible speech, its original language and "
                    "repetitions. Separate different voices within a passage. "
                    "Keep uncertain details explicit. "
                    "Do not guess from context or formatting."
                ),
                "questions": [QUESTION],
                "context": {
                    "reference_status": "AI-assisted draft awaiting human review"
                },
            }
            Records.write(folder / "manifest.json", manifest)
            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "prepare",
                    str(folder / "manifest.json"),
                    str(folder / "case"),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        Records.write(
            output / "requests.json",
            {
                "schema_version": 1,
                "corpus_sha256": corpus.data["content_sha256"],
                "selected_cases": selected,
                "tool_path": str(tool.resolve()),
                "tool_sha256": Records.file_digest(tool),
                "model": "gpt-audio-1.5",
                "max_completion_tokens": 14000,
                "purpose": "Authorized reference drafting; no human approval implied",
            },
        )

    @staticmethod
    def extract(text: str, required_key: str = "edits") -> Record:
        """Extract a complete JSON object without repairing a model response.

        :param text: Original assistant response, including surrounding prose.
        :param required_key: Key identifying the expected structured object.
        :return: First valid object containing the required key.
        :raises ValueError: If the response requires manual inspection.
        """
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = json.JSONDecoder(
                    object_pairs_hook=Records.unique_object
                ).raw_decode(text[index:])
            except ValueError:
                continue
            if isinstance(value, dict) and required_key in value:
                return value
        message = f"No valid structured {required_key} object"
        raise ValueError(f"{message}; preserve and inspect the raw response")

    @classmethod
    def one(cls, root: Path, identity: str, request: Record) -> Record:
        """Collect one authorized request or reuse its retained response.

        :param root: Prepared request-set directory.
        :param identity: Case identity within that request set.
        :param request: Frozen tool path, model and completion settings.
        :return: Durable collected, failed or needs-inspection status.
        """
        LOGGER.debug("Inspecting native-audio request %s", identity)
        folder = root / identity
        status_path = folder / "collection-status.json"
        if status_path.exists():
            return Records.load(status_path)
        case = folder / "case"
        result: Record = {"case_id": identity, "started_at": Records.now()}
        if not (case / "conversation.json").exists():
            with (folder / "collect.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        request["tool_path"],
                        "analyze",
                        str(case),
                        "--model",
                        request["model"],
                        "--max-completion-tokens",
                        str(request["max_completion_tokens"]),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                result.update(
                    status="failed",
                    returncode=completed.returncode,
                    finished_at=Records.now(),
                )
                Records.write(status_path, result)
                return result
        try:
            conversation = Records.load(case / "conversation.json")
            turn = conversation["turns"][-1]
            raw = Records.load(case / turn["response_file"])
            Records.require(
                raw["choices"][0]["finish_reason"] == "stop",
                "Incomplete model response",
            )
            proposals = cls.extract(turn["assistant_text"])
            Records.write(folder / "proposals.json", proposals)
            result.update(
                status="collected",
                proposal_sha256=Records.digest(proposals),
                response_id=raw["id"],
                usage=raw.get("usage"),
            )
        except (ValueError, KeyError) as exc:
            result.update(status="needs_inspection", error=str(exc))
        result["finished_at"] = Records.now()
        Records.write(status_path, result)
        return result

    @classmethod
    def collect(cls, root: Path, concurrency: int) -> None:
        """Resume authorized requests under an exclusive collection lock.

        :param root: Prepared request directory containing requests.json.
        :param concurrency: Concurrent native-audio calls, between one and four.
        :raises ValueError: If configuration changed or another collector owns it.
        """
        LOGGER.debug(
            "Collecting requests from %s with concurrency %s", root, concurrency
        )
        request = Records.load(root / "requests.json")
        Records.require(1 <= concurrency <= 4, "Audio review concurrency must be 1–4")
        Records.require(
            Records.file_digest(Path(request["tool_path"])) == request["tool_sha256"],
            "Audio tool changed after request preparation",
        )
        with (root / ".collection.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "Another audio collector owns this request set"
                ) from exc
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                jobs = {
                    executor.submit(cls.one, root, identity, request): identity
                    for identity in request["selected_cases"]
                }
                for future in as_completed(jobs):
                    result = future.result()
                    print(
                        json.dumps(
                            {"case_id": result["case_id"], "status": result["status"]}
                        ),
                        flush=True,
                    )


def main() -> None:
    """Parse the requested operation and report failures with a nonzero exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", action="store_true", help="Enable diagnostic logging"
    )
    Records.location_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Freeze requests without making API calls"
    )
    prepare.add_argument(
        "--corpus",
        type=Path,
        help="Corpus JSON; defaults to the current reference revision",
    )
    prepare.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: %(default)s)",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--audio-tool", type=Path, required=True)
    prepare.add_argument("--case", action="append", dest="selected", required=True)
    collect = commands.add_parser(
        "collect", help="Collect an already authorized native-audio request set"
    )
    collect.add_argument("--requests", type=Path, required=True)
    collect.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    Records.configure(args)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        if args.command == "prepare":
            AudioReferences.prepare(
                Corpus.load(args.corpus) if args.corpus else Corpus.current(args.root),
                args.root,
                args.output,
                args.audio_tool,
                args.selected,
            )
        else:
            AudioReferences.collect(args.requests, args.concurrency)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if args.debug:
            LOGGER.exception("Audio reference operation failed")
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
