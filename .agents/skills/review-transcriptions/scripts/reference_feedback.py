"""Validate portable review drafts and reconcile explicit human decisions."""

from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import override
import uuid

from reference_core import Corpus, PROJECT, Record, Records, SCHEMA_VERSION


class FeedbackDocument(HTMLParser):
    """Read the data payload without executing an exported review document."""

    def __init__(self) -> None:
        """Initialize inert extraction of embedded feedback payloads."""
        super().__init__()
        self.payloads: list[str] = []
        self.inside: bool = False

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start capturing a feedback payload without interpreting its content."""
        if tag == "script" and dict(attrs).get("id") == "review-feedback":
            self.inside = True
            self.payloads.append("")

    @override
    def handle_data(self, data: str) -> None:
        """Append raw data only while inside the feedback payload element."""
        if self.inside:
            self.payloads[-1] += data

    @override
    def handle_endtag(self, tag: str) -> None:
        """Stop capturing when the payload element closes."""
        if tag == "script":
            self.inside = False


class Feedback:
    """Keep browser proposals separate from accepted reference revisions."""

    REVIEW_MARKER: str = "[REVIEW NEEDED]"
    DRAFT_FIELDS: set[str] = {
        "schema_version",
        "project",
        "corpus_id",
        "corpus_revision",
        "corpus_sha256",
        "session_id",
        "revision",
        "reviewer",
        "active_case",
        "cases",
    }
    CASE_FIELDS: set[str] = {
        "case_id",
        "base_revision",
        "base_sha256",
        "passages",
        "notes",
        "status",
        "reviewed_at",
        "reviewed_by",
    }

    @staticmethod
    def empty(corpus: Corpus) -> Record:
        """Create an empty review session bound to an exact corpus revision."""
        return {
            "schema_version": SCHEMA_VERSION,
            "project": PROJECT,
            "corpus_id": corpus.data["corpus_id"],
            "corpus_revision": corpus.data["revision"],
            "corpus_sha256": corpus.data["content_sha256"],
            "session_id": str(uuid.uuid4()),
            "revision": 0,
            "reviewer": "",
            "active_case": next(iter(corpus.cases)),
            "cases": [],
        }

    @staticmethod
    def proposal(case: Record) -> Record:
        """Copy one case into an editable proposal with no implied completion."""
        passages = deepcopy(case["passages"])
        for passage in passages:
            passage.pop("speaker_change_before", None)
            passage.pop("boundary_basis", None)
        if case["review"]["status"] == "pending":
            for passage in passages:
                note = passage["uncertainty"]
                if note.strip() and Feedback.REVIEW_MARKER not in note:
                    passage["uncertainty"] = f"{Feedback.REVIEW_MARKER} {note}"
        return {
            "case_id": case["id"],
            "base_revision": case["revision"],
            "base_sha256": case["content_sha256"],
            "passages": passages,
            "notes": "",
            "status": "draft",
            "reviewed_at": None,
            "reviewed_by": None,
        }

    @staticmethod
    def snapshot(draft: Record) -> Record:
        """Copy the complete draft without obsolete checkbox fields."""
        snapshot = deepcopy(draft)
        for proposal in snapshot["cases"]:
            for passage in proposal["passages"]:
                passage.pop("speaker_change_before", None)
                passage.pop("boundary_basis", None)
        return snapshot

    @classmethod
    def validate(cls, corpus: Corpus, draft: Record) -> None:
        """Check proposal identities, lineage and explicit completion provenance.

        :param corpus: Exact base revision expected by the review session.
        :param draft: Structured browser edits, including unfinished work.
        :raises ValueError: If edits are stale, conflicting or malformed.
        """
        draft = cls.snapshot(draft)
        Records.require(set(draft) == cls.DRAFT_FIELDS, "Invalid draft fields")
        Records.require(
            draft["schema_version"] == SCHEMA_VERSION, "Unsupported draft schema"
        )
        for key in ("project", "corpus_id"):
            Records.require(draft[key] == corpus.data[key], f"Wrong {key}")
        Records.require(
            draft["corpus_sha256"] == corpus.data["content_sha256"]
            and draft["corpus_revision"] == corpus.data["revision"],
            "Stale or different corpus; preserve this file and reconcile explicitly",
        )
        Records.require(
            isinstance(draft["session_id"], str) and bool(draft["session_id"]),
            "Invalid session",
        )
        Records.require(
            type(draft["revision"]) is int and draft["revision"] >= 0,
            "Invalid draft revision",
        )
        Records.require(isinstance(draft["reviewer"], str), "Invalid reviewer")
        Records.require(draft["active_case"] in corpus.cases, "Unknown active case")
        Records.require(isinstance(draft["cases"], list), "Invalid proposal list")
        proposals = Records.index(draft["cases"], "case_id")
        for identity, proposal in proposals.items():
            Records.require(identity in corpus.cases, f"Unknown case: {identity}")
            Records.require(set(proposal) == cls.CASE_FIELDS, "Invalid proposal fields")
            base = corpus.cases[identity]
            Records.require(
                proposal["base_revision"] == base["revision"]
                and proposal["base_sha256"] == base["content_sha256"],
                f"Stale base: {identity}",
            )
            current = Corpus.passages(
                proposal["passages"],
                base["audio"]["duration"],
                allow_empty=proposal["status"] == "draft",
            )
            original = Records.index(base["passages"])
            for passage_id, passage in current.items():
                if passage_id in original:
                    Records.require(
                        passage["origin_ids"] == original[passage_id]["origin_ids"],
                        "Changed passage lineage",
                    )
                else:
                    Records.require(
                        passage_id.startswith("n-"),
                        "Unknown passage ID; additions must use n- IDs",
                    )
                    Records.require(
                        all(parent in original for parent in passage["origin_ids"]),
                        "Unknown split parent",
                    )
            Records.require(isinstance(proposal["notes"], str), "Invalid notes")
            status = proposal["status"]
            Records.require(
                status in ("draft", "complete", "complete_with_uncertainty"),
                "Invalid review disposition",
            )
            if status == "draft":
                Records.require(
                    proposal["reviewed_at"] is None and proposal["reviewed_by"] is None,
                    "Unfinished draft cannot claim completion",
                )
            else:
                missing_roles = [
                    passage_id
                    for passage_id, passage in current.items()
                    if "role_assignment" not in passage
                ]
                Records.require(
                    not missing_roles,
                    f"Role assignment is required in {identity}: {', '.join(missing_roles)}. "
                    "Select Operator, Caller or Unsure for each block.",
                )
                tagged = [
                    identity
                    for identity, passage in current.items()
                    if cls.REVIEW_MARKER in passage["uncertainty"]
                ]
                Records.require(
                    not tagged,
                    f"Review tagged AI notes in {proposal['case_id']}: {', '.join(tagged)}",
                )
                Records.require(
                    isinstance(proposal["reviewed_by"], str)
                    and bool(proposal["reviewed_by"].strip())
                    and bool(proposal["reviewed_at"]),
                    "Completion needs reviewer and timestamp",
                )

    @classmethod
    def load_package(cls, corpus: Corpus, path: Path) -> Record:
        """Read portable feedback as data and validate it against its reference.

        :param corpus: Expected reference revision.
        :param path: Submitted HTML document or structured JSON package.
        :return: Validated package with exact proposed passages.
        :raises ValueError: If payloads are duplicated, stale or invalid.
        """
        if path.suffix.lower() == ".html":
            document = FeedbackDocument()
            document.feed(path.read_text(encoding="utf-8"))
            Records.require(
                len(document.payloads) == 1, "Missing or duplicate feedback payload"
            )
            package = json.loads(
                document.payloads[0], object_pairs_hook=Records.unique_object
            )
        else:
            package = Records.load(path)
        Records.require(
            set(package)
            == {"format", "schema_version", "exported_at", "review_markdown", "draft"}
            and package["format"] == "transcription-review-feedback"
            and package["schema_version"] == SCHEMA_VERSION,
            "Unsupported feedback package",
        )
        cls.validate(corpus, package["draft"])
        Records.require(
            isinstance(package["review_markdown"], str), "Missing readable review"
        )
        return package

    @staticmethod
    def markdown(corpus: Corpus, draft: Record) -> str:
        """Render original and proposed text alongside dispositions and notes."""
        lines = [
            "# Transcription review",
            "",
            f"Reviewer: {draft['reviewer'] or '(not supplied)'}",
            f"Reference revision: {draft['corpus_revision']}",
            f"Reference hash: {draft['corpus_sha256']}",
            "",
            "Structured proposals in this package are the importable record.",
            "",
        ]
        for proposal in draft["cases"]:
            base = corpus.cases[proposal["case_id"]]
            old = Records.index(base["passages"])
            new = Records.index(proposal["passages"])
            lines.extend(
                [
                    f"## {base['id']}",
                    "",
                    f"Disposition: {proposal['status']}",
                    f"Completed by: {proposal['reviewed_by'] or '(not completed)'}",
                    f"Completed at: {proposal['reviewed_at'] or '(not completed)'}",
                    "",
                ]
            )
            for passage in proposal["passages"]:
                identity = passage["id"]
                previous = old.get(identity)
                if previous == passage:
                    continue
                previous_text = previous["text"] if previous else "(added passage)"
                lines.extend(
                    [
                        f"### {identity} ({passage['start']:.1f}s)",
                        "",
                        f"Original: {previous_text}",
                        f"Proposed: {passage['text']}",
                        *(
                            [
                                "Role assignment: "
                                + json.dumps(passage.get("role_assignment")),
                            ]
                            if "role_assignment" in passage
                            or previous
                            and "role_assignment" in previous
                            else []
                        ),
                        f"Uncertainty: {passage['uncertainty'].strip() or 'none'}",
                        "Previous alternatives: "
                        + json.dumps(previous["alternatives"] if previous else []),
                        f"Proposed alternatives: {json.dumps(passage['alternatives'])}",
                        "",
                    ]
                )
            for identity in old.keys() - new.keys():
                lines.extend([f"Deleted {identity}: {old[identity]['text']}", ""])
            if proposal["notes"]:
                lines.extend(["Notes:", proposal["notes"], ""])
        return "\n".join(lines)

    @classmethod
    def package(cls, corpus: Corpus, draft: Record) -> Record:
        """Validate and bundle exact edits with a derived readable review."""
        draft = cls.snapshot(draft)
        cls.validate(corpus, draft)
        return {
            "format": "transcription-review-feedback",
            "schema_version": SCHEMA_VERSION,
            "exported_at": Records.now(),
            "review_markdown": cls.markdown(corpus, draft),
            "draft": deepcopy(draft),
        }

    @classmethod
    def reconcile(
        cls,
        corpus: Corpus,
        package_path: Path,
        decisions: list[Record],
        actor: str,
        output: Path,
    ) -> Corpus:
        """Apply explicit decisions while preserving the submitted feedback.

        :param corpus: Exact base revision named in the feedback.
        :param package_path: Original submitted feedback document.
        :param decisions: One accept, reject or defer decision per proposed case.
        :param actor: Identity responsible for adjudication.
        :param output: New directory for the revision, feedback and decisions.
        :return: Derived corpus; accepting unfinished work leaves review pending.
        :raises ValueError: If feedback or decisions fail validation.
        """
        package = cls.load_package(corpus, package_path)
        draft = cls.snapshot(package["draft"])
        proposals = Records.index(draft["cases"], "case_id")
        indexed = Records.index(decisions, "case_id")
        Records.require(bool(actor.strip()), "An adjudicator identity is required")
        Records.require(
            set(indexed) == set(proposals), "Decide every proposal exactly once"
        )
        for decision in indexed.values():
            Records.require(
                set(decision) == {"case_id", "decision", "rationale"}
                and decision["decision"] in ("accept", "reject", "defer")
                and isinstance(decision["rationale"], str)
                and bool(decision["rationale"].strip()),
                "Each decision needs accept/reject/defer and a rationale",
            )
        source_hash = Records.file_digest(package_path)
        data = deepcopy(corpus.data)
        for index, base in enumerate(data["cases"]):
            identity = base["id"]
            if identity not in indexed or indexed[identity]["decision"] != "accept":
                continue
            proposal = proposals[identity]
            base["revision"] += 1
            base["passages"] = deepcopy(proposal["passages"])
            base["review"] = {
                "status": (
                    "pending" if proposal["status"] == "draft" else proposal["status"]
                ),
                "actor_type": "human",
                "reviewer": proposal["reviewed_by"] or draft["reviewer"],
                "reviewed_at": proposal["reviewed_at"],
                "feedback_sha256": source_hash,
                "notes": proposal["notes"],
            }
            data["cases"][index] = Records.seal(base)
        data["revision"] += 1
        data["parent_sha256"] = corpus.data["content_sha256"]
        data["created_at"] = Records.now()
        data["reconciliation"] = {
            "feedback_sha256": source_hash,
            "actor": actor,
            "decisions": decisions,
        }
        updated = Corpus(Records.seal(data))
        output.mkdir(parents=True, exist_ok=False)
        with (output / f"feedback{package_path.suffix}").open("xb") as destination:
            destination.write(package_path.read_bytes())
        Records.write(output / "decisions.json", data["reconciliation"])
        updated.save(output / "corpus.json")
        return updated
