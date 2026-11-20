"""Prepare traceable drafts and static review bundles from retained evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import shutil
import wave

import pysubs2

from reference_core import Corpus, PROJECT, Record, Records
from reference_feedback import Feedback


class Preparation:
    """Separate importing candidate text from audio-grounded draft curation."""

    @staticmethod
    def artifact(root: Path, identity: str, path: str, kind: str) -> Record:
        """Record the identity, kind and byte hash of an existing artifact."""
        source = Records.within(root, path)
        return {
            "id": identity,
            "path": path,
            "sha256": Records.file_digest(source),
            "kind": kind,
        }

    @classmethod
    def seed(cls, root: Path, specification: Record, output: Path) -> Corpus:
        """Import candidate SRTs without treating them as reviewed references.

        :param root: Repository containing original audio and evidence.
        :param specification: Corpus identity and per-case source paths.
        :param output: New corpus JSON path.
        :return: Validated corpus with seed-only preparation and pending review.
        :raises ValueError: If source paths or imported records are invalid.
        """
        cases = []
        Records.index(specification["cases"])
        for entry in specification["cases"]:
            audio_path = Records.within(root, entry["audio"])
            with wave.open(str(audio_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            source = cls.artifact(root, "audio", entry["audio"], "original_audio")
            source["duration"] = duration
            evidence = [cls.artifact(root, "seed", entry["seed_srt"], "machine_output")]
            for item in entry["evidence"]:
                evidence.append(
                    cls.artifact(root, item["id"], item["path"], item["kind"])
                )
            subtitles = pysubs2.load(
                str(Records.within(root, entry["seed_srt"])), encoding="utf-8"
            )
            passages = []
            for number, cue in enumerate(subtitles, 1):
                identity = f"p{number:04d}"
                passages.append(
                    {
                        "id": identity,
                        "text": cue.plaintext,
                        "start": min(duration, max(0, cue.start / 1000)),
                        "end": min(duration, max(0, cue.end / 1000)),
                        "origin_ids": [identity],
                        "uncertainty": "",
                        "alternatives": [],
                        "speaker_change_before": False,
                        "boundary_basis": None,
                    }
                )
            case = {
                "id": entry["id"],
                "revision": 1,
                "audio": source,
                "evidence": evidence,
                "passages": passages,
                "tags": entry.get("tags", []),
                "preparation": {
                    "status": "seed_only",
                    "actor_type": "tool",
                    "notes": (
                        "Imported candidate; requires audio-grounded "
                        "curation before handoff."
                    ),
                    "decisions": [],
                },
                "review": {"status": "pending"},
            }
            cases.append(Records.seal(case))
        corpus = Corpus(
            Records.seal(
                {
                    "schema_version": 1,
                    "project": PROJECT,
                    "corpus_id": specification["corpus_id"],
                    "revision": 1,
                    "parent_sha256": None,
                    "created_at": Records.now(),
                    "cases": cases,
                }
            )
        )
        corpus.verify_sources(root)
        corpus.save(output)
        return corpus

    @staticmethod
    def curate(corpus: Corpus, root: Path, proposals: Record, output: Path) -> Corpus:
        """Record explicit assistant curation without assigning human approval.

        :param corpus: Unchanged base revision named by the proposals.
        :param root: Repository containing audio and curation evidence.
        :param proposals: Exact passage snapshots, evidence and rationale.
        :param output: New corpus JSON path.
        :return: Derived draft-ready revision with human review pending.
        :raises ValueError: If proposals are stale, unsupported or invalid.
        """
        Records.require(
            proposals["base_sha256"] == corpus.data["content_sha256"], "Stale curation"
        )
        changes = Records.index(proposals["cases"], "case_id")
        Records.require(set(changes) <= set(corpus.cases), "Unknown curation case")
        data = deepcopy(corpus.data)
        for index, base in enumerate(data["cases"]):
            change = changes.get(base["id"])
            if change is None:
                continue
            Records.require(
                change["base_sha256"] == base["content_sha256"], "Stale case curation"
            )
            base["evidence"].extend(deepcopy(change.get("added_evidence", [])))
            Records.require(
                bool(change["notes"]) and bool(change["evidence_ids"]),
                "Curation needs rationale and evidence",
            )
            evidence = Records.index(base["evidence"])
            Records.require(
                set(change["evidence_ids"]) <= set(evidence),
                "Unknown curation evidence",
            )
            Records.require(
                any(
                    evidence[key]["kind"] == "native_audio_assessment"
                    for key in change["evidence_ids"]
                ),
                "Curation must include original-audio assessment evidence",
            )
            Corpus.passages(change["passages"], base["audio"]["duration"])
            draft = Feedback.empty(corpus)
            edit = Feedback.proposal(corpus.cases[base["id"]])
            edit["passages"] = change["passages"]
            draft["cases"] = [edit]
            Feedback.validate(corpus, draft)
            base["passages"] = deepcopy(change["passages"])
            base["revision"] += 1
            base["review"] = {"status": "pending"}
            base["preparation"] = {
                "status": "draft_ready",
                "actor_type": "assistant",
                "notes": change["notes"],
                "evidence_ids": change["evidence_ids"],
                "decisions": change.get("decisions", []),
                "proposal_sha256": Records.digest(change),
            }
            data["cases"][index] = Records.seal(base)
        data["parent_sha256"] = corpus.data["content_sha256"]
        data["revision"] += 1
        data["created_at"] = Records.now()
        result = Corpus(Records.seal(data))
        result.verify_sources(root)
        result.save(output)
        return result


class Portal:
    """Build a self-contained static view without changing reference data."""

    @staticmethod
    def stamp(corpus: Corpus, output: Path) -> None:
        """Identify application content and build time separately from the corpus.

        :param corpus: Reference revision bundled into the site.
        :param output: Site with application assets and an unstamped HTML template.
        """
        application_hash = Records.digest(
            {
                name: Records.file_digest(output / name)
                for name in ("index.html", "app.js", "data.js", "review.css")
            }
        )
        metadata = {
            "schema_version": 1,
            "site_version": application_hash[:12],
            "built_at": Records.now(),
            "application_sha256": application_hash,
            "corpus_revision": corpus.data["revision"],
            "corpus_sha256": corpus.data["content_sha256"],
        }
        app = output / "app.js"
        app.write_text(
            f"window.reviewAppVersion = {json.dumps(metadata['site_version'])};\n"
            + app.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        asset_files = {}
        for name in ("app.js", "data.js", "review.css"):
            path = output / name
            versioned = f"{path.stem}.{metadata['site_version']}{path.suffix}"
            path.rename(output / versioned)
            asset_files[name] = versioned
        metadata["assets"] = asset_files
        Records.write(output / "build.json", metadata)
        page = output / "index.html"
        label = (
            f"Site version {metadata['site_version']} · Built {metadata['built_at']}"
            f" · Reference revision {metadata['corpus_revision']}"
            ' · <a href="build.json">Build metadata</a>'
        )
        html = page.read_text(encoding="utf-8").replace("__SITE_BUILD__", label)
        for original, versioned in asset_files.items():
            html = html.replace(f'"{original}"', f'"{versioned}"')
        page.write_text(html, encoding="utf-8")

    @staticmethod
    def build(
        corpus: Corpus,
        root: Path,
        assets: Path,
        output: Path,
        selected: list[str] | None = None,
    ) -> None:
        """Bundle verified audio, references and review assets for static hosting.

        :param corpus: Reference revision presented to reviewers.
        :param root: Repository containing its source artifacts.
        :param assets: Directory containing the bundled browser application.
        :param output: New site directory; existing sites are not overwritten.
        :param selected: Case identities to include, or all cases when omitted.
        :raises ValueError: If sources changed or the selection is invalid.
        """
        corpus.verify_sources(root)
        identities = selected if selected is not None else list(corpus.cases)
        Records.require(
            bool(identities) and len(set(identities)) == len(identities),
            "Invalid portal selection",
        )
        Records.require(set(identities) <= set(corpus.cases), "Unknown portal case")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("index.html", "review.css"):
            shutil.copyfile(assets / name, output / name)
        Records.write(output / "corpus.json", corpus.data)
        Records.write(output / "selection.json", identities)
        browser = (
            (assets / "draft.js")
            .read_text()
            .replace("export function ", "function ")
            .replace("export class ", "class ")
        )
        names = "randomId, emptyDraft, proposal, hasUncertainty, validateDraft, feedbackPackage, feedbackHTML, parseFeedback, DraftStore"
        browser = (
            f"const {{{names}}} = (() => {{\n{browser}\nreturn {{{names}}};\n}})();\n"
        )
        browser += "\n" + (assets / "review.js").read_text()
        (output / "app.js").write_text(
            "(() => {\n" + browser + "\n})();\n", encoding="utf-8"
        )
        bundled_evidence = {}
        audio_dir = output / "audio"
        audio_dir.mkdir()
        references = output / "references"
        references.mkdir()
        evidence_dir = output / "evidence"
        evidence_dir.mkdir()
        for identity in identities:
            case = corpus.cases[identity]
            shutil.copyfile(
                Records.within(root, case["audio"]["path"]),
                audio_dir / f"{identity}.wav",
            )
            (references / f"{identity}.srt").write_text(
                Corpus.srt(case), encoding="utf-8"
            )
            (references / f"{identity}.md").write_text(
                Corpus.markdown(case), encoding="utf-8"
            )
            sources = []
            for item in case["evidence"]:
                source = Records.within(root, item["path"])
                sources.append({**item, "contents": source.read_text(encoding="utf-8")})
            Records.write(evidence_dir / f"{identity}.json", sources)
            bundled_evidence[identity] = sources
        payload = {
            "corpus": corpus.data,
            "selection": identities,
            "evidence": bundled_evidence,
        }
        (output / "data.js").write_text(
            "window.reviewData = " + json.dumps(payload, ensure_ascii=True) + ";\n",
            encoding="utf-8",
        )
        Portal.stamp(corpus, output)
