"""Compare retained speech text and evidenced speaker boundaries across backends."""

from __future__ import annotations

from array import array
from collections import Counter
from pathlib import Path
import re
import unicodedata

import pysubs2

from reference_core import Corpus, Record, Records


class Evaluator:
    """Keep edit distance, boundary evidence and human approval distinguishable."""

    POLICY: str = "nfkc-casefold-punctuation-v1"

    @staticmethod
    def words(text: str) -> list[str]:
        """Normalize formatting while retaining accents, digits and repetitions."""
        normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
        return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized)

    @classmethod
    def alignment(cls, reference: list[str], candidate: list[str]) -> Record:
        """Compute minimum word edits and exact matches for boundary checks.

        :param reference: Ordered normalized reference words.
        :param candidate: Ordered normalized output words, independent of cues.
        :return: Edit counts, individual edits and unambiguous match metadata.
        """
        columns = len(candidate) + 1
        previous = array("I", range(columns))
        distances = [previous]
        trace = [bytearray([2] * columns)]
        for row, word in enumerate(reference, 1):
            current = array("I", [row])
            decisions = bytearray([1])
            for column, other in enumerate(candidate, 1):
                choices = (
                    previous[column - 1] + int(word != other),
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
                best = min(choices)
                decisions.append(choices.index(best))
                current.append(best)
            trace.append(decisions)
            distances.append(current)
            previous = current
        row, column = len(reference), len(candidate)
        counts: Counter[str] = Counter()
        matches: dict[int, int] = {}
        edits: list[Record] = []
        while row or column:
            choice = trace[row][column]
            if row and column and choice == 0:
                row -= 1
                column -= 1
                if reference[row] == candidate[column]:
                    matches[row] = column
                else:
                    counts["substitutions"] += 1
                    edits.append(
                        {
                            "kind": "substitution",
                            "reference_index": row,
                            "candidate_index": column,
                            "reference": reference[row],
                            "candidate": candidate[column],
                        }
                    )
            elif row and (not column or choice == 1):
                row -= 1
                counts["deletions"] += 1
                edits.append(
                    {
                        "kind": "deletion",
                        "reference_index": row,
                        "candidate_index": column,
                        "reference": reference[row],
                        "candidate": None,
                    }
                )
            else:
                column -= 1
                counts["insertions"] += 1
                edits.append(
                    {
                        "kind": "insertion",
                        "reference_index": row,
                        "candidate_index": column,
                        "reference": None,
                        "candidate": candidate[column],
                    }
                )
        ambiguous = cls.ambiguous_matches(reference, candidate, distances, matches)
        return {
            "reference_words": len(reference),
            "candidate_words": len(candidate),
            "substitutions": counts["substitutions"],
            "deletions": counts["deletions"],
            "insertions": counts["insertions"],
            "matches": matches,
            "ambiguous_matches": ambiguous,
            "edits": list(reversed(edits)),
        }

    @staticmethod
    def ambiguous_matches(
        reference: list[str],
        candidate: list[str],
        distances: list[array[int]],
        matches: dict[int, int],
    ) -> list[int]:
        """Find anchors that another optimal alignment maps differently or omits.

        :param reference: Normalized reference words.
        :param candidate: Normalized output words.
        :param distances: Forward edit-distance matrix from the chosen alignment.
        :param matches: Exact matches in one minimum-cost traceback.
        :return: Reference indices unsuitable for definitive boundary checks.
        """
        total = distances[-1][-1]
        if not total:
            return []
        following = array("I", range(len(candidate), -1, -1))
        ambiguous = []
        for row in range(len(reference) - 1, -1, -1):
            current = array("I", [0] * (len(candidate) + 1))
            current[-1] = len(reference) - row
            for column in range(len(candidate) - 1, -1, -1):
                current[column] = min(
                    following[column + 1] + int(reference[row] != candidate[column]),
                    following[column] + 1,
                    current[column + 1] + 1,
                )
            if row in matches:
                for column in range(len(candidate) + 1):
                    before = distances[row][column]
                    deletion = before + 1 + following[column] == total
                    other = (
                        column < len(candidate)
                        and column != matches[row]
                        and before
                        + int(reference[row] != candidate[column])
                        + following[column + 1]
                        == total
                    )
                    if deletion or other:
                        ambiguous.append(row)
                        break
            following = current
        return sorted(ambiguous)

    @classmethod
    def compare(cls, case: Record, contents: str) -> Record:
        """Compare words and supplied speaker-change evidence for one recording.

        :param case: Reference text, uncertainty and review provenance.
        :param contents: Candidate SRT contents.
        :return: Mechanical differences, score exclusions and boundary coverage.
        :raises ValueError: If the candidate SRT cannot be meaningfully parsed.
        """
        subtitles = pysubs2.SSAFile.from_string(contents, format_="srt")
        arrows = len(re.findall(r"^\d\d:\d\d:\d\d,\d{3} -->", contents, re.MULTILINE))
        Records.require(
            bool(subtitles) and arrows == len(subtitles), "Malformed or empty SRT"
        )
        structural = []
        candidate: list[str] = []
        owners: list[int] = []
        for number, cue in enumerate(subtitles, 1):
            if cue.start < 0 or cue.end <= cue.start or not cue.plaintext.strip():
                structural.append(
                    {"cue": number, "issue": "invalid cue interval or empty text"}
                )
            words = cls.words(cue.plaintext)
            candidate.extend(words)
            owners.extend([number] * len(words))
        reference: list[str] = []
        boundaries: list[tuple[int, Record]] = []
        for position, passage in enumerate(case["passages"]):
            previous = case["passages"][position - 1] if position else {}
            evidence = Corpus.boundary_evidence(previous, passage)
            if evidence:
                boundaries.append(
                    (len(reference), {**passage, "boundary_basis": evidence})
                )
            reference.extend(cls.words(passage["text"]))
        alignment = cls.alignment(reference, candidate)
        matches = alignment.pop("matches")
        ambiguous = set(alignment.pop("ambiguous_matches"))
        boundary_checks = []
        for offset, passage in boundaries:
            left, right = matches.get(offset - 1), matches.get(offset)
            uncertain_match = bool({offset - 1, offset} & ambiguous)
            cues = (
                [owners[left], owners[right]]
                if left is not None and right is not None and not uncertain_match
                else []
            )
            checked = bool(cues)
            mixed = bool(cues and cues[0] == cues[1])
            boundary_checks.append(
                {
                    "passage_id": passage["id"],
                    "reference_word_offset": offset,
                    "evidence": passage["boundary_basis"],
                    "status": (
                        "mixed" if mixed else "separate" if checked else "unscorable"
                    ),
                    "candidate_cues": cues,
                    "unscorable_reason": (
                        None
                        if checked
                        else (
                            "ambiguous word correspondence"
                            if uncertain_match
                            else "missing boundary words"
                        )
                    ),
                    "reference_context": reference[max(0, offset - 3) : offset + 3],
                }
            )
        reviewed = case["review"]["status"] in ("complete", "complete_with_uncertainty")
        unresolved = any(
            p["uncertainty"].strip() or "[unclear]" in p["text"].casefold()
            for p in case["passages"]
        )
        eligible = reviewed and not unresolved and bool(reference) and not structural
        errors = sum(
            alignment[key] for key in ("insertions", "deletions", "substitutions")
        )
        return {
            "case_id": case["id"],
            "reference_sha256": case["content_sha256"],
            "mechanical": {"structural_issues": structural, "text": alignment},
            "word_error_rate": errors / len(reference) if eligible else None,
            "provisional_word_difference_rate": (
                errors / len(reference) if reference else None
            ),
            "official_score_exclusion": (
                None
                if eligible
                else (
                    "unreviewed reference"
                    if not reviewed
                    else (
                        "unresolved text"
                        if unresolved
                        else "invalid or empty transcript"
                    )
                )
            ),
            "unresolved_reference_passages": [
                {
                    "passage_id": p["id"],
                    "uncertainty": p["uncertainty"],
                    "alternatives": p["alternatives"],
                }
                for p in case["passages"]
                if p["uncertainty"].strip() or "[unclear]" in p["text"].casefold()
            ],
            "speaker_boundaries": boundary_checks,
            "speaker_review_complete": reviewed,
            "semantic_judgment": None,
        }

    @classmethod
    def evaluate(cls, corpus: Corpus, run_path: Path, output: Path) -> Record:
        """Rescore preserved outputs without collecting new transcriptions.

        :param corpus: Reference revision used for this comparison.
        :param run_path: Sealed run manifest with retained output hashes.
        :param output: New report JSON path.
        :return: Sealed report with explicit coverage and official exclusions.
        :raises ValueError: If audio, run identities or retained outputs changed.
        """
        run = Records.load(run_path)
        Records.verify(run)
        Records.require(run["schema_version"] == 1, "Unsupported run schema")
        Records.require(
            run["corpus_id"] == corpus.data["corpus_id"], "Wrong run corpus"
        )
        entries = Records.index(run["cases"], "case_id")
        Records.require(set(entries) <= set(corpus.cases), "Unknown run cases")
        results = []
        for identity, case in corpus.cases.items():
            entry = entries.get(identity)
            if entry is None or entry["status"] != "success":
                results.append(
                    {
                        "case_id": identity,
                        "status": "not_selected" if entry is None else entry["status"],
                        "error": entry.get("error") if entry else None,
                    }
                )
                continue
            Records.require(
                entry["audio_sha256"] == case["audio"]["sha256"],
                f"Changed audio requires recollection: {identity}",
            )
            artifact = entry["srt"]
            path = Records.within(run_path.parent, artifact["path"])
            Records.require(
                Records.file_digest(path) == artifact["sha256"],
                f"Output changed: {identity}",
            )
            try:
                result = cls.compare(case, path.read_text(encoding="utf-8"))
            except ValueError as exc:
                result = {
                    "case_id": identity,
                    "status": "invalid_output",
                    "error": str(exc),
                }
            else:
                result["status"] = "compared"
                result["output_sha256"] = artifact["sha256"]
            results.append(result)
        official = [r for r in results if r.get("word_error_rate") is not None]
        counts = Counter(r["status"] for r in results)
        reference_words = sum(
            r["mechanical"]["text"]["reference_words"] for r in official
        )
        errors = sum(
            r["mechanical"]["text"][key]
            for r in official
            for key in ("substitutions", "insertions", "deletions")
        )
        checks = [b for r in results for b in r.get("speaker_boundaries", [])]
        report = Records.seal(
            {
                "schema_version": 1,
                "created_at": Records.now(),
                "corpus_sha256": corpus.data["content_sha256"],
                "corpus_revision": corpus.data["revision"],
                "run_sha256": run["content_sha256"],
                "backend": run["backend"],
                "normalization_policy": cls.POLICY,
                "summary": {
                    "corpus_cases": len(corpus.cases),
                    "run_cases": len(entries),
                    "statuses": dict(counts),
                    "official_scored_cases": len(official),
                    "official_exclusions": dict(
                        Counter(
                            r["official_score_exclusion"]
                            for r in results
                            if r.get("official_score_exclusion")
                        )
                    ),
                    "unresolved_reference_passages": sum(
                        len(r.get("unresolved_reference_passages", [])) for r in results
                    ),
                    "official_reference_words": reference_words,
                    "official_word_errors": errors,
                    "official_word_error_rate": (
                        errors / reference_words if reference_words else None
                    ),
                    "boundary_statuses": dict(Counter(b["status"] for b in checks)),
                    "human_reviewed_cases": sum(
                        r.get("speaker_review_complete", False) for r in results
                    ),
                    "semantic_judged_cases": 0,
                },
                "cases": results,
            }
        )
        Records.write(output, report)
        return report
