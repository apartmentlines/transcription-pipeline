"""Prepare, review, reconcile and evaluate versioned audio reference transcripts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from reference_core import Corpus, Record, Records
from reference_evaluate import Evaluator
from reference_feedback import Feedback
from reference_prepare import Portal, Preparation
from reference_run import Runs

LOGGER = logging.getLogger(__name__)


class ReviewCommand:
    """Expose independently composable framework operations."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Build explicit subcommands for independent framework operations."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--debug", action="store_true", help="Enable diagnostic logging"
        )
        Records.location_arguments(parser)
        commands = parser.add_subparsers(dest="command", required=True)
        seed = commands.add_parser(
            "seed", help="Import candidates and evidence; curation remains pending"
        )
        seed.add_argument(
            "--spec",
            type=Path,
            help="Source specification; defaults to matching reference audio and SRT files",
        )
        seed.add_argument(
            "--root",
            type=Path,
            required=True,
            help="Repository containing source evidence",
        )
        seed.add_argument(
            "--output", type=Path, required=True, help="New corpus JSON path"
        )
        for name, description in (
            ("validate", "Verify reference records and source hashes"),
            ("curate", "Apply explicit audio-grounded assistant curation"),
            ("portal", "Build a static review site"),
            ("check-feedback", "Validate a portable feedback package"),
            ("reconcile", "Accept, reject or defer human proposals"),
            ("retain", "Import existing backend output without recollecting"),
            ("collect", "Run the production transcriber for selected recordings"),
            ("evaluate", "Rescore retained outputs against this reference revision"),
        ):
            sub = commands.add_parser(name, help=description)
            sub.add_argument(
                "--corpus",
                type=Path,
                help="Corpus JSON; defaults to the current reference revision",
            )
            if name in ("validate", "portal", "collect", "curate"):
                sub.add_argument(
                    "--root",
                    type=Path,
                    default=Path.cwd(),
                    help="Repository root (default: %(default)s)",
                )
            if name not in ("validate", "check-feedback"):
                sub.add_argument("--output", type=Path, required=True)
            if name == "curate":
                sub.add_argument("--proposals", type=Path, required=True)
            if name in ("check-feedback", "reconcile"):
                sub.add_argument("--feedback", type=Path, required=True)
            if name == "reconcile":
                sub.add_argument(
                    "--decisions",
                    type=Path,
                    required=True,
                    help="JSON object with a decisions list",
                )
                sub.add_argument("--actor", required=True, help="Adjudicator identity")
            if name in ("portal", "collect", "retain"):
                sub.add_argument(
                    "--case",
                    action="append",
                    dest="selected",
                    help="Case ID; repeat to select multiple recordings",
                )
            if name in ("collect", "retain"):
                sub.add_argument(
                    "--backend", choices=("qwen3", "whisperx"), required=True
                )
            if name == "retain":
                sub.add_argument("--sources", type=Path, required=True)
                sub.add_argument(
                    "--provenance",
                    type=Path,
                    required=True,
                    help="Source run metadata and audio/output association limits",
                )
            if name == "collect":
                sub.add_argument(
                    "--image",
                    help="Bundled Docker image; omit to use local transcriber",
                )
                sub.add_argument("--manifest", type=Path)
                sub.add_argument("--processing-limit", type=int, required=True)
                sub.add_argument("--qwen-batch-size", type=int)
                sub.add_argument("--qwen-chunk-seconds", type=int)
            if name == "evaluate":
                sub.add_argument("--run", type=Path, required=True)
        return parser

    @staticmethod
    def run(args: argparse.Namespace) -> Record:
        """Execute one parsed operation without automatically advancing workflow.

        :param args: Arguments produced by the framework command parser.
        :return: Structured operation result for the CLI to serialize.
        :raises ValueError: If records or requested operations are invalid.
        """
        LOGGER.debug("Starting %s", args.command)
        if args.command == "seed":
            specification = (
                Records.load(args.spec)
                if args.spec
                else {
                    "corpus_id": "operator-recordings",
                    "cases": [
                        {
                            "id": path.stem,
                            "audio": f"reference-audio/{path.name}",
                            "seed_srt": f"reference-srt/{path.stem}.srt",
                            "evidence": [],
                        }
                        for path in sorted(Records.audio_dir.glob("*.wav"))
                    ],
                }
            )
            corpus = Preparation.seed(args.root, specification, args.output)
            return {
                "corpus_sha256": corpus.data["content_sha256"],
                "cases": len(corpus.cases),
            }
        corpus = Corpus.load(args.corpus) if args.corpus else Corpus.current(Path.cwd())
        if args.command == "validate":
            corpus.verify_sources(args.root)
            return {
                "valid": True,
                "cases": len(corpus.cases),
                "draft_ready": sum(
                    c["preparation"]["status"] == "draft_ready"
                    for c in corpus.cases.values()
                ),
            }
        if args.command == "curate":
            updated = Preparation.curate(
                corpus, args.root, Records.load(args.proposals), args.output
            )
            return {
                "corpus_sha256": updated.data["content_sha256"],
                "revision": updated.data["revision"],
            }
        if args.command == "portal":
            Portal.build(
                corpus,
                args.root,
                Path(__file__).resolve().parents[1] / "assets",
                args.output,
                args.selected,
            )
            return {
                "site": str(args.output),
                "selected_cases": args.selected or list(corpus.cases),
            }
        if args.command == "check-feedback":
            package = Feedback.load_package(corpus, args.feedback)
            return {"valid": True, "proposed_cases": len(package["draft"]["cases"])}
        if args.command == "reconcile":
            updated = Feedback.reconcile(
                corpus,
                args.feedback,
                Records.load(args.decisions)["decisions"],
                args.actor,
                args.output,
            )
            return {
                "corpus_sha256": updated.data["content_sha256"],
                "revision": updated.data["revision"],
            }
        if args.command == "retain":
            run = Runs.retain(
                corpus,
                args.sources,
                args.output,
                args.backend,
                Records.load(args.provenance),
                args.selected,
            )
            return {"run_sha256": run["content_sha256"], "cases": len(run["cases"])}
        if args.command == "collect":
            run = Runs.collect(
                corpus,
                args.root,
                args.output,
                args.backend,
                args.image,
                args.manifest,
                args.processing_limit,
                args.qwen_batch_size,
                args.qwen_chunk_seconds,
                args.selected,
            )
            return {
                "run_sha256": run["content_sha256"],
                "cases": len(run["cases"]),
                "returncode": run["provenance"]["returncode"],
            }
        if args.command == "evaluate":
            return Evaluator.evaluate(corpus, args.run, args.output)["summary"]
        raise ValueError("Unsupported command")


def main() -> None:
    """Run one operation with JSON output and diagnostics on standard error."""
    parser = ReviewCommand.parser()
    args = parser.parse_args()
    Records.configure(args)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        result = ReviewCommand.run(args)
    except (ValueError, KeyError, OSError) as exc:
        if args.debug:
            LOGGER.exception("Framework command failed")
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get("returncode", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
