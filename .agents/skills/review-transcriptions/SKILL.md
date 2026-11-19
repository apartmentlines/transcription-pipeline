---
name: review-transcriptions
description: Operate this project's audio reference corpus, human review site, feedback reconciliation, and Qwen/WhisperX quality comparisons. Use only when explicitly invoked as $review-transcriptions; ordinary unit tests do not require it.
---

# Transcription reference review

Use only when explicitly invoked as `$review-transcriptions`.
Work from the transcription-pipeline repository root in its Python environment.
Read the [review contract](references/contract.md) before revising references or
interpreting scores. Discover exact arguments with:

```text
python .agents/skills/review-transcriptions/scripts/review.py --help
```

Each subcommand also has `--help`; put the optional `--debug` before the
subcommand. The implementation is [scripts/review.py](scripts/review.py).

## Authority and scope

Read [corpus/current.json](../../../../testing-references/transcription-pipeline/corpus/current.json)
to locate the authoritative revision and expected hash. Use `validate` with that
`--corpus` and `--root .` before operating on source artifacts. The
[handoff](../../../test-data/reference-framework/HANDOFF.md) records preparation,
validation, current delivery paths and limitations. Never promote deliberately
modified corpora under `validation/`.

Original audio establishes audible speech. Preserve wording, language,
repetitions and unresolved alternatives. ASR outputs and audio-model judgments
are fallible evidence; assistant curation never constitutes human approval.
Only mixed-speaker passages are grouping defects. Do not score cue lengths,
counts, exact timestamps, or Operator/Caller roles. Speaker-change evidence must
come from listening or an audio assessment, never Python guesses about text.

Stable references live in `~/git/apartmentlines/testing-references/`:
`audio/operator-recordings/` for audio, `transcription-pipeline/srt/` for curated
SRTs, and `transcription-pipeline/corpus/` for immutable revisions and the current
pointer.
Referenced source evidence is preserved by hash under `transcription-pipeline/evidence/`,
with an index mapping historical source paths to those files. Feedback, runs and
generated sites live under the repository's ignored `test-data/reference-framework/`.
Preserve accepted human feedback alongside future stable corpus revisions.
Use `--audio-dir`, `--srt-dir`, `--corpus-dir`, `--evidence-dir`, and `--data-dir` before the
subcommand to override locations. `validate` and `portal` default to the current
corpus. Historical sealed paths are resolved through the documented migration
mapping without changing corpus hashes or invalidating existing feedback. Framework code and synthetic tests live in the
repository. Reference work does not change production transcription behavior.

## Human review and feedback

Open `test-data/reference-framework/review-site/index.html` directly in a browser.
Copy the whole folder, including audio, to share it. It also works on HTTP/HTTPS
static hosting. Optional localhost serving from the repository root:

```text
python -m http.server 8767 --bind 127.0.0.1 --directory test-data/reference-framework/review-site
```

After generating or rebuilding a review site, package the complete site folder
in a `.zip` archive beside it and verify the archive before handing it off.
Include all audio, bundled scripts/data, styles, evidence and reference views;
keep the archive outside the site folder. For the default delivery location:

```text
python -m zipfile -c test-data/reference-framework/review-site.zip test-data/reference-framework/review-site
python -m zipfile -t test-data/reference-framework/review-site.zip
```

Adapt both paths when using another output directory. Preserve earlier delivery
archives when creating a new review revision. Link the ZIP in the handoff and
tell reviewers to extract the entire archive, then open `review-site/index.html`
from the extracted folder. Opening HTML inside an archive preview is insufficient.

Open http://127.0.0.1:8767. Reviewers listen through each recording, correct words,
insert missing speech, remove invented speech, and split passages containing
both voices. Times locate playback; precise timing annotation is unnecessary.
Uncertainty can remain when explicitly completing a review.

Download feedback produces one readable HTML file containing exact structured
edits. Files → Resume from feedback file restores it losslessly. Browser storage
is a convenience; exporting preserves current edits even after storage failure.
Importing a file resumes proposals without changing the reference. Preserve
competing drafts separately when the portal reports a conflict.

For returned feedback:

1. Run `check-feedback` with its exact base `--corpus` and `--feedback` file.
   Preserve stale submissions; compare their actual base revision before
   reconciling differences. Do not change hashes to bypass validation.
2. Inspect proposed text, deletions, splits, uncertainty and completion. Record
   one explicit decision per proposed case in a JSON object with a `decisions`
   list. Each item has `case_id`, `decision` (`accept`, `reject`, or `defer`), and
   a nonempty `rationale`. Ambiguous notes remain notes, not guessed edits.
3. Run `reconcile` with `--corpus`, `--feedback`, `--decisions`, `--actor`, and a
   new `--output` directory. It preserves submitted bytes and the decision ledger
   beside a new `corpus.json`. Accepting an unfinished review leaves it pending.
4. Validate the resulting revision and inspect accepted changes. When it is the
   intended authoritative successor, update `corpus/current.json` with its
   `reference-corpus/` path and content hash. Preserve the previous revision.
   Use `portal --corpus ... --root . --output ...` with a new directory for the
   next review round; use repeated `--case` arguments for a subset. Keep new
   authoritative revisions in the configured corpus directory. After validation,
   refresh the affected stable SRT views from the new portal's `references/`
   directory into the configured SRT directory. The JSON revision remains the
   authority for uncertainty, boundaries and review provenance.

The exact feedback and lineage rules live in the contract and
[scripts/reference_feedback.py](scripts/reference_feedback.py).

## Prepare or refine references

`seed` imports source paths from `--spec` into a new corpus JSON at `--output`;
`--root .` resolves its sources. Without `--spec`, seed discovers matching WAV/SRT
pairs in the configured stable directories. The existing
[source specification](../../../test-data/reference-framework/source-spec.json)
shows the format. Imported text is `seed_only`, not a review-ready reference.

Reconcile candidate differences against the original audio. Keep model edits
as proposals until individually assessed, including plausible deletions,
repetitions and name corrections. Use `curate` with `--corpus`, `--root .`,
`--proposals` and a new corpus JSON `--output`. Proposals contain exact passage
snapshots, base hashes, evidence references and rationale. The frozen
[curation proposals](../../../test-data/reference-framework/curation/r0002/proposals.json)
and [preparation code](scripts/reference_prepare.py) provide a working example
and validation rules. Curation requires native-audio assessment evidence and
leaves human review pending. Preserve raw responses, including rejected ideas.

For additional authorized audio assessments, use the existing
[analyze-call-audio skill](../analyze-call-audio/SKILL.md) and its tool in place.
[scripts/reference_audio.py](scripts/reference_audio.py) has `prepare` and
`collect` commands with `--help`. Preparation freezes requests without paid
calls. Collection uses the frozen tool, model and request settings; it locks the
request set and preserves per-case statuses and raw responses. Inspect existing
statuses and active work before resuming. Failed or malformed responses require
inspection; do not erase history to force another call. Use existing user
approval within its remaining scope; completed requests do not authorize an
additional paid batch. The original 105-call reference collection is complete.

## Compare backend outputs

Use `retain` for existing SRTs named by case identity: provide `--corpus`,
`--sources`, `--backend`, a JSON `--provenance` file describing the original run
and source-association limits, and a new `--output` directory. A retained run
includes copied outputs, hashes, missing cases and its `run.json`.

Use `collect` only for requested fresh execution. It calls the real `transcriber`
CLI with `--corpus`, `--root .`, `--backend`, `--processing-limit`, and a new
`--output` directory. Add `--image` for a bundled Docker image or omit it for the
local executable; `--manifest` overrides the model manifest. Set
`--qwen-batch-size` and `--qwen-chunk-seconds` explicitly for reproducible Qwen
comparisons. Repeated `--case` selects recordings; omission selects all cases.
Use Qwen first, then WhisperX after it exits so they do not compete for the GPU.

Docker collection resolves the image ID before execution. The handoff identifies
the images actually validated and the stale WhisperX image failure. Preserve
`request.json`, logs, partial outputs and exit status. Retry in a new directory;
select unfinished cases rather than overwriting an earlier attempt. A tool wait
ending does not establish that its process ended.

Use `evaluate` with `--corpus`, `--run` pointing to the retained `run.json`, and a
new report JSON `--output`. Existing runs are under `runs/qwen45-baseline` and
`runs/whisperx-baseline`. Reference corrections can rescore them without GPU or
API calls; changed audio or execution settings require a new collection.

The report separates missing, failed and unselected cases, uncertainty,
mechanical word differences, boundary coverage and human review. Official WER
requires human-reviewed, resolved references with valid output. Word comparison
ignores ordinary cue grouping. Ambiguous word mappings and missing boundary
anchors are unscorable, not proof of clean speaker separation. Provisional
comparisons against Qwen-seeded drafts cannot establish a backend accuracy
ranking. Semantic judgments require separate audio review.

Finish with reference/run identities, artifact locations, coverage, observed
results and unresolved limitations. Distinguish preserved-output analysis from
fresh execution and AI drafts from accepted human review.
