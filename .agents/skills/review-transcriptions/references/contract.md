# Reference and review contract

The original WAV is the authority for audible speech. AI-assisted references
are drafts awaiting human review. ASR output, native-audio model judgments,
assistant curation, proposed human corrections and accepted human review remain
separate records. Production transcription and speaker labeling are unchanged.

## Authority and revisions

`corpus.json` is the authoritative case set. Each immutable revision has a
content SHA-256 covering its canonical JSON except `content_sha256`, a revision
number and parent hash. Case IDs remain stable; edited cases receive new
revisions/hashes. Source audio and evidence have relative paths and byte hashes.
Source verification must pass before preparing a portal or collecting outputs.

Each passage has a stable ID, spoken text, approximate navigation interval,
uncertainty, alternatives, and optional speaker-change evidence. Splits preserve
the first passage ID and give new parts `n-` IDs with parent `origin_ids`.
Insertions use new IDs without parents. Deletions remain visible in earlier
revisions and exact feedback. Do not infer speakers using Python text rules.

`speaker_change_before` means an audio-supported change between adjacent
passages. It assigns no identity or role. Other passage breaks are presentation
only. Overlapping times alone never establish a mixed-speaker defect. A split
made to separate voices marks the second part as a speaker change; its replay
interval can remain the original passage's interval.

Importing a candidate creates `seed_only` preparation. Explicit assistant
curation against native-audio evidence creates `draft_ready`, still with human
review pending. Curation must preserve unresolved names, repetitions, language
or speaker transitions; model agreement is not proof. Raw responses remain
unchanged even when a proposed correction is rejected.

## Review and feedback

Human completion explicitly covers listening, words and speaker changes.
The Complete review button permits uncertainty notes and automatically records
completion with uncertainty when those notes or an explicit `[unclear]` remain.
Alternatives and general recording notes alone do not imply uncertainty.
User uncertainty notes never block completion, import or export, including older
drafts marked complete. Initial AI notes in new pending proposals carry
`[REVIEW NEEDED]`; only that marker requires attention before completion, with
links to the affected blocks. Replace the tagged note with review findings or
clear it if resolved. Existing feedback is not retagged. Scoring checks the
actual uncertainty independently of completion status.
Confirmed corrections belong in the main transcript; alternatives do not replace it.
Opening a call, playing audio, importing a file and AI review do not constitute
approval. Reviewers may complete a call with unresolved passages. Their name
and completion timestamp belong to that case, independently of later changes
to the active reviewer name.

Download feedback produces one readable HTML document with an embedded JSON
package. The embedded structured draft is the importable record; the readable
review is its derived view. The document executes no scripts. It can be opened
normally for reading or imported into the site to resume. Export never clears
drafts and includes current editor contents before pending autosaves finish.
An incomplete blank passage is exportable in a draft, but cannot be approved
as reference speech.

The draft identifies project, corpus hash/revision, review session, edit
revision, active case and reviewer. Each proposed case identifies its base
revision/hash and contains proposed ordered passages, disposition and notes.
Browser imports restore proposals only; they do not revise the corpus.

Reconciliation requires explicit accept/reject/defer decisions for every
proposed case, with adjudicator and rationale. Accepted structured edits create
a new revision; notes stay notes and are never guessed text patches. Preserve
the submitted bytes, decision ledger and prior revision. Stale versions,
unknown IDs, duplicate entries, invalid lineage and conflicting edits fail
visibly. Accepting an unfinished draft does not confer human approval.

## Scoring

Compare ordered words across each whole recording. Ignore cue count, line
wrapping, duration and exact boundaries. Normalize Unicode NFKC, case,
whitespace and punctuation; keep accents, digits, repetitions and internal
apostrophes. Do not translate, expand numbers or repair grammar. Raw text and
edit records remain available for adjudicating acceptable wording variations.

Minimum word edit distance is a mechanical comparison, not a semantic verdict.
Report substitutions, insertions and deletions separately. Official WER includes
only human-reviewed, fully resolved recordings with structurally valid output.
Draft comparisons are provisional. Calls with uncertain words are separately
reported with no official score; do not silently remove uncertain words.

Map exact matched reference words to candidate cue IDs. A reference speaker
change is crossed only when its adjacent matched words fall in one candidate
cue. Missing boundary words or equally optimal word alignments that map those
words differently make it unscorable, not clean. This conservative
check follows supplied audio evidence; it does not infer who speaks. Report
checked/unchecked boundaries and human review coverage. No findings cannot
prove speaker purity when boundary coverage is incomplete.

Reports identify the corpus and retained-output hashes, selected cases,
missing/failed outputs, unselected cases, unresolved passages, exclusions and
denominators. Keep semantic judgments
separate from mechanical checks and human approvals. Revised references can
rescore unchanged outputs; changed audio or execution settings require a new
run. These exposed recordings are regression cases, not an untouched holdout.

## Delivery and recovery

The portal contains local HTML/CSS/JavaScript, data and audio. Open index.html directly with file://, or serve it using
ordinary static hosting or localhost; it needs no backend, CDN or cloud API.
IndexedDB is a browser-local convenience, not the authority. Save status follows
completed transactions. Storage failure leaves editing and export in memory.
Compare stored revisions before saving; competing tabs must preserve both
versions through export or a separate session. Compatible saved sessions can
be reopened through Files. Portable feedback also restores a cleared browser.

Stable audio lives under `~/git/apartmentlines/testing-references/audio/operator-recordings/`.
Stable SRTs and versioned corpus records live under that root's
`transcription-pipeline/srt/` and `transcription-pipeline/corpus/` directories.
The corpus retains uncertainty and review provenance that SRT cannot represent.
Working artifacts and generated sites live under the repository's ignored
`test-data/` directory. CLI location overrides precede the subcommand:
`--audio-dir`, `--srt-dir`, `--corpus-dir`, `--evidence-dir`, and `--data-dir`.

Logical source paths use `reference-audio/`, `reference-srt/`, and
`reference-corpus/` prefixes. Historical sealed records retain their original
`audio-files/` paths to preserve hashes and feedback identities; the source
resolver maps those paths to the new storage locations.
Never rewrite old evidence to make paths look current. `test-data/storage-migration.json`
records the relocation and byte verification. Preserve feedback downloads
independently of browser storage.

The portable site uses bundled classic JavaScript and data, with relative audio
paths. It performs no fetch calls or module imports. Copy the whole site folder
for another reviewer; the entry HTML alone does not contain the audio.

Referenced source evidence is stored by hash in `transcription-pipeline/evidence/`.
Its index maps historical paths without changing sealed corpus records.
`--evidence-dir` overrides this location. Archival links under test-data preserve
older working layouts; the stable corpus does not depend on those links.
