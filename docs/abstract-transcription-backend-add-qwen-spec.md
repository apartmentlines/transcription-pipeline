# Feature Specification: Qwen3-ASR + ForcedAligner Backend

**Project:** [apartmentlines/transcription-pipeline](https://github.com/apartmentlines/transcription-pipeline/)  
**Reference commit:** `0b6579a06163991120104665d4cf78266c0f0fd9`  
**Feature:** Add Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B as an alternative to WhisperX  
**Status:** Specification refinement; implementation follows review

**Priority:** External API contracts, clean backend separation, transcription completeness, and reproducibility

---

## 1. Implementation Instructions

When implementation is authorized, implement the feature described in this specification in the existing repository.

Before modifying code:

1. Inspect the current repository, including its processing framework integration, Dockerfiles, CLI entry points, REST interface, model seeding, SRT generation, and tests.
2. Preserve the external contracts defined in section 4.3. Refactor package internals to implement clear backend boundaries.
3. Integrate both backends with the existing processing framework.
4. Verify the exact APIs of the pinned Qwen3-ASR version before implementing its adapter.
5. Implement the feature completely, including code, Docker builds, tests, and documentation.
6. Do not substitute pseudocode, TODOs, or unimplemented stubs for required functionality.
7. Document any requirement that cannot be tested in the available environment.

The desired result is a production-usable second transcription backend, not an experimental script alongside the existing application.

## 2. Objective

Add support for the following model combination:

- **ASR:** `Qwen/Qwen3-ASR-1.7B-hf`
- **Alignment:** `Qwen/Qwen3-ForcedAligner-0.6B-hf`
- **Inference:** Local NVIDIA GPU using PyTorch/Transformers
- **Language support:** English and Spanish, automatically detected
- **Primary output:** SRT subtitles
- **Speaker diarization:** Not required

The WhisperX backend must remain functional, including its ASR, alignment, prompting, and optional diarization capabilities. Both backends adopt the shared output process in section 8.

The two backends must operate in **separate Docker images**, with all required model weights and associated files downloaded during image construction.

Runtime inference must not require Hugging Face or other model-download network access.

### Primary user-facing behavior

Each seeded installation identifies its backend through a model manifest. Normal execution discovers it automatically:

```bash
transcription-processor
transcriber audio.wav --output-dir ./output --output-format srt
```

An optional backend selector asserts the expected backend:

```bash
transcription-processor --backend whisperx
transcription-processor --backend qwen3
```

A standalone transcription command should support the same selector:

```bash
transcriber audio.wav \
  --backend qwen3 \
  --output-dir ./output \
  --output-format srt
```

The REST interface uses the same manifest discovery and accepts an optional backend assertion.

The manifest chooses the application's transcription implementation. Explicit backend assertions validate the selected deployment; model installation happens during seeding or image construction.

Each Docker image contains exactly one backend's model dependencies and weights. Attempting to select a backend not installed in the running image must fail clearly.

## 3. Existing Architecture

The repository currently uses `download-pipeline-processor` as its processing framework.

Relevant files include:

| File | Current responsibility |
|---|---|
| `transcription_pipeline/main.py` | Main CLI, configuration, pipeline orchestration |
| `transcription_pipeline/transcriber.py` | WhisperX transcription, alignment, optional diarization, metadata |
| `transcription_pipeline/processors/transcription_processor.py` | Worker initialization, transcription, SRT formatting |
| `transcription_pipeline/processors/transcription_pre_processor.py` | Audio validation |
| `transcription_pipeline/processors/transcription_post_processor.py` | Result handling, API updates, error classification |
| `transcription_pipeline/rest_interface.py` | One-shot Flask REST execution |
| `transcription_pipeline/seed_models.py` | Download WhisperX and alignment models |
| `transcription_pipeline/constants.py` | Defaults, language restrictions, model configuration |
| `Dockerfile.whisperx` | Multi-stage WhisperX image |
| `docker-compose.whisperx.yaml` | Local build and execution |
| `docker-compose.whisperx.local.yaml` | Execution using the published image |
| `dockerhub-build.sh` | Image building and publication |
| `run-pod.sh` | RunPod initialization and REST execution |
| `pyproject.toml` | Package dependencies and CLI entry points |

### 3.1 Existing processing flow

```text
transcription-processor / REST /run
        |
        v
TranscriptionPipeline
        |
        v
download-pipeline-processor
        |
        +--> Download audio
        |
        +--> TranscriptionPreProcessor
        |       |
        |       +--> WAV validation
        |
        +--> TranscriptionProcessor
        |       |
        |       +--> Transcriber
        |       |       |
        |       |       +--> WhisperX ASR
        |       |       +--> Language validation
        |       |       +--> Forced alignment
        |       |       +--> Optional diarization
        |       |
        |       +--> SRT formatting
        |
        +--> TranscriptionPostProcessor
                |
                +--> Existing application API
```

The new backend must integrate at the transcription stage.

It must not create a separate download queue, duplicate the REST lifecycle, or bypass the post-processor.

### 3.2 Important existing constraints

**Worker initialization**

The pinned `download-pipeline-processor` framework creates a processor instance per worker thread using:

```python
self.processor_class(debug=self.debug)
```

Each existing `TranscriptionProcessor` creates its own `Transcriber`, which loads WhisperX models.

The Qwen backend must account for this because multiple Qwen worker instances will create multiple copies of the ASR and alignment models in GPU memory.

**Current defaults**

- Whisper model: `large-v2`
- Supported languages: `en`, `es`
- Maximum audio duration: 600 seconds
- Minimum audio duration: 5 seconds
- Default processing concurrency: 3
- Default output format: SRT

**Existing output contract**

The processor returns a dictionary containing:

- `id`
- `success`
- `transcription`
- `metadata`

The metadata includes language, segment count, word count, duration, speaking duration, and optional confidence information.

The post-processor submits the SRT text and metadata to the existing application API.

This contract must remain intact.

**Existing REST interface**

The Flask `/run` endpoint accepts configuration and starts `TranscriptionPipeline` in a background thread. The service is designed for one pipeline run per container lifecycle.

The new backend must work through this path as well as the direct CLI.

## 4. Architecture

### 4.1 Backend abstraction

Introduce a small backend abstraction with two implementations:

```text
                   TranscriptionProcessor
                            |
                            v
                   Transcriber Factory
                            |
                 +----------+----------+
                 |                     |
                 v                     v
        WhisperXTranscriber     Qwen3Transcriber
                 |                     |
                 v                     v
        WhisperX + CTC          Qwen3-ASR-1.7B
        alignment               + ForcedAligner
                 |                     |
                 +----------+----------+
                            |
                            v
                 Normalized result schema
                            |
                            v
                    SRT formatting
                            |
                            v
                   Existing post-processor
```

Both implementations must return compatible normalized results.

Suggested structure:

```text
transcription_pipeline/
    backends/
        __init__.py
        base.py
        factory.py
        qwen3.py
        whisperx.py
    subtitles/
        __init__.py
        srt.py
```

These file names are recommendations. Organize code around backend ownership and shared responsibilities; migrate internal imports and tests to the resulting structure.

### 4.2 Interface

Use a typed protocol or abstract interface resembling:

```python
class TranscriptionBackend(Protocol):
    def transcribe(
        self,
        input_file: str | Path,
        context: TranscriptionContext,
    ) -> TranscriptionResult:
        ...
```

`TranscriptionContext` and `TranscriptionResult` are shared typed structures to define during implementation. Context carries recording terminology and community information; each backend translates it into its own prompt. Output paths and subtitle serialization belong to the calling application. Backend-specific model and diarization options belong in typed backend configuration at construction.

The exact interface may differ, but the following must be true:

- Both backends implement a consistent transcription operation.
- Backend-specific dependencies are imported lazily.
- Backend construction is controlled by one factory.
- Both implementations produce the same normalized output contract.
- The processor does not contain large backend-specific inference implementations.
- Backend selection occurs once per pipeline run rather than once per audio file.
- A worker reuses its loaded models for successive files.

### 4.3 External contracts and internal refactoring

Compatibility requirements apply to two boundaries:

1. **The application at `TRANSCRIPTION_DOMAIN`:** preserve authentication, work retrieval, result submission, transcription state transitions, the SRT payload field, and the existing metadata contract.
2. **The Flask REST API:** preserve `/run` and `/status`, authentication, response/status semantics, callbacks, and the one-run container lifecycle. Backend selection and validation are intentional request-contract additions described in section 5.4.

Internal classes, module paths, imports, constructor signatures, CLI options, and tests may change to support a clean design. Migrate their callers together. Remove the old `Transcriber` interface instead of maintaining an alias or compatibility wrapper.

Move WhisperX inference into `backends/whisperx.py` as `WhisperXTranscriber`. Give WhisperX-only concepts explicit ownership: its prompt, batch size, model configuration, seeder, and optional diarization configuration belong under the WhisperX namespace or use a `WHISPERX_` prefix where shared configuration requires one. Reserve generic names for genuinely shared concepts.

Move shared errors, result types, metadata calculation, and subtitle generation into backend-neutral modules. A standalone CLI module handles selection and file output. Existing tests should verify the new boundaries and preserved external behavior.

### 4.4 Processor selection

The processing framework accepts a processor class, not an arbitrary instantiated backend.

A clean implementation is to bind immutable backend configuration to symmetric worker subclasses:

```python
class WhisperXTranscriptionProcessor(TranscriptionProcessor):
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug, backend="whisperx")


class Qwen3TranscriptionProcessor(TranscriptionProcessor):
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug, backend="qwen3")
```

`TranscriptionPipeline._initialize_pipeline()` should select the appropriate processor class:

```python
processor_class = {
    "whisperx": WhisperXTranscriptionProcessor,
    "qwen3": Qwen3TranscriptionProcessor,
}[self.backend]
```

This preserves the upstream processing framework's constructor requirements.

It also avoids relying on a mutable global environment variable to select models inside worker threads.

The exact implementation can differ, but the full resolved configuration, including precision and model paths, must reach each worker reliably. The shared processor has no implicit WhisperX implementation. A configuration-bound class or factory compatible with the pinned framework is also acceptable; verify the constructor contract before choosing it.

## 5. Backend Configuration

### 5.1 Supported identifiers

Use stable backend identifiers:

- `whisperx`
- `qwen3`

Do not make the backend identifier equal to the full Hugging Face model name.

Model versions and model revisions are separate configuration concepts.

### 5.2 Manifest discovery and backend resolution

Use one application-owned manifest per seeded deployment as the source of truth for its backend and model assets. Both image builders and the local seeding command write this manifest after validating the required files.

Resolve its location in this order:

1. CLI `--model-manifest PATH` (including server startup).
2. `TRANSCRIPTION_MODEL_MANIFEST` environment variable.
3. `Path(huggingface_hub.constants.HF_HOME) / "transcription-pipeline" / "manifest.json"`.

With the existing Docker `HF_HOME`, the default becomes:

```text
/opt/models/huggingface/transcription-pipeline/manifest.json
```

Keep application metadata alongside the Hub cache under `HF_HOME`, while model snapshots remain under `huggingface_hub.constants.HF_HUB_CACHE`. This follows Hugging Face's `HF_HOME`/`XDG_CACHE_HOME` resolution and honors a separately configured `HF_HUB_CACHE` for weights. Hugging Face has no application-backend manifest convention; discovery of this namespaced file is this package's responsibility. Set environment configuration before importing Hub constants. See the [Hub configuration documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

Read `schema_version`, `backend`, and the asset inventory from the selected manifest. Resolve asset paths against named roots (`hf_hub_cache`, `torch_home`, and `nltk_data`) so relocating a cache does not require rewriting absolute paths. Validate the schema, backend, installed backend dependencies, model revisions, and required files before work retrieval or GPU initialization. A manifest identifies intended installation contents; file and dependency checks establish their availability.

`--backend` and `TRANSCRIPTION_BACKEND` are optional assertions. CLI takes precedence over the environment assertion; the effective assertion must match the manifest. `/run` may also assert the backend as described below. Use the manifest alone when assertions are omitted. Image startup and local execution share this single resolution process.

An explicitly selected missing manifest, a malformed manifest, or a missing default manifest produces an actionable configuration error identifying the path and seeding command. CLI help and the seeding command work before a manifest exists. The bootstrap command `seed-models --backend BACKEND` selects what to install and writes the manifest to the same resolved location. Inspect the one resolved path rather than scanning model caches or guessing from installed packages. Local installations with multiple deployments select their manifest explicitly; each manifest still describes exactly one backend.

### 5.3 CLI integration

Add optional `--backend` and `--model-manifest` to:

- `transcription-processor`
- `transcriber`

Use clear backend-specific options where appropriate and update deployment commands and documentation together. Retain useful command entry points; their old signatures are not compatibility requirements.

Log the resolved manifest path, backend, model identifiers, and versions at startup.

### 5.4 REST integration

Resolve and validate the server's manifest at startup. `/run` uses that backend when `backend` is omitted. An optional `backend` field asserts that the request reached the intended deployment.

Example:

```json
{
  "api_key": "API_KEY",
  "domain": "example.com",
  "limit": 100,
  "processing_limit": 3
}
```

Requirements:

- Omitted backend uses the server's manifest-backed configuration.
- A supplied null, empty, or non-string backend returns HTTP 400.
- Unknown backend returns HTTP 400.
- A supplied backend that differs from the manifest returns HTTP 400 before starting the pipeline.
- The validated backend is passed to `TranscriptionPipeline`.
- Existing callback and lifecycle behavior remains intact.

The `rest-interface` command accepts the same manifest path and optional backend assertion as the CLI. REST requests cannot select a filesystem manifest path or change the server deployment.

Do not change the existing `/run` or `/status` endpoint names.

## 6. Qwen3-ASR Implementation

### 6.1 Native Transformers inference

Use the official native Transformers checkpoints:

- [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)

Use Python 3.12, CUDA-enabled PyTorch, and `transformers==5.22.2` as the initial dependency baseline. This released version contains both native model classes and the transcription/alignment processor helpers; verify dependency resolution and pin the complete compatible stack during implementation. Support RTX 3090 (Ampere) and newer NVIDIA GPUs with the selected PyTorch CUDA build.

Use BF16, `attn_implementation="sdpa"`, eager model execution, inference mode, and one chunk at a time per worker. PyTorch dispatches SDPA to its available CUDA kernels. This gives the application direct generation-token access and uses standard released model APIs without a separate inference server or custom attention build. The selected runtime consists of Transformers and PyTorch; the `qwen-asr` wrapper, vLLM, external `flash-attn`, and `torch.compile` are outside this initial runtime configuration.

The adapter owns chunk orchestration, language validation, and offsets. Native processor helpers handle model inputs and forced-alignment decoding. This is a deliberate tradeoff: a small explicit chunk loop gives access to termination diagnostics and text assembly without modifying the old wrapper. See the released [processor implementation](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_asr/processing_qwen3_asr.py) and [model implementation](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_asr/modeling_qwen3_asr.py).

### 6.2 Model loading

Load both models from paths baked into the Docker image.

Use the existing image's model storage convention. WhisperX already uses `HF_HOME=/opt/models/huggingface` and `TORCH_HOME=/opt/models/torch`. Qwen snapshots belong in that same Hugging Face cache root within its separate image:

```text
/opt/models/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/<asr-revision>/
/opt/models/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B-hf/snapshots/<aligner-revision>/
```

At build time, use `snapshot_download` with the resolved `HF_HUB_CACHE` and immutable revisions. Record snapshot paths relative to that root in the manifest defined in section 5.2. Resolve them at runtime and pass the resulting local directories to the loaders. Copy cache blobs together with snapshot directories so their relative symlinks remain valid. Keep each image's cache limited to its own backend's required assets.

Local development may configure the cache root; seeding and loading must resolve the same root and manifest. This follows the [Hugging Face cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

The implementation should resemble:

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification
from transformers import AutoProcessor

asr_processor = AutoProcessor.from_pretrained(
    asr_snapshot_path, local_files_only=True,
)
asr_model = AutoModelForMultimodalLM.from_pretrained(
    asr_snapshot_path,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    local_files_only=True,
).eval()
aligner_processor = AutoProcessor.from_pretrained(
    aligner_snapshot_path, local_files_only=True,
)
aligner_model = AutoModelForTokenClassification.from_pretrained(
    aligner_snapshot_path,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    local_files_only=True,
).eval()
```

Treat this as an API reference, not a drop-in final implementation.

Verify the exact keyword arguments against the pinned package version.

Model loading requirements:

- Load the models once per worker.
- Use local filesystem paths.
- Run in inference mode.
- Avoid retaining unnecessary GPU tensors.
- Fail clearly if CUDA is unavailable.
- Fail clearly when required local model files are missing.
- Do not download models at runtime.
- Do not substitute a different model when loading fails.

BF16 is the default for the target hardware. Permit an explicit FP16 configuration if needed, and validate the selected precision at startup. Load each model on the selected CUDA device rather than using automatic CPU offload.

The adapter executes one ASR chunk followed by its alignment before advancing to the next chunk. This batch size of one is separate from `processing_limit=3`, which controls independently loaded worker instances. Release chunk-local GPU inputs, outputs, and generation caches before advancing; keep only CPU text, timings, and diagnostics.

### 6.3 Automatic language detection

Automatic detection is mandatory.

Pass:

```python
language=None
```

to `asr_processor.apply_transcription_request` for each chunk.

The model may detect any of its supported languages. The application must validate its results against English and Spanish.

Handle:

- `English` → `en`
- `Spanish` → `es`
- Mixed English/Spanish recordings
- Empty or unknown detection
- Unsupported detected languages

The application must not force every recording to English or Spanish before transcription.

However, unsupported language results must not be silently relabeled as one of the two supported languages.

For mixed English/Spanish recordings:

- Preserve the detected language information.
- Support recordings containing English and Spanish sections.
- Expose a `detected_languages` metadata field when appropriate.
- Verify that any representation of mixed language is compatible with the existing downstream API.
- If downstream compatibility requires one primary-language value, determine and document that rule rather than choosing arbitrarily.

Qwen's automatic detection is not guaranteed to correctly identify every short or code-switched utterance. Include test cases for both languages and mixed-language recordings.

Validate each chunk's detected language before invoking alignment. Normalize labels, retain the original detection string for diagnostics, and keep `detected_languages` unique in first-occurrence order. These are chunk-level detections; a single detected label is not proof that every word uses that language. Preserve English/Spanish code-switching in the text even when only one language is detected.

An empty-transcript chunk with no detected language contributes an empty interval and requires no alignment. Preserve its position in sample coverage. Nonempty text with unknown or unsupported language requires a classified error. An entirely empty recording follows the existing empty-transcription failure behavior.

### 6.4 Context prompting

The existing WhisperX implementation uses `INITIAL_PROMPT`, formatted with `file_data.record_name`.

Preserve its WhisperX behavior.

For Qwen3, pass the context through the native processor's `prompt` parameter to provide relevant terminology.

A compact, Qwen-specific context is preferable to blindly reusing the entire Whisper prompt.

Suggested context:

```text
This audio is a telephone conversation with an apartment
community answering service. Topics may include maintenance,
emergencies, security, parking, access gates, HVAC, plumbing,
and resident requests. Preserve the spoken language, including
English and Spanish.
```

Optionally include the apartment community name.

Requirements:

- Context may provide vocabulary and domain information.
- Context must not instruct the model to invent missing speech.
- Context must not force English output.
- Keep context concise.
- Preserve the ability to customize context per recording.
- Test whether prompting improves terminology without increasing hallucinations.

Do not translate Spanish speech into English.

Output must be a transcription of the original spoken language.

### 6.5 Audio processing

The current pre-processor validates WAV files between 5 and 600 seconds.

Preserve those existing limits.

For Qwen3:

- Support the WAV formats accepted by the current pipeline.
- Normalize audio to the sample rate and channel format required by Qwen.
- Preserve the original recording's time coordinate system.
- Do not modify the original downloaded file.
- Do not use lossy intermediate audio encoding.
- Do not introduce aggressive speech filtering or VAD that could discard speech.

Use existing audio decoding facilities where practical.

Qwen's official implementation normalizes audio to mono 16 kHz.

Avoid decoding the same recording multiple times when it can be reused safely.

### 6.6 Long recordings and chunking

The native model/processor calls operate on supplied audio chunks; the adapter supplies long-recording orchestration. Use a maximum of 180 seconds per chunk for predictable memory and alignment behavior. This is an application limit, not a claim that native Transformers performs the old wrapper's integrated chunking.

Decode and normalize the complete recording once to mono 16 kHz. Represent chunks as half-open integer sample ranges covering the waveform exactly once. For an interior cut, select a deterministic low-energy boundary within the final five seconds before the 180-second limit; low energy chooses a cut position and never removes samples. Record each chunk's original start sample. Process the final remainder in full, including short remainders.

For each chunk, run ASR, check generation termination, validate language, run forced alignment on the same waveform slice, and add `start_sample / sample_rate` to local timestamps exactly once. Pass sample rate explicitly and disable feature-extractor truncation; verify that the feature mask accounts for the complete slice. Chunk-local context and token limits must fit the pinned model's context capacity.

Assemble authoritative text in chunk order with explicit whitespace separation for English/Spanish. Preserve each chunk's text and offsets until validation completes. Test a boundary between two letter-ending/letter-starting words without whitespace or punctuation, as well as sentence punctuation at a boundary. The initial chunker uses contiguous ranges, so duplicate-word removal is unnecessary.

CPU tests verify complete sample coverage, deterministic boundaries, short tails, text joining, and timestamp offsets on synthetic recordings through 600 seconds. Optional GPU checks assess recognition quality around those boundaries.

### 6.7 Forced alignment

Use the native ASR generation API, followed by the native forced-aligner processor and token-classification model on the same chunk. The flow is:

```python
inputs = asr_processor.apply_transcription_request(
    audio=chunk_audio,
    prompt=context,
    language=None,
    audio_kwargs={"sampling_rate": 16000, "truncation": False},
).to(asr_model.device, asr_model.dtype)
with torch.inference_mode():
    output_ids = asr_model.generate(
        **inputs, max_new_tokens=2048, do_sample=False,
        logits_to_keep=1, return_dict_in_generate=False,
        output_scores=False, forced_eos_token_id=None,
    )
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

# Check termination before decoding; parse the raw text as described below.

aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=chunk_audio,
    transcript=chunk_text,
    language=detected_language,
    audio_kwargs={"sampling_rate": 16000, "truncation": False},
)
aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)
with torch.inference_mode():
    alignment_output = aligner_model(**aligner_inputs)
timings = aligner_processor.decode_forced_alignment(
    logits=alignment_output.logits,
    input_ids=aligner_inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner_model.config.timestamp_token_id,
)[0]
```

These snippets describe API boundaries rather than a complete implementation. Test the exact processor kwargs against the pinned release. Decode the generated suffix as raw text with tokenization-space cleanup disabled, remove only protocol/control tokens, and parse the language header and `<asr_text>` separator. Preserve the transcript body. The pinned processor's parsed-output helper performs repetition cleanup, which can alter real repeated speech; use a small protocol parser to preserve authoritative text and report suspected repetition separately. See the [released processor source](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_asr/processing_qwen3_asr.py).

Alignment returns dictionaries containing:

- `text`
- `start_time`
- `end_time`

Convert them into the normalized word-timestamp representation, adding the chunk offset. Timing repair in the library does not replace application validation of text coverage and timestamp bounds.

Do not treat alignment timestamps as transcription confidence scores.

### 6.8 Punctuation and text preservation

**Critical requirement:** The Qwen forced aligner's text processor removes much punctuation when preparing alignment tokens.

Therefore:

**Do not construct the final SRT simply by joining the aligner's word strings.**

This can lose punctuation, capitalization, and other information present in the ASR transcription.

Implement a reconciliation step that combines:

1. The authoritative ASR transcript text.
2. The aligner's word-level timing information.

The resulting SRT should retain:

- Original language
- Capitalization
- Sentence punctuation
- Spanish accents
- Spanish inverted punctuation (`¿`, `¡`)
- Contractions and apostrophes
- Numbers
- Common abbreviations
- Original word order

The reconciliation should be deterministic and tested.

It must not silently discard words or punctuation when ASR and alignment tokenization differ.

For tokens that cannot be aligned reliably:

- Preserve their transcript text.
- Use an explicitly identified, conservative timing fallback where feasible.
- Record an alignment warning.
- Fail transcription if a usable timestamp sequence cannot be produced.

Do not pretend inferred fallback timestamps have the same precision as model-derived alignment.

## 7. Normalized Transcription Results

Both backends return a shared typed result containing authoritative transcript text, audio duration, normalized timed words, source segments, detected language information, and diagnostics. Words reference character spans in the authoritative text and identify timing provenance (`aligned` or `inferred`). Preserve source segment boundaries and optional speaker labels separately from generated subtitle cues.

The following dictionary illustrates the current metadata and segment concepts; it does not prescribe a legacy internal dictionary interface. Use one canonical word sequence and derive segment views from it to avoid divergent duplicate word lists.

Example:

```python
{
    "language": "en",
    "segments": [
        {
            "start": 0.5,
            "end": 2.4,
            "text": "Hello, how can I help you?",
            "words": [
                {
                    "word": "Hello,",
                    "start": 0.5,
                    "end": 0.9,
                },
            ],
        },
    ],
    "word_segments": [],
    "total_words": 6,
    "total_duration": 2.4,
    "speaking_duration": 1.9,
}
```

The example is illustrative, not a complete transcript.

Requirements:

- `segments` must be ordered chronologically.
- Every segment must contain text, start, and end timestamps.
- Timings must use seconds internally.
- Segment start must be less than segment end.
- Timestamp values must be finite and nonnegative.
- Segments must not extend beyond the actual audio duration, apart from negligible floating-point tolerance.
- The canonical word sequence must contain normalized word timing information where available; any `word_segments` compatibility-shaped view is derived from that sequence.
- `total_words` must reflect actual transcribed words.
- `speaking_duration` must not double-count overlapping segments.
- Preserve external metadata meanings through a shared metadata mapper, independent of subtitle layout. In particular, `total_duration` remains the final speech segment end, while `audio_duration` describes the whole recording. `speaking_duration` measures the union of source speech intervals, and `segments` counts source transcription segments. Add a separate `subtitle_cue_count` for rendered cues. Establish fixtures for these meanings before replacing the old calculation.

WhisperX currently provides optional word-confidence scores.

Qwen's forced aligner does not provide an equivalent calibrated confidence value.

For Qwen results:

- Do not fabricate confidence scores.
- Submit `average_word_confidence: null` when the backend has no equivalent score.
- Do not convert timestamp quality into confidence.

The current `TRANSCRIPTION_DOMAIN` update handler recursively merges submitted metadata into existing metadata (`_al_update_transcription_metadata` in `al_download_file.inc`). Omitting confidence could retain a previous WhisperX score when Qwen retranscribes a recording. Test this update case and explicitly clear inapplicable values. Verify replacement semantics for list-valued additions such as `detected_languages` before submitting them; a shorter numeric array may otherwise retain old entries. Internal language lists can remain structured regardless of the external representation chosen.

Add optional metadata:

```json
{
  "backend": "qwen3",
  "asr_model": "Qwen/Qwen3-ASR-1.7B-hf",
  "alignment_model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
  "detected_languages": ["en"],
  "audio_duration": 180.0,
  "alignment_warnings": 0
}
```

Preserve the existing top-level processor response:

```python
{
    "id": file_data.id,
    "success": True,
    "transcription": srt_content,
    "metadata": metadata,
}
```

Do not change the existing API endpoints or the existing SRT field name.

## 8. SRT Generation

### 8.1 One shared subtitle pipeline

Implement one backend-independent subtitle pipeline used by both backends, the worker pipeline, and the standalone CLI:

```text
Backend ASR text + alignment
        -> normalized transcript with timed text spans
        -> shared cue segmentation
        -> shared SRT serialization
        -> string for API submission / UTF-8 file for standalone output
```

Backend adapters reconcile their model-specific alignment tokens with the authoritative text. The shared layer owns cue boundaries, punctuation placement, line wrapping, timing validation, speaker-label rendering when present, and serialization. Backend implementations return transcription data rather than writing subtitles.

Use a shared policy for partially unaligned text from either backend: preserve it, infer bounded timing only from reliable neighboring timings or source segment bounds, and mark the inference in diagnostics. Give each transcript text span exactly one owner in the resulting cues. When optional speaker labels are rendered, validate transcript preservation separately from those added labels.

Replace both existing `whisperx.utils.get_writer` SRT call sites. The shared subtitle modules import only lightweight backend-neutral dependencies and use the same defaults for WhisperX and Qwen. An identical normalized transcript and subtitle configuration must produce byte-identical SRT through either backend or entry point.

The existing WhisperX writer's exact cue boundaries and line wrapping are not compatibility requirements. Regression tests preserve text, chronology, valid SRT, optional speaker labels, and the downstream API contract. Consolidating the current standalone and worker formatting differences is an intentional change.

Separate cue construction from serialization so any supported VTT output can reuse the same cues. Keep format support explicit; reject unsupported formats before model initialization. Retain standalone SRT and VTT support, which the current CLI advertises, through shared serializers.

Use these existing libraries in the shared layer:

| Responsibility | Selected implementation |
|---|---|
| Sentence and abbreviation boundaries | NLTK Punkt with the English/Spanish `punkt_tab` assets already used by the WhisperX image |
| Timed cue grouping | A small shared policy function combining sentence spans, word timestamps, pauses, speaker changes, and the limits below |
| SRT and VTT serialization | `pysubs2==1.9.1`, using `SSAFile`, `SSAEvent`, and in-memory `to_string` |

`pysubs2` is an MIT-licensed library maintained since 2011, supports both required formats, and has no runtime dependencies. Its 1.9.1 release supports Python 3.12+. It handles subtitle syntax and timestamp serialization; it does not turn arbitrary word timestamps into readable cues. NLTK's `span_tokenize` supplies sentence boundaries without rewriting transcript text. Declare both libraries as shared dependencies and bake the English/Spanish Punkt data into both images at `/opt/models/nltk`; pin/checksum the data artifact used in builds. Sources: [pysubs2 documentation](https://pysubs2.readthedocs.io/en/latest/), [release notes](https://pysubs2.readthedocs.io/en/latest/release-notes.html), [Punkt API](https://www.nltk.org/api/nltk.tokenize.punkt.html).

Stable-ts offers word-timestamp regrouping, but its repository is archived and its required dependencies include `openai-whisper`. That is a poor fit for the shared layer and Qwen dependency isolation. Keep only the application-specific timed grouping policy in this package rather than importing another inference stack or copying its subtitle engine. The SRT-only `srt` library is a reasonable serializer, but `pysubs2` covers both SRT and VTT through one maintained API. See [stable-ts dependency metadata](https://github.com/jianfch/stable-ts/blob/main/setup.py).

Construct plain-text events and configure the writer to preserve literal text rather than interpret transcript braces as ASS styling. Normalize intended line breaks explicitly. Test braces, backslashes, markup-like text, Spanish punctuation, and Unicode against the pinned serializer, including cue count and transcript preservation after serialization. Keep the adapter small; use library timestamp rendering and cue numbering.

### 8.2 Shared subtitle segmentation

Use normalized timed text spans to group text into readable subtitle cues for both backends. Slice the authoritative transcript using those spans; alignment token spelling is used for matching rather than rendered text. Apply only documented whitespace normalization when comparing reconstructed cue text to the transcript.

Initial recommended defaults:

| Setting | Default |
|---|---|
| Maximum lines per cue | 1 |
| Target maximum characters per cue | 42 |
| Target minimum cue duration | 1.0 second |
| Target maximum cue duration | 6.0 seconds |
| Preferred breaks | Sentence and punctuation boundaries |
| Secondary breaks | Natural pauses |
| Pause that forces a cue break | 1.0 second |

These values must be configurable in code.

Use one subtitle line per cue by default for both backends and both entry points. Duration and character targets are soft limits; text preservation and valid timing take precedence. Obtain sentence boundary candidates from the appropriate Punkt tokenizer using the detected source language. Mixed-language fixtures must preserve text even if boundary detection is imperfect. Snap boundary candidates to complete normalized word spans, then apply timing, pause, and speaker-change rules through the shared grouping policy.

Rules:

- Never discard words merely to satisfy a length limit.
- Avoid separating punctuation from its associated word.
- Preserve chronological order.
- Do not generate zero-duration cues.
- Avoid overlapping cue intervals unless unavoidable and explicitly handled.
- Do not merge text across a substantial silence merely to reach a character target.
- Allow exceptions for unusually long spoken words or unavoidable timing constraints.
- Avoid merging complete sentences into one excessively long cue.

### 8.3 SRT timestamps

Use standard SRT timestamps:

```text
HH:MM:SS,mmm
```

Example:

```srt
1
00:00:00,500 --> 00:00:02,400
Hello, how can I help you?

2
00:00:02,700 --> 00:00:04,200
Tengo una emergencia.
```

The formatter must:

- Use UTF-8.
- Use sequential cue numbers starting at 1.
- Use comma-separated milliseconds.
- Round timestamps consistently.
- Handle recordings crossing minute boundaries correctly.
- Preserve non-ASCII characters.
- Produce valid output without trailing malformed cues.

Validate cues again after millisecond quantization: positive floating-point durations can collapse to zero at SRT precision. Merge such a cue with an adjacent compatible cue when possible, or apply a documented one-millisecond adjustment within the audio bounds without introducing overlap. Report a formatting error when neither is possible.

### 8.4 Standalone output

The `transcriber` CLI must use the shared subtitle pipeline with output-directory and output-format options for both backends.

The worker pipeline must continue producing SRT strings for API submission rather than relying on intermediate SRT files.

Writing a temporary file should be optional, not required.

## 9. Transcription Completeness and Diagnostics

An important motivation for this feature is that WhisperX `large-v2` has sometimes omitted entire sections of speech.

This feature must avoid introducing silent data-loss behavior.

### 9.1 Full-audio inference

The Qwen backend should initially use the official ASR inference path without WhisperX-style external VAD preprocessing.

This reduces one potential source of omitted speech.

It does not guarantee perfect transcription.

### 9.2 Validate ASR/alignment consistency

After inference, inspect the outputs for:

- Nonempty ASR transcription with no aligned words.
- Large discrepancies between ASR text and reconstructed subtitle text.
- Missing or malformed timestamp values.
- Words outside the audio time range.
- Large unexplained timestamp discontinuities.
- Alignment output that stops substantially before the final transcribed passage.
- Duplicate or reordered words introduced during subtitle generation.

Do not treat long silence as proof of missing transcription.

A recording can legitimately contain long pauses.

### 9.3 Preserve the raw transcript

Keep the original ASR text available internally until all validation and SRT formatting have completed.

Add an optional debug output mode that can write:

```text
recording.txt
recording.srt
recording.diagnostics.json
```

These files are for debugging and comparison.

They are not required for the normal API-processing path.

### 9.4 Diagnostic metadata

Suggested fields:

```json
{
  "audio_duration": 300.0,
  "last_aligned_word_end": 295.2,
  "unaligned_word_count": 0,
  "suspicious_gap_count": 0,
  "alignment_warnings": 0
}
```

Diagnostics must not expose downloaded audio, resident information, or full transcript text in ordinary application logs.

Do not silently mark invalid or empty subtitle generation as successful.

### 9.5 Boundary testing

Use synthetic audio and mocked inference results to verify text and timestamps around every computed chunk boundary, nominal 180/360-second limits, and the end of a recording near 600 seconds. Low-energy cut selection can move boundaries earlier, so tests must inspect the actual sample ranges. Include words on both sides of cuts, silence-only chunks, and short final remainders. Optional GPU deployment checks use real speech at those positions.

### 9.6 Generation-limit detection

Keep truncation detection with native Transformers: `generate` still stops at `max_new_tokens` even if speech remains. Inspect only generated suffix token IDs, using the effective generation configuration's EOS token IDs. Record generated token count and whether normal EOS termination occurred. Disable forced EOS insertion at the length limit so an inserted marker cannot disguise truncation. Naturally generated EOS on the final allowed token is a normal completion; exhausting the budget without EOS is a truncation error. Treat any other unexpected stop without EOS as incomplete generation as well.

Use `max_new_tokens=2048` per chunk initially. On incomplete generation, fail the recording explicitly with a classified diagnostic rather than submitting partial subtitles. Keep the check local to the adapter; it needs neither a wrapper patch nor retained per-token scores/logits. A later retry policy can be added if deployment evidence warrants it. Unit tests cover early EOS, EOS at the limit, limit without EOS, multiple EOS IDs, and prompt-token exclusion. See [Transformers generation controls](https://huggingface.co/docs/transformers/main_classes/text_generation).

Normal EOS proves only that generation ended normally. Audio coverage and transcript/alignment checks remain separate checks; none guarantees the model recognized every spoken word.

## 10. Docker Architecture

### 10.1 Separate images

Maintain two independently built and independently published runtime images:

```text
WhisperX image
    |
    +-- WhisperX dependencies
    +-- large-v2 model
    +-- English alignment model
    +-- Spanish alignment model
    +-- Existing application code

Qwen3 image
    |
    +-- Native Transformers/PyTorch dependencies
    +-- Qwen3-ASR-1.7B
    +-- Qwen3-ForcedAligner-0.6B
    +-- Existing application code
```

The Qwen image must not download or install WhisperX models.

The WhisperX image must not download or install Qwen models.

Do not create one combined image containing both model families.

### 10.2 Dockerfile strategy

Name the WhisperX image definition `Dockerfile.whisperx`.

Introduce:

```text
Dockerfile.qwen3
```

The new Dockerfile should follow the general multi-stage approach used by the existing image:

```text
base
  |
  v
dependencies-build
  |
  v
dependencies
  |
  v
models
  |
  v
application
  |
  +--> development
  |
  +--> runtime
```

The exact stage organization may be adjusted for Docker layer caching.

Important requirements:

- Prefer a CUDA-compatible Python/PyTorch environment.
- Use a pinned base image.
- Include FFmpeg and required audio libraries.
- Install build tools only in build stages.
- Exclude compiler and test dependencies from the final runtime image.
- Preserve SSH and RunPod tooling if required by the existing deployment workflow.
- Keep dependency installation independent of application-source changes.
- Keep model downloads independent of application-source changes.
- Preserve the current no-GPU-required Docker build behavior.

The Qwen3 image may use a different Python version from the existing WhisperX image if required by the dependency stack.

Python 3.12 is an acceptable initial target.

### 10.3 Dependency isolation

The existing `pyproject.toml` includes WhisperX as a runtime dependency.

This must not cause WhisperX and its dependency stack to be installed in the Qwen image.

Recommended approach:

- Retain shared application dependencies in the package metadata.
- Introduce separate backend-specific, pinned dependency files or optional dependency groups.
- Install only the dependency set required by each image.
- Continue installing the application package without pulling in unrelated backend dependencies.

For example:

```text
requirements-whisperx-frozen.txt
requirements-qwen3-frozen.txt
```

The precise dependency-file structure is flexible.

The required behavior is not:

```text
Install everything, then choose at runtime
```

It is:

```text
Install only the selected backend's dependencies
```

The Qwen image must successfully import and run the application without WhisperX installed.

The WhisperX image retains its own pinned inference dependencies; it does not require Qwen model support. The Qwen image uses native Transformers without the `qwen_asr` package.

### 10.4 Eliminate eager WhisperX imports

The current source contains eager WhisperX imports, including:

- The `whisperx.utils` writer import in `transcriber.py`.
- The `whisperx.utils` writer import in `transcription_processor.py`.
- Public package imports that transitively load these modules.
- Error-handling code that imports the current transcriber module.

Refactor these imports as needed.

A Qwen-only image must be capable of importing:

```python
import transcription_pipeline
from transcription_pipeline.main import TranscriptionPipeline
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from transcription_pipeline.rest_interface import TranscriptionRestInterface
```

without requiring WhisperX.

Likewise, WhisperX startup must not import Qwen's heavy dependencies.

Use lazy imports and a backend-neutral location for shared errors and output types.

### 10.5 Model seeding during Docker build

Introduce a Qwen model seeding implementation.

Suggested organization:

```text
transcription_pipeline/backends/qwen3_seed.py
transcription_pipeline/backends/whisperx_seed.py
```

Expose seeding through a backend-aware command that lazily selects the corresponding implementation. Rename the current generic WhisperX-only `ModelSeeder` to `WhisperXModelSeeder` when retaining a class-based implementation.

At build time, download complete snapshots for:

```text
Qwen/Qwen3-ASR-1.7B-hf
Qwen/Qwen3-ForcedAligner-0.6B-hf
```

Use Hugging Face `snapshot_download` or another officially supported download mechanism.

The seeding process must include:

- Model weights
- Model configuration
- Tokenizer files
- Processor files
- Other assets required for inference
- Any necessary associated configuration files

Download to the cache layout and manifest paths specified in section 6.2. Validate that snapshot symlinks resolve inside the final image.

Do not depend on model files existing in a developer's home-directory cache.

### 10.6 Pin model revisions

**Do not download floating `main` revisions for release images.**

Pin each Hugging Face model to an immutable commit SHA.

Record the selected model revisions in the Docker build configuration and model manifest.

Manifest schema example, stored at the location resolved in section 5.2:

```json
{
  "schema_version": 1,
  "backend": "qwen3",
  "runtime": "transformers",
  "models": {
    "asr": {
      "repo_id": "Qwen/Qwen3-ASR-1.7B-hf",
      "revision": "<ASR_COMMIT_SHA>",
      "root": "hf_hub_cache",
      "path": "models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/<ASR_COMMIT_SHA>"
    },
    "aligner": {
      "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
      "revision": "<ALIGNER_COMMIT_SHA>",
      "root": "hf_hub_cache",
      "path": "models--Qwen--Qwen3-ForcedAligner-0.6B-hf/snapshots/<ALIGNER_COMMIT_SHA>"
    }
  },
  "assets": [
    {"root": "nltk_data", "path": "tokenizers/punkt_tab/english"},
    {"root": "nltk_data", "path": "tokenizers/punkt_tab/spanish"}
  ]
}
```

Both backend seeders write this schema with their own asset inventory. WhisperX records its Whisper, English/Spanish alignment, and any configured diarization assets, including Torchaudio cache files under `torch_home`. Validate inventories without importing the other backend. Write the manifest atomically after successful seeding and asset verification.

The implementer must resolve and verify the actual commit SHAs and pinned shared-data checksum. Placeholders in this schema example are not release values.

Do not invent revisions or substitute branch names.

Model revisions, Python dependencies, and image tags must be reproducible.

A model revision change must produce a new image version rather than silently changing an existing published artifact.

### 10.7 Runtime offline enforcement

Set appropriate offline environment variables in the Qwen runtime image:

```dockerfile
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
```

Load models from local paths.

Do not require:

- Hugging Face credentials
- Hugging Face network access
- Model downloads on first use
- Mounted model-cache volumes

The runtime still requires network access to the existing transcription API and audio download endpoints.

Offline model loading must not interfere with that application network access.

### 10.8 Image verification

During the image build, validate that required model files exist.

A GPU must not be required merely to build the image.

GPU-dependent initialization and inference checks are optional deployment validation, separate from development completion. On this development box, verify manifest contents, snapshot integrity, config/tokenizer/processor loading, and offline resolution without allocating CUDA models.

The image must fail fast during startup if its required model files are missing.

## 11. GPU Memory and Concurrency

### 11.1 Different defaults by backend

Resolve processing defaults after selecting the backend. Keep the defaults independently configurable, even when their initial values match:

| Backend | Default `processing_limit` |
|---|---:|
| WhisperX | 3 |
| Qwen3 | 3 |

Qwen targets a dedicated RTX 3090 or newer NVIDIA GPU with at least 24 GB VRAM, BF16, PyTorch SDPA, inference batch size 1, and one complete ASR/aligner pair per worker.

**Research basis (2026-11-11):** the selected native [ASR weight file](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/main) is approximately 4.08 GB, and the native [aligner weight file](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf/tree/main) is approximately 1.84 GB. Use approximately 5.92 GB (5.51 GiB) of 16-bit weights per worker as a planning baseline.

| Workers | Approximate weight storage alone |
|---|---:|
| 1 | 5.51 GiB |
| 2 | 11.03 GiB |
| 3 | 16.54 GiB |
| 4 | 22.05 GiB |

Keep the default at three: it leaves about 7.5 GiB on a 24 GiB device for activations, KV caches, CUDA libraries, and allocator overhead. Four leaves under 2 GiB before any inference allocations, which is insufficient comfortable headroom for four concurrent long-chunk jobs. These are sizing estimates, not measured peak VRAM or a guarantee of capacity on every workload.

The [ASR configuration](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/raw/main/config.json) specifies 28 decoder layers, 8 KV heads, and head dimension 128. At 16-bit precision, the KV cache alone costs approximately 112 KiB per sequence token per active ASR worker. Audio tokens, generated text, alignment activations, and attention implementation therefore matter; weight fit is only a lower bound.

The development box has no suitable GPU for capacity validation. Implement the researched default and explicit concurrency override, and test configuration, chunk lifetime, and OOM classification with CPU fixtures and mocks. Mark actual peak VRAM and throughput as unmeasured. GPU benchmarking is optional deployment follow-up and is not an implementation acceptance requirement. Real OOM failures must surface clearly with the configured worker count and guidance to lower `--processing-limit`.

Use one resolver for defaults across CLI, REST, Docker, Compose, and RunPod. Represent an omitted `processing_limit` as unset until backend resolution; deployment scripts should omit the option unless intentionally overriding it. This also removes the current inconsistency between the Python default of 3 and the Docker command's explicit value of 2.

### 11.2 Explicit override

Allow users to override Qwen3 concurrency with the existing `--processing-limit` option.

Document that higher concurrency increases GPU memory consumption and that larger GPUs may support an explicitly configured higher limit. An explicit positive override takes precedence over backend defaults.

Do not silently create additional worker instances to improve throughput.

### 11.3 Model lifetime

The expected model lifecycle is:

```text
Worker created
    |
    v
Qwen models loaded once
    |
    v
Transcribe file A
    |
    v
Transcribe file B
    |
    v
Transcribe file C
    |
    v
Worker destroyed
```

Avoid loading and unloading models for every recording.

### 11.4 CUDA errors

Handle GPU errors consistently with the existing processing framework.

Differentiate:

- Out-of-memory errors
- CUDA runtime errors
- Missing GPU
- Invalid model configuration
- Unsupported language
- Invalid audio
- Alignment failures

Extend the shared transcription exception classification where appropriate.

Do not classify every Qwen exception as transient.

A deterministic configuration failure should not be retried indefinitely.

## 12. Deployment Integration

### 12.1 Docker Compose

Add a Qwen-specific Compose workflow.

A separate file is acceptable:

```text
docker-compose.qwen3.yaml
```

Alternatively, Compose profiles may be used if they do not disturb existing services.

The new workflow must:

- Build from `Dockerfile.qwen3`.
- Use the Qwen runtime target.
- Reserve NVIDIA GPU access.
- Discover the baked Qwen manifest using the configured `HF_HOME`.
- Use the shared Qwen default processing limit of 3, unless explicitly overridden.
- Pass through existing API credentials.
- Preserve existing data/download behavior.
- Avoid launching both transcription backends unintentionally.

Example:

```bash
docker compose -f docker-compose.qwen3.yaml build

docker compose -f docker-compose.qwen3.yaml up
```

The WhisperX workflow uses its own namespaced file:

```bash
docker compose -f docker-compose.whisperx.yaml up
```

Published-image execution uses `docker-compose.whisperx.local.yaml` or `docker-compose.qwen3.local.yaml` for the selected backend.

### 12.2 Image publishing

Maintain separately versioned published images.

Repository names:

```text
apartmentlinesproduction/transcription-pipeline-whisperx:<version>

apartmentlinesproduction/transcription-pipeline-qwen3:<version>
```

Derive each repository from the shared `apartmentlinesproduction/transcription-pipeline` prefix and the backend name.

Do not overwrite the existing `v0.1.0` image.

Update or extend `dockerhub-build.sh` to select the appropriate Dockerfile and image repository.

The implementation may use separate build scripts if that is cleaner.

Requirements:

- Explicit image tags
- Reproducible dependency versions
- Pinned model revisions
- Separate publication of each image
- No accidental publication to the wrong repository
- Clear documentation of how to build and push both images

### 12.3 RunPod compatibility

The existing `run-pod.sh` starts SSH and the REST interface.

The Qwen runtime image must preserve the required deployment capabilities.

In particular:

- `run-pod.sh` must work in both images.
- The REST interface must start successfully.
- The application must correctly identify its backend.
- The existing `/run` request lifecycle must work.
- The existing callback behavior must work.
- Existing pod-shutdown behavior must be preserved.

Do not introduce a second, incompatible REST server for Qwen.

## 13. Error Handling and Lifecycle Correctness

The new backend must integrate with the existing `TranscriptionError` and post-processing error behavior.

### 13.1 Model errors

Model loading errors should fail clearly.

Examples:

- Missing local model weights
- Incompatible package version
- CUDA unavailable
- GPU out of memory
- Unsupported precision

Do not download missing files as a fallback.

### 13.2 Processing errors

A transcription may only be marked successful if:

1. ASR completed.
2. Alignment completed or a documented acceptable fallback was applied.
3. The transcript-to-timestamp reconciliation passed validation.
4. Valid SRT content was generated.
5. The result satisfies the existing processor output contract.

A Qwen exception must not be converted into an empty successful result.

### 13.3 Existing pipeline return-code issue

Inspect `TranscriptionPipeline.run()` and the pinned `ProcessingPipeline.run()` implementation.

The upstream framework returns an exit code, while the current wrapper does not explicitly propagate that result.

Ensure a nonzero processing-pipeline return code cannot be reported as a successful REST pipeline execution.

Propagate the result consistently through CLI exit status and the existing REST lifecycle, and update affected tests.

Be aware that some individual file failures are handled through the framework's post-processing path rather than the overall pipeline return code.

Do not assume a zero overall return code proves that every recording succeeded.

### 13.4 Cleanup

Continue to clean up downloaded files through the existing processing framework.

Do not leave intermediate audio copies, large tensors, or temporary subtitle files behind after completed processing.

## 14. Test Requirements

Ordinary tests must run without downloading model weights or requiring a GPU.

Use mocks or small deterministic fixtures for ordinary unit tests.

Provide optional GPU model tests separately. Their absence on this development box does not block completing the implementation.

### 14.1 Backend selection

Test:

- Manifest path precedence, `HF_HOME`/`XDG_CACHE_HOME` defaults, and independent `HF_HUB_CACHE` overrides.
- Automatic backend discovery for both manifests with no backend argument.
- Missing/malformed manifest, unsupported schema, missing assets, and missing backend dependencies.
- Explicit WhisperX assertion.
- Explicit Qwen3 assertion.
- Optional CLI/environment backend assertions and their precedence.
- Unknown backend.
- Backend/image mismatch.
- CLI arguments.
- REST request validation.
- Omitted REST backend, supplied matching backend, and mismatch with server configuration.
- Backend propagation to worker initialization.
- Backend default processing limits and explicit overrides across entry points.

### 14.2 Dependency isolation

Test both image environments.

Qwen image:

- Imports the application without WhisperX installed.
- Loads only Qwen-related model dependencies.
- Does not download WhisperX models.

WhisperX image:

- Imports the application without Qwen installed.
- Preserves existing transcription behavior.
- Does not download Qwen models.

### 14.3 Qwen adapter

Test:

- Model initialization arguments.
- Native Transformers classes, BF16/SDPA defaults, and local-only processor/model loading.
- Local model paths.
- Automatic language detection.
- English output normalization.
- Spanish output normalization.
- Mixed English/Spanish detection.
- Unsupported language handling.
- Forced-alignment conversion.
- Missing timestamps.
- Invalid timestamps.
- Empty transcription.
- Generation-budget exhaustion and normal end-token detection.
- Chunk-boundary word separation.
- Full sample coverage and audio feature-extractor truncation disabled.
- Raw transcript parsing preserves repeated speech without parsed-helper cleanup.
- No fabricated confidence scores.
- Error classification.

### 14.4 Text reconstruction

Include test cases with:

```text
Hello, how are you?
¿Dónde está la llave?
¡Necesito ayuda!
Don't open the door.
The apartment is 204-B.
There's no A/C.
```

Verify that punctuation, capitalization, Unicode, and word order survive alignment and subtitle generation.

### 14.5 SRT formatting

Test:

- Valid sequential cue numbering.
- Millisecond rounding.
- Minute boundaries.
- Long words.
- Punctuation.
- Spanish text.
- Short pauses.
- Long pauses.
- No duplicated words.
- No omitted words.
- Non-overlapping cues.
- Empty input.
- Invalid timing.
- Maximum line width behavior.
- Byte-identical output for the same normalized transcript through either backend and entry point.
- Preservation of WhisperX speaker labels when diarization is enabled.
- Unaligned words retained through the shared fallback policy.
- Positive cue duration and ordering after millisecond quantization.
- Metadata independent of subtitle segmentation settings.
- Retranscription metadata clears stale backend-specific values through the existing external update contract.

### 14.6 Existing test suite

Run and update affected tests, including:

```text
tests/test_main.py
tests/test_transcriber.py
tests/test_rest_interface.py
tests/test_config.py
tests/processors/test_transcription_processor.py
tests/processors/test_transcription_post_processor.py
```

Migrate tests and fixtures to the new abstraction boundaries. Retain useful behavioral coverage, especially external API contracts and WhisperX transcription capability. Replace tests that solely assert obsolete internal names or signatures with tests of their new equivalents.

### 14.7 Docker smoke tests

For both images:

- Build successfully.
- Confirm the expected backend configuration.
- Confirm model files exist.
- Verify application imports.
- Verify CLI help.
- Verify REST interface startup.
- Verify a mismatched backend fails before work begins.

For the Qwen image, verify config, tokenizer, processor, and snapshot resolution from disk with Hugging Face offline mode enabled. GPU weight initialization and inference belong to the optional checks below. Include offline Punkt loading and shared subtitle serialization for both images.

### 14.8 Optional GPU deployment checks

These checks describe a future deployment validation procedure. They are not required on the current development box and do not gate implementation completion. Supply an opt-in test entry point that skips clearly when the required GPU/model assets or recording fixtures are unavailable.

When suitable hardware and recordings are available, exercise the full processing path with:

1. Short English conversation.
2. Short Spanish conversation.
3. English conversation with apartment-maintenance terminology.
4. Spanish conversation with apartment-maintenance terminology.
5. Recording with both English and Spanish.
6. Recording containing long silence.
7. Recording longer than 180 seconds.
8. Recording approaching the existing 600-second limit.
9. Recording with speech around internal chunk boundaries.
10. A known recording where WhisperX previously omitted audible speech, if available.

Verify generated SRT output, not just the ASR text.

For the known WhisperX failure case, compare whether Qwen recovers the omitted passage.

Report the observed comparison; backend choice alone does not establish recognition quality.

## 15. Optional Deployment Measurements

Actual inference performance is unmeasured during development. If GPU deployment measurements are later run, record the following for representative audio files:

- Total audio duration
- ASR inference time
- Alignment time, if separately measurable
- Total processing time
- Peak GPU memory
- Detected language
- Number of transcribed words
- Number of subtitle cues
- Number of alignment warnings

Compare the Qwen backend with the existing WhisperX backend.

Measure real-time factor:

```text
RTF = total_processing_seconds / audio_duration_seconds
```

Lower values indicate faster processing.

Record GPU model, precision, attention implementation, dependency versions, and worker count with any measurements. Comparing one, two, and three workers on representative short and long recordings can guide later tuning. Report only measurements actually collected; GPU capacity, memory-headroom targets, and speed measurements are not development acceptance criteria.

However, the implementation must avoid repeated model initialization and unnecessary audio decoding.

Use performance measurements to guide later optimizations rather than introducing premature batching complexity.

## 16. Proposed File Changes

The following is the recommended change inventory.

| File | Change |
|---|---|
| `transcription_pipeline/backends/base.py` | New shared backend contract |
| `transcription_pipeline/backends/factory.py` | New backend selection logic |
| `transcription_pipeline/backends/qwen3.py` | New Qwen ASR/alignment adapter |
| `transcription_pipeline/backends/whisperx.py` | Move and rename WhisperX implementation and configuration |
| `transcription_pipeline/results.py` | Shared typed results and external metadata mapping |
| `transcription_pipeline/errors.py` | Backend-neutral transcription errors |
| `transcription_pipeline/subtitles/` | NLTK-assisted timed cue grouping and pysubs2 serialization |
| `transcription_pipeline/transcriber.py` | Extract backend implementation; retain only CLI orchestration or move to `cli/transcribe.py` |
| `transcription_pipeline/processors/transcription_processor.py` | Backend factory integration |
| `transcription_pipeline/main.py` | CLI selection, per-backend concurrency, lifecycle |
| `transcription_pipeline/rest_interface.py` | Backend validation and propagation |
| `transcription_pipeline/config.py` | Manifest discovery, validation, and optional backend assertions |
| `transcription_pipeline/constants.py` | Backend identifiers and defaults |
| `transcription_pipeline/backends/qwen3_seed.py` | Build-time Qwen snapshots and manifest |
| `transcription_pipeline/backends/whisperx_seed.py` | Move and rename WhisperX model seeding |
| `transcription_pipeline/processors/transcription_post_processor.py` | Error compatibility and optional metadata |
| `transcription_pipeline/__init__.py` | Avoid eager backend-specific imports |
| `pyproject.toml` | Dependency organization and scripts |
| `requirements-qwen3-frozen.txt` | Pinned Qwen dependency stack |
| `Dockerfile.qwen3` | Separate Qwen runtime image |
| `docker-compose.qwen3.yaml` | Qwen development/production execution |
| `dockerhub-build.sh` | Support independent image builds |
| `README.md` | Backend selection and deployment documentation |
| `tests/` | New Qwen, selection, subtitle, and regression tests |

This inventory is not mandatory file-for-file.

Implement the smallest coherent design that satisfies all requirements.

## 17. Documentation Requirements

Update the README to describe:

### Backend selection

Explain the difference between:

- Discovering a backend from the deployment manifest.
- Asserting the expected backend through optional CLI/REST configuration.
- Selecting the Docker image containing that backend.

Make clear that a WhisperX-only image cannot execute Qwen merely by changing a CLI argument.

### Local execution

Document seeding a manifest and running WhisperX and Qwen locally. Explain `HF_HOME`, `HF_HUB_CACHE`, manifest-path overrides, and how to select among multiple local manifests.

### Docker execution

Document how to build and run each image independently.

### Model installation

Explain that models are embedded during Docker build and are unavailable for runtime downloading.

### GPU requirements

Explain how to configure NVIDIA GPU access.

Document Qwen processing concurrency of 3 for the target 24 GB GPU, batch size 1, and explicit overrides. Clearly distinguish the section 11.1 weight-based estimate from measured peak memory and publish the actual validation results when available.

### Language behavior

Document automatic English/Spanish detection, unsupported-language handling, and mixed-language behavior.

### Limitations

Document:

- No diarization in the new Qwen backend.
- The application's 180-second chunk bound and word-boundary limitations.
- Model alignment limitations.
- Potential subtitle timing imperfections.
- Automatic language-detection limitations.
- Lack of Qwen word-confidence scores.
- The need for GPU testing to establish actual performance.

## 18. Acceptance Criteria

Implementation completion requires the behaviors below, verified through CPU fixtures, mocked model responses, and the non-GPU build/import/offline checks. Actual recognition quality, CUDA weight loading, peak VRAM, and inference throughput remain explicitly unverified until optional deployment checks run; those checks do not block completion on this development box.

### External contracts and supported workflows

- [ ] Existing WhisperX Docker workflow still builds.
- [ ] Existing WhisperX transcription remains functional.
- [ ] Documented CLI commands work with the revised backend configuration.
- [ ] `/run` and `/status`, authentication, callbacks, and lifecycle semantics remain functional.
- [ ] `/run` works without a backend field and validates an explicitly supplied field against the manifest.
- [ ] Existing post-processing/API payloads remain compatible.

### Backend selection

- [ ] Each image discovers its backend from the baked manifest without CLI/REST backend arguments.
- [ ] `--backend whisperx` and `--backend qwen3` validate the expected deployment.
- [ ] Manifest-path overrides and optional environment assertions work.
- [ ] REST backend assertions work.
- [ ] Incorrect image/backend combinations fail clearly.
- [ ] The correct backend is selected before workers initialize.

### Qwen functionality

- [ ] Qwen3-ASR-1.7B transcribes English.
- [ ] Qwen3-ASR-1.7B transcribes Spanish.
- [ ] Automatic language detection works.
- [ ] Qwen3-ForcedAligner-0.6B produces word timestamps.
- [ ] Word timestamps map correctly to the original audio timeline.
- [ ] The Qwen backend generates valid SRT.
- [ ] Both backends and both entry points use one shared subtitle pipeline.
- [ ] Punctuation and accented characters are preserved.
- [ ] Long recordings are supported through the existing 600-second limit.
- [ ] No diarization is required.
- [ ] No missing-model runtime downloads occur.

### Docker

- [ ] WhisperX and Qwen use separate runtime images.
- [ ] The Qwen image includes both required models.
- [ ] The Qwen image does not require WhisperX.
- [ ] The WhisperX image does not require Qwen.
- [ ] Model revisions are pinned.
- [ ] Runtime model loading works offline.
- [ ] Local builds do not require a GPU.
- [ ] Both images can be published independently.
- [ ] RunPod REST execution remains functional.

### Reliability

- [ ] Qwen defaults to 3 workers with inference batch size 1; sizing assumptions and unmeasured GPU capacity are documented.
- [ ] Qwen models are reused across jobs within a worker.
- [ ] Invalid transcription results cannot silently become successful SRTs.
- [ ] Alignment discrepancies are detected and reported.
- [ ] Generation-budget exhaustion and chunk-boundary text joining are validated.
- [ ] Model failures propagate through the processing framework.
- [ ] The backend does not silently skip an audio interval during its own preprocessing or chunk assembly.
- [ ] CUDA and configuration failures are distinguishable.

### Testing

- [ ] Existing unit tests pass.
- [ ] New backend-selection tests pass.
- [ ] New Qwen adapter tests pass.
- [ ] New SRT tests pass.
- [ ] Docker smoke tests pass.
- [ ] Optional NVIDIA GPU checks are provided and reported as untested on this development box; development completion requires the CPU/unit and build checks above.

## 19. Delivery Scope

Deliver file-based English/Spanish transcription through the existing download framework, CLI, and Flask service. Discover the backend from its deployment manifest and run it in its own image with pretrained local weights and reusable worker instances. Qwen uses native Transformers with automatic language detection and forced alignment; WhisperX retains its existing inference capabilities. Both use shared NLTK-assisted cue grouping and pysubs2 serialization, then submit results through the existing application API and database contract.

## 20. Recommended Implementation Order

Implement in this order to minimize risk:

1. **Deployment manifest.** Add manifest seeding, discovery, validation, backend resolution, and optional backend assertions.

2. **Dependency isolation.** Remove eager WhisperX imports from shared application paths. Verify the application can import without WhisperX installed.

3. **Qwen adapter.** Implement native model loading, lossless chunk orchestration, generation termination checks, language detection, forced alignment, and normalized output.

4. **Shared subtitles.** Implement punctuation-preserving normalization and timed cue grouping using NLTK boundary spans; use pysubs2 serialization and migrate both backends and entry points.

5. **Pipeline integration.** Connect the Qwen implementation to the existing transcription processor, worker framework, CLI, and REST interface.

6. **Docker model baking.** Add the separate image, pinned dependencies, immutable model snapshots, and offline verification.

7. **Deployment integration.** Add Compose configuration, image publishing, and RunPod compatibility.

8. **Testing and documentation.** Run compatibility tests, add Qwen-specific tests, validate the image, and update the README.

Do not begin broad refactoring of unrelated code.

## 21. Final Deliverables

When implementation is finished, provide:

1. A summary of the architectural changes.
2. A list of files added and modified.
3. The exact CLI commands to run WhisperX and Qwen3.
4. The exact Docker build and execution commands for each backend.
5. The model and Python dependency versions selected.
6. The pinned Hugging Face revision SHAs.
7. The tests executed and their results.
8. GPU integration test results, if available.
9. Any remaining limitations or known risks.

Do not claim that GPU inference, offline loading, or complete SRT generation was verified unless those tests were actually executed.

---

## 22. Reference Documentation

**Target repository**

- https://github.com/apartmentlines/transcription-pipeline
- https://github.com/apartmentlines/transcription-pipeline/tree/0b6579a06163991120104665d4cf78266c0f0fd9

**Processing framework**

- https://github.com/apartmentlines/download-pipeline-processor
- Dependency version currently referenced by the application: `v0.1.1`

**Native Qwen implementation**

- https://github.com/huggingface/transformers/releases/tag/v5.22.2
- https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_asr/processing_qwen3_asr.py
- https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_asr/modeling_qwen3_asr.py
- https://huggingface.co/docs/transformers/main_classes/text_generation

**Models**

- https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
- https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf

**Model caching and offline operation**

- https://huggingface.co/docs/transformers/installation
- https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables

**Shared subtitles**

- https://pysubs2.readthedocs.io/en/latest/
- https://pysubs2.readthedocs.io/en/latest/api-reference.html
- https://www.nltk.org/api/nltk.tokenize.punkt.html

**Implementation principle:** Deliver two clearly namespaced, independently deployable transcription backends with one shared subtitle pipeline. Refactor package internals freely within this feature while preserving the `TRANSCRIPTION_DOMAIN` integration and the Flask endpoint contracts defined in section 4.3. Keep dependencies and model assets isolated by image. Prioritize complete transcription and preserved text.
