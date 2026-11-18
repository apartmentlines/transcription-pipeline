# Deployment guide

Start with the [README](../README.md) for installation and ordinary pipeline usage.

## Backend and model configuration

Each installation has one backend, recorded by `seed-models` in a manifest. Runtime discovers that manifest in this order:

1. `--model-manifest PATH`.
2. `TRANSCRIPTION_MODEL_MANIFEST`.
3. `HF_HOME/transcription-pipeline/manifest.json`, using Hugging Face's configured `HF_HOME`.

`--backend` and `TRANSCRIPTION_BACKEND` assert the installed backend; the CLI option takes precedence. To run another backend, use its environment/image and manifest.

| Setting | Purpose |
|---|---|
| `HF_HOME` | Hugging Face data root; `/opt/models/huggingface` in Docker |
| `HF_HUB_CACHE` | Hub snapshots and blobs; defaults to `HF_HOME/hub` |
| `TORCH_HOME` | WhisperX alignment weights; `/opt/models/torch` in Docker |
| `NLTK_DATA` | Sentence tokenizer data; defaults to `HF_HOME/nltk`, `/opt/models/nltk` in Docker |
| `TRANSCRIPTION_DEVICE` | Pipeline/REST CUDA device; default `cuda:0` |
| `TRANSCRIPTION_PRECISION` | Pipeline/REST Qwen precision; `bfloat16` (default) or `float16` |

The standalone `transcriber` uses `--device` and `--precision` for those settings. Preserve Hub blobs when moving snapshots: snapshot files can be relative symlinks. Reseed missing assets; inference does not download replacements. Docker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

For explicit manifest locations:

```bash
seed-models --backend qwen3 --model-manifest ./manifests/qwen3.json
transcriber recording.wav --model-manifest ./manifests/qwen3.json --output-dir ./output
```

Model identities and revisions are maintained in [constants.py](../transcription_pipeline/constants.py); installed assets are recorded in the manifest. Use the corresponding frozen requirements file for dependency versions.

## Qwen shared engine

Qwen uses one vLLM ASR engine with FlashAttention and one Transformers aligner with SDPA. The ordinary image includes the complete stack; the old optional FlashAttention wheel and `TRANSCRIPTION_QWEN_ATTENTION` setting are no longer used.

```bash
docker build -f Dockerfile.qwen3 --target runtime -t transcription-pipeline-qwen3:shared-engine .
mkdir -p test-data/transcriptions
docker run --rm --gpus all --ipc=host \
  -v "${HOME}/git/apartmentlines/testing-references/audio/operator-recordings:/data:ro" \
  -v "${PWD}/test-data/transcriptions:/output" \
  transcription-pipeline-qwen3:shared-engine \
  transcriber /data/first.wav /data/second.wav --output-dir /output --debug-output
```

For direct installation, use Python 3.12, FFmpeg, GCC/G++, and the pinned `requirements-qwen3-frozen.txt`, then install the project and run `seed-models --backend qwen3` as shown in the README.

| CLI option | Environment variable | Qwen default |
|---|---|---|
| `--processing-limit` | `TRANSCRIPTION_PROCESSING_LIMIT` | 4 simultaneous recordings |
| `--qwen-batch-size` | `TRANSCRIPTION_QWEN_BATCH_SIZE` | 16 active audio chunks |
| `--qwen-chunk-seconds` | `TRANSCRIPTION_QWEN_CHUNK_SECONDS` | 45 seconds (allowed: 1–90) |

These options work with both `transcriber` and `transcription-processor`. Explicit options override environment defaults. REST `/run` accepts `processing_limit`, `qwen_batch_size`, and `qwen_chunk_seconds` with the same meanings. Each recording retains its own context and output. WhisperX still owns a separate model per worker and defaults to three workers.

Incomplete Qwen generation receives one retry with a neighboring chunk where available. The entire retry window is re-transcribed and aligned; partial text is never appended to the result. Use `--debug-output` to inspect retry diagnostics.

## GPU compatibility and concurrency

Use an Ampere or newer NVIDIA GPU (RTX 3090 and newer). The pinned Qwen stack uses CUDA 13 and requires an R580 or newer compatible host driver; see [NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The 3090 has been tested; throughput and memory usage on other GPUs and MIG slices must be measured separately.

vLLM compiles kernels for the assigned GPU at startup. Mount a persistent cache at `/root/.cache/vllm` to reuse compilation across runs on compatible hardware. The image includes the required compiler. Start with the defaults and lower the chunk batch size if memory is constrained.

### MIG selection

Find the assigned instance UUID with `nvidia-smi -L`, then expose it to the container:

```bash
docker run --rm --gpus "device=MIG-<UUID>" \
  -e TRANSCRIPTION_DEVICE=cuda:0 \
  -e TRANSCRIPTION_API_KEY -e TRANSCRIPTION_DOMAIN \
  transcription-pipeline-qwen3:shared-engine transcription-processor --limit 100
```

For direct Python execution, set `CUDA_VISIBLE_DEVICES` to the MIG UUID before launching Python. In Compose, replace the GPU reservation's `count: all` with `device_ids: ["MIG-<UUID>"]`. When the provider already isolates the instance, keep its visibility settings. A single visible instance is `cuda:0`.

The `1g.24gb` profile provides 24 GB and one quarter of the parent GPU's SMs; tune concurrency for the slice. See NVIDIA's [container selection](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html#running-containers) and [RTX PRO 6000 MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-mig-profiles.html#rtx-pro-6000-blackwell-mig-profiles).

## REST and RunPod

Start the REST service in the selected backend environment:

```bash
rest-interface --host 0.0.0.0 --port 8080
```

Use `GET /status` to inspect state and `POST /run` to start processing. An example request body:

```json
{
  "api_key": "YOUR_API_KEY",
  "domain": "example.com",
  "limit": 100,
  "processing_limit": 3
}
```

The server manifest selects the backend. An optional `backend` field must match it; requests cannot select a filesystem manifest. Configure `--model-manifest` when starting the server if needed. Set the server's expected API key with `--api-key` or `TRANSCRIPTION_API_KEY`. Include `callback_url` in the request body to receive a completion notification. Each server process handles one run and exits after completion.

RunPod starts either image with `bash ./run-pod.sh`; forwarded arguments configure `rest-interface`. The wrapper handles SSH setup, completion, and pod shutdown.

## Recording context and diarization

Each backend has a built-in apartment-maintenance prompt in [constants.py](../transcription_pipeline/constants.py). The pipeline supplies the recording's `record_name` as its community. The standalone CLI can supply the community and append extra terminology:

```bash
transcriber recording.wav --community "Oak Ridge" \
  --terminology "Building names: Juniper, Sycamore" --output-dir ./output
```

Custom terminology is currently a standalone CLI option; pipeline workers use the built-in prompt and community without additional terminology.

To enable WhisperX speaker diarization, accept the gated models' terms, provide `HF_TOKEN`, and seed:

```bash
seed-models --backend whisperx --whisperx-diarization
```

A seeded diarizer is enabled for that installation. Use a separately seeded manifest without diarization to disable it. Default images omit diarization assets; Qwen does not provide diarization.

## Troubleshooting and deployment checks

For Qwen CUDA out-of-memory errors, reduce `--qwen-batch-size` or `--qwen-chunk-seconds`; for WhisperX, reduce `--processing-limit`. For missing model assets, reseed the selected backend in its environment. For unexpected output, save text and diagnostics alongside the subtitles:

```bash
transcriber recording.wav --output-dir ./comparison --debug-output
```

This writes `.srt`, `.txt`, and `.diagnostics.json`. The pipeline accepts WAV recordings from 5 to 600 seconds and supports English and Spanish. Compare output with the recording when assessing recognition or alignment quality.

### Deployment checks

Smoke-test a locally built image without network access or a GPU:

```bash
docker build -f Dockerfile.whisperx --target runtime -t transcription-pipeline-whisperx:local .
docker run --rm --network none transcription-pipeline-whisperx:local \
  python -m transcription_pipeline.smoke
```

For Qwen, substitute `Dockerfile.qwen3` and `transcription-pipeline-qwen3:local`. These checks verify bundled assets, dependencies, application startup, and subtitle generation without loading CUDA model weights.

Optional GPU checks run against a seeded installation and representative recordings:

```bash
TRANSCRIPTION_GPU_TESTS=1 \
TRANSCRIPTION_GPU_TEST_AUDIO_DIR=/path/to/fixtures \
TRANSCRIPTION_GPU_TEST_OUTPUT_DIR=./gpu-results \
pytest -q tests/integration/test_gpu.py
```

Required fixture names are listed in [test_gpu.py](../tests/integration/test_gpu.py). Inspect the saved subtitles and metrics; unavailable hardware, manifests, or recordings cause explicit skips.
