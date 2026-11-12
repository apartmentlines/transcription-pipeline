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

## Qwen FlashAttention

Qwen defaults to PyTorch SDPA. To use FlashAttention 2 locally, install the optional pinned wheel after the Qwen dependencies:

```bash
pip install --no-deps --only-binary=:all: --require-hashes -r requirements-qwen3-flash-attention.txt
TRANSCRIPTION_QWEN_ATTENTION=flash_attention_2 transcriber recording.wav --output-dir ./output
```

The wheel requires Linux x86-64, CPython 3.12, Torch 2.8 with C++11 ABI, and CUDA 12, matching the Qwen dependency lock. Both BF16 and FP16 are supported.

For Docker:

```bash
docker build -f Dockerfile.qwen3 --target runtime-flash-attention \
  -t transcription-pipeline-qwen3:flash-attention .
docker run --rm --gpus all \
  -e TRANSCRIPTION_API_KEY -e TRANSCRIPTION_DOMAIN \
  transcription-pipeline-qwen3:flash-attention transcription-processor --limit 100
```

This target defaults `TRANSCRIPTION_QWEN_ATTENTION` to `flash_attention_2`. Set it to `sdpa` to switch back. The ordinary `runtime` target omits FlashAttention; selecting it there fails with installation instructions. This setting applies to both Qwen models and all entry points.

## GPU compatibility and concurrency

The pinned Qwen stack targets the following deployment GPUs:

| GPU | Compute capability | FlashAttention kernel |
|---|---|---|
| GeForce RTX 3090 | 8.6 | compatible `sm_80` |
| RTX A6000 | 8.6 | compatible `sm_80` |
| A40 | 8.6 | compatible `sm_80` |
| GeForce RTX 4090 | 8.9 | compatible `sm_80` |
| RTX PRO 4000 Blackwell | 12.0 | `sm_120` |
| RTX PRO 4500 Blackwell | 12.0 | `sm_120` |
| RTX PRO 4500 Blackwell Server Edition | 12.0 | `sm_120` |
| RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb | 12.0 | `sm_120` |

Architecture coverage follows [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus), [Ampere/Ada binary compatibility](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html#application-compatibility-on-the-nvidia-ada-gpu-architecture), and the [pinned FlashAttention build targets](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/setup.py). The packaged Torch and FlashAttention binaries include compatible kernels. This does not establish measured throughput or memory use on each card.

Use a host driver supporting the GPU and the CUDA 12.8 PyTorch runtime. RTX PRO 6000 Blackwell MIG additionally requires NVIDIA driver R575, at least 575.51.03, and NVIDIA Container Toolkit; see [MIG prerequisites](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html#prerequisites).

Both backends default to three workers. Qwen's model weights alone occupy roughly 16.54 GiB across three workers; the remaining memory must accommodate activations, caches, and CUDA overhead. Adjust `--processing-limit` or REST `processing_limit` for available memory and observed throughput. Three-worker peak usage has not been measured on this development machine.

### MIG selection

Find the assigned instance UUID with `nvidia-smi -L`, then expose it to the container:

```bash
docker run --rm --gpus "device=MIG-<UUID>" \
  -e TRANSCRIPTION_DEVICE=cuda:0 \
  -e TRANSCRIPTION_API_KEY -e TRANSCRIPTION_DOMAIN \
  transcription-pipeline-qwen3:flash-attention transcription-processor --limit 100
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

For CUDA out-of-memory errors, reduce the processing limit. For missing model assets, reseed the selected backend in its environment. For unexpected output, save text and diagnostics alongside the subtitles:

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
