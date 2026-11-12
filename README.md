# Transcription Pipeline

Transcribe English and Spanish recordings with WhisperX or Qwen3 ASR and forced alignment. Uses [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) to download recordings, process them concurrently, and submit subtitles to the configured API.

## Requirements

- Python 3.12 for Qwen; Python 3.13 for the frozen WhisperX dependencies.
- FFmpeg and an NVIDIA GPU for transcription.
- Docker with NVIDIA Container Toolkit for container execution.

## Installation

Use a separate virtual environment for each backend. For WhisperX:

```bash
pip install -r requirements-whisperx-frozen.txt
pip install --no-deps -e .
seed-models --backend whisperx
```

For Qwen, use `requirements-qwen3-frozen.txt` and `seed-models --backend qwen3` instead. Seeding downloads models without requiring a GPU; Docker builds already do this.

## Usage

### Running the pipeline

```bash
transcription-processor --api-key YOUR_API_KEY --domain example.com --debug
```

Alternatively, set `TRANSCRIPTION_API_KEY` and `TRANSCRIPTION_DOMAIN`. Use `--help` for all options, including download and processing limits.

The seeded manifest selects the backend automatically. Runtime uses cached models and fails if required assets are missing. An optional `--backend` checks that the installation matches the requested backend.

### Transcribing a local file

```bash
transcriber recording.wav --output-dir ./output
```

Produces SRT; use `--output-format vtt` for WebVTT. `--community` and `--terminology` add recording-specific context.

### Docker

Run from the repository root with `TRANSCRIPTION_API_KEY` and `TRANSCRIPTION_DOMAIN` set in your shell or `.env`. Builds download dependencies and models without requiring a GPU.

Build and run WhisperX:

```bash
./setup.sh -b whisperx
./dev.sh -b whisperx
```

Build and run Qwen:

```bash
./setup.sh -b qwen3
./dev.sh -b qwen3
```

All three wrappers (`setup.sh`, `dev.sh`, `test.sh`) accept `-b whisperx|qwen3`, defaulting to WhisperX. Put wrapper options first; remaining arguments go to Compose. For build flags, use `./setup.sh -b qwen3 -- --no-cache`. Backend projects and test images are isolated.

To use the published WhisperX image:

```bash
docker compose -f docker-compose.whisperx.local.yaml pull
docker compose -f docker-compose.whisperx.local.yaml up
```

To use the published Qwen image:

```bash
docker compose -f docker-compose.qwen3.local.yaml pull
docker compose -f docker-compose.qwen3.local.yaml up
```

Override the command for an ad hoc run:

```bash
docker compose -f docker-compose.whisperx.local.yaml \
  run --rm transcription-local-service transcription-processor --limit 1
```

Build and publish backend images (requires repository push access):

```bash
docker login
./dockerhub-build.sh -b whisperx -t v0.1.0 -p
./dockerhub-build.sh -b qwen3 -t v0.1.0 -p
```

Images use `apartmentlinesproduction/transcription-pipeline-whisperx` and `apartmentlinesproduction/transcription-pipeline-qwen3`. Omit `-p` to build without publishing.

See the [deployment guide](docs/deployment.md) for FlashAttention, supported GPUs and MIG, cache configuration, REST/RunPod, and troubleshooting.

## Development

After installing the selected backend's dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
basedpyright
flake8 --config ~/.flake8 transcription_pipeline tests
basht tests/shell/dockerhub-build.bash
basht tests/shell/compose.bash
```

The ordinary tests require neither a GPU nor downloaded models. To test in a backend's development image (substitute `whisperx` as needed):

```bash
./test.sh -b qwen3 build
./test.sh -b qwen3 run --rm transcription-processor pytest
```

See [deployment checks](docs/deployment.md#deployment-checks) for offline image smoke tests and optional GPU tests.
