# Transcription Pipeline

A processing pipeline for transcribing audio files and handling post-processing tasks. This package leverages the [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) framework to manage the downloading, processing, and post-processing of audio files.

## Features

- Concurrent processing of audio files.
- Configurable download queue and processing limits.
- Retry logic with exponential backoff for network requests.
- Customizable logging with debug support.
- Environment variable and command-line argument configurations.

## Requirements

- Python 3.12 or later.
- [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) package

## Installation

Ensure [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) package is installed locally.

```bash
pip install -e .

# ...or for development...
pip install -e ".[dev]"
```

## Usage

### Running the pipeline directly

Can be run as a command-line script or integrated into other Python applications.

Run `seed-models` before first use; Docker builds already do this. Runtime loading of Whisper and Hugging Face alignment models uses only cached files and fails if the required files are missing.

#### Command-Line Usage

Run the pipeline by executing the `transcription-processor` script:

```bash
transcription-processor [OPTIONS]
```

**Options**

Run with the `--help` arg for a list of all CLI options.

**Example**

```bash
transcription-processor --api-key YOUR_API_KEY --domain example.com --debug
```

#### Environment Variables

You can set the following environment variables instead of providing command-line arguments:

- `TRANSCRIPTION_API_KEY`
- `TRANSCRIPTION_DOMAIN`

### Docker

Run from the repo root with `TRANSCRIPTION_API_KEY` and `TRANSCRIPTION_DOMAIN` set in your shell or `.env`. Local execution requires Docker GPU support for an NVIDIA GPU.

Build and push the prepared image to `apartmentlinesproduction/transcription-pipeline:v0.1.0` (requires repository push access):

```bash
docker login
./dockerhub-build.sh
```

The multi-stage build installs runtime dependencies and seeds Whisper, English/Spanish alignment models and NLTK tokenizer data on CPU; building requires no GPU. Dependencies and model data are cached independently of application code. The final image includes the model data, SSH and `runpodctl`, but excludes compiler and test tooling. Optional speaker-diarization weights are not seeded.

For a local build without publishing, run `./setup.sh` (or `docker compose build`), then `docker compose up`. No setup container or `docker commit` is needed. Pip downloads use a BuildKit cache.

Run the prepared image locally:

```bash
# Pull when using another machine or refreshing an older local image.
docker compose -f docker-compose.local.yaml pull
docker compose -f docker-compose.local.yaml up
```

For an ad hoc run, override the command without editing the Compose file:

```bash
docker compose -f docker-compose.local.yaml \
  run --rm transcription-local-service \
  transcription-processor --limit 1
```

`run` creates a new container from the prepared image; `--rm` removes it afterward. The supplied command replaces the entire default command.

## Development

### Running Tests

```bash
pip install -r requirements-frozen.txt -r requirements-dev.txt
pytest
```

To test inside Docker with the development target:

```bash
./test.sh build
./test.sh run --rm transcription-processor pytest
```
