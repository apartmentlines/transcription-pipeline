# Transcription Pipeline

A processing pipeline for transcribing audio files and handling post-processing tasks. This package leverages the [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) framework to manage the downloading, processing, and post-processing of audio files.

## Features

- Concurrent processing of audio files.
- Configurable download queue and processing limits.
- Retry logic with exponential backoff for network requests.
- Customizable logging with debug support.
- Environment variable and command-line argument configurations.

## Requirements

- Python 3.9 or later.
- [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) package

## Installation

Ensure [download-pipeline-processor](https://github.com/apartmentlines/download-pipeline-processor) package is installed locally.

```bash
pip install -e .

# ...or for development...
pip install -e ".[dev]"
```

## Usage

The `transcription_pipeline` can be run as a command-line script or integrated into other Python applications.

### Command-Line Usage

Run the pipeline by executing the `main.py` script:

```bash
transcription-processor [OPTIONS]
```

**Options**

Run with the `--help` arg for a list of all CLI options.

**Example**

```bash
transcription-processor --api-key YOUR_API_KEY --domain example.com --debug
```

### Environment Variables

You can set the following environment variables instead of providing command-line arguments:

- `TRANSCRIPTION_API_KEY`
- `TRANSCRIPTION_DOMAIN`

## Development

### Running Tests

```bash
pytest tests/
```
