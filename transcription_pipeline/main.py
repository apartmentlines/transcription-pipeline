import argparse
from pathlib import Path
from typing import Optional
from download_pipeline_processor.processing_pipeline import ProcessingPipeline
from transcription_pipeline.processors.transcription_processor import TranscriptionProcessor
from transcription_pipeline.processors.transcription_post_processor import TranscriptionPostProcessor
from transcription_pipeline.utils import (
    configure_logging,
    positive_int,
    fail_hard,
)
from transcription_pipeline.config import load_configuration, set_environment_variables
from transcription_pipeline.constants import (
    DEFAULT_PROCESSING_LIMIT,
    DEFAULT_DOWNLOAD_QUEUE_SIZE,
    DEFAULT_DOWNLOAD_CACHE,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument("--limit", type=positive_int, help="Only transcribe this many files, default unlimited")
    parser.add_argument('--api-key', type=str, help="API key (also can be provided as TRANSCRIPTION_API_KEY environment variable)")
    parser.add_argument('--domain', type=str, help="Transcription domain used for REST operations (also can be provided as TRANSCRIPTION_DOMAIN environment variable)")
    parser.add_argument("--processing-limit", type=positive_int, default=DEFAULT_PROCESSING_LIMIT, help="Maximum concurrent processing threads, default %(default)s")
    parser.add_argument("--download-queue-size", type=positive_int, default=DEFAULT_DOWNLOAD_QUEUE_SIZE, help="Maximum size of downloaded files queue, default %(default)s")
    parser.add_argument("--download-cache", type=Path, default=DEFAULT_DOWNLOAD_CACHE, help="Directory to cache downloaded files, default %(default)s")
    parser.add_argument("--simulate-downloads", action="store_true", help="Simulate downloads instead of performing actual downloads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

import requests
from transcription_pipeline.utils import get_request

def retrieve_file_data(domain: str) -> list[dict]:
    url = f"https://my.{domain}/al/transcriptions/retrieve/operator-recordings"
    try:
        response = get_request(url)
        resp_json = response.json()
        if resp_json.get('success'):
            files = resp_json.get('files', [])
            if not files:
                fail_hard("No files to process.")
            return files
        else:
            fail_hard("Failed to retrieve files.")
    except Exception as e:
        fail_hard(f"Error retrieving files: {e}")

def prepare_file_data(files: list[dict], api_key: str, limit: Optional[int]) -> list[dict]:
    for file in files:
        separator = '&' if '?' in file['url'] else '?'
        file['url'] += f"{separator}api_key={api_key}"
    if limit:
        files = files[:limit]
    return files

def main() -> None:
    args = parse_arguments()
    configure_logging(args.debug)
    api_key = None
    domain = None
    try:
        api_key, domain = load_configuration(args)
        set_environment_variables(api_key, domain)
    except ValueError as e:
        fail_hard(str(e))
        return
    files = retrieve_file_data(domain)
    files = prepare_file_data(files, api_key, args.limit)
    pipeline = ProcessingPipeline(
        processor_class=TranscriptionProcessor,
        post_processor_class=TranscriptionPostProcessor,
        processing_limit=args.processing_limit,
        download_queue_size=args.download_queue_size,
        download_cache=args.download_cache,
        simulate_downloads=args.simulate_downloads,
    )
    pipeline.run(files)

if __name__ == "__main__":
    main()
