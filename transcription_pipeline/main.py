import argparse
from pathlib import Path
from typing import Optional, List
from copy import deepcopy
from download_pipeline_processor.processing_pipeline import ProcessingPipeline
from download_pipeline_processor.logger import Logger
from transcription_pipeline.processors.transcription_pre_processor import (
    TranscriptionPreProcessor,
)
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from transcription_pipeline.processors.transcription_post_processor import (
    TranscriptionPostProcessor,
)
from transcription_pipeline.utils import (
    get_request,
    positive_int,
    fail_hard,
)
from transcription_pipeline.config import load_configuration, set_environment_variables
from transcription_pipeline.constants import (
    DEFAULT_PROCESSING_LIMIT,
    DEFAULT_DOWNLOAD_QUEUE_SIZE,
    DEFAULT_DOWNLOAD_CACHE,
    TRANSCRIPTION_STATE_READY,
)


class TranscriptionPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        limit: Optional[int] = None,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
        processing_limit: int = DEFAULT_PROCESSING_LIMIT,
        download_queue_size: int = DEFAULT_DOWNLOAD_QUEUE_SIZE,
        download_cache: Path = DEFAULT_DOWNLOAD_CACHE,
        simulate_downloads: bool = False,
        debug: bool = False,
    ) -> None:
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.api_key = api_key
        self.domain = domain
        self.limit = limit
        self.min_id = min_id
        self.max_id = max_id
        self.processing_limit = processing_limit
        self.download_queue_size = download_queue_size
        self.download_cache = download_cache
        self.simulate_downloads = simulate_downloads
        self.debug = debug
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> ProcessingPipeline:
        return ProcessingPipeline(
            pre_processor_class=TranscriptionPreProcessor,
            processor_class=TranscriptionProcessor,
            post_processor_class=TranscriptionPostProcessor,
            processing_limit=self.processing_limit,
            download_queue_size=self.download_queue_size,
            download_cache=self.download_cache,
            simulate_downloads=self.simulate_downloads,
            debug=self.debug,
        )

    def build_retrieve_request_url(self) -> str:
        return f"https://{self.domain}/al/transcriptions/retrieve/operator-recordings/{TRANSCRIPTION_STATE_READY}"

    def build_retrieve_request_params(self) -> dict:
        params = {"api_key": self.api_key}
        if self.limit is not None:
            params["limit"] = str(self.limit)
        if self.min_id is not None:
            params["min_id"] = str(self.min_id)
        if self.max_id is not None:
            params["max_id"] = str(self.max_id)
        return params

    def retrieve_file_data(self) -> List[dict] | None:
        url = self.build_retrieve_request_url()
        try:
            params = self.build_retrieve_request_params()
            log_params = deepcopy(params)
            log_params["api_key"] = "REDACTED"
            self.log.debug(
                f"Retrieving file data from URL: {url}, params: {log_params}"
            )
            response = get_request(url, params)
            resp_json = response.json()
            if resp_json.get("success"):
                files = resp_json.get("files", [])
                self.log.info(f"Retrieved {len(files)} files for processing")
                return files
            else:
                fail_hard("Failed to retrieve files.")
        except Exception as e:
            fail_hard(f"Error retrieving files: {e}")

    def prepare_file_data(self, files: List[dict]) -> List[dict]:
        self.log.debug("Preparing file data with API key")
        for file in files:
            separator = "&" if "?" in file["url"] else "?"
            file["url"] += f"{separator}api_key={self.api_key}"
            # NOTE: This is a hack to force downloading from S3 specific to the Apartment Lines
            # infrastructure.
            if "metadata" in file and "call_uuid" in file["metadata"] and file["metadata"]["call_uuid"] == "N/A":
                self.log.info(
                    f"File must be downloaded from S3. Adding from_s3=1 to {file['url']}"
                )
                file["url"] += "&from_s3=1"
        return files

    def setup_configuration(self) -> None:
        self.log.debug("Setting up configuration")
        if not self.api_key or not self.domain:
            fail_hard("API key and domain must be provided")
        set_environment_variables(self.api_key, self.domain)
        self.log.info("Configuration loaded successfully")

    def run(self) -> None:
        self.log.info("Starting transcription pipeline")
        self.setup_configuration()
        files = self.retrieve_file_data()
        if not files:
            self.log.info("No files to process")
            return
        files = self.prepare_file_data(files)
        self.log.info("Starting pipeline execution")
        self.pipeline.run(files)
        self.log.info("Transcription pipeline completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Only transcribe this many files, default unlimited",
    )
    parser.add_argument(
        "--min-id",
        type=positive_int,
        help="Only process transcriptions with ID >= this value",
    )
    parser.add_argument(
        "--max-id",
        type=positive_int,
        help="Only process transcriptions with ID <= this value",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (also can be provided as TRANSCRIPTION_API_KEY environment variable)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Transcription domain used for REST operations (also can be provided as TRANSCRIPTION_DOMAIN environment variable)",
    )
    parser.add_argument(
        "--processing-limit",
        type=positive_int,
        default=DEFAULT_PROCESSING_LIMIT,
        help="Maximum concurrent processing threads, default %(default)s",
    )
    parser.add_argument(
        "--download-queue-size",
        type=positive_int,
        default=DEFAULT_DOWNLOAD_QUEUE_SIZE,
        help="Maximum size of downloaded files queue, default %(default)s",
    )
    parser.add_argument(
        "--download-cache",
        type=Path,
        default=DEFAULT_DOWNLOAD_CACHE,
        help="Directory to cache downloaded files, default %(default)s",
    )
    parser.add_argument(
        "--simulate-downloads",
        action="store_true",
        help="Simulate downloads instead of performing actual downloads",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        api_key, domain = load_configuration(args)
    except ValueError as e:
        fail_hard(str(e))
        return

    pipeline = TranscriptionPipeline(
        api_key=api_key,
        domain=domain,
        limit=args.limit,
        min_id=args.min_id,
        max_id=args.max_id,
        processing_limit=args.processing_limit,
        download_queue_size=args.download_queue_size,
        download_cache=args.download_cache,
        simulate_downloads=args.simulate_downloads,
        debug=args.debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
