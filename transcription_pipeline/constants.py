from pathlib import Path
import tempfile

DEFAULT_PROCESSING_LIMIT = 3
DEFAULT_DOWNLOAD_QUEUE_SIZE = 10
DEFAULT_DOWNLOAD_CACHE = Path(tempfile.gettempdir()) / "processing-pipeline-download-cache"
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 0.5  # in seconds
DOWNLOAD_TIMEOUT = 30  # in seconds
