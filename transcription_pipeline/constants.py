from pathlib import Path
import tempfile

# Pipeline constants
DEFAULT_PROCESSING_LIMIT = 3
DEFAULT_DOWNLOAD_QUEUE_SIZE = 10
DEFAULT_DOWNLOAD_CACHE = (
    Path(tempfile.gettempdir()) / "processing-pipeline-download-cache"
)
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 0.5  # in seconds
DOWNLOAD_TIMEOUT = 30  # in seconds

# Audio validation constants
MIN_AUDIO_DURATION = 5  # seconds
MAX_AUDIO_DURATION = 600  # seconds

# Transcription model constants
DEFAULT_WHISPER_MODEL = "large-v2"
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_OUTPUT_DIR = None
DEFAULT_OUTPUT_FORMAT = "srt"
DEFAULT_BATCH_SIZE = 16

# Transcription state constants
TRANSCRIPTION_STATE_NOT_TRANSCRIBABLE = "not-transcribable"
TRANSCRIPTION_STATE_READY = "ready"
TRANSCRIPTION_STATE_ACTIVE = "active"
TRANSCRIPTION_STATE_COMPLETE = "complete"
