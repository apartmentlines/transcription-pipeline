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
VALID_LANGUAGES = ["en", "es"]
INITIAL_PROMPT = """At %s, answering service operators engage in conversations with residents about maintenance emergencies and security issues.

Frequent emergencies include no A/C or heat, electrical problems, lockouts, gas leaks, and overflowing toilets.

Residents might say "There's a water leak in my apartment", "I have no hot water", "I hear a smoke alarm", or "I'm locked out of my
unit".

Operators often respond with "I'll dispatch maintenance", "I'll notify courtesy patrol", or "Please provide your apartment number
and a callback number."

Common discussions also involve loud noise, parking, elevator issues, and gate failures. Phrases like "stopped up sink", "broken
stove", "no power", or "access key won't work" are typical.

Operators assist residents by contacting maintenance or courtesy patrol services as needed."""

# Transcription state constants
TRANSCRIPTION_STATE_NOT_TRANSCRIBABLE = "not-transcribable"
TRANSCRIPTION_STATE_READY = "ready"
TRANSCRIPTION_STATE_ACTIVE = "active"
TRANSCRIPTION_STATE_COMPLETE = "complete"
