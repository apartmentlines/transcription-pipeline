"""Shared configuration, model identities, and backend-specific defaults."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Literal

DEFAULT_INFERENCE_DEVICE = "cuda:0"
DEFAULT_INFERENCE_PRECISION = "bfloat16"
DEFAULT_QWEN_ATTENTION: Literal["sdpa", "flash_attention_2"] = "sdpa"
DEFAULT_COMMUNITY = "an apartment community"
DEFAULT_TERMINOLOGY = ""
MANIFEST_SCHEMA_VERSION = 1
MODEL_DOWNLOAD_TIMEOUT = 60
LANGUAGE_NAMES = {"en": "english", "es": "spanish"}
PUNKT_PARAMETER_FILES = (
    "abbrev_types.txt",
    "collocations.tab",
    "ortho_context.tab",
    "sent_starters.txt",
)
QWEN_ATTENTION_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")
QWEN_CHUNK_SEARCH_SECONDS = 5
QWEN_ENERGY_FRAME_MILLISECONDS = 20
QWEN_LANGUAGE_CODES = {"english": "en", "spanish": "es", "en": "en", "es": "es"}
WHISPERX_VAD_ONSET = 0.5
WHISPERX_VAD_OFFSET = 0.363
WHISPERX_VAD_REPO_ID = "whisperx-3.8.6/pyannote-vad"
SUBTITLE_MAX_CHARACTERS = 42
SUBTITLE_MAX_LINES = 1
SUBTITLE_MIN_DURATION = 1.0
SUBTITLE_MAX_DURATION = 6.0
SUBTITLE_PAUSE_BREAK = 1.0
SUBTITLE_RENDER_SPEAKERS = True
SUBTITLE_TOKENIZER_CACHE_SIZE = 4
MILLISECONDS_PER_SECOND = 1000
TIMESTAMP_TOLERANCE = 1e-6
ALIGNMENT_MIN_CHARACTER_COVERAGE = 0.7
SOURCE_SPEECH_PAUSE_SECONDS = 1.0
SUSPICIOUS_GAP_SECONDS = 30

# Pipeline constants
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
WHISPERX_NUM_SPEAKERS = 2
WHISPERX_BATCH_SIZE = 16
VALID_LANGUAGES = ["en", "es"]
WHISPERX_INITIAL_PROMPT = """At %s, answering service operators engage in conversations with residents about maintenance emergencies and security issues.

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

# REST interface constants
DEFAULT_REST_HOST = "0.0.0.0"
DEFAULT_REST_PORT = 8080
DEFAULT_REST_FAILURE_EXIT_CODE = 1

BACKENDS = ("whisperx", "qwen3")

PROCESSING_LIMITS = {"whisperx": 3, "qwen3": 3}

QWEN_SAMPLE_RATE = 16000

QWEN_MAX_CHUNK_SECONDS = 180

QWEN_GENERATION_BUDGET = 2048

QWEN_CONTEXT = (
    "Telephone conversation with an apartment community answering service. "
    "Topics: maintenance, emergencies, security, parking, access gates, HVAC, "
    "plumbing, and resident requests. Preserve the spoken language, "
    "including English and Spanish."
)

QWEN_MODELS = {
    "asr": ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"),
    "aligner": (
        "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        "c07281df297b9905d24a508279258cccf987a064",
    ),
}

WHISPERX_ASR = (
    "Systran/faster-whisper-large-v2",
    "f0fe81560cb8b68660e564f55dd99207059c092e",
)

WHISPERX_ALIGNMENT = {
    "en": (
        "WAV2VEC2_ASR_BASE_960H",
        "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
        "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d",
    ),
    "es": (
        "VOXPOPULI_ASR_BASE_10K_ES",
        "wav2vec2_voxpopuli_base_10k_asr_es.pt",
        "272a92b156b78e697e6c7cf7c64f274250a36d0047440e27a895a377d3af818f",
    ),
}

WHISPERX_DIARIZATION = {
    "diarizer": (
        "pyannote/speaker-diarization-3.1",
        "84fd25912480287da0247647c3d2b4853cb3ee5d",
    ),
    "diarizer_segmentation": (
        "pyannote/segmentation-3.0",
        "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
    ),
    "diarizer_embedding": (
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "837717ddb9ff5507820346191109dc79c958d614",
    ),
}

PUNKT_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"

PUNKT_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
