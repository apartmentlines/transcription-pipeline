"""WhisperX inference with locally seeded models and shared result normalization."""

from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from transcription_pipeline.constants import (
    VALID_LANGUAGES,
    WHISPERX_INITIAL_PROMPT,
    WHISPERX_VAD_OFFSET,
    WHISPERX_VAD_ONSET,
)
from ..deployment import Deployment, ManifestStore
from ..errors import ConfigurationError, TranscriptionError
from ..results import (
    AlignmentToken,
    ResultBuilder,
    TranscriptionContext,
    TranscriptionResult,
)


class WhisperXTranscriber:
    """Reuse a Whisper model, language aligners, and optional diarizer per worker."""

    deployment: Deployment
    log: logging.Logger
    torch: ModuleType
    whisperx: ModuleType
    model: Any
    diarization_model: Any

    def __init__(self, deployment: Deployment, debug: bool = False) -> None:
        self.deployment = deployment
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG if debug else logging.INFO)
        self.torch, self.whisperx = self._dependencies()
        nltk = importlib.import_module("nltk")
        nltk.data.path.insert(0, str(ManifestStore.roots()["nltk_data"]))
        device, _, index = deployment.device.partition(":")
        vad_module = importlib.import_module("whisperx.vads.pyannote")
        vad = vad_module.Pyannote(
            self.torch.device(deployment.device),
            model_fp=str(deployment.model("vad").path),
            vad_onset=WHISPERX_VAD_ONSET,
            vad_offset=WHISPERX_VAD_OFFSET,
        )
        self.model = self.whisperx.load_model(
            str(deployment.model("asr").path),
            device,
            device_index=int(index or 0),
            compute_type="float16",
            local_files_only=True,
            vad_model=vad,
        )
        self.alignment_models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.diarization_model = None
        diarization = next((m for m in deployment.models if m.role == "diarizer"), None)
        if diarization is not None:
            self.diarization_model = self._load_diarization(diarization.path)

    def _load_diarization(self, snapshot: Path) -> Any:
        """Resolve pinned nested models in a relocatable, temporary local config."""
        yaml = importlib.import_module("yaml")
        module = importlib.import_module("whisperx.diarize")
        configuration = yaml.safe_load((snapshot / "config.yaml").read_text())
        parameters = configuration["pipeline"]["params"]
        for key in ("segmentation", "embedding"):
            parameters[key] = str(
                self.deployment.model(f"diarizer_{key}").path / "pytorch_model.bin"
            )
        with tempfile.TemporaryDirectory(
            prefix="transcription-diarization-"
        ) as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            return module.DiarizationPipeline(
                model_name=str(path), device=self.deployment.device
            )

    @staticmethod
    def _dependencies() -> tuple[ModuleType, ModuleType]:
        """Load WhisperX only after its deployment has been selected."""
        return importlib.import_module("torch"), importlib.import_module("whisperx")

    def load_alignment_model(self, language: str) -> tuple[Any, dict[str, Any]]:
        """Load each language once from its manifest-recorded asset."""
        if language not in self.alignment_models:
            asset = self.deployment.model(f"aligner_{language}")
            model_name = str(asset.path) if asset.path.is_dir() else asset.repo_id
            self.alignment_models[language] = self.whisperx.load_align_model(
                language_code=language,
                device=self.deployment.device,
                model_name=model_name,
                model_dir=str(asset.path.parent) if asset.path.is_file() else None,
                model_cache_only=True,
            )
        return self.alignment_models[language]

    def transcribe(
        self, input_file: str | Path, context: TranscriptionContext
    ) -> TranscriptionResult:
        """Preserve source ASR text through alignment and optional speaker assignment."""
        if not Path(input_file).is_file():
            raise ConfigurationError(f"Audio file does not exist: {input_file}")
        try:
            audio = self.whisperx.load_audio(str(input_file))
            self.model.options.initial_prompt = (
                WHISPERX_INITIAL_PROMPT % context.community
            ) + (f"\n{context.terminology}" if context.terminology else "")
            with self.torch.inference_mode():
                base = self.model.transcribe(
                    audio, batch_size=self.deployment.whisperx_batch_size
                )
                language = base.get("language")
                if language not in VALID_LANGUAGES:
                    raise TranscriptionError(
                        f"Unsupported detected language: {language}", code="language"
                    )
                aligner, metadata = self.load_alignment_model(language)
                speakers = None
                if self.diarization_model is not None:
                    speakers = self.diarization_model(
                        audio, num_speakers=self.deployment.whisperx_num_speakers
                    )
                builder = ResultBuilder()
                for source in base["segments"]:
                    if not source["text"].strip():
                        continue
                    aligned = self.whisperx.align(
                        [source],
                        aligner,
                        metadata,
                        audio,
                        self.deployment.device,
                        return_char_alignments=True,
                    )
                    if speakers is not None:
                        aligned = self.whisperx.assign_word_speakers(speakers, aligned)
                    tokens = [
                        AlignmentToken(
                            word["word"],
                            (
                                word.get("start")
                                if word.get("score") is not None
                                else None
                            ),
                            word.get("end") if word.get("score") is not None else None,
                            word.get("speaker"),
                            word.get("score"),
                            "aligned" if word.get("score") is not None else "inferred",
                        )
                        for segment in aligned["segments"]
                        for word in segment.get("words", [])
                    ]
                    builder.append(
                        source["text"],
                        tokens,
                        source["start"],
                        source["end"],
                        language,
                    )
            return builder.build(
                len(audio) / 16000,
                "whisperx",
                self.deployment.model("asr").repo_id,
                self.deployment.model(f"aligner_{language}").repo_id,
            )
        except (TranscriptionError, ConfigurationError):
            raise
        except Exception as exc:
            raise TranscriptionError(exc, code="inference") from exc
