"""CPU-only deployment verification for prepared offline runtime images."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys

import numpy as np

from .deployment import Deployment, ManifestStore
from .errors import ConfigurationError
from .results import AlignmentToken, ResultBuilder
from .rest_interface import TranscriptionRestInterface, create_app
from .subtitles import SubtitleRenderer


class DeploymentSmokeTest:
    """Exercise configuration, real processors, serializers, and HTTP startup."""

    @staticmethod
    def run() -> dict[str, object]:
        """Verify the installation without loading model weights onto a GPU."""
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise ConfigurationError("Run smoke checks with HF_HUB_OFFLINE=1")
        deployment = ManifestStore.load()
        other = "whisperx" if deployment.backend == "qwen3" else "qwen3"
        try:
            ManifestStore.load(backend=other)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("Mismatched backend was accepted")
        for command in ("main", "transcriber", "rest_interface", "seed_models"):
            subprocess.run(
                [sys.executable, "-m", f"transcription_pipeline.{command}", "--help"],
                check=True,
                capture_output=True,
            )
        if deployment.backend == "qwen3":
            if importlib.util.find_spec("whisperx") is not None:
                raise AssertionError("Qwen runtime unexpectedly contains WhisperX")
            DeploymentSmokeTest._qwen_processors(deployment)
            if deployment.qwen_attention == "flash_attention_2":
                importlib.import_module("flash_attn")
                importlib.import_module("flash_attn_2_cuda")
        else:
            for module in (
                "whisperx.asr",
                "whisperx.alignment",
                "whisperx.diarize",
                "whisperx.vads.pyannote",
            ):
                importlib.import_module(module)
        builder = ResultBuilder()
        builder.append(
            "Hello, world!",
            [AlignmentToken("hello", 0, 1), AlignmentToken("world", 1, 2)],
            0,
            2,
            "en",
        )
        builder.append(
            "¡Hola, mundo!",
            [AlignmentToken("hola", 3, 4), AlignmentToken("mundo", 4, 5)],
            3,
            5,
            "es",
        )
        result = builder.build(5, deployment.backend, "fixture", "fixture")
        for format_ in ("srt", "vtt"):
            contents = SubtitleRenderer().render(result, format_)
            if "Hello, world!" not in contents or "¡Hola, mundo!" not in contents:
                raise AssertionError("Subtitle text was altered")
        interface = TranscriptionRestInterface(deployment=deployment)
        app = create_app(
            interface.manager,
            interface.validator,
            "smoke-key",
            logging.getLogger(__name__),
        )
        client = app.test_client()
        status = client.get("/status")
        if status.status_code != 200 or status.json != {
            "status": "idle",
            "exit_code": None,
        }:
            raise AssertionError("REST startup status failed")
        rejected = client.post(
            "/run",
            json={"api_key": "smoke-key", "domain": "unused.invalid", "backend": other},
        )
        if (
            rejected.status_code != 400
            or interface.manager.get_status()["status"] != "idle"
        ):
            raise AssertionError("REST backend mismatch started work")
        return {
            "backend": deployment.backend,
            "manifest": str(deployment.manifest_path),
            "models": {model.role: model.revision for model in deployment.models},
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "pysubs2": importlib.metadata.version("pysubs2"),
            "qwen_attention": (
                deployment.qwen_attention if deployment.backend == "qwen3" else None
            ),
            "flash_attn": (
                importlib.metadata.version("flash-attn")
                if deployment.qwen_attention == "flash_attention_2"
                else None
            ),
            "offline_cpu_checks": "passed",
            "gpu_inference": "not tested",
        }

    @staticmethod
    def _qwen_processors(deployment: Deployment) -> None:
        """Exercise the exact released native processor helpers on CPU audio."""
        from .constants import QWEN_MAX_CHUNK_SECONDS, QWEN_SAMPLE_RATE

        transformers = importlib.import_module("transformers")
        audio = np.zeros(QWEN_MAX_CHUNK_SECONDS * QWEN_SAMPLE_RATE, dtype=np.float32)
        for role in ("asr", "aligner"):
            path = str(deployment.model(role).path)
            transformers.AutoConfig.from_pretrained(path, local_files_only=True)
            processor = transformers.AutoProcessor.from_pretrained(
                path, local_files_only=True
            )
            if role == "asr":
                inputs = processor.apply_transcription_request(
                    audio=audio,
                    language=None,
                    prompt="apartment maintenance",
                    processor_kwargs={
                        "audio_kwargs": {
                            "sampling_rate": QWEN_SAMPLE_RATE,
                            "truncation": False,
                        }
                    },
                )
            else:
                inputs, words = processor.prepare_forced_aligner_inputs(
                    audio=audio,
                    transcript="¿Dónde está la llave?",
                    language="es",
                    processor_kwargs={
                        "audio_kwargs": {
                            "sampling_rate": QWEN_SAMPLE_RATE,
                            "truncation": False,
                        }
                    },
                )
                if not words or not words[0]:
                    raise AssertionError("Aligner processor produced no words")
            frames = int(inputs["input_features_mask"].sum().item())
            if frames < len(audio) // processor.feature_extractor.hop_length:
                raise AssertionError("Native processor truncated the supplied waveform")


def main() -> None:
    """Print an auditable smoke result or return a failure exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    print(json.dumps(DeploymentSmokeTest.run(), indent=2))


if __name__ == "__main__":
    main()
