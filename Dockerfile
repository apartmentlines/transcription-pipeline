# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.13.18-slim-bookworm

FROM ${BASE_IMAGE} AS repository-key
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /nvidia-cuda.gpg

FROM ${BASE_IMAGE} AS runtime-base
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    PATH=/opt/venv/bin:${PATH} \
    HF_HOME=/opt/models/huggingface \
    TORCH_HOME=/opt/models/torch
COPY --from=repository-key /nvidia-cuda.gpg /usr/share/keyrings/nvidia-cuda.gpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg openssh-server procps \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
        > /etc/apt/sources.list.d/nvidia-cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcudnn9-cuda-12=9.27.0.42-1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.0.1/runpodctl-linux-amd64 \
    && chmod 755 /usr/local/bin/runpodctl

FROM runtime-base AS dependencies-build
RUN apt-get update && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
COPY requirements-frozen.txt /tmp/requirements-frozen.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements-frozen.txt \
    && python -m pip check

FROM runtime-base AS dependencies
COPY --from=dependencies-build /opt/venv /opt/venv

FROM dependencies AS models
WORKDIR /model-seeding
COPY transcription_pipeline/constants.py transcription_pipeline/seed_models.py ./transcription_pipeline/
RUN python -m transcription_pipeline.seed_models \
    --device cpu --compute-type int8 --skip-diarization
RUN python -m nltk.downloader --dir /opt/models/nltk --exit-on-error punkt_tab \
    && rm /opt/models/nltk/tokenizers/punkt_tab.zip

FROM dependencies AS application
ENV NLTK_DATA=/opt/models/nltk
COPY --from=models /opt/models /opt/models
WORKDIR /usr/src/transcription-pipeline
COPY pyproject.toml README.md LICENSE ./
COPY transcription_pipeline ./transcription_pipeline
COPY run-pod.sh ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip check
RUN printf '%s\n' \
    'export PATH=/opt/venv/bin:${PATH}' \
    'export HF_HOME=${HF_HOME:-/opt/models/huggingface}' \
    'export TORCH_HOME=${TORCH_HOME:-/opt/models/torch}' \
    'export NLTK_DATA=${NLTK_DATA:-/opt/models/nltk}' \
    > /etc/profile.d/transcription-venv.sh
CMD ["transcription-processor", "--limit", "1000", "--processing-limit", "2"]

FROM application AS development
RUN apt-get update && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements-dev.txt \
    && python -m pip check
COPY tests ./tests

FROM application AS runtime
