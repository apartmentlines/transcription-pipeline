ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /usr/src/transcription-pipeline
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;9.0a"

RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  ffmpeg \
  gcc \
  git \
  gnupg \
  net-tools \
  openssh-server \
  procps \
  python3-dev \
  tree \
  vim

RUN \
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg && \
  echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
    > /etc/apt/sources.list.d/nvidia-cuda.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends cudnn9-cuda-12=9.27.0.42-1 && \
  rm -rf /var/lib/apt/lists/*

RUN apt clean

RUN curl -L -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.0.1/runpodctl-linux-amd64 && \
  chmod 755 /usr/local/bin/runpodctl

COPY . .
