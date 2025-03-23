ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /usr/src/transcription-pipeline
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;9.0a"

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ffmpeg \
  gcc \
  git \
  net-tools \
  procps \
  python3-dev \
  tree \
  vim

RUN curl -o cudnn.deb https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-debian12-9.8.0_1.0-1_amd64.deb && \
  dpkg -i cudnn.deb && \
  rm -v cudnn.deb && \
  cp /var/cudnn-local-repo-debian12-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
  apt-get update && apt-get install -y --no-install-recommends cudnn9-cuda-12=9.8.0.87-1 && \
  rm -rv /var/cudnn-local-repo-debian12-9.8.0

RUN apt clean

RUN curl -L -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v1.14.4/runpodctl-linux-amd64 && \
  chmod 755 /usr/local/bin/runpodctl

COPY . .
