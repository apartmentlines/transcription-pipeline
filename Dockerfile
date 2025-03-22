ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /usr/src/transcription-pipeline
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

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

COPY . .

RUN pip install --no-cache-dir -r requirements-initial.txt && \
  pip install --no-cache-dir -e . && \
  pip install --no-cache-dir -r requirements-adjustments.txt
# RUN seed-models --debug
