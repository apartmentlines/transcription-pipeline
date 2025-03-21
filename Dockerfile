ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /usr/src/transcription-pipeline

RUN mkdir -pv /var/local/data/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ffmpeg \
  gcc \
  net-tools \
  procps \
  python3-dev \
  tree \
  vim

RUN wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-debian12-9.8.0_1.0-1_amd64.deb
RUN dpkg -i cudnn-local-repo-debian12-9.8.0_1.0-1_amd64.deb
RUN rm cudnn-local-repo-debian12-9.8.0_1.0-1_amd64.deb
RUN cp /var/cudnn-local-repo-debian12-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get install -y --no-install-recommends cudnn9-cuda-12=9.8.0.87-1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

COPY . .

RUN pip config --global set global.cache-dir "/var/local/data/cache/pip"
RUN pip install -r requirements-initial.txt
RUN pip install -e .
RUN pip install -r requirements-adjustments.txt
# RUN seed-models --debug

RUN apt cache clean
