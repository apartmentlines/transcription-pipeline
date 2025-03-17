ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /usr/src/transcription-pipeline

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends curl net-tools
RUN apt-get update && apt-get install -y --no-install-recommends vim procps

COPY . .

RUN pip install --no-cache-dir -r requirements-initial.txt
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r requirements-adjustments.txt
