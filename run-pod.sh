#!/usr/bin/env bash

limit=${1}
processing_limit=${2}

transcription-processor --limit "${limit}" --processing-limit "${processing_limit}"
runpodctl stop pod "${RUNPOD_POD_ID}"
