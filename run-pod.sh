#!/usr/bin/env bash

# limit=${1}
# processing_limit=${2}
ssh_dir="${HOME}/.ssh"

mkdir -p "${ssh_dir}" && \
  chmod 700 "${ssh_dir}" && \
  echo "${PUBLIC_KEY}" > "${ssh_dir}/authorized_keys" && \
  chmod 600 "${ssh_dir}/authorized_keys" && \
  service ssh start

# transcription-processor --limit "${limit}" --processing-limit "${processing_limit}"
sleep 1h
runpodctl stop pod "${RUNPOD_POD_ID}"
