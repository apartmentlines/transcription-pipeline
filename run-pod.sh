#!/usr/bin/env bash

setup_ssh() {
  local ssh_dir="${HOME}/.ssh"
  mkdir -pv "${ssh_dir}" && \
    chmod -v 700 "${ssh_dir}" && \
    echo "${PUBLIC_KEY}" > "${ssh_dir}/authorized_keys" && \
    chmod -v 600 "${ssh_dir}/authorized_keys" && \
    service ssh start
}

run_service() {
  echo "Running REST interface for transcription processor..."
  rest-interface "$@"
}

stop_pod() {
  if [ -f "/tmp/abort-pod-stop" ]; then
    echo "Aborting pod stop..."
    return
  fi
  if [ -n "${RUNPOD_POD_ID}" ]; then
    echo "Stopping pod ${RUNPOD_POD_ID}..."
    runpodctl stop pod "${RUNPOD_POD_ID}"
  fi
}

trap stop_pod EXIT SIGINT SIGTERM ERR

setup_ssh
run_service "$@"
