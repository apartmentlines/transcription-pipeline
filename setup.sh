#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

main() {
  printf 'Building the prepared local transcription image\n'
  docker compose --project-directory "${SCRIPT_DIR}" -f "${SCRIPT_DIR}/docker-compose.yaml" build "$@"
}

main "$@"
