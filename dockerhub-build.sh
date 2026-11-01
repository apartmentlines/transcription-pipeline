#!/usr/bin/env bash

set -euo pipefail

readonly DOCKERHUB_REPOSITORY="apartmentlinesproduction/transcription-pipeline"
readonly DOCKERHUB_TAG="v0.1.0"
readonly DOCKERHUB_IMAGE="${DOCKERHUB_REPOSITORY}:${DOCKERHUB_TAG}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

build_image() {
  printf 'Building prepared image %s\n' "${DOCKERHUB_IMAGE}"
  docker build --target runtime --tag "${DOCKERHUB_IMAGE}" "${SCRIPT_DIR}"
}

push_image() {
  printf 'Pushing %s\n' "${DOCKERHUB_IMAGE}"
  docker push "${DOCKERHUB_IMAGE}"
}

main() {
  build_image
  push_image
  printf 'Published %s\n' "${DOCKERHUB_IMAGE}"
}

main "$@"
