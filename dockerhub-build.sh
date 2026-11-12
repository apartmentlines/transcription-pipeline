#!/usr/bin/env bash
set -euo pipefail
readonly DOCKERHUB_REPOSITORY="apartmentlinesproduction/transcription-pipeline"
readonly DEFAULT_TAG="v0.1.0"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

usage() {
  printf 'Usage: %s [-b whisperx|qwen3] [-t TAG] [-p]\n' "${0}"
  printf 'Build one runtime image; -p also publishes it to Docker Hub.\n'
}

main() {
  local backend="whisperx" tag="${DEFAULT_TAG}" publish=0 option dockerfile image
  while getopts ':b:t:ph' option; do
    case "${option}" in
      b) backend="${OPTARG}" ;;
      t) tag="${OPTARG}" ;;
      p) publish=1 ;;
      h) usage; return 0 ;;
      *) usage >&2; return 2 ;;
    esac
  done
  shift "$((OPTIND - 1))"
  if (( $# != 0 )); then
    usage >&2
    return 2
  fi
  case "${backend}" in
    whisperx|qwen3) dockerfile="Dockerfile.${backend}" ;;
    *) printf 'Unknown backend: %s\n' "${backend}" >&2; return 2 ;;
  esac
  if [[ -z "${tag}" ]]; then
    printf 'Image tag must be nonempty.\n' >&2
    return 2
  fi
  image="${DOCKERHUB_REPOSITORY}-${backend}:${tag}"
  printf 'Building %s from %s\n' "${image}" "${dockerfile}"
  docker build --file "${SCRIPT_DIR}/${dockerfile}" --target runtime --tag "${image}" "${SCRIPT_DIR}"
  if (( publish )); then
    printf 'Publishing %s\n' "${image}"
    docker push "${image}"
  fi
}

main "$@"
