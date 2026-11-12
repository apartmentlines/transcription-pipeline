#!/usr/bin/env bash
set -euo pipefail
readonly DEFAULT_BACKEND="whisperx"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

usage() {
  printf 'Usage: ./%s.sh [-b whisperx|qwen3] [-h] [--] [COMPOSE_ARGS...]\n' "${1}"
  printf 'Backend defaults to %s. Put wrapper options before Compose arguments.\n' "${DEFAULT_BACKEND}"
  if [[ "${1}" == setup ]]; then
    printf 'Builds the runtime image. Example: ./setup.sh -b qwen3 -- --no-cache\n'
  else
    printf 'Defaults to up. Example: ./%s.sh -b qwen3 run --rm transcription-processor\n' "${1}"
  fi
}

main() {
  local mode="${1}" backend="${DEFAULT_BACKEND}" option project
  local OPTIND=1
  local -a command
  shift
  case "${mode}" in
    dev|setup|test) ;;
    *) printf 'Unknown wrapper: %s\n' "${mode}" >&2; return 2 ;;
  esac
  while getopts ':b:h' option; do
    case "${option}" in
      b) backend="${OPTARG}" ;;
      h) usage "${mode}"; return 0 ;;
      *) usage "${mode}" >&2; return 2 ;;
    esac
  done
  shift "$((OPTIND - 1))"
  case "${backend}" in
    whisperx|qwen3) ;;
    *) printf 'Unknown backend: %s\n' "${backend}" >&2; return 2 ;;
  esac
  project="transcription-pipeline-${backend}"
  if [[ "${mode}" == test ]]; then
    project="${project}-test"
  fi
  command=(docker compose --project-directory "${SCRIPT_DIR}" --project-name "${project}" -f "${SCRIPT_DIR}/docker-compose.${backend}.yaml")
  if [[ "${mode}" == test ]]; then
    command+=(-f "${SCRIPT_DIR}/docker-compose.test.yaml")
  fi
  if [[ "${mode}" == setup ]]; then
    command+=(build)
  elif (( $# == 0 )); then
    set -- up
  fi
  printf 'Using %s backend (%s project)\n' "${backend}" "${project}" >&2
  "${command[@]}" "${@}"
}

main "${@}"
