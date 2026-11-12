T_builds_whisperx_without_publishing() {
  local output
  docker() { printf 'DOCKER %s\n' "${*}"; }
  export -f docker
  output=$(bash ./dockerhub-build.sh -b whisperx -t test-version) || return 1
  [[ "${output}" == *"Dockerfile.whisperx --target runtime --tag apartmentlinesproduction/transcription-pipeline-whisperx:test-version"* ]] || return 1
  [[ "${output}" != *"DOCKER push"* ]]
}

T_publishes_qwen_only_when_requested() {
  local output
  docker() { printf 'DOCKER %s\n' "${*}"; }
  export -f docker
  output=$(bash ./dockerhub-build.sh -b qwen3 -t test-version -p) || return 1
  [[ "${output}" == *"Dockerfile.qwen3 --target runtime --tag apartmentlinesproduction/transcription-pipeline-qwen3:test-version"* ]] || return 1
  [[ "${output}" == *"DOCKER push apartmentlinesproduction/transcription-pipeline-qwen3:test-version"* ]]
}

T_rejects_unknown_backend_before_building() {
  local output status
  docker() { printf 'DOCKER %s\n' "${*}"; }
  export -f docker
  output=$(bash ./dockerhub-build.sh -b invalid 2>&1)
  status=${?}
  [[ "${status}" -eq 2 && "${output}" != *"DOCKER "* ]]
}

T_build_failure_prevents_publish() {
  local output status
  docker() { printf 'DOCKER %s\n' "${*}"; return 9; }
  export -f docker
  output=$(bash ./dockerhub-build.sh -b qwen3 -p 2>&1)
  status=${?}
  [[ "${status}" -eq 9 && "${output}" != *"DOCKER push"* ]]
}
