mock_compose() {
  docker() { printf 'DOCKER'; printf ' <%s>' "${@}"; printf '\n'; }
  export -f docker
}

T_defaults_to_whisperx_with_expected_action() {
  local wrapper output action
  mock_compose
  for wrapper in dev setup test; do
    action=up
    [[ "${wrapper}" != setup ]] || action=build
    output=$(bash "./${wrapper}.sh") || return 1
    [[ "${output}" == *"/docker-compose.whisperx.yaml>"* && "${output}" == *"<${action}>" ]] || return 1
  done
}

T_selects_both_backends_for_every_wrapper() {
  local wrapper backend output project
  mock_compose
  for wrapper in dev setup test; do
    for backend in whisperx qwen3; do
      project="transcription-pipeline-${backend}"
      [[ "${wrapper}" != test ]] || project="${project}-test"
      output=$(bash "./${wrapper}.sh" -b "${backend}") || return 1
      [[ "${output}" == *"<--project-name> <${project}>"* ]] || return 1
      [[ "${output}" == *"/docker-compose.${backend}.yaml>"* ]] || return 1
      if [[ "${wrapper}" == test ]]; then
        [[ "${output}" == *"/docker-compose.test.yaml>"* ]] || return 1
      else
        [[ "${output}" != *"/docker-compose.test.yaml>"* ]] || return 1
      fi
    done
  done
}

T_preserves_compose_argument_boundaries() {
  local output
  mock_compose
  output=$(bash ./test.sh -b qwen3 run --rm transcription-processor pytest -k 'one or two') || return 1
  [[ "${output}" == *"<run> <--rm> <transcription-processor> <pytest> <-k> <one or two>" ]] || return 1
  output=$(bash ./setup.sh -b qwen3 -- --build-arg 'LABEL=two words') || return 1
  [[ "${output}" == *"<build> <--build-arg> <LABEL=two words>" ]]
}

T_rejects_bad_wrapper_options_before_docker() {
  local wrapper output status
  mock_compose
  for wrapper in dev setup test; do
    output=$(bash "./${wrapper}.sh" -b invalid 2>&1)
    status=${?}
    [[ "${status}" -eq 2 && "${output}" != *DOCKER* ]] || return 1
    output=$(bash "./${wrapper}.sh" -b 2>&1)
    status=${?}
    [[ "${status}" -eq 2 && "${output}" != *DOCKER* ]] || return 1
    output=$(bash "./${wrapper}.sh" -x 2>&1)
    status=${?}
    [[ "${status}" -eq 2 && "${output}" != *DOCKER* ]] || return 1
  done
}

T_help_does_not_invoke_docker() {
  local wrapper output
  mock_compose
  for wrapper in dev setup test; do
    output=$(bash "./${wrapper}.sh" -h) || return 1
    [[ "${output}" == *"Usage: ./${wrapper}.sh"* && "${output}" != *DOCKER* ]] || return 1
  done
}

T_runs_from_another_directory() {
  local repo_path="${PWD}" output
  mock_compose
  output=$(cd /tmp && bash "${repo_path}/dev.sh" -b qwen3 config --quiet) || return 1
  [[ "${output}" == *"<--project-directory> <${repo_path}>"* ]] || return 1
  [[ "${output}" == *"<${repo_path}/docker-compose.qwen3.yaml> <config> <--quiet>" ]]
}

T_propagates_docker_failure() {
  local wrapper status
  docker() { return 9; }
  export -f docker
  for wrapper in dev setup test; do
    bash "./${wrapper}.sh" -b qwen3 >/dev/null 2>&1
    status=${?}
    [[ "${status}" -eq 9 ]] || return 1
  done
}
