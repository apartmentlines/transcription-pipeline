#!/usr/bin/env bash

base_command="docker compose -f docker-compose.yaml -f docker-compose.test.yaml"

if [ $# -eq 0 ]; then
  ${base_command} up
else
  ${base_command} "$@"
fi
