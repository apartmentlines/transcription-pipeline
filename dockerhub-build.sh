#!/usr/bin/env bash

DOCKERHUB_USERNAME="apartmentlinesproduction"
DOCKERHUB_PROJECT="transcription-pipeline"
LOCAL_DOCKER_IMAGE="transcription-pipeline-transcription-processor-1"
DOCKERHUB_PROJECT_TAG="v0.1.0"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "Building docker image ${LOCAL_DOCKER_IMAGE}"
docker compose build
echo "Running setup script on ${LOCAL_DOCKER_IMAGE}"
"${SCRIPT_DIR}/setup.sh"
echo "Committing ${LOCAL_DOCKER_IMAGE} changes to ${DOCKERHUB_USERNAME}/${DOCKERHUB_PROJECT}:${DOCKERHUB_PROJECT_TAG}"
docker commit ${LOCAL_DOCKER_IMAGE} ${DOCKERHUB_USERNAME}/${DOCKERHUB_PROJECT}:${DOCKERHUB_PROJECT_TAG}
echo "Pushing changes to ${DOCKERHUB_USERNAME}/${DOCKERHUB_PROJECT}:${DOCKERHUB_PROJECT_TAG}"
docker push ${DOCKERHUB_USERNAME}/${DOCKERHUB_PROJECT}:${DOCKERHUB_PROJECT_TAG}
