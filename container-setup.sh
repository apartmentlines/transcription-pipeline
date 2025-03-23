#!/usr/bin/env bash

cache_dir="/var/local/data/cache"
pip_cache_dir="${cache_dir}/pip"

mkdir -p ${cache_dir}

pip install --cache-dir ${pip_cache_dir} -r requirements-initial.txt && \
  pip install --cache-dir ${pip_cache_dir} -e . && \
  pip install --cache-dir ${pip_cache_dir} -r requirements-adjustments.txt && \
  seed-models --debug
