#!/usr/bin/env bash

set -e

GCC_VERSION=$1

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  gcc-$GCC_VERSION \
  g++-$GCC_VERSION \
  jq \
  unzip \
  wget
