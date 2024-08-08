#!/usr/bin/env bash

set -e

TORCH_VARIANT=$1
TORCH_VERSION=$2

cd /usr/local
wget -q "https://download.pytorch.org/libtorch/$TORCH_VARIANT/libtorch-cxx11-abi-shared-with-deps-$TORCH_VERSION%2B$TORCH_VARIANT.zip"
unzip -q libtorch-*.zip
rm libtorch-*.zip
