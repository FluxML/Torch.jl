#!/usr/bin/env bash

set -e

CUDA_VERSION=$1

CUDA_VERSION_MAJOR_MINOR=$(echo $CUDA_VERSION | cut -d . -f 1-2)
TMP_PROJECT=$(mktemp -d)
cd $TMP_PROJECT
touch Project.toml
cat <<EOT > LocalPreferences.toml
[CUDA_Runtime_jll]
version = "$CUDA_VERSION_MAJOR_MINOR"
EOT
CUDA_ROOT=$(julia --project --eval '
    using Pkg
    CUDA_VERSION = VersionNumber(ENV["CUDA_VERSION"])
    CUDA_SDK_jll_pkg = :CUDA_SDK_jll
    if CUDA_VERSION < v"11.4"
        CUDA_SDK_jll_pkg = :CUDA_full_jll
    end
    Pkg.add(name=string(CUDA_SDK_jll_pkg), version=ENV["CUDA_VERSION"])
    @eval using $CUDA_SDK_jll_pkg
    println(@eval $CUDA_SDK_jll_pkg.artifact_dir)
')
ln -s $CUDA_ROOT/cuda /usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
