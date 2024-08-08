#!/usr/bin/env bash

set -e

CUDA_VERSION=$1
CUDNN_VERSION=$2

CUDA_VERSION_MAJOR_MINOR=$(echo $CUDA_VERSION | cut -d . -f 1-2)
TMP_PROJECT=$(mktemp -d)
cd $TMP_PROJECT
touch Project.toml
cat <<EOT > LocalPreferences.toml
[CUDA_Runtime_jll]
version = "$CUDA_VERSION_MAJOR_MINOR"
EOT
CUDNN_ROOT=$(julia --project --eval '
    using Pkg;
    CUDA_VERSION = VersionNumber(ENV["CUDA_VERSION"])
    if CUDA_VERSION < v"11"
        Pkg.add(name="CUDA_Runtime_jll", version="0.2")
    elseif CUDA_VERSION < v"11.4"
        Pkg.add(name="CUDA_Runtime_jll", version="0.7")
    else
        Pkg.add(name="CUDA_Runtime_jll")
    end
    Pkg.add(name="CUDNN_jll", version=ENV["CUDNN_VERSION"]);
    using CUDNN_jll;
    println(CUDNN_jll.artifact_dir)')
for F in $CUDNN_ROOT/include/cudnn*.h; do ln -sf $F /usr/local/cuda/include/$(basename $F); done
for F in $CUDNN_ROOT/lib/libcudnn*; do ln -sf $F /usr/local/cuda/lib64/$(basename $F); done
