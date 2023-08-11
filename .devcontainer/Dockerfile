ARG BASE_IMAGE_VARIANT=debian
ARG BASE_IMAGE_VERSION=11

ARG BASE_IMAGE_TAG=$BASE_IMAGE_VARIANT-$BASE_IMAGE_VERSION

FROM mcr.microsoft.com/devcontainers/cpp:$BASE_IMAGE_TAG

ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8.9.4

ARG OCAML_VERSION=4
ARG OPAM_VERSION=2

ARG TORCH_VARIANT
ARG TORCH_VERSION=2.1.1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && apt-get satisfy -y "ocaml (>= $OCAML_VERSION)" "opam (>= $OPAM_VERSION)" \
    && rm -rf /var/lib/apt/lists/*

ENV JULIA_DEPOT_PATH=/opt/julia_depot
ENV JULIAUP_DEPOT_PATH=/opt/juliaup
RUN curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.9 --path /opt/juliaup --yes
ENV PATH=/opt/juliaup/bin:$PATH

ENV CUDA_VERSION=$CUDA_VERSION
ENV CUDNN_VERSION=$CUDNN_VERSION

RUN <<EOF
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
EOF
ENV PATH=$PATH:/usr/local/cuda/bin

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [ -z "$TORCH_VARIANT" ]; then export TORCH_VARIANT="cu$(echo $CUDA_VERSION | cut -d . -f 1-2 | tr -d '.')"; fi \
    && cd /usr/local \
    && wget -q "https://download.pytorch.org/libtorch/$TORCH_VARIANT/libtorch-cxx11-abi-shared-with-deps-$TORCH_VERSION%2B$TORCH_VARIANT.zip" \
    && unzip -q libtorch-*.zip \
    && rm libtorch-*.zip

ENV CMAKE_PREFIX_PATH=/usr/local/libtorch
