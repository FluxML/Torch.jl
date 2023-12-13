ARG BASE_IMAGE_VARIANT=debian
ARG BASE_IMAGE_VERSION=11

ARG BASE_IMAGE_TAG=$BASE_IMAGE_VARIANT-$BASE_IMAGE_VERSION

FROM mcr.microsoft.com/devcontainers/cpp:$BASE_IMAGE_TAG

ARG CUDA_VERSION=11.8.0

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

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [ -z "$TORCH_VARIANT" ]; then export TORCH_VARIANT="cu$(echo $CUDA_VERSION | cut -d . -f 1-2 | tr -d '.')"; fi \
    && cd /usr/local \
    && wget -q "https://download.pytorch.org/libtorch/$TORCH_VARIANT/libtorch-cxx11-abi-shared-with-deps-$TORCH_VERSION%2B$TORCH_VARIANT.zip" \
    && unzip -q libtorch-*.zip \
    && rm libtorch-*.zip

ENV CMAKE_PREFIX_PATH=/usr/local/libtorch
