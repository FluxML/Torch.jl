ARG BASE_IMAGE_TAG

FROM mcr.microsoft.com/devcontainers/cpp:$BASE_IMAGE_TAG

ARG OCAML_VERSION
ARG OPAM_VERSION
ARG TORCH_VERSION

RUN sudo apt-get update \
    && sudo apt-get satisfy -y "ocaml (>= $OCAML_VERSION)" "opam (>= $OPAM_VERSION)" \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/local \
    && sudo wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$TORCH_VERSION%2Bcpu.zip \
    && sudo unzip libtorch-*.zip \
    && sudo rm libtorch-*.zip

ENV CMAKE_PREFIX_PATH=/usr/local/libtorch
