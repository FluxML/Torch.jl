ARG BASE_IMAGE_VARIANT=debian
ARG BASE_IMAGE_VERSION=11

ARG BASE_IMAGE_TAG=$BASE_IMAGE_VARIANT-$BASE_IMAGE_VERSION

FROM mcr.microsoft.com/devcontainers/cpp:$BASE_IMAGE_TAG

ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8.9.4

ARG GCC_VERSION=10

ARG OCAML_VERSION=4
ARG OPAM_VERSION=2

ARG TORCH_VARIANT
ARG TORCH_VERSION=2.1.1

COPY .dev /opt/container

RUN /opt/container/install_build_deps.sh $GCC_VERSION \
    && apt-get satisfy -y "ocaml (>= $OCAML_VERSION)" "opam (>= $OPAM_VERSION)" \
    && rm -rf /var/lib/apt/lists/*

ENV JULIA_DEPOT_PATH=/opt/julia_depot
ENV JULIAUP_DEPOT_PATH=/opt/juliaup
RUN curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.11 --path /opt/juliaup --yes
ENV PATH=/opt/juliaup/bin:$PATH

ENV CUDA_VERSION=$CUDA_VERSION
ENV CUDNN_VERSION=$CUDNN_VERSION

RUN /opt/container/install_cuda_sdk.sh $CUDA_VERSION
ENV PATH=$PATH:/usr/local/cuda/bin

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export TORCH_VARIANT="cu$(echo $CUDA_VERSION | cut -d . -f 1-2 | tr -d '.')" \
    && /opt/container/install_torch.sh $TORCH_VARIANT $TORCH_VERSION

ENV CMAKE_PREFIX_PATH=/usr/local/libtorch
