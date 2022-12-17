FROM mcr.microsoft.com/devcontainers/cpp

ARG TORCH_VERSION

RUN cd /usr/local \
    && sudo wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$TORCH_VERSION%2Bcpu.zip \
    && sudo unzip libtorch-*.zip \
    && sudo rm libtorch-*.zip

ENV CMAKE_PREFIX_PATH=/usr/local/libtorch
