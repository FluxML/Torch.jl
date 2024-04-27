sudo chown -R vscode:vscode /opt/juliaup /opt/julia_depot

./.dev/install_cudnn.sh $CUDA_VERSION $CUDNN_VERSION

opam init --disable-sandboxing --auto-setup
