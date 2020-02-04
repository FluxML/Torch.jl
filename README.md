# Torch.jl
Sensible extensions for exposing torch in Julia.

This package is aimed at providing the `tensor` type, which offloads all computations over to torch.

**Note:**
* Needs a working libtorch v1.4 installation, with CUDA (if desired)
  - Will be alleviated with a move to artifacts

## Acknowledgements
Takes a lot of inspiration from existing such projects - ocaml-torch for generating the wrappers.
