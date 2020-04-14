# Torch.jl
Sensible extensions for exposing torch in Julia.

This package is aimed at providing the `tensor` type, which offloads all computations over to [PyTorch](https://pytorch.org).

**Note:**
* Needs a working libtorch v1.4 installation, with CUDA (if desired)
  - Will be alleviated with a move to artifacts
* For the time being please follow the build instructions [here](build/README.md)

## Usage Example

```julia
using Metalhead, Metalhead.Flux, Torch

resnet = ResNet()
tresnet = Flux.fmap(Torch.to_tensor, resnet.layers)

ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
tip = tensor(ip, dev = 0) # 0 => GPU:0 in Torch

tresnet(tip);
```

## Acknowledgements
Takes a lot of inspiration from existing such projects - ocaml-torch for generating the wrappers.
