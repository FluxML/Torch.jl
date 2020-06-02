# Torch.jl
Sensible extensions for exposing torch in Julia.

This package is aimed at providing the `Tensor` type, which offloads all computations over to [PyTorch](https://pytorch.org).

**Note:**
* Needs a machine with a CUDA GPU
  * will need lazy artifacts function without a GPU

## Usage Example

```julia
using Metalhead, Metalhead.Flux, Torch

resnet = ResNet()
tresnet = Flux.fmap(Torch.to_tensor, resnet.layers)

ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
tip = tensor(ip, dev = 0) # 0 => GPU:0 in Torch

tresnet(tip);

# Taking gradients
gs = gradient(x -> sum(tresnet(x)), tip);
```

## Acknowledgements
Takes a lot of inspiration from existing such projects - ocaml-torch for generating the wrappers.
