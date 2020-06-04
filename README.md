# Torch.jl
Sensible extensions for exposing torch in Julia.

This package is aimed at providing the `Tensor` type, which offloads all computations over to [PyTorch](https://pytorch.org).

**Note:**
* Needs a machine with a CUDA GPU
  * will need lazy artifacts function without a GPU

## Quick Start

To add the package, from the Julia REPL, enter the Pkg prompt by typing `]` and execute the following:
```julia
pkg> add Torch
```

Or via Julia's package manager Pkg.
```julia
julia> using Pkg; Pkg.add("Torch");
```

## Usage Example

```julia
using Metalhead, Metalhead.Flux, Torch

resnet = ResNet()
```

We can move our object over to Torch via a simple call to `torch`

```julia
tresnet = resnet.layers |> torch
```

Or if we need more control over the device to be used like so:

```julia
ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
tip = tensor(ip, dev = 0) # 0 => GPU:0 in Torch
cpu_tensor = tensor(ip, dev = -1) # -1 => CPU:0
```

Calling into the model is done via the usual Flux mechanism.

```julia
tresnet(tip);
```

We can take gradients using Zygote as well

```julia
gs = gradient(x -> sum(tresnet(x)), tip);

# Or

ps = Flux.params(tresnet);
gs = gradient(ps) do
  sum(tresnet(tip))
end
```

## Contributing and Issues

Please feel free to open issues you might encounter in the issue tracker.
I would also appreciate contributions through PRs toward corrections, increased
coverage, docs, etc. Testing currently runs on Linux, but that can be expanded
as need arises.

## Acknowledgements
Takes a lot of inspiration from existing such projects - ocaml-torch for generating the wrappers.
