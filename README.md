# Torch.jl

Sensible extensions for exposing torch in Julia.

This package is aimed at providing the `Tensor` type, which offloads all computations over to [ATen](https://pytorch.org/cppdocs/), the foundational tensor library for PyTorch, written in C++.

## Supported platforms

| **Operating System** | **Architecture** | **Acceleration Runtime** |
| --- | --- | --- |
| macOS | aarch64 | - |
| macOS | x86_64 | - |
| Linux (glibc) | aarch64 | - |
| Linux (glibc) | x86_64 | CUDA 10.2 |
| Linux (glibc) | x86_64 | CUDA 11.3 |

Windows support is pending, cf. [issue #26](https://github.com/FluxML/Torch.jl/issues/26).

The binary dependencies are available for Linux (glibc) on i686 (32-bit), but not all tests run succesfully.

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
using Torch: torch

resnet = ResNet(18)
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
Contributions through PRs toward corrections, increased
coverage, docs, etc. are most welcome.

## Acknowledgements

Takes a lot of inspiration from existing such projects, in particular [ocaml-torch](https://github.com/LaurentMazare/ocaml-torch), and [tch-rs](https://github.com/LaurentMazare/tch-rs).
