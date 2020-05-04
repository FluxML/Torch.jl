module Torch

using CUDAapi

if has_cuda()
  using Torch_jll
else
  @warn "Torch currently requires a GPU to operate normally."
end

export Tensor, tensor, Scalar

using ZygoteRules
using ZygoteRules: @adjoint
using NNlib
using NNlib: PoolDims
using Requires

TURN_ON_LOGGING = false

# include("wrap2.jl")
include("error.jl")

include("wrap/libtorch_common.jl")
include("wrap/libdoeye_caml_generated.jl")

# sync + clear empty cache
const clear_cache = at_empty_cache
const sync = at_sync

include("tensor.jl")
include("scalar.jl")
include("ops.jl")
include("nnlib.jl")
include("normalise.jl")
include("broadcast.jl")
include("statistics.jl")

include("utils.jl")

@init @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin

  function (tbn::Flux.BatchNorm)(x::Tensor)
    tbn.λ.(Torch.batchnorm(x, tbn.γ,  tbn.β,  tbn.μ, tbn.σ², 0, tbn.momentum, tbn.ϵ, 1))
  end

  Torch.tensor(x::Flux.Zygote.FillArrays.Fill; kwargs...) = Torch.tensor(collect(x); kwargs...)
end

end # module
