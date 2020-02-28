module Torch

export Tensor, tensor, Scalar

using YAML
using Clang
using ZygoteRules
using ZygoteRules: @adjoint
using NNlib
using NNlib: PoolDims

TURN_ON_LOGGING = false

# include("wrap2.jl")

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
end # module
