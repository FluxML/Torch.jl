module Torch

export Tensor, tensor

using YAML
using Clang
# using CUDAdrv

const TURN_ON_LOGGING = true

# include("wrap2.jl")

include("wrap/libtorch_common.jl")
include("wrap/libdoeye_caml_generated.jl")

include("tensor.jl")
# include("scalar.jl")
include("ops.jl")
include("statistics.jl")
include("nnlib.jl")
include("broadcast.jl")

end # module
