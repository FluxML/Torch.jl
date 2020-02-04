module Torch

export Tensor, tensor

using YAML
using Clang

const PATH = "/home/dhairyagandhi96/temp/aten/torch_cpp/ocaml-torch/src/wrapper" # /build" # /libdoeye_caml.so"

# include("wrap2.jl")

include("libtorch_common.jl")
include("libdoeye_caml_generated.jl")

include("tensor.jl")
include("ops.jl")
include("nnlib.jl")
include("broadcast.jl")

end # module
