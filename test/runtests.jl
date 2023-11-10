using Test

@testset verbose=true "Torch" begin
    include("flux_tests.jl")
    include("tensor_movement_tests.jl")
    include("tensor_nnlib_tests.jl")
end
