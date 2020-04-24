using Test, Flux, Metalhead
using Torch
using Torch: Tensor, tensor

@testset "Movement" begin
  r = rand(Float32, 3,3)
  tr = tensor(r, dev = 0)
  @test tr isa Tensor
  @test tr .* tr isa Tensor

  cr = collect(tr)
  @test cr isa Array
end


