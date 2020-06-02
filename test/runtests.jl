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

@testset "Flux" begin
  resnet = ResNet()
  tresnet = Flux.fmap(Torch.to_tensor, resnet.layers)

  ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
  tip = tensor(ip, dev = 0) # 0 => GPU:0 in Torch

  top = tresnet(tip)
  op = resnet.layers(ip)

  gs = gradient(() -> sum(tresnet(tip)), Flux.params(tresnet))
  @test top isa Tensor
  @test size(top) == size(op)
  @test gs isa Flux.Zygote.Grads
end
