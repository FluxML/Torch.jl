using Flux
using Metalhead
using Test
using Torch
using Torch: Tensor, tensor

if Torch.cuda_is_available() && Torch.cudnn_is_available() && Torch.cuda_device_count() > 0
    torch_device = 0 # GPU 0
else
    torch_device = -1 # CPU
end

@testset "Flux" begin
    resnet = ResNet(18)
    tresnet = Flux.fmap(x -> Torch.to_tensor(x; dev = torch_device), resnet.layers)

    ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
    tip = tensor(ip, dev = torch_device)

    top = tresnet(tip)
    op = resnet.layers(ip)

    @test top isa Tensor
    @test size(top) == size(op)

    if torch_device == -1 # CPU
        @test_broken false # gradient(...): Could not run 'aten::batch_norm_backward_elemt' with arguments from the 'CPU' backend.
    else
        gs = gradient(() -> sum(tresnet(tip)), Flux.params(tresnet))
        @test gs isa Flux.Zygote.Grads
    end
end
