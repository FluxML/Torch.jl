using Test
using Torch: Tensor, tensor

if Torch.cuda_is_available() && Torch.cudnn_is_available() && Torch.cuda_device_count() > 0
    torch_device = 0 # GPU 0
else
    torch_device = -1 # CPU
end

@testset "Movement" begin
    r = rand(Float32, 3,3)
    tr = tensor(r, dev = torch_device)
    @test tr isa Tensor
    @test tr .* tr isa Tensor

    cr = collect(tr)
    @test cr isa Array
end
