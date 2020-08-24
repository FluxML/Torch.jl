using Test
using NNlib
using Torch: tensor


@testset "DepthwiseConv" begin
    for kernel_width in [1, 3, 5],
        kernel_height in [1, 2, 4],
        in_channels in [1, 2],
        out_channels in [1, 2]

        kernel = rand(-9.0f0:9.0f0, kernel_height, kernel_width, 1, in_channels)

        for height in [5, 6],
            width in [5, 7]

            test_input = rand(-9.0f0:9.0f0, height, width, in_channels, 1)
            x = tensor(test_input, dev = 0)
            w = tensor(kernel, dev = 0)

            expected_output = NNlib.depthwiseconv(test_input, kernel, pad = (0,0), stride = (1,1 ), dilation = (1, 1), flipped = true)
            test_output = NNlib.depthwiseconv(x, w, pad = (0,0), stride = (1,1 ), dilation = (1, 1))

            test_output = Array(test_output)
            @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
        end
    end
end


@testset "Conv with padding" begin
    for kernel_width in [1, 2, 3, 5],
        kernel_height in [1, 2, 3, 5],
        in_channels in [1, 2],
        out_channels in [1, 2]

        num_coefficients = (kernel_width * kernel_height * in_channels * out_channels)
        kernel = reshape(1.0f0:num_coefficients, kernel_height, kernel_width, in_channels, out_channels)
        kernel = collect(kernel)
        pad = size(kernel)[1:2] .÷ 2

        for height in [1, 2, 3, 4],
            width in [1, 2, 3, 5]

            test_input = zeros(Float32, height, width, in_channels, 1)
            test_input[(height + 1) ÷ 2, (width + 1) ÷ 2, 1, 1] = 1
            x = tensor(test_input, dev = 0)
            w = tensor(kernel, dev = 0)

            cdims = NNlib.DenseConvDims(size(test_input),
                                        size(kernel),
                                        stride=(1, 1),
                                        padding=pad,
                                        dilation=(1, 1),
                                        flipkernel = true)

            expected_output = NNlib.conv(test_input, kernel, cdims)
            test_output     = NNlib.conv(x,          w,      cdims)

            test_output = Array(test_output)
            @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
        end
    end
end


@testset "Conv with stride" begin
    for kernel_width in [1, 3, 4],
        kernel_height in [1, 2, 5],
        in_channels in [1],
        out_channels in [1],
        row_stride in [1, 2, 4],
        column_stride in [1, 3, 5]

        kernel = fill(1.0f0, kernel_height, kernel_width, in_channels, out_channels)
        kernel = collect(kernel)

        for height in 13:(13 + row_stride - 1),
            width in 15:(15 + column_stride - 1)

            sz_in = [height, width, in_channels, 1]
            test_input = reshape(1.0f0:prod(sz_in), height, width, in_channels, 1)
            test_input = collect(test_input)
            x = tensor(test_input, dev = 0)
            w = tensor(kernel, dev = 0)

            cdims = NNlib.DenseConvDims(size(test_input),
                                        size(kernel),
                                        stride=(row_stride, column_stride),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        flipkernel = true)

            expected_output = NNlib.conv(test_input, kernel, cdims)
            test_output     = NNlib.conv(x,          w,      cdims)

            test_output = Array(test_output)
            @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
        end
    end
end


@testset "Conv with dilation" begin
    for kernel_width in 1,
        kernel_height in 1:9,
        in_channels in 1,
        out_channels in 1,
        row_stride in 1:11,
        column_stride in 1,
        row_rate in 1:4,
        column_rate in 1

        if kernel_height * row_rate > 13
            continue
        end

        kernel = fill(1.0f0, kernel_height, kernel_width, in_channels, out_channels)
        kernel = collect(kernel)

        for height in 13:(13 + row_stride - 1),
            width in [1]

            sz_in = [height, width, in_channels, 1]
            test_input = reshape(1.0f0:prod(sz_in), height, width, in_channels, 1)
            test_input = collect(test_input)
            x = tensor(test_input, dev = 0)
            w = tensor(kernel, dev = 0)

            cdims = NNlib.DenseConvDims(size(test_input),
                                        size(kernel),
                                        stride=(row_stride, column_stride),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        flipkernel = true)

            expected_output = NNlib.conv(test_input, kernel, cdims)
            test_output     = NNlib.conv(x,          w,      cdims)

            test_output = Array(test_output)
            @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
        end
    end
end


@testset "Pooling" begin
    for fn in (NNlib.maxpool, NNlib.meanpool),
        column_span in 1:3,
        row_span in 1:3,
        column_stride in 1:3,
        row_stride in 1:3,
        pad in (false, true)

        if pad
            padding = (row_span, column_span) .÷ 2
        else
            padding = (0, 0)
        end

        for height in (1:2) * row_span * row_stride,
            width in (1:2) * column_span * column_stride,
            channels in 1:2

            test_input = rand(0.0f0:9.0f0, height, width, channels, 1)
            x = tensor(test_input, dev = 0)

            pdims = NNlib.PoolDims(size(test_input),
                                   (row_span, column_span),
                                   padding=padding,
                                   stride=(row_stride, column_stride))

            expected_output = fn(test_input, pdims)
            test_output     = fn(x,          pdims)

            test_output = Array(test_output)
            @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
        end
    end
end


@testset "Activations" begin
    for fn in (NNlib.relu, NNlib.tanh, NNlib.sigmoid, NNlib.leakyrelu, NNlib.softmax),
        height in [1, 2, 3, 4, 7],
        width in [1, 2, 3, 5, 6],
        channels in 1:3

        test_input = rand(-9.0f0:9.0f0, height, width, channels, 1)
        x = tensor(test_input, dev = 0)

        if fn == NNlib.softmax
            expected_output = fn(test_input, dims = 3)
            test_output     = fn(x, dims = 3)
        else
            expected_output = fn.(test_input)
            test_output     = fn(x)
        end

        test_output = Array(test_output)
        @test maximum(abs.(test_output - expected_output)) < 10 * eps(Float32)
    end
end
