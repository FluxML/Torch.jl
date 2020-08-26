import Base: +, -, *, /
using LinearAlgebra

for (op,fn) in zip((:+, :-, :/, :*), (atg_add, atg_sub, atg_div, atg_matmul))
  @eval function $op(t1::Tensor{T,N}, t2::Tensor{T,K}) where {T,N,K}
    ptr = Ref(Ptr{Cvoid}())
    rank = Ref{Cint}(-1)

    $fn(ptr, t1.ptr, t2.ptr)

    # TODO: using `rank` here causes compiler to emit error
    # make shape checking more robust
    # at_dim(rank, ptr[])
    Tensor{T,max(N,K)}(ptr[], on(t1))
  end
end

for op in (:+, :-, :/, :*)
  @eval function $op(t::Tensor{T,N}, r::S) where {T,N,S <: Real}
    i = T[r]
    t2 = tensor(i, dev = on(t))
    res = $op(t, t2)
    res
  end
end

# Basic LinAlg handling

function LinearAlgebra.adjoint!(dest::Tensor, src::TensorMatrix{T}) where {T}
  ptr = Ref(Ptr{Cvoid}())
  p = parent(src)
  atg_t(ptr, p.ptr)
  t = TensorMatrix{T}(ptr[], on(p))
  at_copy_(dest.ptr, t.ptr)
end

function Base.copyto!(dest::Array, src::Union{Adjoint{T, <:TensorMatrix{T}},
                                              Transpose{T, <:TensorMatrix{T}}}) where T
  t = tensor(similar(src), dev = on(parent(src)))
  LinearAlgebra.adjoint!(t, src)
  copyto!(dest, t)
end

*(a::Fill, b::Tensor) = tensor(a,dev = on(b)) * b
*(a::Fill, b::LinearAlgebra.Adjoint{T,<:Tensor{T}}) where T = tensor(a,dev = on(parent(b))) * b
*(a::LinearAlgebra.Adjoint{T,<:Tensor{T}}, b::Fill) where T = a * tensor(b,dev = on(parent(a)))

function *(a::Tensor, b::Union{Adjoint{T, <:TensorMatrix{T}},
                               Transpose{T, <:TensorMatrix{T}}}) where T
  ptr = Ref(Ptr{Cvoid}())
  p = Tensor(size(b)..., dev = on(a))
  LinearAlgebra.adjoint!(p, b.parent)
  a * p
end

function *(a::Union{Adjoint{T, <:TensorMatrix{T}},
                    Transpose{T, <:TensorMatrix{T}}}, b::Tensor) where T
  ptr = Ref(Ptr{Cvoid}())
  p = Tensor(size(a)..., dev = on(b))
  LinearAlgebra.adjoint!(p, a.parent)
  p * b
end

function Base.maximum(t::Tensor{T}; dims = :) where T
  ptr = Ref(Ptr{Cvoid}())
  atg_max(ptr, t.ptr)
  Tensor{T,0}(ptr[], on(t))
end

function Base.sqrt(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_sqrt(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function Base.cat(ts::Tensor{T,N}...; dims = 1) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  ts_arr = [i.ptr for i in ts]
  atg_cat(ptr, ts_arr, length(ts_arr), N - dims)
  Tensor{T,N}(ptr[], on(ts[1]))
end

# TODO: Use a macro to generate wrappers
function conv2d(input::Tensor{T}, filter::Tensor{T,N}, bias::Tensor{T};
                stride = [1],
                padding = [0],
                dilation = [1],
                groups = 1) where {T,N}

  ptr = Ref(Ptr{Cvoid}())

  atg_conv2d(ptr, input.ptr, filter.ptr, bias.ptr,
             reverse(stride), length(stride),
             reverse(padding), length(padding),
             reverse(dilation), length(dilation),
             groups)

  Tensor{T,N}(ptr[], on(input))
end

function conv_transpose_2d(input::Tensor{T}, filter::Tensor{T,N}, bias::Tensor{T};
        stride = [1],
        padding = [0],
        output_padding = [0],
        dilation = [1],
        groups = 1) where {T,N}

  ptr = Ref(Ptr{Cvoid}())

  atg_conv_transpose2d(ptr, input.ptr, filter.ptr, bias.ptr,
						reverse(stride), length(stride),
                       reverse(padding), length(padding),
                       reverse(output_padding), length(output_padding),
                       groups,
                       reverse(dilation), length(dilation))

  Tensor{T,N}(ptr[], on(input))
end

function _depthwise_conv2d(input::Tensor{T}, filter::Tensor{T,N}, bias::Tensor{T};
                           stride = [1],
                           padding = [0],
                           dilation = [1]) where {T,N}

    # When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    # this operation is also termed in literature as depthwise convolution.

    c_in = size(input)[end - 1]  # number of input channels
    c_out = size(filter)[end]    # number of output channels
    @assert mod(c_in, c_out) == 0 "Invalid kernel size for depthwise convolution"

    groups = c_in
    ptr = Ref(Ptr{Cvoid}())

    atg_conv2d(ptr, input.ptr, filter.ptr, bias.ptr,
               reverse(stride), length(stride),
               reverse(padding), length(padding),
               reverse(dilation), length(dilation),
               groups)

    Tensor{T,N}(ptr[], on(input))
end

function _softmax(input::Tensor{T,N}, dims = 1, dtype = options[T]) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_softmax(ptr, input.ptr, N - dims, dtype)
  Tensor{T,N}(ptr[], on(input))
end

function _meanpool(t::Tensor{T,N}, kernel_size; stride = [1] , padding = [0]) where {T,N}
  k = collect(kernel_size)
  s = collect(stride)
  p = collect(padding)
  ptr = Ref(Ptr{Cvoid}())

  atg_avg_pool2d(ptr, t.ptr,
                 reverse(k), length(k),
                 reverse(s), length(s),
                 reverse(p), length(p),
                 0,  # ceil_mode
                 1,  # count_include_pad
                 prod(k)  # divisor_override
  )
  Tensor{T,N}(ptr[], on(t))
end

function _maxpool(t::Tensor{T,N}, kernel_size; stride = [1], padding = [0], dilation = [1]) where {T,N}
  k = collect(kernel_size)
  s = collect(stride)
  p = collect(padding)
  d = collect(dilation)
  ptr = Ref(Ptr{Cvoid}())

  atg_max_pool2d(ptr, t.ptr,
                 reverse(k), length(k),
                 reverse(s), length(s),
                 reverse(p), length(p),
                 reverse(d), length(d),
                 0,  # ceil_mode
  )
  Tensor{T,N}(ptr[], on(t))
end

function _maxpool(t::Tensor{T,M}, pdims::PoolDims{N,K,S,P,D};
                  ceil_mode = 0) where {N,K,S,P,D, T,M}
  k = collect(NNlib.kernel_size(pdims))
  s = collect(S)
  p = [P[1];P[3]]
  d = collect(D)

  ptr = Ref(Ptr{Cvoid}())

  atg_max_pool2d(ptr, t.ptr,
                 reverse(k), length(k),
                 reverse(s), length(s),
                 reverse(p), length(p),
                 reverse(d), length(d),
                 ceil_mode,                # ceil_mode
  )

  Tensor{T,M}(ptr[], on(t))
end

function _maxpool_with_inds(t::Tensor{T,M}, pdims::PoolDims{N,K,S,P,D};
                            ceil_mode = 0) where {N,K,S,P,D, T,M}
  k = collect(NNlib.kernel_size(pdims))
  s = collect(S)
  p = [P[1];P[3]]
  d = collect(D)

  ptr = [Ptr{Cvoid}(), Ptr{Cvoid}()]

  atg_max_pool2d_with_indices(ptr, t.ptr,
                              reverse(k), length(k),
                              reverse(s), length(s),
                              reverse(p), length(p),
                              reverse(d), length(d),
                              ceil_mode,
  )

  Tensor{T,M}(ptr[1], on(t)), Tensor{T,M}(ptr[2], on(t))
end

function _upsample_nearest2d(t::Tensor{T,N}, output_size) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_upsample_nearest2d(ptr, t.ptr,
                         reverse(output_size), length(output_size),
  )
  Tensor{T,N}(ptr[], on(t))
end

function _upsample_bilinear2d(t::Tensor{T,N}, output_size, align_corners = true) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_upsample_bilinear2d(ptr, t.ptr,
                         reverse(output_size), length(output_size),
                         align_corners,
  )
  Tensor{T,N}(ptr[], on(t))
end

function _upsample_bicubic2d(t::Tensor{T,N}, output_size, align_corners = true) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_upsample_bicubic2d(ptr, t.ptr,
                         reverse(output_size), length(output_size),
                         align_corners,
  )
  Tensor{T,N}(ptr[], on(t))
end

function upsample(t::Tensor{T,N}, output_size, mode) where {T,N}
    if mode == :NEAREST
        _upsample_nearest2d(t, output_size)
    elseif mode == :LINEAR
        _upsample_bilinear2d(t, output_size)
    elseif mode == :CUBIC
        _upsample_bicubic2d(t, output_size)
    else
       error("Unsupported mode $(mode).")
    end
end

function pad(t::Tensor{T,N}, padding) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  p = collect(padding)

  atg_constant_pad_nd(ptr, t.ptr,
                      p, length(p),
  )
  Tensor{T,N}(ptr[], on(t))
end

function _chunk(t::Tensor{T,N}, chunks=2, dims=1) where {T,N}
  ts = [Ptr{Cvoid}() for _ in 1:chunks]
  atg_chunk(ts, t.ptr, chunks, N - dims)
  [Tensor{T,N}(ts[i], on(t)) for i in 1:chunks]
end
