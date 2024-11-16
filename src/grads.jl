import NNlib: ∇conv_data, ∇conv_filter

@adjoint function tensor(x; kwargs...)
  tensor(x; kwargs...), Δ -> (collect(Δ),)
end

function cudnn_convolution_backward_bias(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_cudnn_convolution_backward_bias(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

const ∇conv_bias = cudnn_convolution_backward_bias

function ∇conv_data(dy::AbstractArray, w::Tensor{T},
                    cdims::DenseConvDims{M,K,S,P,D};
                    groups = 1,
                    benchmark = 0,
                    deterministic = 0,
                    allow_tf32 = 0) where {M,K,S,P,D,T}

  ptr = Ref(Ptr{Cvoid}())
  dy_ = tensor(dy, dev = on(w))
  padding          = NNlib.padding(cdims)
  padding          = [padding[1];padding[3]]
  stride           = collect(NNlib.stride(cdims))
  dilation         = collect(NNlib.dilation(cdims))

  s = reverse([NNlib.input_size(cdims)...,
               NNlib.channels_in(cdims),
               size(dy_, ndims(dy_))])

  atg_cudnn_convolution_backward_input(ptr,
                                       s, length(s),
                                       dy_.ptr, w.ptr,
                                       padding,  length(padding),
                                       stride,   length(stride),
                                       dilation, length(dilation),
                                       groups, benchmark, deterministic, allow_tf32)
  Tensor{T,ndims(dy_)}(ptr[], on(dy_))
end

function ∇conv_filter(w::Tensor{T}, dy::AbstractArray{T},
                      cdims::DenseConvDims{M,K,S,P,D};
                      groups = 1,
                      benchmark = 0,
                      deterministic = 0,
                      allow_tf32 = 0) where {M,K,S,P,D,T}

  dy_ = tensor(dy, dev = on(w))
  ptr = Ref(Ptr{Cvoid}())
  padding          = NNlib.padding(cdims)
  padding          = [padding[1];padding[3]]
  stride           = collect(NNlib.stride(cdims))
  dilation         = collect(NNlib.dilation(cdims))

  s = reverse([NNlib.kernel_size(cdims)...,
               NNlib.channels_in(cdims),
               NNlib.channels_out(cdims)])

  atg_cudnn_convolution_backward_weight(ptr,
                                        s, length(s),
                                        dy_.ptr, w.ptr,
                                        padding,  length(padding),
                                        stride,   length(stride),
                                        dilation, length(dilation),
                                        groups, benchmark, deterministic, allow_tf32)

  Tensor{T,ndims(dy_)}(ptr[], on(dy_))
end

function NNlib.∇maxpool(dy::AbstractArray{T}, y::Tensor{T,M}, x::Tensor{T,M},
                        pdims::PoolDims{N,K,S,P,D};
                        ceil_mode = 0,
                        indices=nothing) where {N,K,S,P,D, T,M}

  dy_ = tensor(dy, dev = on(y))
  ptr = Ref(Ptr{Cvoid}())
  kernel = collect(NNlib.kernel_size(pdims))
  stride = collect(NNlib.stride(pdims))
  padding = NNlib.padding(pdims)
  padding = Int[padding[1];padding[3]]
  dilation = collect(NNlib.dilation(pdims))
  atg_max_pool2d_with_indices_backward(ptr, dy_.ptr, x.ptr,
                          kernel, length(kernel),
                          stride, length(stride),
                          padding, length(padding),
                          dilation, length(dilation),
                          ceil_mode,
                          indices.ptr
  )

  mp = Tensor{T,N}(ptr[], on(x))
  reshape(mp, reverse(size(mp))...)
end

@adjoint function NNlib.maxpool(t::Tensor, pdims::PoolDims; ceil_mode = 0)
  y, inds = _maxpool_with_inds(t, pdims, ceil_mode = ceil_mode)
  y, Δ -> begin
    (∇maxpool(Δ, y, t, pdims, ceil_mode = ceil_mode, indices = inds), nothing)
  end
end

function NNlib.∇meanpool(dy::AbstractArray, y::Tensor{T,M}, x::Tensor{T,M},
                         pdims::PoolDims{N,K,S,P,D};
                         ceil_mode = 0,
                         count_include_pad = 1,
                         divisor_override = 1) where {N,K,S,P,D, T,M}

  ptr = Ref(Ptr{Cvoid}())
  dy_ = dy isa Base.ReshapedArray ? reshape(parent(dy), dy.dims...) : tensor(dy, dev = on(y))
  kernel = collect(NNlib.kernel_size(pdims))
  stride = collect(NNlib.stride(pdims))
  padding = NNlib.padding(pdims)
  padding = [padding[1];padding[3]]

  atg_avg_pool2d_backward(ptr,
                          dy_.ptr, x.ptr,
                          kernel, length(kernel),
                          stride, length(stride),
                          padding, length(padding),
                          ceil_mode,
                          count_include_pad,
                          divisor_override)

  Tensor{T,M}(ptr[], on(x))
end

function ∇sigmoid(dy::AbstractArray, t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  dy_ = tensor(dy, dev = on(t))
  atg_sigmoid_backward(ptr, dy_.ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

@adjoint function NNlib.sigmoid(t::Tensor)
  x = sigmoid(t)
  x, Δ -> (∇sigmoid(Δ, x),)
end

function ∇leaky_relu(dy::AbstractArray, x::Tensor{T,N}, slope) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  dy_ = tensor(dy, dev = on(x))
  self_is_result = 0
  atg_leaky_relu_backward(ptr, dy_.ptr, x.ptr, Scalar(slope).ptr, self_is_result)
  Tensor{T,N}(ptr[], on(x))
end

@adjoint function NNlib.relu(t::Tensor{T,N}) where {T,N}
  NNlib.relu(t), Δ -> (∇leaky_relu(Δ, t, zero(T)),)
end
