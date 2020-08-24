using NNlib
using NNlib: expand
using NNlib: PoolDims

import NNlib: conv, depthwiseconv

function NNlib.conv(x::Tensor{xT, N}, w::Tensor, b::Tensor{T},
                    cdims::DenseConvDims{M,K,C_in,C_out,S,P,D,F}) where {T,N,xT,M,K,C_in,C_out,S,P,D,F}
  op = conv2d(x, w, b, stride = collect(S), padding = [P[1];P[3]], dilation = collect(D))
  op
end

function NNlib.conv(x::Tensor, w::Tensor, cdims::DenseConvDims)
  b = zeros(Tensor{Float32}, size(w)[end], dev = on(w))
  op = conv(x, w, b, cdims)
  op
end

function NNlib.depthwiseconv(x::Tensor{xT, N}, w::Tensor, b::Tensor{T};
                             stride = 1, pad = 0, dilation = 1) where {T, N, xT}
  op = _depthwise_conv2d(x, w, b, stride = collect(stride), padding = collect(pad),
                         dilation = collect(dilation))
  op
end

function NNlib.depthwiseconv(x::Tensor, w::Tensor; stride = 1, pad = 0, dilation = 1)
  b = zeros(Tensor{Float32}, size(w)[end], dev = on(w))
  op = depthwiseconv(x, w, b, stride = collect(stride), pad = collect(pad),
                     dilation = collect(dilation))
  op
end

function NNlib.relu(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_relu(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.leakyrelu(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_leaky_relu(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.sigmoid(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_sigmoid(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.tanh(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_tanh(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.softmax(t::Tensor{T,N}; dims = 1) where {T,N}
  _softmax(t, dims, options[T])
end

function NNlib.∇softmax(Δ, xs::Tensor; dims = 1)
  t = tensor(Δ, dev = on(xs))
  sf = softmax(xs, dims=dims)
  sf .* (t .- sum(t .* sf, dims = dims))
end

function NNlib.meanpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  ks = collect(NNlib.kernel_size(pdims))
  stride = collect(S)
  padding = [P[1];P[3]]
  # op_sz = NNlib.output_size(pdims)

  _meanpool(t, ks, stride=stride, padding=padding)
end

function NNlib.maxpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  ks = collect(NNlib.kernel_size(pdims))
  stride = collect(S)
  padding = [P[1];P[3]]
  dilation = collect(D)

  _maxpool(t, ks, stride=stride, padding=padding, dilation=dilation)
end
