using NNlib
using NNlib: expand

import NNlib: conv

function NNlib.conv(x::Tensor{xT, N}, w::Tensor{T,N}, b::Tensor{T}, cdims::DenseConvDims; stride = 1, pad = 1, dilation = 1) where {T,N, xT}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(N-2), pad)
  dilation = expand(Val(N-2), dilation)
  op = conv2d(x, w, b, stride = collect(stride), padding = collect(pad), dilation = collect(dilation))
  # free!(tx)
  # free!(b)
  op
end

function NNlib.conv(x::Tensor, w::Tensor, cdims::DenseConvDims; stride = 1, pad = 1, dilation = 1)
  b = zeros(Tensor{Float32}, size(w)[end], dev = :gpu)
  op = conv(x, w, b, cdims, stride = stride, pad = pad, dilation = dilation)
  free!(b)
  op
end

function NNlib.conv(x::Array{xT, N}, w::Tensor{T,N}, cdims::DenseConvDims; stride = 1, pad = 1, dilation = 1) where {T,N, xT}
  tx = tensor(x, dev = :gpu)
  b = zeros(Tensor{Float32}, size(w)[end], dev = :gpu)
  op = NNlib.conv(tx, w, b, cdims, stride = stride, pad = pad, dilation = dilation)
  free!(tx)
  free!(b)
  op
end

function NNlib.relu(t::Tensor{T,N}) where {T,N}
  po = [Ptr{Cvoid}()]
  ppo = pointer(po)

  atg_relu(ppo, t.ptr)
  Tensor{T,N}(po[1], on(t))
end

function NNlib.softmax(t::Tensor{T,N}; dims = 1) where {T,N}
  softmax(t, dims, options[T])
end
