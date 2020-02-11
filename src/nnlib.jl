using NNlib
using NNlib: expand
using NNlib: PoolDims

import NNlib: conv

function NNlib.conv(x::Tensor{xT, N}, w::Tensor{T,N}, b::Tensor{T}, cdims::DenseConvDims{M,K,C_in,C_out,S,P,D,F}; stride = 1, pad = 0, dilation = 1) where {T,N, xT,  M,K,C_in,C_out,S,P,D,F}
  op = conv2d(x, w, b, stride = collect(S), padding = [P[1];P[3]], dilation = collect(dilation))
  op
end

function NNlib.conv(x::Tensor, w::Tensor, cdims::DenseConvDims; stride = 1, pad = 0, dilation = 1)
  b = zeros(Tensor{Float32}, size(w)[end], dev = :gpu)
  op = conv(x, w, b, cdims, stride = stride, pad = pad, dilation = dilation)
  op
end

function NNlib.relu(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_relu(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.softmax(t::Tensor{T,N}; dims = 1) where {T,N}
  _softmax(t, dims, options[T])
end

function NNlib.meanpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  ks = collect(NNlib.kernel_size(pdims))
  stride = collect(S)
  pad = [P[1];P[3]]
  op_sz = NNlib.output_size(pdims)

  _meanpool(t, ks, stride, pad, op_sz)
end

function NNlib.maxpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  ks = collect(NNlib.kernel_size(pdims))
  stride = collect(S)
  pad = [P[1];P[3]]
  dilation = collect(D)
  op_sz = NNlib.output_size(pdims)

  _maxpool(t, ks, stride, pad, dilation, op_sz)
end
