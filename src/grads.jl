import NNlib: ∇conv_data

function cudnn_convolution_backward_bias(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_cudnn_convolution_backward_bias(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

const ∇conv_bias = cudnn_convolution_backward_bias

function ∇conv_data(dy::Tensor{T}, w::Tensor{T},
                    cdims::DenseConvDims{M,K,C_in,C_out,S,P,D,F};
                    groups = 1,
                    benchmark = 0,
                    deterministic = 0) where {M,K,C_in,C_out,S,P,D,F, T}

  ptr = Ref(Ptr{Cvoid}())
  padding          = [P[1];P[3]]
  stride           = collect(S)
  dilation         = collect(D)

  s = reverse([NNlib.input_size(cdims)...,
               NNlib.channels_in(cdims),
               size(dy, ndims(dy))])

  atg_cudnn_convolution_backward_input(ptr,
                                       s, length(s),
                                       dy.ptr, w.ptr,
                                       padding,  length(padding),
                                       stride,   length(stride),
                                       dilation, length(dilation),
                                       groups, benchmark, deterministic)
  Tensor{T,ndims(dy)}(ptr[], on(dy))
end

function ∇conv_filter(w::Tensor{T}, dy::Tensor{T},
                      cdims::DenseConvDims{M,K,C_in,C_out,S,P,D,F};
                      groups = 1,
                      benchmark = 0,
                      deterministic = 0) where {M,K,C_in,C_out,S,P,D,F, T}

  ptr = Ref(Ptr{Cvoid}())
  padding          = [P[1];P[3]]
  stride           = collect(S)
  dilation         = collect(D)

  s = reverse([NNlib.kernel_size(cdims)...,
               NNlib.channels_in(cdims),
               NNlib.channels_out(cdims)])

  atg_cudnn_convolution_backward_weight(ptr,
                                        s, length(s),
                                        dy.ptr, w.ptr,
                                        padding,  length(padding),
                                        stride,   length(stride),
                                        dilation, length(dilation),
                                        groups, benchmark, deterministic)

  Tensor{T,ndims(dy)}(ptr[], on(dy))
end
