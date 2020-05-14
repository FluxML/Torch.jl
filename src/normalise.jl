function batchnorm(input::Tensor{T,N}, weight, bias,
                   running_mean, running_var,
                   training, momentum,
                   ep, cudnn_enabled) where {T,N}

  ptr = Ref(Ptr{Cvoid}())
  atg_batch_norm(ptr, input.ptr,
                 weight.ptr, bias.ptr,
                 running_mean.ptr, running_var.ptr,
                 training, momentum,
                 ep, cudnn_enabled)

  Tensor{T,N}(ptr[], on(input))
end

function ∇batchnorm(dy::AbstractArray, input::Tensor{T,N},
                    running_mean, invstd, weight::Tensor{T,S},
                    input_g = 1, weight_g = 1, bias_g = 1) where {T,N,S}

  ptr = [Ptr{Cvoid}() for i = 1:4]
  dy_ = tensor(dy, dev = on(input))

  atg_batch_norm_backward_reduce(ptr, dy_.ptr, input.ptr,
                                 running_mean.ptr, invstd.ptr,
                                 weight.ptr,
                                 input_g, weight_g, bias_g)

  rank = Ref{Cint}(-1)
  [Tensor{T,S}(i, on(input)) for i in ptr[1:4]]
end


function ∇batchnorm_element(dy::AbstractArray, input::Tensor{T,N},
                    running_mean, invstd, weight::Tensor{T,S},
                    mean_dy = nothing, mean_dy_xmu = nothing) where {T,N,S}

  ptr = Ref(Ptr{Cvoid}())
  dy_ = tensor(dy, dev = on(input))

  atg_batch_norm_backward_elemt(ptr, dy_.ptr, input.ptr,
                                 running_mean.ptr, invstd.ptr,
                                 weight.ptr,
                                 weight.ptr, weight.ptr)

  Tensor{T,N}(ptr[], on(input))
end 
