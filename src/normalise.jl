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
