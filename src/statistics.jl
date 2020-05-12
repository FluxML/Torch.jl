using Statistics

function Statistics.mean(t::Tensor{T,N}; dims = :) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  if dims isa Colon
    atg_mean(ptr, t.ptr, options[T])
    Tensor{T,0}(ptr[], on(t))
  else
    atg_mean1(ptr, t.ptr, dims, length(dims), dims[1], options[T])
    Tensor{T,N-length(dims)}(ptr[], on(t))
  end
end

function Statistics.sum(t::Tensor{T,N}; dims = :) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  if dims isa Colon
    atg_sum(ptr, t.ptr, options[T])
    Tensor{T,0}(ptr[], on(t))
  else
    atg_sum1(ptr, t.ptr, dims, length(dims), dims[1], options[T])
    Tensor{T,N-length(dims)}(ptr[], on(t))
  end
end
