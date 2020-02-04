using Statistics

function Statistics.mean(t::Tensor{T,N}; dims = :) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  if dims isa Colon
    atg_mean(ptr, t.ptr, options[T])
  else
    atg_mean1(ptr, t.ptr, dims, length(dims), dims[1], options[T])
  end

  Tensor{T,N}(ptr[], on(t))
end

function Statistics.sum(t::Tensor{T,N}; dims = :) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  if dims isa Colon
    atg_sum(ptr, t.ptr, options[T])
  else
    atg_sum1(ptr, t.ptr, dims, length(dims), dims[1], options[T])
  end

  Tensor{T,N}(ptr[], on(t))
end
