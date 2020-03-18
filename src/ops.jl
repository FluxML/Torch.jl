import Base: +, -, *, /

for (op,fn) in zip((:+, :-, :/, :*), (atg_add, atg_sub, atg_div, atg_matmul))
  @eval function $op(t1::Tensor{T,N}, t2::Tensor{T,K}) where {T,N,K}
    ptr = Ref(Ptr{Cvoid}())

    $fn(ptr, t1.ptr, t2.ptr)
    Tensor{T,N}(ptr[], on(t1))
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

function Base.sqrt(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_sqrt(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function Base.cat(ts::Tensor{T,N}...; dims = 1) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  ts_arr = [i.ptr for i in ts]
  atg_cat(ptr, ts_arr, length(ts_arr), dims - 1)
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
                stride, length(stride),
                padding, length(padding),
                dilation, length(dilation),
                groups)

  Tensor{T,N}(ptr[], on(input))
end

function _softmax(input::Tensor{T,N}, dims = 1, dtype = options[T]) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_softmax(ptr, input.ptr, dims - 1, dtype)
  Tensor{T,N}(ptr[], on(input))
end

function _meanpool(t::Tensor{T,N}, k, s, p, op_sz) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_avg_pool2d(ptr, t.ptr,
                 k, length(k),
                 s, length(s),
                 p, length(p),
                 0,                # ceil_mode
                 1,                # count_include_pad
                 1                 # divisor_override
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
                 k, length(k),
                 s, length(s),
                 p, length(p),
                 d, length(d),
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
                 k, length(k),
                 s, length(s),
                 p, length(p),
                 d, length(d),
                 ceil_mode,                # ceil_mode
  )

  Tensor{T,M}(ptr[1], on(t)), Tensor{T,M}(ptr[2], on(t))
end
