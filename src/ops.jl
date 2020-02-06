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
    $op(t, t2)
  end
end

function Base.sqrt(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_sqrt(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

# TODO: Use a macro to generate wrappers
function conv2d(input::Tensor{T}, filter::Tensor{T,N}, bias::Tensor{T};
		stride = [1],
		padding = [1],
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
                 1   # divisor_override
  )

  Tensor{T,N}(ptr[], on(t))
end

function _maxpool(t::Tensor{T,N}, k, s, p, d, op_sz) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_max_pool2d(ptr, t.ptr,
                 k, length(k),
                 s, length(s),
                 p, length(p),
                 d, length(d),
                 0,                # ceil_mode
  )

  Tensor{T,N}(ptr[], on(t))
end

function Base.cos(input::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_cos(ppo, input.ptr)
  Tensor{T,N}(ptr[], on(input))
end
