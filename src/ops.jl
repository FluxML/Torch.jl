import Base: +, -, *, /

for (op,fn) in zip((:+, :-, :/, :*), (atg_add, atg_sub, atg_div, atg_matmul))
  @eval function $op(t1::Tensor{T,N}, t2::Tensor{T,K}) where {T,N,K}
    o = Ref(Ptr{Cvoid}())
    po = [o.x]
    ppo = pointer(po)

    # @show "ew"  
    # @show size(t1), size(t2)
    # global gt1 = t1
    # global gt2 = t2
    $fn(ppo, t1.ptr, t2.ptr)
    Tensor{T,N}(po[1], on(t1))
  end
end

# TODO: Use a macro to generate wrappers
function conv2d(input::Tensor{T}, filter::Tensor{T,N}, bias::Tensor{T};
		stride = [1],
		padding = [1],
		dilation = [1],
		groups = 1) where {T,N}

  po = [Ptr{Cvoid}()]
  ppo = pointer(po)

  atg_conv2d(ppo, input.ptr, filter.ptr, bias.ptr,
                stride, length(stride),
                padding, length(padding),
                dilation, length(dilation),
                groups)

  Tensor{T,N}(po[1], on(input))
end

function softmax(input::Tensor{T,N}, dims = 1, dtype = options[T]) where {T,N}
  po = [Ptr{Cvoid}()]
  ppo = pointer(po)

  atg_softmax(ppo, input.ptr, dims - 1, dtype)
  Tensor{T,N}(po[1], on(input))
end
