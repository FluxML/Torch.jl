# Do we really need to have broadcasting defined for every operation we do with Tensors?
# They don't really lend themselves well to the AbstractArray interface.

import Base.Broadcast
import Base.Broadcast: broadcasted

for op in (:+, :-, :*, :/)
  @eval function broadcasted(::typeof($op), t1::Tensor{T}, t2::Tensor{T}) where T
    $op(t1, t2)
  end
end

function broadcasted(::typeof(*), t1::Tensor{T,N}, t2::Tensor) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_mul(ptr, t1.ptr, t2.ptr)
  Tensor{T,N}(ptr[], on(t1))
end

broadcasted(::typeof(NNlib.relu), t::Tensor) = NNlib.relu(t)
broadcasted(::typeof(identity), t::Tensor) = identity(t)

for op in (:+, :-, :*, :/)
  @eval function broadcasted(::typeof($op), t::Tensor, args...)
    $op(t, args...)
  end
end
