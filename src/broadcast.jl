# Do we really need to have broadcasting defined for every operation we do with Tensors?
# They don't really lend themselves well to the AbstractArray interface.

import Base.Broadcast
import Base.Broadcast: broadcasted, BroadcastStyle

# struct TensorStyle <: BroadcastStyle end
# Base.BroadcastStyle(::Type{Tensor}) = TensorStyle()

for op in (:+, :-, :/)
  @eval function broadcasted(::typeof($op), t1::Tensor, t2::Tensor)
    $op(t1, t2)
  end
end

function broadcasted(::typeof(*), t1::Tensor{T,4}, t2::Tensor) where {T}
  ptr = Ref(Ptr{Cvoid}())

  atg_mul(ptr, t1.ptr, t2.ptr)
  Tensor{T,4}(ptr[], on(t1))
end

broadcasted(::typeof(NNlib.relu), t::Tensor) = NNlib.relu(t)
broadcasted(::typeof(identity), t::Tensor) = identity(t)

for op in (:+, :-, :*, :/)
  @eval function broadcasted(::typeof($op), t::Tensor, args...)
    $op(t, args...)
  end
end

broadcasted(::typeof(sqrt), t::Tensor) = sqrt(t)

function broadcasted(::typeof(copy), t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_clone(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end
