# Do we really need to have broadcasting defined for every operation we do with Tensors?
# They don't really lend themselves well to the AbstractArray interface.

import Base.Broadcast
import Base.Broadcast: broadcasted

for op in (:+, :-, :*, :/)
  @eval function broadcasted(::typeof($op), t1::Tensor{T}, t2::Tensor{T}) where T
    $op(t1, t2)
  end
end

broadcasted(::typeof(NNlib.relu), t::Tensor) = NNlib.relu(t)
broadcasted(::typeof(identity), t::Tensor) = identity(t)

for op in (:+, :-, :*, :/)
  @eval function broadcasted(::typeof($op), t::Tensor, args...)
    $op(t, args...)
  end
end
