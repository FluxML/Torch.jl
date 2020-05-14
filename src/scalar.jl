mutable struct Scalar{T}
  ptr::Ptr{Cvoid}
  device::Int
end

function Scalar(r::T; dev = -1) where T <: Real
  ptr = Ref(Ptr{Cvoid}())

  ats_float(ptr, float.(r))
  Scalar{T}(ptr[], dev)
end

free!(s::Scalar) = ats_free(s.ptr)
