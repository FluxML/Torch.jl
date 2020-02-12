mutable struct Scalar{T}
  ptr::Ptr{Cvoid}
  device::Symbol

  function Scalar{T}(r::T, dev = :cpu) where T <: Real
    ptr = Ref(Ptr{Cvoid}())
    atg_scalar_tensor(ptr, [r], options[T], device[dev])
    obj = new(ptr[], dev)
    finalizer(async_free!, obj)
    obj
  end
end

Scalar(r::T; dev = :cpu) where T = Scalar{T}(r, dev)

free!(s::Scalar) = ats_free(s.ptr)
