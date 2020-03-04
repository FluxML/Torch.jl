const options = Dict(
  Float32 => 6,
  Int64 => 4,
  Float64 => 7,)

at_grad_set_enabled(0)

function no_grad(f; flag = 0)
  at_no_grad(flag)
  f()
end

async_free!(x) = let x = x, ptr = x.ptr, oid = objectid(x)
  @async begin
    free!(x)
  end
  return
end

mutable struct Tensor{T, N} <: AbstractArray{T,N}
  ptr::Union{Ptr,CuPtr}
  device::Int

  function Tensor{T,N}(ptr::Union{Ptr,CuPtr}, dev::Int) where {T,N}
    cuptr = Base.convert(CuPtr{Cvoid}, Base.bitcast(UInt, ptr))
    obj = new(cuptr, dev)
    finalizer(async_free!, obj)
    # TURN_ON_LOGGING == true && (logdict[ptr] = (size(obj), stacktrace()))
    obj
  end
end

TensorVector{T} = Tensor{T, 1}
TensorMatrix{T} = Tensor{T, 2}
TensorVecOrMat{T} = Union{TensorVector{T}, TensorMatrix{T}}

function Tensor(::Type{T}, sz::Int...; dev = -1) where T
  ptr = Ref(Ptr{Cvoid}())
  dtype = options[T]
  sz = reverse(collect(sz))
  # sz = length(sz) == 1 ? [sz;1] : sz
  mem = dev
  d = Ref(pointer(sz))
  len = length(sz)

  # atg_rand
  atg_zeros(ptr, d.x, len, dtype, mem)
  Tensor{T, len}(ptr[], dev)
end

Tensor(sz::Int...; dev = -1) = Tensor(Float32, sz..., dev = dev)
Tensor(sz::Int; dev = -1) = Tensor(Float32, Int(sz), dev = dev)

# function Tensor{T,N}(ptr::Ptr) where {T,N}
#   Tensor{T,N}(ptr, on(ptr))
# end

function Base.size(t::Tensor)
  dims = at_dim(t.ptr)
  sz = zeros(Int32, dims)
  at_shape(t.ptr, pointer(sz))
  if t isa TensorMatrix
    Int.(tuple(sz...))
  else
    reverse(Int.(tuple(sz...)))
  end
end

function Base.size(t::Tensor, dim::Int)
  sz = size(t)
  dim <= length(sz) ? sz[dim] : 1
end

Base.length(t::Tensor) = prod(size(t))

Base.IndexStyle(::Type{<:Tensor}) = IndexCartesian()
# function Base.getindex(t::Tensor{T,N}, I::Vararg{Int,N}) where {T,N}
#   # @show reverse!(collect(I)) .- 1, size(t)
#   # at_double_value_at_indexes(t.ptr, reverse!(collect(I)) .- 1, N)
#   zero(T)
# end

function Base.similar(t::Tensor, ::Type{K}, sz::Int...) where {K}
  Tensor(K, sz..., dev = on(t))
end
Base.similar(t::Tensor{T,N}) where {T,N} = Tensor(T,size(t)..., dev = on(t))
Base.similar(t::Tensor{T,N}, sz::Int...) where {T,N} = similar(t, T, sz..., dev = on(t))

function Base.copy(t::Tensor{T,N}) where {T,N}
  sz = size(t)
  z = zeros(T, sz...)
  copyto!(z, t)
end

function Base.copyto!(dest::AbstractArray, src::Tensor)
  at_copy_data(src.ptr, dest, length(dest), sizeof(eltype(dest)))
  dest
end

function Base.reshape(t::Tensor{T,N}, dims::Union{Colon, Int}...) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  dims = Colon() in dims ? Base._reshape_uncolon(t, dims) : dims
  dims = length(dims) == 2 ? collect(dims) : reverse(collect(dims))
  atg_reshape(ptr, t.ptr, dims, length(dims))
  Tensor{T,length(dims)}(ptr[], on(t))
end

function Base.zero(t::Tensor{T,N}) where {T, N}
  ptr = Ref(Ptr{Cvoid}())

  atg_zeros_like(ptr, t.ptr)
  Tensor{T, N}(ptr[], on(t))
end

function Base.rand(::Type{Tensor{T}}, sz::Int...; dev = -1) where T <: Real

  ptr = Ref(Ptr{Cvoid}())
  dtype = options[T]
  sz = collect(sz)
  mem = dev
  len = length(sz)

  # atg_rand
  atg_rand(ptr, sz, len, dtype, mem)
  Tensor{T, len}(ptr[], dev)
end

Base.zeros(::Type{Tensor{T}}, sz::Int...; dev = -1) where T =
  Tensor(T, sz..., dev = dev)

function tensor(x::AbstractArray{T,N}; dev = -1) where {T,N}

  sz = if N == 2
    collect(size(x))
  elseif N == 1
    [collect(size(x));1]
  else
    collect(size(x)) |> reverse
  end
  # d = Ref(pointer(sz))
  el_sz_in_bytes = sizeof(eltype(x))
  nd = ndims(x)
  typ = options[T] 
  parr = Ref(pointer(x))

  op = at_tensor_of_data(parr.x, sz, nd, el_sz_in_bytes, typ)
  opt = Tensor{Float32, N}(op, dev)
  to(opt, dev = dev)
end
# tensor(x) = x

function to(x::Tensor{T,N}; dev = -1) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  atg_to(ptr, x.ptr, dev)
  Tensor{Float32,N}(ptr[], dev)
end

on(t::Tensor) = t.device

function free!(t::Tensor)
  # TURN_ON_LOGGING && delete!(logdict, t.ptr)
  at_free(t.ptr)
end
free!(ptr::Ptr) = at_free(ptr)
