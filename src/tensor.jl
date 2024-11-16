const options = Dict(
  Int32 => 3,
  Int64 => 4,
  Float32 => 6,
  Float64 => 7,)

let was_enabled = Ref{Int32}()
  at_grad_set_enabled(was_enabled, 0)
end

struct TorchGPUOOMError <: Exception end

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
  ptr::Ptr
  device::Int

  function Tensor{T,N}(ptr::Ptr, dev::Int) where {T,N}
    obj = new(ptr, dev)
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
  sz = length(sz) == 2 ? collect(sz) : reverse(collect(sz))
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

function Base.ndims(t::Tensor)
  i = Int32[-1]
  at_dim(i, t.ptr)
  Int(i[1])
end

function Base.size(t::Tensor)
  dims = ndims(t)
  sz = zeros(Int32, dims)
  at_shape(t.ptr, pointer(sz))
  # s = Int.(tuple(sz...))
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
Base.similar(t::Tensor{T,N}, sz::Int...) where {T,N} = similar(t, T, sz...)
Base.similar(t::Tensor, dims::Tuple) = similar(t, dims...)

function Base.copy(t::Tensor{T,N}) where {T,N}
  sz = size(t)
  z = zeros(T, sz...)
  copyto!(z, t)
end

function Base.copyto!(dest::AbstractArray, src::Tensor)
  at_copy_data(src.ptr, dest, length(dest), sizeof(eltype(dest)))
  dest
end
Base.copyto!(dest::Tensor, src::Tensor) = at_copy_(dest.ptr, src.ptr)

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

  ptr = Ref(Ptr{Cvoid}())
  at_tensor_of_data(ptr, parr.x, sz, nd, el_sz_in_bytes, typ)
  opt = Tensor{Float32, N}(ptr[], dev)
  to(opt, dev = dev)
end
# tensor(x) = x
tensor(x::Fill; kwargs...) = tensor(collect(x); kwargs...)
tensor(x::Tensor; kwargs...) = x

Base.print_array(io::IO, t::Tensor) = Base.print_array(io, collect(t))
Base.show_vector(io::IO, t::Tensor) = Base.show_vector(io, collect(t))

function from_blob(x::AbstractArray{T,N}; dev = -1) where {T,N}
  sz = reverse(collect(size(x)))
  st = reverse(collect(strides(x)))
  ptr = Ref(Ptr{Cvoid}())
  at_from_blob(ptr, pointer(x), sz, length(sz), st, length(st), dev)
  Tensor{T,N}(ptr[], dev)
end

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
