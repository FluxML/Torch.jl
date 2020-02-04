options = Dict(
  Float32 => 6,
  Int64 => 4)

device = Dict(
  :gpu => 0,
  :cpu => -1)

# TODO: Tensor <: AbstractArray
struct Tensor{T, N} <: AbstractArray{T,N}
  ptr::Ptr{Cvoid}
  device::Symbol
end
TensorVector{T} = Tensor{T, 1}
TensorMatrix{T} = Tensor{T, 2}
TensorVecOrMat{T} = Union{TensorVector{T}, TensorMatrix{T}}

function Tensor(::Type{T}, sz::Int...; dev = :cpu) where T
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)
  dtype = options[T]
  sz = reverse(collect(sz))
  sz = length(sz) == 1 ? [1; sz] : sz
  mem = device[dev]
  d = Ref(pointer(sz))
  len = length(sz)

  # atg_rand
  atg_zeros(ppo, d.x, len, dtype, mem)
  Tensor{T, len}(po[1], dev)
end

Tensor(sz::Int...; dev = :cpu) = Tensor(Float32, sz..., dev = dev)
Tensor(sz::Int; dev = :cpu) = Tensor(Float32, Int(sz), dev = dev)

function Base.size(t::Tensor)
  dims = at_dim(t.ptr)
  sz = zeros(Int32, dims)
  at_shape(t.ptr, pointer(sz))
  # s = Int.(tuple(sz...))
  # @show sz
  if t isa TensorMatrix
    Int.(tuple(sz...))
  else
    reverse(Int.(tuple(sz...)))
  end
end

function Base.size(t::Tensor, dim::Int)
  sz = size(t)
  dim < length(sz) ? sz[dim] : 1
end

Base.length(t::Tensor) = prod(size(t))

Base.IndexStyle(::Type{<:Tensor}) = IndexCartesian()
function Base.getindex(t::Tensor{T,N}, I::Vararg{Int,N}) where {T,N}
  # @show reverse!(collect(I)) .- 1, size(t)
  # at_double_value_at_indexes(t.ptr, reverse!(collect(I)) .- 1, N)
  zero(T)
end

function Base.similar(t::Tensor, ::Type{K}, sz::Int...) where {K}
  Tensor(K, sz..., dev = on(t))
end
Base.similar(t::Tensor{T,N}) where {T,N} = Tensor(T,size(t)..., dev = on(t))
Base.similar(t::Tensor{T,N}, sz::Int...) where {T,N} = similar(t, T, sz..., dev = on(t))

function Base.copy(t::Tensor{T}) where T
  z = zeros(T, reverse(size(t))...)
  copyto!(z, t)
end

function Base.copyto!(dest::AbstractArray, src::Tensor)
  at_copy_data(src.ptr, dest, length(dest), sizeof(eltype(dest)))
  dest
end

function Base.reshape(t::Tensor{T,N}, dims::Union{Colon, Int}...) where {T,N}
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)

  # @show size(t)
  # @show dims
  dims = Colon() in dims ? Base._reshape_uncolon(t, dims) : dims
  dims = reverse(collect(dims))
  d = Ref(pointer(dims))
  atg_reshape(ppo, t.ptr, d.x, length(dims))
  Tensor{T,length(dims)}(po[1], on(t))
end

function Base.reshape(t::Tensor{T,N}, dims::Int...) where {T,N}
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)

  dims = reverse(collect(dims))
  d = Ref(pointer(dims))
  atg_reshape(ppo, t.ptr, d.x, length(dims))
  Tensor{T,length(dims)}(po[1], on(t))
end

function Base.zero(t::Tensor{T,N}) where {T, N}
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)

  atg_zeros_like(ppo, t.ptr)
  Tensor{T, N}(po[1], on(t))
end

function Base.rand(::Type{Tensor{T}}, sz::Int...; dev = :cpu) where T <: Real
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)
  dtype = options[T]
  sz = collect(sz)
  mem = device[dev]
  d = Ref(pointer(sz))
  len = length(sz)

  # atg_rand
  atg_rand(ppo, d.x, len, dtype, mem)
  Tensor{T, len}(po[1], dev)
end

Base.zeros(::Type{Tensor{T}}, sz::Int...; dev = :cpu) where T =
  Tensor(T, sz..., dev = dev)

function tensor(x::AbstractArray{T,N}; dev = :cpu) where {T,N}
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)

  sz = if N == 2
    collect(size(x))
  elseif N == 1
    reverse([1;collect(size(x))])
  else
    collect(size(x)) |> reverse
  end
  @info sz
  d = Ref(pointer(sz))
  el_sz_in_bytes = sizeof(eltype(x))
  nd = ndims(x)
  typ = options[T] 
  parr = Ref(pointer(x))

  op = at_tensor_of_data(parr.x, d.x, nd, el_sz_in_bytes, typ)
  opt = Tensor{Float32, N}(op, dev)
  opt = to(opt, dev = dev)
end
# tensor(x) = x

# function tensor(x::AbstractVector{T}; dev = :cpu) where T
#   o = Ref(Ptr{Cvoid}())
#   po = [o.x]
#   ppo = pointer(po)
# 
#   sz = collect(size(x))
#   sz = reverse([sz; 1])
#   d = Ref(pointer(sz))
#   el_sz_in_bytes = sizeof(T)
#   nd = 1
#   typ = options[T]
#   parr = Ref(pointer(x))
# 
#   op = at_tensor_of_data(parr.x, d.x, nd, el_sz_in_bytes, typ)
#   opt = Tensor{Float32, 1}(op, dev)
#   opt = to(opt, dev = dev)
# end

function to(x::Tensor{T,N}; dev = :cpu) where {T,N}
  o = Ref(Ptr{Cvoid}())
  po = [o.x]
  ppo = pointer(po)

  atg_to(ppo, x.ptr, device[dev])
  Tensor{Float32,N}(po[1], dev)
end

on(t::Tensor) = t.device

free!(t::Tensor) = at_free(t.ptr)
