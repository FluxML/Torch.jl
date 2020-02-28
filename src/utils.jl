to_tensor(x::AbstractArray) = tensor(x, dev = 0)
to_tensor(x) = x
