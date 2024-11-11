to_tensor(x::AbstractArray; dev = -1) = tensor(x; dev = dev)
to_tensor(x; dev = -1) = x

function cuda_device_count()
    result = Ref{Int32}()
    Wrapper.atc_cuda_device_count(result)
    return result[]
end

function cuda_is_available()
    result = Ref{Int32}()
    Wrapper.atc_cuda_is_available(result)
    return result[] == 1
end

function cudnn_is_available()
    result = Ref{Int32}()
    Wrapper.atc_cudnn_is_available(result)
    return result[] == 1
end
