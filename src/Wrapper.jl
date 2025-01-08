module Wrapper

using TorchCAPI_jll
export TorchCAPI_jll

using CEnum

function get_error()
  err = cglobal((:myerr, libtorch_c_api), Cstring) |> unsafe_load
  unsafe_string(err)
end

macro runtime_error_check(ex)
  quote
    x = $ex
    if x == 1
      cs = get_error()
      flush_error()
      throw(cs)
    end
  end |> esc
end


const tensor = Ptr{Cvoid}

const optimizer = Ptr{Cvoid}

const scalar = Ptr{Cvoid}

const _module = Ptr{Cvoid}

const ivalue = Ptr{Cvoid}

function get_last_error(arg1)
    @runtime_error_check ccall((:get_last_error, libtorch_c_api), Cint, (Ptr{Cchar},), arg1)
end

# no prototype is found for this function at torch_api.h:30:5, please use with caution
function flush_error()
    @runtime_error_check ccall((:flush_error, libtorch_c_api), Cint, ())
end

function at_manual_seed(arg1)
    @runtime_error_check ccall((:at_manual_seed, libtorch_c_api), Cint, (Int64,), arg1)
end

function at_new_tensor(arg1)
    @runtime_error_check ccall((:at_new_tensor, libtorch_c_api), Cint, (Ptr{tensor},), arg1)
end

# no prototype is found for this function at torch_api.h:34:5, please use with caution
function at_empty_cache()
    @runtime_error_check ccall((:at_empty_cache, libtorch_c_api), Cint, ())
end

function at_no_grad(flag)
    @runtime_error_check ccall((:at_no_grad, libtorch_c_api), Cint, (Cint,), flag)
end

# no prototype is found for this function at torch_api.h:36:5, please use with caution
function at_sync()
    @runtime_error_check ccall((:at_sync, libtorch_c_api), Cint, ())
end

function at_from_blob(arg1, data, dims, ndims, strides, nstrides, dev)
    @runtime_error_check ccall((:at_from_blob, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cvoid}, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, data, dims, ndims, strides, nstrides, dev)
end

function at_tensor_of_data(arg1, vs, dims, ndims, element_size_in_bytes, type)
    @runtime_error_check ccall((:at_tensor_of_data, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cvoid}, Ptr{Int64}, Cint, Cint, Cint), arg1, vs, dims, ndims, element_size_in_bytes, type)
end

function at_copy_data(tensor_, vs, numel, element_size_in_bytes)
    @runtime_error_check ccall((:at_copy_data, libtorch_c_api), Cint, (tensor, Ptr{Cvoid}, Int64, Cint), tensor_, vs, numel, element_size_in_bytes)
end

function at_float_vec(arg1, values, value_len, type)
    @runtime_error_check ccall((:at_float_vec, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cdouble}, Cint, Cint), arg1, values, value_len, type)
end

function at_int_vec(arg1, values, value_len, type)
    @runtime_error_check ccall((:at_int_vec, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint), arg1, values, value_len, type)
end

function at_defined(arg1, arg2)
    @runtime_error_check ccall((:at_defined, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

function at_is_sparse(arg1, arg2)
    @runtime_error_check ccall((:at_is_sparse, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

function at_device(arg1, arg2)
    @runtime_error_check ccall((:at_device, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

function at_dim(arg1, arg2)
    @runtime_error_check ccall((:at_dim, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

function at_shape(arg1, arg2)
    @runtime_error_check ccall((:at_shape, libtorch_c_api), Cint, (tensor, Ptr{Cint}), arg1, arg2)
end

function at_stride(arg1, arg2)
    @runtime_error_check ccall((:at_stride, libtorch_c_api), Cint, (tensor, Ptr{Cint}), arg1, arg2)
end

function at_scalar_type(arg1, arg2)
    @runtime_error_check ccall((:at_scalar_type, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

# no prototype is found for this function at torch_api.h:51:5, please use with caution
function at_autocast_clear_cache()
    @runtime_error_check ccall((:at_autocast_clear_cache, libtorch_c_api), Cint, ())
end

function at_autocast_decrement_nesting(arg1)
    @runtime_error_check ccall((:at_autocast_decrement_nesting, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function at_autocast_increment_nesting(arg1)
    @runtime_error_check ccall((:at_autocast_increment_nesting, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function at_autocast_is_enabled(arg1)
    @runtime_error_check ccall((:at_autocast_is_enabled, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function at_autocast_set_enabled(arg1, b)
    @runtime_error_check ccall((:at_autocast_set_enabled, libtorch_c_api), Cint, (Ptr{Cint}, Cint), arg1, b)
end

function at_backward(arg1, arg2, arg3)
    @runtime_error_check ccall((:at_backward, libtorch_c_api), Cint, (tensor, Cint, Cint), arg1, arg2, arg3)
end

function at_requires_grad(arg1, arg2)
    @runtime_error_check ccall((:at_requires_grad, libtorch_c_api), Cint, (Ptr{Cint}, tensor), arg1, arg2)
end

function at_grad_set_enabled(arg1, arg2)
    @runtime_error_check ccall((:at_grad_set_enabled, libtorch_c_api), Cint, (Ptr{Cint}, Cint), arg1, arg2)
end

function at_get(arg1, arg2, index)
    @runtime_error_check ccall((:at_get, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, arg2, index)
end

function at_fill_double(arg1, arg2)
    @runtime_error_check ccall((:at_fill_double, libtorch_c_api), Cint, (tensor, Cdouble), arg1, arg2)
end

function at_fill_int64(arg1, arg2)
    @runtime_error_check ccall((:at_fill_int64, libtorch_c_api), Cint, (tensor, Int64), arg1, arg2)
end

function at_double_value_at_indexes(arg1, arg2, indexes, indexes_len)
    @runtime_error_check ccall((:at_double_value_at_indexes, libtorch_c_api), Cint, (Ptr{Cdouble}, tensor, Ptr{Cint}, Cint), arg1, arg2, indexes, indexes_len)
end

function at_int64_value_at_indexes(arg1, arg2, indexes, indexes_len)
    @runtime_error_check ccall((:at_int64_value_at_indexes, libtorch_c_api), Cint, (Ptr{Int64}, tensor, Ptr{Cint}, Cint), arg1, arg2, indexes, indexes_len)
end

function at_set_double_value_at_indexes(arg1, indexes, indexes_len, v)
    @runtime_error_check ccall((:at_set_double_value_at_indexes, libtorch_c_api), Cint, (tensor, Ptr{Cint}, Cint, Cdouble), arg1, indexes, indexes_len, v)
end

function at_set_int64_value_at_indexes(arg1, indexes, indexes_len, v)
    @runtime_error_check ccall((:at_set_int64_value_at_indexes, libtorch_c_api), Cint, (tensor, Ptr{Cint}, Cint, Int64), arg1, indexes, indexes_len, v)
end

function at_copy_(dst, src)
    @runtime_error_check ccall((:at_copy_, libtorch_c_api), Cint, (tensor, tensor), dst, src)
end

function at_print(arg1)
    @runtime_error_check ccall((:at_print, libtorch_c_api), Cint, (tensor,), arg1)
end

function at_to_string(arg1, arg2, line_size)
    @runtime_error_check ccall((:at_to_string, libtorch_c_api), Cint, (Ptr{Ptr{Cchar}}, tensor, Cint), arg1, arg2, line_size)
end

function at_save(arg1, filename)
    @runtime_error_check ccall((:at_save, libtorch_c_api), Cint, (tensor, Ptr{Cchar}), arg1, filename)
end

function at_load(arg1, filename)
    @runtime_error_check ccall((:at_load, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cchar}), arg1, filename)
end

function at_get_num_threads(arg1)
    @runtime_error_check ccall((:at_get_num_threads, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function at_set_num_threads(n_threads)
    @runtime_error_check ccall((:at_set_num_threads, libtorch_c_api), Cint, (Cint,), n_threads)
end

function at_save_multi(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_save_multi, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_multi(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_load_multi, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_multi_(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_load_multi_, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_callback(filename, f)
    @runtime_error_check ccall((:at_load_callback, libtorch_c_api), Cint, (Ptr{Cchar}, Ptr{Cvoid}), filename, f)
end

function at_free(arg1)
    @runtime_error_check ccall((:at_free, libtorch_c_api), Cint, (tensor,), arg1)
end

function at_run_backward(tensors, ntensors, inputs, ninputs, outputs, keep_graph, create_graph)
    @runtime_error_check ccall((:at_run_backward, libtorch_c_api), Cint, (Ptr{tensor}, Cint, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint), tensors, ntensors, inputs, ninputs, outputs, keep_graph, create_graph)
end

function ato_adam(arg1, learning_rate, beta1, beta2, weight_decay, eps)
    @runtime_error_check ccall((:ato_adam, libtorch_c_api), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble), arg1, learning_rate, beta1, beta2, weight_decay, eps)
end

function ato_rmsprop(arg1, learning_rate, alpha, eps, weight_decay, momentum, centered)
    @runtime_error_check ccall((:ato_rmsprop, libtorch_c_api), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint), arg1, learning_rate, alpha, eps, weight_decay, momentum, centered)
end

function ato_sgd(arg1, learning_rate, momentum, dampening, weight_decay, nesterov)
    @runtime_error_check ccall((:ato_sgd, libtorch_c_api), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble, Cint), arg1, learning_rate, momentum, dampening, weight_decay, nesterov)
end

function ato_add_parameters(arg1, arg2, ntensors)
    @runtime_error_check ccall((:ato_add_parameters, libtorch_c_api), Cint, (optimizer, Ptr{tensor}, Cint), arg1, arg2, ntensors)
end

function ato_set_learning_rate(arg1, learning_rate)
    @runtime_error_check ccall((:ato_set_learning_rate, libtorch_c_api), Cint, (optimizer, Cdouble), arg1, learning_rate)
end

function ato_set_momentum(arg1, momentum)
    @runtime_error_check ccall((:ato_set_momentum, libtorch_c_api), Cint, (optimizer, Cdouble), arg1, momentum)
end

function ato_zero_grad(arg1)
    @runtime_error_check ccall((:ato_zero_grad, libtorch_c_api), Cint, (optimizer,), arg1)
end

function ato_step(arg1)
    @runtime_error_check ccall((:ato_step, libtorch_c_api), Cint, (optimizer,), arg1)
end

function ato_free(arg1)
    @runtime_error_check ccall((:ato_free, libtorch_c_api), Cint, (optimizer,), arg1)
end

function ats_int(arg1, arg2)
    @runtime_error_check ccall((:ats_int, libtorch_c_api), Cint, (Ptr{scalar}, Int64), arg1, arg2)
end

function ats_float(arg1, arg2)
    @runtime_error_check ccall((:ats_float, libtorch_c_api), Cint, (Ptr{scalar}, Cdouble), arg1, arg2)
end

function ats_free(arg1)
    @runtime_error_check ccall((:ats_free, libtorch_c_api), Cint, (scalar,), arg1)
end

function atc_cuda_device_count(arg1)
    @runtime_error_check ccall((:atc_cuda_device_count, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function atc_cuda_is_available(arg1)
    @runtime_error_check ccall((:atc_cuda_is_available, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function atc_cudnn_is_available(arg1)
    @runtime_error_check ccall((:atc_cudnn_is_available, libtorch_c_api), Cint, (Ptr{Cint},), arg1)
end

function atc_set_benchmark_cudnn(b)
    @runtime_error_check ccall((:atc_set_benchmark_cudnn, libtorch_c_api), Cint, (Cint,), b)
end

function atm_load(arg1, arg2)
    @runtime_error_check ccall((:atm_load, libtorch_c_api), Cint, (Ptr{_module}, Ptr{Cchar}), arg1, arg2)
end

function atm_forward(arg1, arg2, tensors, ntensors)
    @runtime_error_check ccall((:atm_forward, libtorch_c_api), Cint, (Ptr{tensor}, _module, Ptr{tensor}, Cint), arg1, arg2, tensors, ntensors)
end

function atm_forward_(arg1, arg2, ivalues, nivalues)
    @runtime_error_check ccall((:atm_forward_, libtorch_c_api), Cint, (Ptr{ivalue}, _module, Ptr{ivalue}, Cint), arg1, arg2, ivalues, nivalues)
end

function atm_free(arg1)
    @runtime_error_check ccall((:atm_free, libtorch_c_api), Cint, (_module,), arg1)
end

function ati_none(arg1)
    @runtime_error_check ccall((:ati_none, libtorch_c_api), Cint, (Ptr{ivalue},), arg1)
end

function ati_tensor(arg1, arg2)
    @runtime_error_check ccall((:ati_tensor, libtorch_c_api), Cint, (Ptr{ivalue}, tensor), arg1, arg2)
end

function ati_bool(arg1, arg2)
    @runtime_error_check ccall((:ati_bool, libtorch_c_api), Cint, (Ptr{ivalue}, Cint), arg1, arg2)
end

function ati_int(arg1, arg2)
    @runtime_error_check ccall((:ati_int, libtorch_c_api), Cint, (Ptr{ivalue}, Int64), arg1, arg2)
end

function ati_double(arg1, arg2)
    @runtime_error_check ccall((:ati_double, libtorch_c_api), Cint, (Ptr{ivalue}, Cdouble), arg1, arg2)
end

function ati_tuple(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_tuple, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_string(arg1, arg2)
    @runtime_error_check ccall((:ati_string, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{Cchar}), arg1, arg2)
end

function ati_generic_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_generic_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_generic_dict(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_generic_dict, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_int_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_int_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{Int64}, Cint), arg1, arg2, arg3)
end

function ati_double_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_double_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{Cdouble}, Cint), arg1, arg2, arg3)
end

function ati_bool_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_bool_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{Cchar}, Cint), arg1, arg2, arg3)
end

function ati_string_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_string_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{Ptr{Cchar}}, Cint), arg1, arg2, arg3)
end

function ati_tensor_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_tensor_list, libtorch_c_api), Cint, (Ptr{ivalue}, Ptr{tensor}, Cint), arg1, arg2, arg3)
end

function ati_to_tensor(arg1, arg2)
    @runtime_error_check ccall((:ati_to_tensor, libtorch_c_api), Cint, (Ptr{tensor}, ivalue), arg1, arg2)
end

function ati_to_int(arg1, arg2)
    @runtime_error_check ccall((:ati_to_int, libtorch_c_api), Cint, (Ptr{Int64}, ivalue), arg1, arg2)
end

function ati_to_double(arg1, arg2)
    @runtime_error_check ccall((:ati_to_double, libtorch_c_api), Cint, (Ptr{Cdouble}, ivalue), arg1, arg2)
end

function ati_to_string(arg1, arg2)
    @runtime_error_check ccall((:ati_to_string, libtorch_c_api), Cint, (Ptr{Ptr{Cchar}}, ivalue), arg1, arg2)
end

function ati_to_bool(arg1, arg2)
    @runtime_error_check ccall((:ati_to_bool, libtorch_c_api), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_length(arg1, arg2)
    @runtime_error_check ccall((:ati_length, libtorch_c_api), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_tuple_length(arg1, arg2)
    @runtime_error_check ccall((:ati_tuple_length, libtorch_c_api), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_to_tuple(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_tuple, libtorch_c_api), Cint, (ivalue, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_to_generic_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_generic_list, libtorch_c_api), Cint, (ivalue, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_to_generic_dict(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_generic_dict, libtorch_c_api), Cint, (ivalue, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_to_int_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_int_list, libtorch_c_api), Cint, (ivalue, Ptr{Int64}, Cint), arg1, arg2, arg3)
end

function ati_to_double_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_double_list, libtorch_c_api), Cint, (ivalue, Ptr{Cdouble}, Cint), arg1, arg2, arg3)
end

function ati_to_bool_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_bool_list, libtorch_c_api), Cint, (ivalue, Ptr{Cchar}, Cint), arg1, arg2, arg3)
end

function ati_to_tensor_list(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_tensor_list, libtorch_c_api), Cint, (ivalue, Ptr{tensor}, Cint), arg1, arg2, arg3)
end

function ati_tag(arg1, arg2)
    @runtime_error_check ccall((:ati_tag, libtorch_c_api), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_free(arg1)
    @runtime_error_check ccall((:ati_free, libtorch_c_api), Cint, (ivalue,), arg1)
end

function atg___and__(arg1, self, other)
    @runtime_error_check ccall((:atg___and__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___and__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___and__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___iand__(arg1, self, other)
    @runtime_error_check ccall((:atg___iand__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___iand__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___iand__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___ilshift__(arg1, self, other)
    @runtime_error_check ccall((:atg___ilshift__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___ilshift__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___ilshift__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___ior__(arg1, self, other)
    @runtime_error_check ccall((:atg___ior__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___ior__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___ior__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___irshift__(arg1, self, other)
    @runtime_error_check ccall((:atg___irshift__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___irshift__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___irshift__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___ixor__(arg1, self, other)
    @runtime_error_check ccall((:atg___ixor__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___ixor__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___ixor__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___lshift__(arg1, self, other)
    @runtime_error_check ccall((:atg___lshift__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___lshift__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___lshift__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___or__(arg1, self, other)
    @runtime_error_check ccall((:atg___or__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___or__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___or__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___rshift__(arg1, self, other)
    @runtime_error_check ccall((:atg___rshift__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___rshift__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___rshift__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg___xor__(arg1, self, other)
    @runtime_error_check ccall((:atg___xor__, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg___xor__tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg___xor__tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg__adaptive_avg_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg__adaptive_avg_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg__adaptive_avg_pool2d_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg__adaptive_avg_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg__adaptive_avg_pool3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg__adaptive_avg_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg__adaptive_avg_pool3d_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg__adaptive_avg_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg__add_batch_dim(arg1, self, batch_dim, level)
    @runtime_error_check ccall((:atg__add_batch_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, batch_dim, level)
end

function atg__add_relu(arg1, self, other)
    @runtime_error_check ccall((:atg__add_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg__add_relu_(arg1, self, other)
    @runtime_error_check ccall((:atg__add_relu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg__add_relu_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg__add_relu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg__add_relu_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg__add_relu_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg__add_relu_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg__add_relu_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg__aminmax(arg1, self)
    @runtime_error_check ccall((:atg__aminmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__aminmax_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg__aminmax_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg__amp_update_scale_(arg1, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval)
    @runtime_error_check ccall((:atg__amp_update_scale_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Cdouble, Int64), arg1, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval)
end

function atg__baddbmm_mkl_(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg__baddbmm_mkl_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg__cast_byte(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_byte, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_char(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_char, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_double(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_double, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_float(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_float, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_half(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_half, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_int(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_long(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_long, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cast_short(arg1, self, non_blocking)
    @runtime_error_check ccall((:atg__cast_short, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, non_blocking)
end

function atg__cat(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg__cat, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg__cat_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg__cat_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg__cdist_backward(arg1, grad, x1, x2, p, cdist)
    @runtime_error_check ccall((:atg__cdist_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, tensor), arg1, grad, x1, x2, p, cdist)
end

function atg__cholesky_solve_helper(arg1, self, A, upper)
    @runtime_error_check ccall((:atg__cholesky_solve_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, A, upper)
end

function atg__coalesce(arg1, self)
    @runtime_error_check ccall((:atg__coalesce, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__coalesced_(arg1, self, coalesced)
    @runtime_error_check ccall((:atg__coalesced_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, coalesced)
end

function atg__compute_linear_combination(arg1, input, coefficients)
    @runtime_error_check ccall((:atg__compute_linear_combination, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, input, coefficients)
end

function atg__compute_linear_combination_out(arg1, out, input, coefficients)
    @runtime_error_check ccall((:atg__compute_linear_combination_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, input, coefficients)
end

function atg__conj(arg1, self)
    @runtime_error_check ccall((:atg__conj, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__conj_physical(arg1, self)
    @runtime_error_check ccall((:atg__conj_physical, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__conv_depthwise2d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg__conv_depthwise2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg__conv_depthwise2d_backward(arg1, grad_input, grad_weight, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg__conv_depthwise2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_weight, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg__conv_depthwise2d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg__conv_depthwise2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg__convert_indices_from_coo_to_csr(arg1, self, size, out_int32)
    @runtime_error_check ccall((:atg__convert_indices_from_coo_to_csr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, size, out_int32)
end

function atg__convert_indices_from_coo_to_csr_out(arg1, out, self, size, out_int32)
    @runtime_error_check ccall((:atg__convert_indices_from_coo_to_csr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, size, out_int32)
end

function atg__convolution(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups, benchmark, deterministic, cudnn_enabled, allow_tf32)
    @runtime_error_check ccall((:atg__convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups, benchmark, deterministic, cudnn_enabled, allow_tf32)
end

function atg__convolution_deprecated(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups, benchmark, deterministic, cudnn_enabled)
    @runtime_error_check ccall((:atg__convolution_deprecated, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups, benchmark, deterministic, cudnn_enabled)
end

function atg__convolution_mode(arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg__convolution_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Cchar}, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
end

function atg__convolution_nogroup(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len)
    @runtime_error_check ccall((:atg__convolution_nogroup, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len)
end

function atg__copy_from(arg1, self, dst, non_blocking)
    @runtime_error_check ccall((:atg__copy_from, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, dst, non_blocking)
end

function atg__copy_from_and_resize(arg1, self, dst)
    @runtime_error_check ccall((:atg__copy_from_and_resize, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, dst)
end

function atg__ctc_loss(arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, zero_infinity)
    @runtime_error_check ccall((:atg__ctc_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint), arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, zero_infinity)
end

function atg__ctc_loss_backward(arg1, grad, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, neg_log_likelihood, log_alpha, blank, zero_infinity)
    @runtime_error_check ccall((:atg__ctc_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor, tensor, Int64, Cint), arg1, grad, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, neg_log_likelihood, log_alpha, blank, zero_infinity)
end

function atg__cudnn_ctc_loss(arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, deterministic, zero_infinity)
    @runtime_error_check ccall((:atg__cudnn_ctc_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, deterministic, zero_infinity)
end

function atg__cudnn_init_dropout_state(arg1, dropout, train, dropout_seed, options_kind, options_device)
    @runtime_error_check ccall((:atg__cudnn_init_dropout_state, libtorch_c_api), Cint, (Ptr{tensor}, Cdouble, Cint, Int64, Cint, Cint), arg1, dropout, train, dropout_seed, options_kind, options_device)
end

function atg__cudnn_rnn(arg1, input, weight_data, weight_len, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
    @runtime_error_check ccall((:atg__cudnn_rnn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64, tensor, tensor, tensor, Int64, Int64, Int64, Int64, Cint, Cdouble, Cint, Cint, Ptr{Int64}, Cint, tensor), arg1, input, weight_data, weight_len, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
end

function atg__cudnn_rnn_flatten_weight(arg1, weight_arr_data, weight_arr_len, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional)
    @runtime_error_check ccall((:atg__cudnn_rnn_flatten_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64, Int64, Int64, Int64, Int64, Int64, Cint, Cint), arg1, weight_arr_data, weight_arr_len, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional)
end

function atg__det_lu_based_helper(arg1, self)
    @runtime_error_check ccall((:atg__det_lu_based_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__det_lu_based_helper_backward_helper(arg1, det_grad, det, self, lu, pivs)
    @runtime_error_check ccall((:atg__det_lu_based_helper_backward_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor), arg1, det_grad, det, self, lu, pivs)
end

function atg__dim_arange(arg1, like, dim)
    @runtime_error_check ccall((:atg__dim_arange, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, like, dim)
end

function atg__dirichlet_grad(arg1, x, alpha, total)
    @runtime_error_check ccall((:atg__dirichlet_grad, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, x, alpha, total)
end

function atg__embedding_bag(arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint, tensor, Cint, Int64), arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
end

function atg__embedding_bag_backward(arg1, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Int64, Cint, Int64, Cint, tensor, Int64), arg1, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx)
end

function atg__embedding_bag_dense_backward(arg1, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag_dense_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Cint, Int64, tensor, Int64), arg1, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx)
end

function atg__embedding_bag_forward_only(arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag_forward_only, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint, tensor, Cint, Int64), arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
end

function atg__embedding_bag_per_sample_weights_backward(arg1, grad, weight, indices, offsets, offset2bag, mode, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag_per_sample_weights_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Int64), arg1, grad, weight, indices, offsets, offset2bag, mode, padding_idx)
end

function atg__embedding_bag_sparse_backward(arg1, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx)
    @runtime_error_check ccall((:atg__embedding_bag_sparse_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Cint, Int64, tensor, Int64), arg1, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx)
end

function atg__empty_affine_quantized(arg1, size_data, size_len, options_kind, options_device, scale, zero_point)
    @runtime_error_check ccall((:atg__empty_affine_quantized, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint, Cdouble, Int64), arg1, size_data, size_len, options_kind, options_device, scale, zero_point)
end

function atg__empty_per_channel_affine_quantized(arg1, size_data, size_len, scales, zero_points, axis, options_kind, options_device)
    @runtime_error_check ccall((:atg__empty_per_channel_affine_quantized, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Int64, Cint, Cint), arg1, size_data, size_len, scales, zero_points, axis, options_kind, options_device)
end

function atg__euclidean_dist(arg1, x1, x2)
    @runtime_error_check ccall((:atg__euclidean_dist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, x1, x2)
end

function atg__fake_quantize_learnable_per_channel_affine(arg1, self, scale, zero_point, axis, quant_min, quant_max, grad_factor)
    @runtime_error_check ccall((:atg__fake_quantize_learnable_per_channel_affine, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Int64, Cdouble), arg1, self, scale, zero_point, axis, quant_min, quant_max, grad_factor)
end

function atg__fake_quantize_learnable_per_channel_affine_backward(arg1, grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor)
    @runtime_error_check ccall((:atg__fake_quantize_learnable_per_channel_affine_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, Int64, Cdouble), arg1, grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor)
end

function atg__fake_quantize_learnable_per_tensor_affine(arg1, self, scale, zero_point, quant_min, quant_max, grad_factor)
    @runtime_error_check ccall((:atg__fake_quantize_learnable_per_tensor_affine, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cdouble), arg1, self, scale, zero_point, quant_min, quant_max, grad_factor)
end

function atg__fake_quantize_learnable_per_tensor_affine_backward(arg1, grad, self, scale, zero_point, quant_min, quant_max, grad_factor)
    @runtime_error_check ccall((:atg__fake_quantize_learnable_per_tensor_affine_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, Cdouble), arg1, grad, self, scale, zero_point, quant_min, quant_max, grad_factor)
end

function atg__fake_quantize_per_tensor_affine_cachemask_tensor_qparams(arg1, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max)
    @runtime_error_check ccall((:atg__fake_quantize_per_tensor_affine_cachemask_tensor_qparams, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64), arg1, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max)
end

function atg__fft_c2c(arg1, self, dim_data, dim_len, normalization, forward)
    @runtime_error_check ccall((:atg__fft_c2c, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, normalization, forward)
end

function atg__fft_c2c_out(arg1, out, self, dim_data, dim_len, normalization, forward)
    @runtime_error_check ccall((:atg__fft_c2c_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, out, self, dim_data, dim_len, normalization, forward)
end

function atg__fft_c2r(arg1, self, dim_data, dim_len, normalization, last_dim_size)
    @runtime_error_check ccall((:atg__fft_c2r, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, self, dim_data, dim_len, normalization, last_dim_size)
end

function atg__fft_c2r_out(arg1, out, self, dim_data, dim_len, normalization, last_dim_size)
    @runtime_error_check ccall((:atg__fft_c2r_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, out, self, dim_data, dim_len, normalization, last_dim_size)
end

function atg__fft_r2c(arg1, self, dim_data, dim_len, normalization, onesided)
    @runtime_error_check ccall((:atg__fft_r2c, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, normalization, onesided)
end

function atg__fft_r2c_out(arg1, out, self, dim_data, dim_len, normalization, onesided)
    @runtime_error_check ccall((:atg__fft_r2c_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, out, self, dim_data, dim_len, normalization, onesided)
end

function atg__fused_dropout(arg1, self, p)
    @runtime_error_check ccall((:atg__fused_dropout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg__fused_moving_avg_obs_fq_helper(arg1, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)
    @runtime_error_check ccall((:atg__fused_moving_avg_obs_fq_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble, Int64, Int64, Int64, Cint, Cint), arg1, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)
end

function atg__fw_primal(arg1, self, level)
    @runtime_error_check ccall((:atg__fw_primal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, level)
end

function atg__gather_sparse_backward(arg1, self, dim, index, grad)
    @runtime_error_check ccall((:atg__gather_sparse_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, grad)
end

function atg__grid_sampler_2d_cpu_fallback(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg__grid_sampler_2d_cpu_fallback, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg__grid_sampler_2d_cpu_fallback_backward(arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg__grid_sampler_2d_cpu_fallback_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg__index_copy_(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg__index_copy_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg__index_put_impl_(arg1, self, indices_data, indices_len, values, accumulate, unsafe)
    @runtime_error_check ccall((:atg__index_put_impl_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, Cint, Cint), arg1, self, indices_data, indices_len, values, accumulate, unsafe)
end

function atg__indices(arg1, self)
    @runtime_error_check ccall((:atg__indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__inverse_helper(arg1, self)
    @runtime_error_check ccall((:atg__inverse_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__linalg_inv_out_helper_(arg1, self, infos_lu, infos_getri)
    @runtime_error_check ccall((:atg__linalg_inv_out_helper_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, infos_lu, infos_getri)
end

function atg__linalg_qr_helper(arg1, self, mode)
    @runtime_error_check ccall((:atg__linalg_qr_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}), arg1, self, mode)
end

function atg__log_softmax(arg1, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__log_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, half_to_float)
end

function atg__log_softmax_backward_data(arg1, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__log_softmax_backward_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, grad_output, output, dim, self)
end

function atg__log_softmax_backward_data_out(arg1, out, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__log_softmax_backward_data_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, tensor), arg1, out, grad_output, output, dim, self)
end

function atg__log_softmax_out(arg1, out, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__log_softmax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, half_to_float)
end

function atg__logcumsumexp(arg1, self, dim)
    @runtime_error_check ccall((:atg__logcumsumexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg__logcumsumexp_out(arg1, out, self, dim)
    @runtime_error_check ccall((:atg__logcumsumexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, dim)
end

function atg__lu_with_info(arg1, self, pivot, check_errors)
    @runtime_error_check ccall((:atg__lu_with_info, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, pivot, check_errors)
end

function atg__make_dual(arg1, primal, tangent, level)
    @runtime_error_check ccall((:atg__make_dual, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, primal, tangent, level)
end

function atg__make_per_channel_quantized_tensor(arg1, self, scale, zero_point, axis)
    @runtime_error_check ccall((:atg__make_per_channel_quantized_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, self, scale, zero_point, axis)
end

function atg__make_per_tensor_quantized_tensor(arg1, self, scale, zero_point)
    @runtime_error_check ccall((:atg__make_per_tensor_quantized_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64), arg1, self, scale, zero_point)
end

function atg__masked_scale(arg1, self, mask, scale)
    @runtime_error_check ccall((:atg__masked_scale, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, self, mask, scale)
end

function atg__mkldnn_reshape(arg1, self, shape_data, shape_len)
    @runtime_error_check ccall((:atg__mkldnn_reshape, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, shape_data, shape_len)
end

function atg__mkldnn_transpose(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg__mkldnn_transpose, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg__mkldnn_transpose_(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg__mkldnn_transpose_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg__neg_view(arg1, self)
    @runtime_error_check ccall((:atg__neg_view, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__nnpack_spatial_convolution(arg1, input, weight, bias, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg__nnpack_spatial_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, input, weight, bias, padding_data, padding_len, stride_data, stride_len)
end

function atg__nnpack_spatial_convolution_backward_input(arg1, input, grad_output, weight, padding_data, padding_len)
    @runtime_error_check ccall((:atg__nnpack_spatial_convolution_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, input, grad_output, weight, padding_data, padding_len)
end

function atg__nnpack_spatial_convolution_backward_weight(arg1, input, weightsize_data, weightsize_len, grad_output, padding_data, padding_len)
    @runtime_error_check ccall((:atg__nnpack_spatial_convolution_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint), arg1, input, weightsize_data, weightsize_len, grad_output, padding_data, padding_len)
end

function atg__pack_padded_sequence(arg1, input, lengths, batch_first)
    @runtime_error_check ccall((:atg__pack_padded_sequence, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, input, lengths, batch_first)
end

function atg__pack_padded_sequence_backward(arg1, grad, input_size_data, input_size_len, batch_sizes, batch_first)
    @runtime_error_check ccall((:atg__pack_padded_sequence_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, tensor, Cint), arg1, grad, input_size_data, input_size_len, batch_sizes, batch_first)
end

function atg__pad_packed_sequence(arg1, data, batch_sizes, batch_first, padding_value, total_length)
    @runtime_error_check ccall((:atg__pad_packed_sequence, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, scalar, Int64), arg1, data, batch_sizes, batch_first, padding_value, total_length)
end

function atg__pdist_backward(arg1, grad, self, p, pdist)
    @runtime_error_check ccall((:atg__pdist_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, tensor), arg1, grad, self, p, pdist)
end

function atg__pin_memory(arg1, self, device)
    @runtime_error_check ccall((:atg__pin_memory, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, device)
end

function atg__remove_batch_dim(arg1, self, level, batch_size, out_dim)
    @runtime_error_check ccall((:atg__remove_batch_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, level, batch_size, out_dim)
end

function atg__reshape_alias(arg1, self, size_data, size_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg__reshape_alias, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, size_data, size_len, stride_data, stride_len)
end

function atg__reshape_from_tensor(arg1, self, shape)
    @runtime_error_check ccall((:atg__reshape_from_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, shape)
end

function atg__rowwise_prune(arg1, weight, mask, compressed_indices_dtype)
    @runtime_error_check ccall((:atg__rowwise_prune, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, weight, mask, compressed_indices_dtype)
end

function atg__s_where(arg1, condition, self, other)
    @runtime_error_check ccall((:atg__s_where, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, condition, self, other)
end

function atg__sample_dirichlet(arg1, self)
    @runtime_error_check ccall((:atg__sample_dirichlet, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__saturate_weight_to_fp16(arg1, weight)
    @runtime_error_check ccall((:atg__saturate_weight_to_fp16, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, weight)
end

function atg__segment_reduce_backward(arg1, grad, output, data, reduce, lengths, axis)
    @runtime_error_check ccall((:atg__segment_reduce_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Cchar}, tensor, Int64), arg1, grad, output, data, reduce, lengths, axis)
end

function atg__shape_as_tensor(arg1, self)
    @runtime_error_check ccall((:atg__shape_as_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__slow_conv2d_backward(arg1, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, finput)
    @runtime_error_check ccall((:atg__slow_conv2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, finput)
end

function atg__sobol_engine_draw(arg1, quasi, n, sobolstate, dimension, num_generated, dtype)
    @runtime_error_check ccall((:atg__sobol_engine_draw, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, Int64, Int64, Cint), arg1, quasi, n, sobolstate, dimension, num_generated, dtype)
end

function atg__sobol_engine_ff_(arg1, self, n, sobolstate, dimension, num_generated)
    @runtime_error_check ccall((:atg__sobol_engine_ff_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, Int64, Int64), arg1, self, n, sobolstate, dimension, num_generated)
end

function atg__sobol_engine_initialize_state_(arg1, self, dimension)
    @runtime_error_check ccall((:atg__sobol_engine_initialize_state_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dimension)
end

function atg__sobol_engine_scramble_(arg1, self, ltm, dimension)
    @runtime_error_check ccall((:atg__sobol_engine_scramble_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, ltm, dimension)
end

function atg__softmax(arg1, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, half_to_float)
end

function atg__softmax_backward_data(arg1, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__softmax_backward_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, grad_output, output, dim, self)
end

function atg__softmax_backward_data_out(arg1, grad_input, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__softmax_backward_data_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, tensor), arg1, grad_input, grad_output, output, dim, self)
end

function atg__softmax_out(arg1, out, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__softmax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, half_to_float)
end

function atg__solve_helper(arg1, self, A)
    @runtime_error_check ccall((:atg__solve_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, A)
end

function atg__sparse_addmm(arg1, self, sparse, dense)
    @runtime_error_check ccall((:atg__sparse_addmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, sparse, dense)
end

function atg__sparse_coo_tensor_unsafe(arg1, indices, values, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg__sparse_coo_tensor_unsafe, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, indices, values, size_data, size_len, options_kind, options_device)
end

function atg__sparse_coo_tensor_with_dims(arg1, sparse_dim, dense_dim, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg__sparse_coo_tensor_with_dims, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Ptr{Int64}, Cint, Cint, Cint), arg1, sparse_dim, dense_dim, size_data, size_len, options_kind, options_device)
end

function atg__sparse_coo_tensor_with_dims_and_tensors(arg1, sparse_dim, dense_dim, size_data, size_len, indices, values, options_kind, options_device)
    @runtime_error_check ccall((:atg__sparse_coo_tensor_with_dims_and_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Ptr{Int64}, Cint, tensor, tensor, Cint, Cint), arg1, sparse_dim, dense_dim, size_data, size_len, indices, values, options_kind, options_device)
end

function atg__sparse_csr_tensor_unsafe(arg1, crow_indices, col_indices, values, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg__sparse_csr_tensor_unsafe, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, crow_indices, col_indices, values, size_data, size_len, options_kind, options_device)
end

function atg__sparse_log_softmax(arg1, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__sparse_log_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, half_to_float)
end

function atg__sparse_log_softmax_backward_data(arg1, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__sparse_log_softmax_backward_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, grad_output, output, dim, self)
end

function atg__sparse_log_softmax_int(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg__sparse_log_softmax_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg__sparse_mask_helper(arg1, t, mask_indices)
    @runtime_error_check ccall((:atg__sparse_mask_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, t, mask_indices)
end

function atg__sparse_mm(arg1, sparse, dense)
    @runtime_error_check ccall((:atg__sparse_mm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, sparse, dense)
end

function atg__sparse_softmax(arg1, self, dim, half_to_float)
    @runtime_error_check ccall((:atg__sparse_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, half_to_float)
end

function atg__sparse_softmax_backward_data(arg1, grad_output, output, dim, self)
    @runtime_error_check ccall((:atg__sparse_softmax_backward_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, grad_output, output, dim, self)
end

function atg__sparse_softmax_int(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg__sparse_softmax_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg__sparse_sparse_matmul(arg1, self, other)
    @runtime_error_check ccall((:atg__sparse_sparse_matmul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg__sparse_sum(arg1, self)
    @runtime_error_check ccall((:atg__sparse_sum, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__sparse_sum_backward(arg1, grad, self, dim_data, dim_len)
    @runtime_error_check ccall((:atg__sparse_sum_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad, self, dim_data, dim_len)
end

function atg__sparse_sum_dim(arg1, self, dim_data, dim_len)
    @runtime_error_check ccall((:atg__sparse_sum_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dim_data, dim_len)
end

function atg__sparse_sum_dim_dtype(arg1, self, dim_data, dim_len, dtype)
    @runtime_error_check ccall((:atg__sparse_sum_dim_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, dtype)
end

function atg__sparse_sum_dtype(arg1, self, dtype)
    @runtime_error_check ccall((:atg__sparse_sum_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg__stack(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg__stack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg__stack_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg__stack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg__standard_gamma(arg1, self)
    @runtime_error_check ccall((:atg__standard_gamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__standard_gamma_grad(arg1, self, output)
    @runtime_error_check ccall((:atg__standard_gamma_grad, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, output)
end

function atg__svd_helper(arg1, self, some, compute_uv)
    @runtime_error_check ccall((:atg__svd_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, some, compute_uv)
end

function atg__symeig_helper(arg1, self, eigenvectors, upper)
    @runtime_error_check ccall((:atg__symeig_helper, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, eigenvectors, upper)
end

function atg__test_ambiguous_defaults(arg1, dummy, a, b)
    @runtime_error_check ccall((:atg__test_ambiguous_defaults, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, dummy, a, b)
end

function atg__test_ambiguous_defaults_b(arg1, dummy, a, b)
    @runtime_error_check ccall((:atg__test_ambiguous_defaults_b, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Cchar}), arg1, dummy, a, b)
end

function atg__test_optional_filled_intlist(arg1, values, addends_data, addends_len)
    @runtime_error_check ccall((:atg__test_optional_filled_intlist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, values, addends_data, addends_len)
end

function atg__test_optional_intlist(arg1, values, addends_data, addends_len)
    @runtime_error_check ccall((:atg__test_optional_intlist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, values, addends_data, addends_len)
end

function atg__test_serialization_subcmul(arg1, self, other)
    @runtime_error_check ccall((:atg__test_serialization_subcmul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg__test_string_default(arg1, dummy, a, b)
    @runtime_error_check ccall((:atg__test_string_default, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}, Ptr{Cchar}), arg1, dummy, a, b)
end

function atg__thnn_differentiable_gru_cell_backward(arg1, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias)
    @runtime_error_check ccall((:atg__thnn_differentiable_gru_cell_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias)
end

function atg__thnn_differentiable_lstm_cell_backward(arg1, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy)
    @runtime_error_check ccall((:atg__thnn_differentiable_lstm_cell_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor), arg1, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy)
end

function atg__thnn_fused_gru_cell(arg1, input_gates, hidden_gates, hx, input_bias, hidden_bias)
    @runtime_error_check ccall((:atg__thnn_fused_gru_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor), arg1, input_gates, hidden_gates, hx, input_bias, hidden_bias)
end

function atg__thnn_fused_gru_cell_backward(arg1, grad_hy, workspace, has_bias)
    @runtime_error_check ccall((:atg__thnn_fused_gru_cell_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, grad_hy, workspace, has_bias)
end

function atg__thnn_fused_lstm_cell(arg1, input_gates, hidden_gates, cx, input_bias, hidden_bias)
    @runtime_error_check ccall((:atg__thnn_fused_lstm_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor), arg1, input_gates, hidden_gates, cx, input_bias, hidden_bias)
end

function atg__thnn_fused_lstm_cell_backward(arg1, grad_hy, grad_cy, cx, cy, workspace, has_bias)
    @runtime_error_check ccall((:atg__thnn_fused_lstm_cell_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint), arg1, grad_hy, grad_cy, cx, cy, workspace, has_bias)
end

function atg__to_copy(arg1, self, options_kind, options_device, non_blocking)
    @runtime_error_check ccall((:atg__to_copy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint), arg1, self, options_kind, options_device, non_blocking)
end

function atg__to_cpu(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg__to_cpu, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg__trilinear(arg1, i1, i2, i3, expand1_data, expand1_len, expand2_data, expand2_len, expand3_data, expand3_len, sumdim_data, sumdim_len, unroll_dim)
    @runtime_error_check ccall((:atg__trilinear, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, i1, i2, i3, expand1_data, expand1_len, expand2_data, expand2_len, expand3_data, expand3_len, sumdim_data, sumdim_len, unroll_dim)
end

function atg__unique(arg1, self, sorted, return_inverse)
    @runtime_error_check ccall((:atg__unique, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, sorted, return_inverse)
end

function atg__unique2(arg1, self, sorted, return_inverse, return_counts)
    @runtime_error_check ccall((:atg__unique2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint), arg1, self, sorted, return_inverse, return_counts)
end

function atg__unpack_dual(arg1, dual, level)
    @runtime_error_check ccall((:atg__unpack_dual, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, dual, level)
end

function atg__unsafe_view(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg__unsafe_view, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg__values(arg1, self)
    @runtime_error_check ccall((:atg__values, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg__weight_norm(arg1, v, g, dim)
    @runtime_error_check ccall((:atg__weight_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, v, g, dim)
end

function atg__weight_norm_cuda_interface(arg1, v, g, dim)
    @runtime_error_check ccall((:atg__weight_norm_cuda_interface, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, v, g, dim)
end

function atg__weight_norm_cuda_interface_backward(arg1, grad_w, saved_v, saved_g, saved_norms, dim)
    @runtime_error_check ccall((:atg__weight_norm_cuda_interface_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_w, saved_v, saved_g, saved_norms, dim)
end

function atg__weight_norm_differentiable_backward(arg1, grad_w, saved_v, saved_g, saved_norms, dim)
    @runtime_error_check ccall((:atg__weight_norm_differentiable_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_w, saved_v, saved_g, saved_norms, dim)
end

function atg_abs(arg1, self)
    @runtime_error_check ccall((:atg_abs, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_abs_(arg1, self)
    @runtime_error_check ccall((:atg_abs_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_abs_out(arg1, out, self)
    @runtime_error_check ccall((:atg_abs_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_absolute(arg1, self)
    @runtime_error_check ccall((:atg_absolute, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_absolute_(arg1, self)
    @runtime_error_check ccall((:atg_absolute_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_absolute_out(arg1, out, self)
    @runtime_error_check ccall((:atg_absolute_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_acos(arg1, self)
    @runtime_error_check ccall((:atg_acos, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acos_(arg1, self)
    @runtime_error_check ccall((:atg_acos_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acos_out(arg1, out, self)
    @runtime_error_check ccall((:atg_acos_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_acosh(arg1, self)
    @runtime_error_check ccall((:atg_acosh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acosh_(arg1, self)
    @runtime_error_check ccall((:atg_acosh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acosh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_acosh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_adaptive_avg_pool1d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool2d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool3d_backward(arg1, grad_input, grad_output, self)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, self)
end

function atg_adaptive_avg_pool3d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool1d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool2d_backward(arg1, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, indices)
end

function atg_adaptive_max_pool2d_backward_grad_input(arg1, grad_input, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, indices)
end

function atg_adaptive_max_pool2d_out(arg1, out, indices, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, indices, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool3d_backward(arg1, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, indices)
end

function atg_adaptive_max_pool3d_backward_grad_input(arg1, grad_input, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, indices)
end

function atg_adaptive_max_pool3d_out(arg1, out, indices, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, indices, self, output_size_data, output_size_len)
end

function atg_add(arg1, self, other)
    @runtime_error_check ccall((:atg_add, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_add_(arg1, self, other)
    @runtime_error_check ccall((:atg_add_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_add_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_add_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_add_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_add_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_add_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_add_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_addbmm(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_addbmm_(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_addbmm_out(arg1, out, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, batch1, batch2)
end

function atg_addcdiv(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcdiv_(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcdiv_out(arg1, out, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, tensor1, tensor2)
end

function atg_addcmul(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcmul_(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcmul_out(arg1, out, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, tensor1, tensor2)
end

function atg_addmm(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_addmm_(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_addmm_out(arg1, out, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat1, mat2)
end

function atg_addmv(arg1, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat, vec)
end

function atg_addmv_(arg1, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat, vec)
end

function atg_addmv_out(arg1, out, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat, vec)
end

function atg_addr(arg1, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, vec1, vec2)
end

function atg_addr_(arg1, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, vec1, vec2)
end

function atg_addr_out(arg1, out, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, vec1, vec2)
end

function atg_affine_grid_generator(arg1, theta, size_data, size_len, align_corners)
    @runtime_error_check ccall((:atg_affine_grid_generator, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, theta, size_data, size_len, align_corners)
end

function atg_affine_grid_generator_backward(arg1, grad, size_data, size_len, align_corners)
    @runtime_error_check ccall((:atg_affine_grid_generator_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, grad, size_data, size_len, align_corners)
end

function atg_alias(arg1, self)
    @runtime_error_check ccall((:atg_alias, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_align_as(arg1, self, other)
    @runtime_error_check ccall((:atg_align_as, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_align_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_align_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_all(arg1, self)
    @runtime_error_check ccall((:atg_all, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_all_all_out(arg1, out, self)
    @runtime_error_check ccall((:atg_all_all_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_all_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_all_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_all_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_all_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_alpha_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_alpha_dropout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_alpha_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_alpha_dropout_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_amax(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_amax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_amax_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_amax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_amin(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_amin, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_amin_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_amin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_aminmax(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_aminmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_aminmax_out(arg1, min, max, self, dim, keepdim)
    @runtime_error_check ccall((:atg_aminmax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, min, max, self, dim, keepdim)
end

function atg_angle(arg1, self)
    @runtime_error_check ccall((:atg_angle, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_angle_out(arg1, out, self)
    @runtime_error_check ccall((:atg_angle_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_any(arg1, self)
    @runtime_error_check ccall((:atg_any, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_any_all_out(arg1, out, self)
    @runtime_error_check ccall((:atg_any_all_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_any_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_any_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_any_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_any_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_arange(arg1, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange, libtorch_c_api), Cint, (Ptr{tensor}, scalar, Cint, Cint), arg1, _end, options_kind, options_device)
end

function atg_arange_out(arg1, out, _end)
    @runtime_error_check ccall((:atg_arange_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, out, _end)
end

function atg_arange_start(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange_start, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_arange_start_out(arg1, out, start, _end)
    @runtime_error_check ccall((:atg_arange_start_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, out, start, _end)
end

function atg_arange_start_step(arg1, start, _end, step, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange_start_step, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, scalar, Cint, Cint), arg1, start, _end, step, options_kind, options_device)
end

function atg_arccos(arg1, self)
    @runtime_error_check ccall((:atg_arccos, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arccos_(arg1, self)
    @runtime_error_check ccall((:atg_arccos_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arccos_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arccos_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_arccosh(arg1, self)
    @runtime_error_check ccall((:atg_arccosh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arccosh_(arg1, self)
    @runtime_error_check ccall((:atg_arccosh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arccosh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arccosh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_arcsin(arg1, self)
    @runtime_error_check ccall((:atg_arcsin, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arcsin_(arg1, self)
    @runtime_error_check ccall((:atg_arcsin_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arcsin_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arcsin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_arcsinh(arg1, self)
    @runtime_error_check ccall((:atg_arcsinh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arcsinh_(arg1, self)
    @runtime_error_check ccall((:atg_arcsinh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arcsinh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arcsinh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_arctan(arg1, self)
    @runtime_error_check ccall((:atg_arctan, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arctan_(arg1, self)
    @runtime_error_check ccall((:atg_arctan_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arctan_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arctan_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_arctanh(arg1, self)
    @runtime_error_check ccall((:atg_arctanh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arctanh_(arg1, self)
    @runtime_error_check ccall((:atg_arctanh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_arctanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_arctanh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_argmax(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_argmax_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_argmin(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmin, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_argmin_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_argsort(arg1, self, dim, descending)
    @runtime_error_check ccall((:atg_argsort, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, descending)
end

function atg_as_strided(arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
    @runtime_error_check ccall((:atg_as_strided, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
end

function atg_as_strided_(arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
    @runtime_error_check ccall((:atg_as_strided_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
end

function atg_asin(arg1, self)
    @runtime_error_check ccall((:atg_asin, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asin_(arg1, self)
    @runtime_error_check ccall((:atg_asin_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asin_out(arg1, out, self)
    @runtime_error_check ccall((:atg_asin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_asinh(arg1, self)
    @runtime_error_check ccall((:atg_asinh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asinh_(arg1, self)
    @runtime_error_check ccall((:atg_asinh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asinh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_asinh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_atan(arg1, self)
    @runtime_error_check ccall((:atg_atan, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atan2(arg1, self, other)
    @runtime_error_check ccall((:atg_atan2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_atan2_(arg1, self, other)
    @runtime_error_check ccall((:atg_atan2_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_atan2_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_atan2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_atan_(arg1, self)
    @runtime_error_check ccall((:atg_atan_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atan_out(arg1, out, self)
    @runtime_error_check ccall((:atg_atan_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_atanh(arg1, self)
    @runtime_error_check ccall((:atg_atanh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atanh_(arg1, self)
    @runtime_error_check ccall((:atg_atanh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_atanh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_atleast_1d(arg1, self)
    @runtime_error_check ccall((:atg_atleast_1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atleast_1d_sequence(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_atleast_1d_sequence, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_atleast_2d(arg1, self)
    @runtime_error_check ccall((:atg_atleast_2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atleast_2d_sequence(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_atleast_2d_sequence, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_atleast_3d(arg1, self)
    @runtime_error_check ccall((:atg_atleast_3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atleast_3d_sequence(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_atleast_3d_sequence, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_avg_pool1d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad)
    @runtime_error_check ccall((:atg_avg_pool1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad)
end

function atg_avg_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_out(arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_out(arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_baddbmm(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_baddbmm_(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_baddbmm_out(arg1, out, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, batch1, batch2)
end

function atg_bartlett_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_bartlett_window, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_bartlett_window_periodic(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_bartlett_window_periodic, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_batch_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble, Cint), arg1, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
end

function atg_batch_norm_backward_elemt(arg1, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count)
    @runtime_error_check ccall((:atg_batch_norm_backward_elemt, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor), arg1, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count)
end

function atg_batch_norm_backward_reduce(arg1, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g)
    @runtime_error_check ccall((:atg_batch_norm_backward_reduce, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cint, Cint), arg1, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g)
end

function atg_batch_norm_elemt(arg1, input, weight, bias, mean, invstd, eps)
    @runtime_error_check ccall((:atg_batch_norm_elemt, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, input, weight, bias, mean, invstd, eps)
end

function atg_batch_norm_elemt_out(arg1, out, input, weight, bias, mean, invstd, eps)
    @runtime_error_check ccall((:atg_batch_norm_elemt_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, out, input, weight, bias, mean, invstd, eps)
end

function atg_batch_norm_gather_stats(arg1, input, mean, invstd, running_mean, running_var, momentum, eps, count)
    @runtime_error_check ccall((:atg_batch_norm_gather_stats, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble, Cdouble, Int64), arg1, input, mean, invstd, running_mean, running_var, momentum, eps, count)
end

function atg_batch_norm_gather_stats_with_counts(arg1, input, mean, invstd, running_mean, running_var, momentum, eps, counts)
    @runtime_error_check ccall((:atg_batch_norm_gather_stats_with_counts, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble, Cdouble, tensor), arg1, input, mean, invstd, running_mean, running_var, momentum, eps, counts)
end

function atg_batch_norm_stats(arg1, input, eps)
    @runtime_error_check ccall((:atg_batch_norm_stats, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, input, eps)
end

function atg_batch_norm_update_stats(arg1, input, running_mean, running_var, momentum)
    @runtime_error_check ccall((:atg_batch_norm_update_stats, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble), arg1, input, running_mean, running_var, momentum)
end

function atg_bernoulli(arg1, self)
    @runtime_error_check ccall((:atg_bernoulli, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bernoulli_(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, p)
end

function atg_bernoulli_float_(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli_float_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_bernoulli_out(arg1, out, self)
    @runtime_error_check ccall((:atg_bernoulli_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_bernoulli_p(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli_p, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_bilinear(arg1, input1, input2, weight, bias)
    @runtime_error_check ccall((:atg_bilinear, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, input1, input2, weight, bias)
end

function atg_binary_cross_entropy(arg1, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, self, target, weight, reduction)
end

function atg_binary_cross_entropy_backward(arg1, grad_output, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, weight, reduction)
end

function atg_binary_cross_entropy_backward_grad_input(arg1, grad_input, grad_output, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, weight, reduction)
end

function atg_binary_cross_entropy_out(arg1, out, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, out, self, target, weight, reduction)
end

function atg_binary_cross_entropy_with_logits(arg1, self, target, weight, pos_weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_with_logits, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, self, target, weight, pos_weight, reduction)
end

function atg_binary_cross_entropy_with_logits_backward(arg1, grad_output, self, target, weight, pos_weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_with_logits_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, weight, pos_weight, reduction)
end

function atg_bincount(arg1, self, weights, minlength)
    @runtime_error_check ccall((:atg_bincount, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, weights, minlength)
end

function atg_binomial(arg1, count, prob)
    @runtime_error_check ccall((:atg_binomial, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, count, prob)
end

function atg_bitwise_and(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_and, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_and_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_and_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_and_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_and_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_bitwise_and_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_and_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_and_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_and_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_and_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_and_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_bitwise_left_shift(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_left_shift_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_left_shift_scalar_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_scalar_tensor, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_bitwise_left_shift_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_bitwise_left_shift_tensor_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_tensor_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_left_shift_tensor_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_tensor_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_left_shift_tensor_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_left_shift_tensor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_bitwise_not(arg1, self)
    @runtime_error_check ccall((:atg_bitwise_not, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bitwise_not_(arg1, self)
    @runtime_error_check ccall((:atg_bitwise_not_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bitwise_not_out(arg1, out, self)
    @runtime_error_check ccall((:atg_bitwise_not_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_bitwise_or(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_or, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_or_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_or_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_or_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_or_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_bitwise_or_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_or_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_or_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_or_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_or_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_or_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_bitwise_right_shift(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_right_shift_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_right_shift_scalar_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_scalar_tensor, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_bitwise_right_shift_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_bitwise_right_shift_tensor_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_tensor_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_right_shift_tensor_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_tensor_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_right_shift_tensor_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_right_shift_tensor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_bitwise_xor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_xor_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_xor_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_bitwise_xor_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_xor_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_xor_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_blackman_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_blackman_window, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_blackman_window_periodic(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_blackman_window_periodic, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_block_diag(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_block_diag, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_bmm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_bmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_bmm_out(arg1, out, self, mat2)
    @runtime_error_check ccall((:atg_bmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mat2)
end

function atg_broadcast_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_broadcast_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_broadcast_to(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_broadcast_to, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_bucketize(arg1, self, boundaries, out_int32, right)
    @runtime_error_check ccall((:atg_bucketize, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, self, boundaries, out_int32, right)
end

function atg_bucketize_scalar(arg1, self, boundaries, out_int32, right)
    @runtime_error_check ccall((:atg_bucketize_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor, Cint, Cint), arg1, self, boundaries, out_int32, right)
end

function atg_bucketize_tensor_out(arg1, out, self, boundaries, out_int32, right)
    @runtime_error_check ccall((:atg_bucketize_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, out, self, boundaries, out_int32, right)
end

function atg_cartesian_prod(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_cartesian_prod, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_cat(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_cat, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg_cat_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_cat_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg_cauchy_(arg1, self, median, sigma)
    @runtime_error_check ccall((:atg_cauchy_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, median, sigma)
end

function atg_cdist(arg1, x1, x2, p, compute_mode)
    @runtime_error_check ccall((:atg_cdist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64), arg1, x1, x2, p, compute_mode)
end

function atg_ceil(arg1, self)
    @runtime_error_check ccall((:atg_ceil, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ceil_(arg1, self)
    @runtime_error_check ccall((:atg_ceil_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ceil_out(arg1, out, self)
    @runtime_error_check ccall((:atg_ceil_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_celu(arg1, self)
    @runtime_error_check ccall((:atg_celu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_celu_(arg1, self)
    @runtime_error_check ccall((:atg_celu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_chain_matmul(arg1, matrices_data, matrices_len)
    @runtime_error_check ccall((:atg_chain_matmul, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, matrices_data, matrices_len)
end

function atg_chain_matmul_out(arg1, out, matrices_data, matrices_len)
    @runtime_error_check ccall((:atg_chain_matmul_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, matrices_data, matrices_len)
end

function atg_channel_shuffle(arg1, self, groups)
    @runtime_error_check ccall((:atg_channel_shuffle, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, groups)
end

function atg_cholesky(arg1, self, upper)
    @runtime_error_check ccall((:atg_cholesky, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, upper)
end

function atg_cholesky_inverse(arg1, self, upper)
    @runtime_error_check ccall((:atg_cholesky_inverse, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, upper)
end

function atg_cholesky_inverse_out(arg1, out, self, upper)
    @runtime_error_check ccall((:atg_cholesky_inverse_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, upper)
end

function atg_cholesky_out(arg1, out, self, upper)
    @runtime_error_check ccall((:atg_cholesky_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, upper)
end

function atg_cholesky_solve(arg1, self, input2, upper)
    @runtime_error_check ccall((:atg_cholesky_solve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, input2, upper)
end

function atg_cholesky_solve_out(arg1, out, self, input2, upper)
    @runtime_error_check ccall((:atg_cholesky_solve_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, self, input2, upper)
end

function atg_choose_qparams_optimized(arg1, input, numel, n_bins, ratio, bit_width)
    @runtime_error_check ccall((:atg_choose_qparams_optimized, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cdouble, Int64), arg1, input, numel, n_bins, ratio, bit_width)
end

function atg_chunk(arg1, self, chunks, dim)
    @runtime_error_check ccall((:atg_chunk, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, chunks, dim)
end

function atg_clamp(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clamp_(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clamp_max(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, max)
end

function atg_clamp_max_(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, max)
end

function atg_clamp_max_out(arg1, out, self, max)
    @runtime_error_check ccall((:atg_clamp_max_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, max)
end

function atg_clamp_max_tensor(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, max)
end

function atg_clamp_max_tensor_(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, max)
end

function atg_clamp_max_tensor_out(arg1, out, self, max)
    @runtime_error_check ccall((:atg_clamp_max_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, max)
end

function atg_clamp_min(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, min)
end

function atg_clamp_min_(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, min)
end

function atg_clamp_min_out(arg1, out, self, min)
    @runtime_error_check ccall((:atg_clamp_min_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, min)
end

function atg_clamp_min_tensor(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, min)
end

function atg_clamp_min_tensor_(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, min)
end

function atg_clamp_min_tensor_out(arg1, out, self, min)
    @runtime_error_check ccall((:atg_clamp_min_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, min)
end

function atg_clamp_out(arg1, out, self, min, max)
    @runtime_error_check ccall((:atg_clamp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, out, self, min, max)
end

function atg_clamp_tensor(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, min, max)
end

function atg_clamp_tensor_(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, min, max)
end

function atg_clamp_tensor_out(arg1, out, self, min, max)
    @runtime_error_check ccall((:atg_clamp_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, min, max)
end

function atg_clip(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clip, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clip_(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clip_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clip_out(arg1, out, self, min, max)
    @runtime_error_check ccall((:atg_clip_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, out, self, min, max)
end

function atg_clip_tensor(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clip_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, min, max)
end

function atg_clip_tensor_(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clip_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, min, max)
end

function atg_clip_tensor_out(arg1, out, self, min, max)
    @runtime_error_check ccall((:atg_clip_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, min, max)
end

function atg_clone(arg1, self)
    @runtime_error_check ccall((:atg_clone, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_coalesce(arg1, self)
    @runtime_error_check ccall((:atg_coalesce, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_col2im(arg1, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_backward(arg1, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_backward_grad_input(arg1, grad_input, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_out(arg1, out, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col_indices(arg1, self)
    @runtime_error_check ccall((:atg_col_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_column_stack(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_column_stack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_column_stack_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_column_stack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_combinations(arg1, self, r, with_replacement)
    @runtime_error_check ccall((:atg_combinations, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, r, with_replacement)
end

function atg_complex(arg1, real, imag)
    @runtime_error_check ccall((:atg_complex, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, real, imag)
end

function atg_complex_out(arg1, out, real, imag)
    @runtime_error_check ccall((:atg_complex_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, real, imag)
end

function atg_concat(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_concat, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg_concat_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_concat_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg_conj(arg1, self)
    @runtime_error_check ccall((:atg_conj, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conj_physical(arg1, self)
    @runtime_error_check ccall((:atg_conj_physical, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conj_physical_(arg1, self)
    @runtime_error_check ccall((:atg_conj_physical_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conj_physical_out(arg1, out, self)
    @runtime_error_check ccall((:atg_conj_physical_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_constant_pad_nd(arg1, self, pad_data, pad_len)
    @runtime_error_check ccall((:atg_constant_pad_nd, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, pad_data, pad_len)
end

function atg_contiguous(arg1, self)
    @runtime_error_check ccall((:atg_contiguous, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conv1d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv1d_padding(arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv1d_padding, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Cchar}, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
end

function atg_conv2d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv2d_padding(arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv2d_padding, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Cchar}, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
end

function atg_conv3d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv3d_padding(arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv3d_padding, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Cchar}, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding, dilation_data, dilation_len, groups)
end

function atg_conv_depthwise3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_depthwise3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_conv_depthwise3d_backward(arg1, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_depthwise3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_conv_tbc(arg1, self, weight, bias, pad)
    @runtime_error_check ccall((:atg_conv_tbc, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, self, weight, bias, pad)
end

function atg_conv_tbc_backward(arg1, self, input, weight, bias, pad)
    @runtime_error_check ccall((:atg_conv_tbc_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, self, input, weight, bias, pad)
end

function atg_conv_transpose1d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_conv_transpose2d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_conv_transpose3d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_convolution(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
    @runtime_error_check ccall((:atg_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
end

function atg_convolution_overrideable(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
    @runtime_error_check ccall((:atg_convolution_overrideable, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
end

function atg_copy_sparse_to_sparse_(arg1, self, src, non_blocking)
    @runtime_error_check ccall((:atg_copy_sparse_to_sparse_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, src, non_blocking)
end

function atg_copysign(arg1, self, other)
    @runtime_error_check ccall((:atg_copysign, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_copysign_(arg1, self, other)
    @runtime_error_check ccall((:atg_copysign_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_copysign_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_copysign_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_copysign_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_copysign_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_copysign_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_copysign_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_copysign_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_copysign_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_corrcoef(arg1, self)
    @runtime_error_check ccall((:atg_corrcoef, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cos(arg1, self)
    @runtime_error_check ccall((:atg_cos, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cos_(arg1, self)
    @runtime_error_check ccall((:atg_cos_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cos_out(arg1, out, self)
    @runtime_error_check ccall((:atg_cos_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_cosh(arg1, self)
    @runtime_error_check ccall((:atg_cosh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cosh_(arg1, self)
    @runtime_error_check ccall((:atg_cosh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cosh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_cosh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_cosine_embedding_loss(arg1, input1, input2, target, margin, reduction)
    @runtime_error_check ccall((:atg_cosine_embedding_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Int64), arg1, input1, input2, target, margin, reduction)
end

function atg_cosine_similarity(arg1, x1, x2, dim, eps)
    @runtime_error_check ccall((:atg_cosine_similarity, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cdouble), arg1, x1, x2, dim, eps)
end

function atg_cov(arg1, self, correction, fweights, aweights)
    @runtime_error_check ccall((:atg_cov, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, correction, fweights, aweights)
end

function atg_cross(arg1, self, other, dim)
    @runtime_error_check ccall((:atg_cross, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, other, dim)
end

function atg_cross_entropy_loss(arg1, self, target, weight, reduction, ignore_index, label_smoothing)
    @runtime_error_check ccall((:atg_cross_entropy_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cdouble), arg1, self, target, weight, reduction, ignore_index, label_smoothing)
end

function atg_cross_out(arg1, out, self, other, dim)
    @runtime_error_check ccall((:atg_cross_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, other, dim)
end

function atg_crow_indices(arg1, self)
    @runtime_error_check ccall((:atg_crow_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ctc_loss(arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, reduction, zero_infinity)
    @runtime_error_check ccall((:atg_ctc_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Int64, Cint), arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, reduction, zero_infinity)
end

function atg_ctc_loss_tensor(arg1, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
    @runtime_error_check ccall((:atg_ctc_loss_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, Cint), arg1, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
end

function atg_cudnn_affine_grid_generator(arg1, theta, n, C, H, W)
    @runtime_error_check ccall((:atg_cudnn_affine_grid_generator, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, theta, n, C, H, W)
end

function atg_cudnn_affine_grid_generator_backward(arg1, grad, n, C, H, W)
    @runtime_error_check ccall((:atg_cudnn_affine_grid_generator_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, grad, n, C, H, W)
end

function atg_cudnn_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    @runtime_error_check ccall((:atg_cudnn_batch_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
end

function atg_cudnn_batch_norm_backward(arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace)
    @runtime_error_check ccall((:atg_cudnn_batch_norm_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble, tensor), arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace)
end

function atg_cudnn_convolution(arg1, self, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, self, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_add_relu(arg1, self, weight, z, alpha, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_cudnn_convolution_add_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, weight, z, alpha, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_cudnn_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_deprecated(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_deprecated, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_deprecated2(arg1, self, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_deprecated2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_relu(arg1, self, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_cudnn_convolution_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_cudnn_convolution_transpose(arg1, self, weight, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, self, weight, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_transpose_backward_input(arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_transpose_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic, allow_tf32)
end

function atg_cudnn_convolution_transpose_deprecated(arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_deprecated, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_transpose_deprecated2(arg1, self, weight, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_deprecated2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_grid_sampler(arg1, self, grid)
    @runtime_error_check ccall((:atg_cudnn_grid_sampler, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, grid)
end

function atg_cudnn_grid_sampler_backward(arg1, self, grid, grad_output)
    @runtime_error_check ccall((:atg_cudnn_grid_sampler_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, grid, grad_output)
end

function atg_cummax(arg1, self, dim)
    @runtime_error_check ccall((:atg_cummax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_cummax_out(arg1, values, indices, self, dim)
    @runtime_error_check ccall((:atg_cummax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, values, indices, self, dim)
end

function atg_cummaxmin_backward(arg1, grad, input, indices, dim)
    @runtime_error_check ccall((:atg_cummaxmin_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad, input, indices, dim)
end

function atg_cummin(arg1, self, dim)
    @runtime_error_check ccall((:atg_cummin, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_cummin_out(arg1, values, indices, self, dim)
    @runtime_error_check ccall((:atg_cummin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, values, indices, self, dim)
end

function atg_cumprod(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumprod, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumprod_(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumprod_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumprod_backward(arg1, grad, input, dim, output)
    @runtime_error_check ccall((:atg_cumprod_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, grad, input, dim, output)
end

function atg_cumprod_out(arg1, out, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumprod_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, dtype)
end

function atg_cumsum(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumsum, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumsum_(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumsum_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumsum_out(arg1, out, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumsum_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, dtype)
end

function atg_cumulative_trapezoid(arg1, y, dim)
    @runtime_error_check ccall((:atg_cumulative_trapezoid, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, y, dim)
end

function atg_cumulative_trapezoid_x(arg1, y, x, dim)
    @runtime_error_check ccall((:atg_cumulative_trapezoid_x, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, y, x, dim)
end

function atg_data(arg1, self)
    @runtime_error_check ccall((:atg_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_deg2rad(arg1, self)
    @runtime_error_check ccall((:atg_deg2rad, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_deg2rad_(arg1, self)
    @runtime_error_check ccall((:atg_deg2rad_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_deg2rad_out(arg1, out, self)
    @runtime_error_check ccall((:atg_deg2rad_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_dequantize(arg1, self)
    @runtime_error_check ccall((:atg_dequantize, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_dequantize_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_dequantize_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_det(arg1, self)
    @runtime_error_check ccall((:atg_det, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_detach(arg1, self)
    @runtime_error_check ccall((:atg_detach, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_detach_(arg1, self)
    @runtime_error_check ccall((:atg_detach_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_diag(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_diag, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_diag_backward(arg1, grad, input_sizes_data, input_sizes_len, diagonal)
    @runtime_error_check ccall((:atg_diag_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64), arg1, grad, input_sizes_data, input_sizes_len, diagonal)
end

function atg_diag_embed(arg1, self, offset, dim1, dim2)
    @runtime_error_check ccall((:atg_diag_embed, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, offset, dim1, dim2)
end

function atg_diag_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_diag_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_diagflat(arg1, self, offset)
    @runtime_error_check ccall((:atg_diagflat, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, offset)
end

function atg_diagonal(arg1, self, offset, dim1, dim2)
    @runtime_error_check ccall((:atg_diagonal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, offset, dim1, dim2)
end

function atg_diagonal_backward(arg1, grad_output, input_sizes_data, input_sizes_len, offset, dim1, dim2)
    @runtime_error_check ccall((:atg_diagonal_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64, Int64), arg1, grad_output, input_sizes_data, input_sizes_len, offset, dim1, dim2)
end

function atg_diff(arg1, self, n, dim, prepend, append)
    @runtime_error_check ccall((:atg_diff, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, tensor, tensor), arg1, self, n, dim, prepend, append)
end

function atg_diff_out(arg1, out, self, n, dim, prepend, append)
    @runtime_error_check ccall((:atg_diff_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, tensor, tensor), arg1, out, self, n, dim, prepend, append)
end

function atg_digamma(arg1, self)
    @runtime_error_check ccall((:atg_digamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_digamma_(arg1, self)
    @runtime_error_check ccall((:atg_digamma_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_digamma_out(arg1, out, self)
    @runtime_error_check ccall((:atg_digamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_dist(arg1, self, other)
    @runtime_error_check ccall((:atg_dist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div(arg1, self, other)
    @runtime_error_check ccall((:atg_div, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div_(arg1, self, other)
    @runtime_error_check ccall((:atg_div_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_div_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_div_out_mode(arg1, out, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_div_out_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Cchar}), arg1, out, self, other, rounding_mode)
end

function atg_div_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_div_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_div_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_div_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_div_scalar_mode(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_div_scalar_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_div_scalar_mode_(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_div_scalar_mode_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_div_tensor_mode(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_div_tensor_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_div_tensor_mode_(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_div_tensor_mode_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_divide(arg1, self, other)
    @runtime_error_check ccall((:atg_divide, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_divide_(arg1, self, other)
    @runtime_error_check ccall((:atg_divide_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_divide_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_divide_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_divide_out_mode(arg1, out, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_divide_out_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Cchar}), arg1, out, self, other, rounding_mode)
end

function atg_divide_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_divide_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_divide_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_divide_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_divide_scalar_mode(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_divide_scalar_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_divide_scalar_mode_(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_divide_scalar_mode_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_divide_tensor_mode(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_divide_tensor_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_divide_tensor_mode_(arg1, self, other, rounding_mode)
    @runtime_error_check ccall((:atg_divide_tensor_mode_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, self, other, rounding_mode)
end

function atg_dot(arg1, self, tensor_)
    @runtime_error_check ccall((:atg_dot, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, tensor_)
end

function atg_dot_out(arg1, out, self, tensor_)
    @runtime_error_check ccall((:atg_dot_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, tensor_)
end

function atg_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_dropout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_dropout_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_dsplit(arg1, self, sections)
    @runtime_error_check ccall((:atg_dsplit, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, sections)
end

function atg_dsplit_array(arg1, self, indices_data, indices_len)
    @runtime_error_check ccall((:atg_dsplit_array, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, indices_data, indices_len)
end

function atg_dstack(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_dstack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_dstack_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_dstack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_eig(arg1, self, eigenvectors)
    @runtime_error_check ccall((:atg_eig, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, eigenvectors)
end

function atg_eig_e(arg1, e, v, self, eigenvectors)
    @runtime_error_check ccall((:atg_eig_e, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, e, v, self, eigenvectors)
end

function atg_einsum(arg1, equation, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_einsum, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cchar}, Ptr{tensor}, Cint), arg1, equation, tensors_data, tensors_len)
end

function atg_elu(arg1, self)
    @runtime_error_check ccall((:atg_elu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_elu_(arg1, self)
    @runtime_error_check ccall((:atg_elu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_elu_backward(arg1, grad_output, alpha, scale, input_scale, is_result, self_or_result)
    @runtime_error_check ccall((:atg_elu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar, scalar, Cint, tensor), arg1, grad_output, alpha, scale, input_scale, is_result, self_or_result)
end

function atg_elu_backward_grad_input(arg1, grad_input, grad_output, alpha, scale, input_scale, is_result, self_or_result)
    @runtime_error_check ccall((:atg_elu_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar, scalar, Cint, tensor), arg1, grad_input, grad_output, alpha, scale, input_scale, is_result, self_or_result)
end

function atg_elu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_elu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_embedding(arg1, weight, indices, padding_idx, scale_grad_by_freq, sparse)
    @runtime_error_check ccall((:atg_embedding, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Cint), arg1, weight, indices, padding_idx, scale_grad_by_freq, sparse)
end

function atg_embedding_backward(arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
    @runtime_error_check ccall((:atg_embedding_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint, Cint), arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
end

function atg_embedding_bag(arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
    @runtime_error_check ccall((:atg_embedding_bag, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint, tensor, Cint), arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
end

function atg_embedding_bag_padding_idx(arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
    @runtime_error_check ccall((:atg_embedding_bag_padding_idx, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint, tensor, Cint, Int64), arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
end

function atg_embedding_dense_backward(arg1, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
    @runtime_error_check ccall((:atg_embedding_dense_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
end

function atg_embedding_renorm_(arg1, self, indices, max_norm, norm_type)
    @runtime_error_check ccall((:atg_embedding_renorm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble), arg1, self, indices, max_norm, norm_type)
end

function atg_embedding_sparse_backward(arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq)
    @runtime_error_check ccall((:atg_embedding_sparse_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq)
end

function atg_empty(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_empty_like(arg1, self)
    @runtime_error_check ccall((:atg_empty_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_empty_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_empty_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_empty_quantized(arg1, size_data, size_len, qtensor, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty_quantized, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, Cint, Cint), arg1, size_data, size_len, qtensor, options_kind, options_device)
end

function atg_empty_strided(arg1, size_data, size_len, stride_data, stride_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty_strided, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, stride_data, stride_len, options_kind, options_device)
end

function atg_eq(arg1, self, other)
    @runtime_error_check ccall((:atg_eq, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_eq_(arg1, self, other)
    @runtime_error_check ccall((:atg_eq_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_eq_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_eq_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_eq_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_eq_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_eq_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_eq_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_eq_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_eq_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_erf(arg1, self)
    @runtime_error_check ccall((:atg_erf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erf_(arg1, self)
    @runtime_error_check ccall((:atg_erf_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erf_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erf_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_erfc(arg1, self)
    @runtime_error_check ccall((:atg_erfc, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfc_(arg1, self)
    @runtime_error_check ccall((:atg_erfc_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erfc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_erfinv(arg1, self)
    @runtime_error_check ccall((:atg_erfinv, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfinv_(arg1, self)
    @runtime_error_check ccall((:atg_erfinv_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfinv_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erfinv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_exp(arg1, self)
    @runtime_error_check ccall((:atg_exp, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp2(arg1, self)
    @runtime_error_check ccall((:atg_exp2, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp2_(arg1, self)
    @runtime_error_check ccall((:atg_exp2_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp2_out(arg1, out, self)
    @runtime_error_check ccall((:atg_exp2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_exp_(arg1, self)
    @runtime_error_check ccall((:atg_exp_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp_out(arg1, out, self)
    @runtime_error_check ccall((:atg_exp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_expand(arg1, self, size_data, size_len, implicit)
    @runtime_error_check ccall((:atg_expand, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, size_data, size_len, implicit)
end

function atg_expand_as(arg1, self, other)
    @runtime_error_check ccall((:atg_expand_as, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_expm1(arg1, self)
    @runtime_error_check ccall((:atg_expm1, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_expm1_(arg1, self)
    @runtime_error_check ccall((:atg_expm1_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_expm1_out(arg1, out, self)
    @runtime_error_check ccall((:atg_expm1_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_exponential_(arg1, self, lambd)
    @runtime_error_check ccall((:atg_exponential_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, lambd)
end

function atg_eye(arg1, n, options_kind, options_device)
    @runtime_error_check ccall((:atg_eye, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, n, options_kind, options_device)
end

function atg_eye_m(arg1, n, m, options_kind, options_device)
    @runtime_error_check ccall((:atg_eye_m, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Cint, Cint), arg1, n, m, options_kind, options_device)
end

function atg_eye_m_out(arg1, out, n, m)
    @runtime_error_check ccall((:atg_eye_m_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, out, n, m)
end

function atg_eye_out(arg1, out, n)
    @runtime_error_check ccall((:atg_eye_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, out, n)
end

function atg_fake_quantize_per_channel_affine(arg1, self, scale, zero_point, axis, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_channel_affine, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Int64), arg1, self, scale, zero_point, axis, quant_min, quant_max)
end

function atg_fake_quantize_per_channel_affine_cachemask(arg1, self, scale, zero_point, axis, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_channel_affine_cachemask, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Int64), arg1, self, scale, zero_point, axis, quant_min, quant_max)
end

function atg_fake_quantize_per_channel_affine_cachemask_backward(arg1, grad, mask)
    @runtime_error_check ccall((:atg_fake_quantize_per_channel_affine_cachemask_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, mask)
end

function atg_fake_quantize_per_tensor_affine(arg1, self, scale, zero_point, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Int64, Int64), arg1, self, scale, zero_point, quant_min, quant_max)
end

function atg_fake_quantize_per_tensor_affine_cachemask(arg1, self, scale, zero_point, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine_cachemask, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Int64, Int64), arg1, self, scale, zero_point, quant_min, quant_max)
end

function atg_fake_quantize_per_tensor_affine_cachemask_backward(arg1, grad, mask)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine_cachemask_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, mask)
end

function atg_fake_quantize_per_tensor_affine_tensor_qparams(arg1, self, scale, zero_point, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine_tensor_qparams, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, scale, zero_point, quant_min, quant_max)
end

function atg_fbgemm_linear_fp16_weight(arg1, input, packed_weight, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_fp16_weight, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, packed_weight, bias)
end

function atg_fbgemm_linear_fp16_weight_fp32_activation(arg1, input, packed_weight, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_fp16_weight_fp32_activation, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, packed_weight, bias)
end

function atg_fbgemm_linear_int8_weight(arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_int8_weight, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor), arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
end

function atg_fbgemm_linear_int8_weight_fp32_activation(arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_int8_weight_fp32_activation, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor), arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
end

function atg_fbgemm_pack_gemm_matrix_fp16(arg1, input)
    @runtime_error_check ccall((:atg_fbgemm_pack_gemm_matrix_fp16, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, input)
end

function atg_fbgemm_pack_quantized_matrix(arg1, input)
    @runtime_error_check ccall((:atg_fbgemm_pack_quantized_matrix, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, input)
end

function atg_fbgemm_pack_quantized_matrix_kn(arg1, input, K, n)
    @runtime_error_check ccall((:atg_fbgemm_pack_quantized_matrix_kn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, input, K, n)
end

function atg_feature_alpha_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_feature_alpha_dropout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_feature_alpha_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_feature_alpha_dropout_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_feature_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_feature_dropout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_feature_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_feature_dropout_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_fft_fft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_fft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_fft2(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_fft2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_fft2_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_fft2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_fft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_fft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_fftfreq(arg1, n, d, options_kind, options_device)
    @runtime_error_check ccall((:atg_fft_fftfreq, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cdouble, Cint, Cint), arg1, n, d, options_kind, options_device)
end

function atg_fft_fftfreq_out(arg1, out, n, d)
    @runtime_error_check ccall((:atg_fft_fftfreq_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cdouble), arg1, out, n, d)
end

function atg_fft_fftn(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_fftn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_fftn_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_fftn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_fftshift(arg1, self, dim_data, dim_len)
    @runtime_error_check ccall((:atg_fft_fftshift, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dim_data, dim_len)
end

function atg_fft_hfft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_hfft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_hfft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_hfft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_ifft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_ifft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_ifft2(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_ifft2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_ifft2_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_ifft2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_ifft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_ifft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_ifftn(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_ifftn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_ifftn_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_ifftn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_ifftshift(arg1, self, dim_data, dim_len)
    @runtime_error_check ccall((:atg_fft_ifftshift, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dim_data, dim_len)
end

function atg_fft_ihfft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_ihfft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_ihfft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_ihfft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_irfft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_irfft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_irfft2(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_irfft2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_irfft2_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_irfft2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_irfft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_irfft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_irfftn(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_irfftn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_irfftn_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_irfftn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_rfft(arg1, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_rfft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Cchar}), arg1, self, n, dim, norm)
end

function atg_fft_rfft2(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_rfft2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_rfft2_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_rfft2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_rfft_out(arg1, out, self, n, dim, norm)
    @runtime_error_check ccall((:atg_fft_rfft_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Ptr{Cchar}), arg1, out, self, n, dim, norm)
end

function atg_fft_rfftfreq(arg1, n, d, options_kind, options_device)
    @runtime_error_check ccall((:atg_fft_rfftfreq, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cdouble, Cint, Cint), arg1, n, d, options_kind, options_device)
end

function atg_fft_rfftfreq_out(arg1, out, n, d)
    @runtime_error_check ccall((:atg_fft_rfftfreq_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cdouble), arg1, out, n, d)
end

function atg_fft_rfftn(arg1, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_rfftn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fft_rfftn_out(arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
    @runtime_error_check ccall((:atg_fft_rfftn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Cchar}), arg1, out, self, s_data, s_len, dim_data, dim_len, norm)
end

function atg_fill_(arg1, self, value)
    @runtime_error_check ccall((:atg_fill_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, value)
end

function atg_fill_diagonal_(arg1, self, fill_value, wrap)
    @runtime_error_check ccall((:atg_fill_diagonal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Cint), arg1, self, fill_value, wrap)
end

function atg_fill_tensor_(arg1, self, value)
    @runtime_error_check ccall((:atg_fill_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, value)
end

function atg_fix(arg1, self)
    @runtime_error_check ccall((:atg_fix, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_fix_(arg1, self)
    @runtime_error_check ccall((:atg_fix_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_fix_out(arg1, out, self)
    @runtime_error_check ccall((:atg_fix_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_flatten(arg1, self, start_dim, end_dim)
    @runtime_error_check ccall((:atg_flatten, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, start_dim, end_dim)
end

function atg_flatten_dense_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_flatten_dense_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_flip(arg1, self, dims_data, dims_len)
    @runtime_error_check ccall((:atg_flip, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dims_data, dims_len)
end

function atg_fliplr(arg1, self)
    @runtime_error_check ccall((:atg_fliplr, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_flipud(arg1, self)
    @runtime_error_check ccall((:atg_flipud, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_float_power(arg1, self, exponent)
    @runtime_error_check ccall((:atg_float_power, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_float_power_(arg1, self, exponent)
    @runtime_error_check ccall((:atg_float_power_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_float_power_scalar(arg1, self, exponent)
    @runtime_error_check ccall((:atg_float_power_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, exponent)
end

function atg_float_power_scalar_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_float_power_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, exponent)
end

function atg_float_power_tensor_(arg1, self, exponent)
    @runtime_error_check ccall((:atg_float_power_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_float_power_tensor_scalar(arg1, self, exponent)
    @runtime_error_check ccall((:atg_float_power_tensor_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_float_power_tensor_scalar_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_float_power_tensor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, exponent)
end

function atg_float_power_tensor_tensor_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_float_power_tensor_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, exponent)
end

function atg_floor(arg1, self)
    @runtime_error_check ccall((:atg_floor, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_floor_(arg1, self)
    @runtime_error_check ccall((:atg_floor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_floor_divide(arg1, self, other)
    @runtime_error_check ccall((:atg_floor_divide, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_floor_divide_(arg1, self, other)
    @runtime_error_check ccall((:atg_floor_divide_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_floor_divide_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_floor_divide_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_floor_divide_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_floor_divide_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_floor_divide_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_floor_divide_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_floor_out(arg1, out, self)
    @runtime_error_check ccall((:atg_floor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_fmax(arg1, self, other)
    @runtime_error_check ccall((:atg_fmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmax_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmax_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_fmin(arg1, self, other)
    @runtime_error_check ccall((:atg_fmin, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmin_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_fmod(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_fmod_(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_fmod_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmod_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_fmod_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmod_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmod_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmod_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_frac(arg1, self)
    @runtime_error_check ccall((:atg_frac, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frac_(arg1, self)
    @runtime_error_check ccall((:atg_frac_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frac_out(arg1, out, self)
    @runtime_error_check ccall((:atg_frac_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_fractional_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool2d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool2d_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool2d_output(arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_output, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool3d(arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool3d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool3d_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool3d_output(arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_output, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_frexp(arg1, self)
    @runtime_error_check ccall((:atg_frexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frexp_tensor_out(arg1, mantissa, exponent, self)
    @runtime_error_check ccall((:atg_frexp_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, mantissa, exponent, self)
end

function atg_frobenius_norm(arg1, self)
    @runtime_error_check ccall((:atg_frobenius_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frobenius_norm_dim(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_frobenius_norm_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_frobenius_norm_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_frobenius_norm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_from_file(arg1, filename, shared, size, options_kind, options_device)
    @runtime_error_check ccall((:atg_from_file, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Cchar}, Cint, Int64, Cint, Cint), arg1, filename, shared, size, options_kind, options_device)
end

function atg_full(arg1, size_data, size_len, fill_value, options_kind, options_device)
    @runtime_error_check ccall((:atg_full, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, scalar, Cint, Cint), arg1, size_data, size_len, fill_value, options_kind, options_device)
end

function atg_full_like(arg1, self, fill_value)
    @runtime_error_check ccall((:atg_full_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, fill_value)
end

function atg_full_out(arg1, out, size_data, size_len, fill_value)
    @runtime_error_check ccall((:atg_full_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, scalar), arg1, out, size_data, size_len, fill_value)
end

function atg_fused_moving_avg_obs_fake_quant(arg1, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)
    @runtime_error_check ccall((:atg_fused_moving_avg_obs_fake_quant, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble, Int64, Int64, Int64, Cint, Cint), arg1, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)
end

function atg_gather(arg1, self, dim, index, sparse_grad)
    @runtime_error_check ccall((:atg_gather, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, Cint), arg1, self, dim, index, sparse_grad)
end

function atg_gather_backward(arg1, grad, self, dim, index, sparse_grad)
    @runtime_error_check ccall((:atg_gather_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, Cint), arg1, grad, self, dim, index, sparse_grad)
end

function atg_gather_out(arg1, out, self, dim, index, sparse_grad)
    @runtime_error_check ccall((:atg_gather_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, Cint), arg1, out, self, dim, index, sparse_grad)
end

function atg_gcd(arg1, self, other)
    @runtime_error_check ccall((:atg_gcd, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gcd_(arg1, self, other)
    @runtime_error_check ccall((:atg_gcd_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gcd_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_gcd_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_ge(arg1, self, other)
    @runtime_error_check ccall((:atg_ge, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ge_(arg1, self, other)
    @runtime_error_check ccall((:atg_ge_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ge_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ge_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_ge_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_ge_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ge_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_ge_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ge_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ge_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_gelu(arg1, self)
    @runtime_error_check ccall((:atg_gelu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_gelu_backward(arg1, grad, self)
    @runtime_error_check ccall((:atg_gelu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, self)
end

function atg_gelu_backward_grad_input(arg1, grad_input, grad, self)
    @runtime_error_check ccall((:atg_gelu_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad, self)
end

function atg_gelu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_gelu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_geometric_(arg1, self, p)
    @runtime_error_check ccall((:atg_geometric_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_geqrf(arg1, self)
    @runtime_error_check ccall((:atg_geqrf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_geqrf_a(arg1, a, tau, self)
    @runtime_error_check ccall((:atg_geqrf_a, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, a, tau, self)
end

function atg_ger(arg1, self, vec2)
    @runtime_error_check ccall((:atg_ger, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, vec2)
end

function atg_ger_out(arg1, out, self, vec2)
    @runtime_error_check ccall((:atg_ger_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, vec2)
end

function atg_glu(arg1, self, dim)
    @runtime_error_check ccall((:atg_glu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_glu_backward(arg1, grad_output, self, dim)
    @runtime_error_check ccall((:atg_glu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, grad_output, self, dim)
end

function atg_glu_backward_grad_input(arg1, grad_input, grad_output, self, dim)
    @runtime_error_check ccall((:atg_glu_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, dim)
end

function atg_glu_out(arg1, out, self, dim)
    @runtime_error_check ccall((:atg_glu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, dim)
end

function atg_grad(arg1, self)
    @runtime_error_check ccall((:atg_grad, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_greater(arg1, self, other)
    @runtime_error_check ccall((:atg_greater, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_greater_(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_greater_equal(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_equal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_greater_equal_(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_equal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_greater_equal_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_greater_equal_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_greater_equal_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_equal_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_greater_equal_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_equal_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_greater_equal_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_greater_equal_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_greater_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_greater_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_greater_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_greater_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_greater_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_greater_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_greater_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_grid_sampler(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_2d(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_2d_backward(arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_3d(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_3d_backward(arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_group_norm(arg1, input, num_groups, weight, bias, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_group_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, Cdouble, Cint), arg1, input, num_groups, weight, bias, eps, cudnn_enabled)
end

function atg_gru(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_gru, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_gru_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_gru_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_gru_data(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_gru_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_gt(arg1, self, other)
    @runtime_error_check ccall((:atg_gt, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_gt_(arg1, self, other)
    @runtime_error_check ccall((:atg_gt_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_gt_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_gt_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_gt_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_gt_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gt_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_gt_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gt_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_gt_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_hamming_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_hamming_window_periodic(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window_periodic, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_hamming_window_periodic_alpha(arg1, window_length, periodic, alpha, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window_periodic_alpha, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cdouble, Cint, Cint), arg1, window_length, periodic, alpha, options_kind, options_device)
end

function atg_hamming_window_periodic_alpha_beta(arg1, window_length, periodic, alpha, beta, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window_periodic_alpha_beta, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cdouble, Cdouble, Cint, Cint), arg1, window_length, periodic, alpha, beta, options_kind, options_device)
end

function atg_hann_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_hann_window, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_hann_window_periodic(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_hann_window_periodic, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_hardshrink(arg1, self)
    @runtime_error_check ccall((:atg_hardshrink, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardshrink_backward(arg1, grad_out, self, lambd)
    @runtime_error_check ccall((:atg_hardshrink_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_out, self, lambd)
end

function atg_hardshrink_backward_grad_input(arg1, grad_input, grad_out, self, lambd)
    @runtime_error_check ccall((:atg_hardshrink_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, grad_input, grad_out, self, lambd)
end

function atg_hardshrink_out(arg1, out, self)
    @runtime_error_check ccall((:atg_hardshrink_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_hardsigmoid(arg1, self)
    @runtime_error_check ccall((:atg_hardsigmoid, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardsigmoid_(arg1, self)
    @runtime_error_check ccall((:atg_hardsigmoid_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardsigmoid_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_hardsigmoid_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_hardsigmoid_backward_grad_input(arg1, grad_input, grad_output, self)
    @runtime_error_check ccall((:atg_hardsigmoid_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, self)
end

function atg_hardsigmoid_out(arg1, out, self)
    @runtime_error_check ccall((:atg_hardsigmoid_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_hardswish(arg1, self)
    @runtime_error_check ccall((:atg_hardswish, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardswish_(arg1, self)
    @runtime_error_check ccall((:atg_hardswish_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardswish_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_hardswish_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_hardswish_out(arg1, out, self)
    @runtime_error_check ccall((:atg_hardswish_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_hardtanh(arg1, self)
    @runtime_error_check ccall((:atg_hardtanh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardtanh_(arg1, self)
    @runtime_error_check ccall((:atg_hardtanh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardtanh_backward(arg1, grad_output, self, min_val, max_val)
    @runtime_error_check ccall((:atg_hardtanh_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, grad_output, self, min_val, max_val)
end

function atg_hardtanh_backward_grad_input(arg1, grad_input, grad_output, self, min_val, max_val)
    @runtime_error_check ccall((:atg_hardtanh_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar), arg1, grad_input, grad_output, self, min_val, max_val)
end

function atg_hardtanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_hardtanh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_heaviside(arg1, self, values)
    @runtime_error_check ccall((:atg_heaviside, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, values)
end

function atg_heaviside_(arg1, self, values)
    @runtime_error_check ccall((:atg_heaviside_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, values)
end

function atg_heaviside_out(arg1, out, self, values)
    @runtime_error_check ccall((:atg_heaviside_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, values)
end

function atg_hinge_embedding_loss(arg1, self, target, margin, reduction)
    @runtime_error_check ccall((:atg_hinge_embedding_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64), arg1, self, target, margin, reduction)
end

function atg_histc(arg1, self, bins)
    @runtime_error_check ccall((:atg_histc, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, bins)
end

function atg_histc_out(arg1, out, self, bins)
    @runtime_error_check ccall((:atg_histc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, bins)
end

function atg_hsplit(arg1, self, sections)
    @runtime_error_check ccall((:atg_hsplit, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, sections)
end

function atg_hsplit_array(arg1, self, indices_data, indices_len)
    @runtime_error_check ccall((:atg_hsplit_array, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, indices_data, indices_len)
end

function atg_hspmm(arg1, mat1, mat2)
    @runtime_error_check ccall((:atg_hspmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, mat1, mat2)
end

function atg_hspmm_out(arg1, out, mat1, mat2)
    @runtime_error_check ccall((:atg_hspmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, mat1, mat2)
end

function atg_hstack(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_hstack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_hstack_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_hstack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_huber_loss(arg1, self, target, reduction, delta)
    @runtime_error_check ccall((:atg_huber_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cdouble), arg1, self, target, reduction, delta)
end

function atg_huber_loss_backward(arg1, grad_output, self, target, reduction, delta)
    @runtime_error_check ccall((:atg_huber_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cdouble), arg1, grad_output, self, target, reduction, delta)
end

function atg_huber_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction, delta)
    @runtime_error_check ccall((:atg_huber_loss_backward_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Cdouble), arg1, grad_input, grad_output, self, target, reduction, delta)
end

function atg_huber_loss_out(arg1, out, self, target, reduction, delta)
    @runtime_error_check ccall((:atg_huber_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cdouble), arg1, out, self, target, reduction, delta)
end

function atg_hypot(arg1, self, other)
    @runtime_error_check ccall((:atg_hypot, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_hypot_(arg1, self, other)
    @runtime_error_check ccall((:atg_hypot_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_hypot_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_hypot_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_i0(arg1, self)
    @runtime_error_check ccall((:atg_i0, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_i0_(arg1, self)
    @runtime_error_check ccall((:atg_i0_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_i0_out(arg1, out, self)
    @runtime_error_check ccall((:atg_i0_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_igamma(arg1, self, other)
    @runtime_error_check ccall((:atg_igamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_igamma_(arg1, self, other)
    @runtime_error_check ccall((:atg_igamma_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_igamma_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_igamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_igammac(arg1, self, other)
    @runtime_error_check ccall((:atg_igammac, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_igammac_(arg1, self, other)
    @runtime_error_check ccall((:atg_igammac_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_igammac_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_igammac_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_im2col(arg1, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_backward(arg1, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_backward_grad_input(arg1, grad_input, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_out(arg1, out, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_imag(arg1, self)
    @runtime_error_check ccall((:atg_imag, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_index(arg1, self, indices_data, indices_len)
    @runtime_error_check ccall((:atg_index, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, self, indices_data, indices_len)
end

function atg_index_add(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_add, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_add_(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_add_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_add_alpha(arg1, self, dim, index, source, alpha)
    @runtime_error_check ccall((:atg_index_add_alpha, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, scalar), arg1, self, dim, index, source, alpha)
end

function atg_index_add_alpha_(arg1, self, dim, index, source, alpha)
    @runtime_error_check ccall((:atg_index_add_alpha_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, scalar), arg1, self, dim, index, source, alpha)
end

function atg_index_copy(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_copy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_copy_(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_copy_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_fill(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_index_fill_(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_index_fill_int_tensor(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill_int_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, value)
end

function atg_index_fill_int_tensor_(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill_int_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, value)
end

function atg_index_put(arg1, self, indices_data, indices_len, values, accumulate)
    @runtime_error_check ccall((:atg_index_put, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, Cint), arg1, self, indices_data, indices_len, values, accumulate)
end

function atg_index_put_(arg1, self, indices_data, indices_len, values, accumulate)
    @runtime_error_check ccall((:atg_index_put_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, Cint), arg1, self, indices_data, indices_len, values, accumulate)
end

function atg_index_select(arg1, self, dim, index)
    @runtime_error_check ccall((:atg_index_select, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor), arg1, self, dim, index)
end

function atg_index_select_backward(arg1, grad, self_sizes_data, self_sizes_len, dim, index)
    @runtime_error_check ccall((:atg_index_select_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, tensor), arg1, grad, self_sizes_data, self_sizes_len, dim, index)
end

function atg_index_select_out(arg1, out, self, dim, index)
    @runtime_error_check ccall((:atg_index_select_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, out, self, dim, index)
end

function atg_indices(arg1, self)
    @runtime_error_check ccall((:atg_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_infinitely_differentiable_gelu_backward(arg1, grad, self)
    @runtime_error_check ccall((:atg_infinitely_differentiable_gelu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, self)
end

function atg_inner(arg1, self, other)
    @runtime_error_check ccall((:atg_inner, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_inner_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_inner_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_instance_norm(arg1, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_instance_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble, Cint), arg1, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
end

function atg_int_repr(arg1, self)
    @runtime_error_check ccall((:atg_int_repr, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_inverse(arg1, self)
    @runtime_error_check ccall((:atg_inverse, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_inverse_out(arg1, out, self)
    @runtime_error_check ccall((:atg_inverse_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_isclose(arg1, self, other, rtol, atol, equal_nan)
    @runtime_error_check ccall((:atg_isclose, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble, Cint), arg1, self, other, rtol, atol, equal_nan)
end

function atg_isfinite(arg1, self)
    @runtime_error_check ccall((:atg_isfinite, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isin(arg1, elements, test_elements, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, elements, test_elements, assume_unique, invert)
end

function atg_isin_scalar_tensor(arg1, element, test_elements, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin_scalar_tensor, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor, Cint, Cint), arg1, element, test_elements, assume_unique, invert)
end

function atg_isin_scalar_tensor_out(arg1, out, element, test_elements, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin_scalar_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor, Cint, Cint), arg1, out, element, test_elements, assume_unique, invert)
end

function atg_isin_tensor_scalar(arg1, elements, test_element, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin_tensor_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Cint, Cint), arg1, elements, test_element, assume_unique, invert)
end

function atg_isin_tensor_scalar_out(arg1, out, elements, test_element, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin_tensor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, Cint, Cint), arg1, out, elements, test_element, assume_unique, invert)
end

function atg_isin_tensor_tensor_out(arg1, out, elements, test_elements, assume_unique, invert)
    @runtime_error_check ccall((:atg_isin_tensor_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, out, elements, test_elements, assume_unique, invert)
end

function atg_isinf(arg1, self)
    @runtime_error_check ccall((:atg_isinf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isnan(arg1, self)
    @runtime_error_check ccall((:atg_isnan, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isneginf(arg1, self)
    @runtime_error_check ccall((:atg_isneginf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isneginf_out(arg1, out, self)
    @runtime_error_check ccall((:atg_isneginf_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_isposinf(arg1, self)
    @runtime_error_check ccall((:atg_isposinf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isposinf_out(arg1, out, self)
    @runtime_error_check ccall((:atg_isposinf_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_isreal(arg1, self)
    @runtime_error_check ccall((:atg_isreal, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_istft(arg1, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex)
    @runtime_error_check ccall((:atg_istft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, tensor, Cint, Cint, Cint, Int64, Cint), arg1, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex)
end

function atg_kaiser_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_kaiser_window, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_kaiser_window_beta(arg1, window_length, periodic, beta, options_kind, options_device)
    @runtime_error_check ccall((:atg_kaiser_window_beta, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cdouble, Cint, Cint), arg1, window_length, periodic, beta, options_kind, options_device)
end

function atg_kaiser_window_periodic(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_kaiser_window_periodic, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_kl_div(arg1, self, target, reduction, log_target)
    @runtime_error_check ccall((:atg_kl_div, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, self, target, reduction, log_target)
end

function atg_kl_div_backward(arg1, grad_output, self, target, reduction, log_target)
    @runtime_error_check ccall((:atg_kl_div_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, grad_output, self, target, reduction, log_target)
end

function atg_kron(arg1, self, other)
    @runtime_error_check ccall((:atg_kron, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_kron_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_kron_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_kthvalue(arg1, self, k, dim, keepdim)
    @runtime_error_check ccall((:atg_kthvalue, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cint), arg1, self, k, dim, keepdim)
end

function atg_kthvalue_values(arg1, values, indices, self, k, dim, keepdim)
    @runtime_error_check ccall((:atg_kthvalue_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, values, indices, self, k, dim, keepdim)
end

function atg_l1_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_l1_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_l1_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_l1_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_layer_norm(arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps, cudnn_enable)
    @runtime_error_check ccall((:atg_layer_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, tensor, tensor, Cdouble, Cint), arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps, cudnn_enable)
end

function atg_lcm(arg1, self, other)
    @runtime_error_check ccall((:atg_lcm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lcm_(arg1, self, other)
    @runtime_error_check ccall((:atg_lcm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lcm_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_lcm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_ldexp(arg1, self, other)
    @runtime_error_check ccall((:atg_ldexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ldexp_(arg1, self, other)
    @runtime_error_check ccall((:atg_ldexp_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ldexp_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ldexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_le(arg1, self, other)
    @runtime_error_check ccall((:atg_le, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_le_(arg1, self, other)
    @runtime_error_check ccall((:atg_le_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_le_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_le_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_le_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_le_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_le_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_le_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_le_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_le_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_leaky_relu(arg1, self)
    @runtime_error_check ccall((:atg_leaky_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_leaky_relu_(arg1, self)
    @runtime_error_check ccall((:atg_leaky_relu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_leaky_relu_backward(arg1, grad_output, self, negative_slope, self_is_result)
    @runtime_error_check ccall((:atg_leaky_relu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, Cint), arg1, grad_output, self, negative_slope, self_is_result)
end

function atg_leaky_relu_backward_grad_input(arg1, grad_input, grad_output, self, negative_slope, self_is_result)
    @runtime_error_check ccall((:atg_leaky_relu_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, Cint), arg1, grad_input, grad_output, self, negative_slope, self_is_result)
end

function atg_leaky_relu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_leaky_relu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_lerp(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, _end, weight)
end

function atg_lerp_(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, _end, weight)
end

function atg_lerp_scalar_out(arg1, out, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, out, self, _end, weight)
end

function atg_lerp_tensor(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, _end, weight)
end

function atg_lerp_tensor_(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, _end, weight)
end

function atg_lerp_tensor_out(arg1, out, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, _end, weight)
end

function atg_less(arg1, self, other)
    @runtime_error_check ccall((:atg_less, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_less_(arg1, self, other)
    @runtime_error_check ccall((:atg_less_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_less_equal(arg1, self, other)
    @runtime_error_check ccall((:atg_less_equal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_less_equal_(arg1, self, other)
    @runtime_error_check ccall((:atg_less_equal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_less_equal_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_less_equal_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_less_equal_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_less_equal_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_less_equal_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_less_equal_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_less_equal_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_less_equal_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_less_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_less_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_less_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_less_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_less_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_less_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_less_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_less_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_lgamma(arg1, self)
    @runtime_error_check ccall((:atg_lgamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_lgamma_(arg1, self)
    @runtime_error_check ccall((:atg_lgamma_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_lgamma_out(arg1, out, self)
    @runtime_error_check ccall((:atg_lgamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_linalg_cholesky(arg1, self, upper)
    @runtime_error_check ccall((:atg_linalg_cholesky, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, upper)
end

function atg_linalg_cholesky_ex(arg1, self, upper, check_errors)
    @runtime_error_check ccall((:atg_linalg_cholesky_ex, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, upper, check_errors)
end

function atg_linalg_cholesky_ex_l(arg1, L, info, self, upper, check_errors)
    @runtime_error_check ccall((:atg_linalg_cholesky_ex_l, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, L, info, self, upper, check_errors)
end

function atg_linalg_cholesky_out(arg1, out, self, upper)
    @runtime_error_check ccall((:atg_linalg_cholesky_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, upper)
end

function atg_linalg_cond(arg1, self, p)
    @runtime_error_check ccall((:atg_linalg_cond, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, p)
end

function atg_linalg_cond_out(arg1, out, self, p)
    @runtime_error_check ccall((:atg_linalg_cond_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, p)
end

function atg_linalg_cond_p_str(arg1, self, p)
    @runtime_error_check ccall((:atg_linalg_cond_p_str, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}), arg1, self, p)
end

function atg_linalg_cond_p_str_out(arg1, out, self, p)
    @runtime_error_check ccall((:atg_linalg_cond_p_str_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, out, self, p)
end

function atg_linalg_det(arg1, self)
    @runtime_error_check ccall((:atg_linalg_det, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_linalg_det_out(arg1, out, self)
    @runtime_error_check ccall((:atg_linalg_det_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_linalg_eig(arg1, self)
    @runtime_error_check ccall((:atg_linalg_eig, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_linalg_eig_out(arg1, eigenvalues, eigenvectors, self)
    @runtime_error_check ccall((:atg_linalg_eig_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, eigenvalues, eigenvectors, self)
end

function atg_linalg_eigh(arg1, self, UPLO)
    @runtime_error_check ccall((:atg_linalg_eigh, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}), arg1, self, UPLO)
end

function atg_linalg_eigh_eigvals(arg1, eigvals, eigvecs, self, UPLO)
    @runtime_error_check ccall((:atg_linalg_eigh_eigvals, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Cchar}), arg1, eigvals, eigvecs, self, UPLO)
end

function atg_linalg_eigvals(arg1, self)
    @runtime_error_check ccall((:atg_linalg_eigvals, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_linalg_eigvals_out(arg1, out, self)
    @runtime_error_check ccall((:atg_linalg_eigvals_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_linalg_eigvalsh(arg1, self, UPLO)
    @runtime_error_check ccall((:atg_linalg_eigvalsh, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}), arg1, self, UPLO)
end

function atg_linalg_eigvalsh_out(arg1, out, self, UPLO)
    @runtime_error_check ccall((:atg_linalg_eigvalsh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Cchar}), arg1, out, self, UPLO)
end

function atg_linalg_householder_product(arg1, input, tau)
    @runtime_error_check ccall((:atg_linalg_householder_product, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, input, tau)
end

function atg_linalg_householder_product_out(arg1, out, input, tau)
    @runtime_error_check ccall((:atg_linalg_householder_product_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, input, tau)
end

function atg_linalg_inv(arg1, self)
    @runtime_error_check ccall((:atg_linalg_inv, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_linalg_inv_ex(arg1, self, check_errors)
    @runtime_error_check ccall((:atg_linalg_inv_ex, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, check_errors)
end

function atg_linalg_inv_ex_inverse(arg1, inverse, info, self, check_errors)
    @runtime_error_check ccall((:atg_linalg_inv_ex_inverse, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, inverse, info, self, check_errors)
end

function atg_linalg_inv_out(arg1, out, self)
    @runtime_error_check ccall((:atg_linalg_inv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_linalg_lstsq(arg1, self, b, rcond, driver)
    @runtime_error_check ccall((:atg_linalg_lstsq, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Ptr{Cchar}), arg1, self, b, rcond, driver)
end

function atg_linalg_lstsq_out(arg1, solution, residuals, rank, singular_values, self, b, rcond, driver)
    @runtime_error_check ccall((:atg_linalg_lstsq_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble, Ptr{Cchar}), arg1, solution, residuals, rank, singular_values, self, b, rcond, driver)
end

function atg_linalg_matmul(arg1, self, other)
    @runtime_error_check ccall((:atg_linalg_matmul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_linalg_matmul_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_linalg_matmul_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_linalg_matrix_power(arg1, self, n)
    @runtime_error_check ccall((:atg_linalg_matrix_power, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, n)
end

function atg_linalg_matrix_power_out(arg1, out, self, n)
    @runtime_error_check ccall((:atg_linalg_matrix_power_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, n)
end

function atg_linalg_matrix_rank(arg1, self, tol, hermitian)
    @runtime_error_check ccall((:atg_linalg_matrix_rank, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, tol, hermitian)
end

function atg_linalg_matrix_rank_out(arg1, out, self, tol, hermitian)
    @runtime_error_check ccall((:atg_linalg_matrix_rank_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cint), arg1, out, self, tol, hermitian)
end

function atg_linalg_matrix_rank_out_tol_tensor(arg1, out, input, tol, hermitian)
    @runtime_error_check ccall((:atg_linalg_matrix_rank_out_tol_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, input, tol, hermitian)
end

function atg_linalg_matrix_rank_tol_tensor(arg1, input, tol, hermitian)
    @runtime_error_check ccall((:atg_linalg_matrix_rank_tol_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, input, tol, hermitian)
end

function atg_linalg_multi_dot(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_linalg_multi_dot, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_linalg_multi_dot_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_linalg_multi_dot_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_linalg_pinv(arg1, self, rcond, hermitian)
    @runtime_error_check ccall((:atg_linalg_pinv, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, rcond, hermitian)
end

function atg_linalg_pinv_out(arg1, out, self, rcond, hermitian)
    @runtime_error_check ccall((:atg_linalg_pinv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cint), arg1, out, self, rcond, hermitian)
end

function atg_linalg_pinv_out_rcond_tensor(arg1, out, self, rcond, hermitian)
    @runtime_error_check ccall((:atg_linalg_pinv_out_rcond_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, self, rcond, hermitian)
end

function atg_linalg_pinv_rcond_tensor(arg1, self, rcond, hermitian)
    @runtime_error_check ccall((:atg_linalg_pinv_rcond_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, rcond, hermitian)
end

function atg_linalg_qr(arg1, self, mode)
    @runtime_error_check ccall((:atg_linalg_qr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}), arg1, self, mode)
end

function atg_linalg_qr_out(arg1, Q, R, self, mode)
    @runtime_error_check ccall((:atg_linalg_qr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Cchar}), arg1, Q, R, self, mode)
end

function atg_linalg_slogdet(arg1, self)
    @runtime_error_check ccall((:atg_linalg_slogdet, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_linalg_slogdet_out(arg1, sign, logabsdet, self)
    @runtime_error_check ccall((:atg_linalg_slogdet_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, sign, logabsdet, self)
end

function atg_linalg_solve(arg1, input, other)
    @runtime_error_check ccall((:atg_linalg_solve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, input, other)
end

function atg_linalg_solve_out(arg1, out, input, other)
    @runtime_error_check ccall((:atg_linalg_solve_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, input, other)
end

function atg_linalg_svd(arg1, self, full_matrices)
    @runtime_error_check ccall((:atg_linalg_svd, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, full_matrices)
end

function atg_linalg_svd_u(arg1, U, S, Vh, self, full_matrices)
    @runtime_error_check ccall((:atg_linalg_svd_u, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint), arg1, U, S, Vh, self, full_matrices)
end

function atg_linalg_svdvals(arg1, input)
    @runtime_error_check ccall((:atg_linalg_svdvals, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, input)
end

function atg_linalg_svdvals_out(arg1, out, input)
    @runtime_error_check ccall((:atg_linalg_svdvals_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, input)
end

function atg_linalg_tensorinv(arg1, self, ind)
    @runtime_error_check ccall((:atg_linalg_tensorinv, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, ind)
end

function atg_linalg_tensorinv_out(arg1, out, self, ind)
    @runtime_error_check ccall((:atg_linalg_tensorinv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, ind)
end

function atg_linalg_tensorsolve(arg1, self, other, dims_data, dims_len)
    @runtime_error_check ccall((:atg_linalg_tensorsolve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, self, other, dims_data, dims_len)
end

function atg_linalg_tensorsolve_out(arg1, out, self, other, dims_data, dims_len)
    @runtime_error_check ccall((:atg_linalg_tensorsolve_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, other, dims_data, dims_len)
end

function atg_linear(arg1, input, weight, bias)
    @runtime_error_check ccall((:atg_linear, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, weight, bias)
end

function atg_linear_out(arg1, out, input, weight, bias)
    @runtime_error_check ccall((:atg_linear_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, input, weight, bias)
end

function atg_linspace(arg1, start, _end, steps, options_kind, options_device)
    @runtime_error_check ccall((:atg_linspace, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, Int64, Cint, Cint), arg1, start, _end, steps, options_kind, options_device)
end

function atg_linspace_out(arg1, out, start, _end, steps)
    @runtime_error_check ccall((:atg_linspace_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar, Int64), arg1, out, start, _end, steps)
end

function atg_log(arg1, self)
    @runtime_error_check ccall((:atg_log, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10(arg1, self)
    @runtime_error_check ccall((:atg_log10, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10_(arg1, self)
    @runtime_error_check ccall((:atg_log10_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log10_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log1p(arg1, self)
    @runtime_error_check ccall((:atg_log1p, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log1p_(arg1, self)
    @runtime_error_check ccall((:atg_log1p_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log1p_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log1p_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log2(arg1, self)
    @runtime_error_check ccall((:atg_log2, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log2_(arg1, self)
    @runtime_error_check ccall((:atg_log2_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log2_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_(arg1, self)
    @runtime_error_check ccall((:atg_log_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log_normal_(arg1, self, mean, std)
    @runtime_error_check ccall((:atg_log_normal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, mean, std)
end

function atg_log_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_sigmoid(arg1, self)
    @runtime_error_check ccall((:atg_log_sigmoid, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log_sigmoid_backward(arg1, grad_output, self, buffer)
    @runtime_error_check ccall((:atg_log_sigmoid_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, buffer)
end

function atg_log_sigmoid_backward_grad_input(arg1, grad_input, grad_output, self, buffer)
    @runtime_error_check ccall((:atg_log_sigmoid_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, buffer)
end

function atg_log_sigmoid_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log_sigmoid_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_softmax(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_log_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_logaddexp(arg1, self, other)
    @runtime_error_check ccall((:atg_logaddexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logaddexp2(arg1, self, other)
    @runtime_error_check ccall((:atg_logaddexp2, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logaddexp2_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logaddexp2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logaddexp_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logaddexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logcumsumexp(arg1, self, dim)
    @runtime_error_check ccall((:atg_logcumsumexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_logcumsumexp_out(arg1, out, self, dim)
    @runtime_error_check ccall((:atg_logcumsumexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, dim)
end

function atg_logdet(arg1, self)
    @runtime_error_check ccall((:atg_logdet, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_and(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_and, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_and_(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_and_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_and_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logical_and_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logical_not(arg1, self)
    @runtime_error_check ccall((:atg_logical_not, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_not_(arg1, self)
    @runtime_error_check ccall((:atg_logical_not_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_not_out(arg1, out, self)
    @runtime_error_check ccall((:atg_logical_not_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_logical_or(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_or, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_or_(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_or_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_or_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logical_or_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logical_xor(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_xor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_xor_(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_xor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_xor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logical_xor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logit(arg1, self, eps)
    @runtime_error_check ccall((:atg_logit, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, eps)
end

function atg_logit_(arg1, self, eps)
    @runtime_error_check ccall((:atg_logit_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, eps)
end

function atg_logit_backward(arg1, grad_output, self, eps)
    @runtime_error_check ccall((:atg_logit_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, grad_output, self, eps)
end

function atg_logit_backward_grad_input(arg1, grad_input, grad_output, self, eps)
    @runtime_error_check ccall((:atg_logit_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble), arg1, grad_input, grad_output, self, eps)
end

function atg_logit_out(arg1, out, self, eps)
    @runtime_error_check ccall((:atg_logit_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, out, self, eps)
end

function atg_logspace(arg1, start, _end, steps, base, options_kind, options_device)
    @runtime_error_check ccall((:atg_logspace, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, Int64, Cdouble, Cint, Cint), arg1, start, _end, steps, base, options_kind, options_device)
end

function atg_logspace_out(arg1, out, start, _end, steps, base)
    @runtime_error_check ccall((:atg_logspace_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar, Int64, Cdouble), arg1, out, start, _end, steps, base)
end

function atg_logsumexp(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_logsumexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_logsumexp_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_logsumexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_lstm(arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_lstm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_lstm_cell(arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_lstm_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, tensor, tensor, tensor), arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh)
end

function atg_lstm_data(arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_lstm_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_lstsq(arg1, self, A)
    @runtime_error_check ccall((:atg_lstsq, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, A)
end

function atg_lstsq_x(arg1, X, qr, self, A)
    @runtime_error_check ccall((:atg_lstsq_x, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, X, qr, self, A)
end

function atg_lt(arg1, self, other)
    @runtime_error_check ccall((:atg_lt, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_lt_(arg1, self, other)
    @runtime_error_check ccall((:atg_lt_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_lt_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_lt_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_lt_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_lt_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lt_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_lt_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lt_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_lt_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_lu_solve(arg1, self, LU_data, LU_pivots)
    @runtime_error_check ccall((:atg_lu_solve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, LU_data, LU_pivots)
end

function atg_lu_solve_out(arg1, out, self, LU_data, LU_pivots)
    @runtime_error_check ccall((:atg_lu_solve_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, LU_data, LU_pivots)
end

function atg_lu_unpack(arg1, LU_data, LU_pivots, unpack_data, unpack_pivots)
    @runtime_error_check ccall((:atg_lu_unpack, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, LU_data, LU_pivots, unpack_data, unpack_pivots)
end

function atg_lu_unpack_out(arg1, P, L, U, LU_data, LU_pivots, unpack_data, unpack_pivots)
    @runtime_error_check ccall((:atg_lu_unpack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cint), arg1, P, L, U, LU_data, LU_pivots, unpack_data, unpack_pivots)
end

function atg_margin_ranking_loss(arg1, input1, input2, target, margin, reduction)
    @runtime_error_check ccall((:atg_margin_ranking_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Int64), arg1, input1, input2, target, margin, reduction)
end

function atg_masked_fill(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, mask, value)
end

function atg_masked_fill_(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, mask, value)
end

function atg_masked_fill_tensor(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, value)
end

function atg_masked_fill_tensor_(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, value)
end

function atg_masked_scatter(arg1, self, mask, source)
    @runtime_error_check ccall((:atg_masked_scatter, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, source)
end

function atg_masked_scatter_(arg1, self, mask, source)
    @runtime_error_check ccall((:atg_masked_scatter_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, source)
end

function atg_masked_select(arg1, self, mask)
    @runtime_error_check ccall((:atg_masked_select, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mask)
end

function atg_masked_select_backward(arg1, grad, input, mask)
    @runtime_error_check ccall((:atg_masked_select_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad, input, mask)
end

function atg_masked_select_out(arg1, out, self, mask)
    @runtime_error_check ccall((:atg_masked_select_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mask)
end

function atg_matmul(arg1, self, other)
    @runtime_error_check ccall((:atg_matmul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_matmul_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_matmul_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_matrix_exp(arg1, self)
    @runtime_error_check ccall((:atg_matrix_exp, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_matrix_exp_backward(arg1, self, grad)
    @runtime_error_check ccall((:atg_matrix_exp_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, grad)
end

function atg_matrix_power(arg1, self, n)
    @runtime_error_check ccall((:atg_matrix_power, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, n)
end

function atg_matrix_power_out(arg1, out, self, n)
    @runtime_error_check ccall((:atg_matrix_power_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, n)
end

function atg_matrix_rank(arg1, self, symmetric)
    @runtime_error_check ccall((:atg_matrix_rank, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, symmetric)
end

function atg_matrix_rank_tol(arg1, self, tol, symmetric)
    @runtime_error_check ccall((:atg_matrix_rank_tol, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, tol, symmetric)
end

function atg_max(arg1, self)
    @runtime_error_check ccall((:atg_max, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_max_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_max_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_max_dim_max(arg1, max, max_values, self, dim, keepdim)
    @runtime_error_check ccall((:atg_max_dim_max, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, max, max_values, self, dim, keepdim)
end

function atg_max_other(arg1, self, other)
    @runtime_error_check ccall((:atg_max_other, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_max_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_max_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_max_pool1d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool1d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool1d_with_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d_with_indices_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool2d_with_indices_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool2d_with_indices_out(arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d_with_indices_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool3d_with_indices_backward_grad_input(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool3d_with_indices_out(arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_unpool2d(arg1, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_backward(arg1, grad_output, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_backward_grad_input(arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_out(arg1, out, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool3d(arg1, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_backward(arg1, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_backward_grad_input(arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_out(arg1, out, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_maximum(arg1, self, other)
    @runtime_error_check ccall((:atg_maximum, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_maximum_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_maximum_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_mean(arg1, self, dtype)
    @runtime_error_check ccall((:atg_mean, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_mean_dim(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_mean_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_mean_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_mean_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_median(arg1, self)
    @runtime_error_check ccall((:atg_median, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_median_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_median_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_median_dim_values(arg1, values, indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_median_dim_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, keepdim)
end

function atg_meshgrid(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_meshgrid, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_meshgrid_indexing(arg1, tensors_data, tensors_len, indexing)
    @runtime_error_check ccall((:atg_meshgrid_indexing, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Ptr{Cchar}), arg1, tensors_data, tensors_len, indexing)
end

function atg_min(arg1, self)
    @runtime_error_check ccall((:atg_min, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_min_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_min_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_min_dim_min(arg1, min, min_indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_min_dim_min, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, min, min_indices, self, dim, keepdim)
end

function atg_min_other(arg1, self, other)
    @runtime_error_check ccall((:atg_min_other, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_min_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_min_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_minimum(arg1, self, other)
    @runtime_error_check ccall((:atg_minimum, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_minimum_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_minimum_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_miopen_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    @runtime_error_check ccall((:atg_miopen_batch_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
end

function atg_miopen_batch_norm_backward(arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon)
    @runtime_error_check ccall((:atg_miopen_batch_norm_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon)
end

function atg_miopen_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_backward_bias(arg1, grad_output)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_bias, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, grad_output)
end

function atg_miopen_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose(arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose_backward_input(arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution_backward_weight, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_rnn(arg1, input, weight_data, weight_len, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
    @runtime_error_check ccall((:atg_miopen_rnn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64, tensor, tensor, Int64, Int64, Int64, Cint, Cdouble, Cint, Cint, Ptr{Int64}, Cint, tensor), arg1, input, weight_data, weight_len, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
end

function atg_mish(arg1, self)
    @runtime_error_check ccall((:atg_mish, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_mish_(arg1, self)
    @runtime_error_check ccall((:atg_mish_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_mish_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_mish_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_mish_out(arg1, out, self)
    @runtime_error_check ccall((:atg_mish_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_mkldnn_adaptive_avg_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_mkldnn_adaptive_avg_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_mkldnn_adaptive_avg_pool2d_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_mkldnn_adaptive_avg_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_mkldnn_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_mkldnn_convolution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
end

function atg_mkldnn_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
    @runtime_error_check ccall((:atg_mkldnn_convolution_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
end

function atg_mkldnn_convolution_backward_weights(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
    @runtime_error_check ccall((:atg_mkldnn_convolution_backward_weights, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
end

function atg_mkldnn_linear(arg1, self, weight, bias)
    @runtime_error_check ccall((:atg_mkldnn_linear, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, weight, bias)
end

function atg_mkldnn_linear_backward_input(arg1, input_size_data, input_size_len, grad_output, weight)
    @runtime_error_check ccall((:atg_mkldnn_linear_backward_input, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor), arg1, input_size_data, input_size_len, grad_output, weight)
end

function atg_mkldnn_linear_backward_weights(arg1, grad_output, input, weight, bias_defined)
    @runtime_error_check ccall((:atg_mkldnn_linear_backward_weights, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, grad_output, input, weight, bias_defined)
end

function atg_mkldnn_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_mkldnn_max_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_mkldnn_max_pool2d_backward(arg1, grad_output, output, input, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_mkldnn_max_pool2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output, input, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_mkldnn_max_pool3d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_mkldnn_max_pool3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_mkldnn_max_pool3d_backward(arg1, grad_output, output, input, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_mkldnn_max_pool3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output, input, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_mkldnn_reorder_conv2d_weight(arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_mkldnn_reorder_conv2d_weight, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
end

function atg_mkldnn_reorder_conv3d_weight(arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_mkldnn_reorder_conv3d_weight, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
end

function atg_mm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_mm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_mm_out(arg1, out, self, mat2)
    @runtime_error_check ccall((:atg_mm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mat2)
end

function atg_mode(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_mode, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_mode_values(arg1, values, indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_mode_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, keepdim)
end

function atg_moveaxis(arg1, self, source_data, source_len, destination_data, destination_len)
    @runtime_error_check ccall((:atg_moveaxis, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, source_data, source_len, destination_data, destination_len)
end

function atg_moveaxis_int(arg1, self, source, destination)
    @runtime_error_check ccall((:atg_moveaxis_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, source, destination)
end

function atg_movedim(arg1, self, source_data, source_len, destination_data, destination_len)
    @runtime_error_check ccall((:atg_movedim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, source_data, source_len, destination_data, destination_len)
end

function atg_movedim_int(arg1, self, source, destination)
    @runtime_error_check ccall((:atg_movedim_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, source, destination)
end

function atg_mse_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_mse_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_mse_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_mse_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_msort(arg1, self)
    @runtime_error_check ccall((:atg_msort, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_msort_out(arg1, out, self)
    @runtime_error_check ccall((:atg_msort_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_mul(arg1, self, other)
    @runtime_error_check ccall((:atg_mul, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_mul_(arg1, self, other)
    @runtime_error_check ccall((:atg_mul_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_mul_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_mul_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_mul_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_mul_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_mul_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_mul_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_multi_margin_loss_backward(arg1, grad_output, self, target, p, margin, weight, reduction)
    @runtime_error_check ccall((:atg_multi_margin_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, tensor, Int64), arg1, grad_output, self, target, p, margin, weight, reduction)
end

function atg_multi_margin_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, p, margin, weight, reduction)
    @runtime_error_check ccall((:atg_multi_margin_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor, Int64), arg1, grad_input, grad_output, self, target, p, margin, weight, reduction)
end

function atg_multilabel_margin_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_multilabel_margin_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_multilabel_margin_loss_backward(arg1, grad_output, self, target, reduction, is_target)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, tensor), arg1, grad_output, self, target, reduction, is_target)
end

function atg_multilabel_margin_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, reduction, is_target)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, tensor), arg1, grad_input, grad_output, self, target, reduction, is_target)
end

function atg_multilabel_margin_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_multinomial(arg1, self, num_samples, replacement)
    @runtime_error_check ccall((:atg_multinomial, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, num_samples, replacement)
end

function atg_multinomial_out(arg1, out, self, num_samples, replacement)
    @runtime_error_check ccall((:atg_multinomial_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, num_samples, replacement)
end

function atg_multiply(arg1, self, other)
    @runtime_error_check ccall((:atg_multiply, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_multiply_(arg1, self, other)
    @runtime_error_check ccall((:atg_multiply_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_multiply_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_multiply_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_multiply_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_multiply_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_multiply_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_multiply_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_mv(arg1, self, vec)
    @runtime_error_check ccall((:atg_mv, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, vec)
end

function atg_mv_out(arg1, out, self, vec)
    @runtime_error_check ccall((:atg_mv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, vec)
end

function atg_mvlgamma(arg1, self, p)
    @runtime_error_check ccall((:atg_mvlgamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, p)
end

function atg_mvlgamma_(arg1, self, p)
    @runtime_error_check ccall((:atg_mvlgamma_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, p)
end

function atg_mvlgamma_out(arg1, out, self, p)
    @runtime_error_check ccall((:atg_mvlgamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, p)
end

function atg_nan_to_num(arg1, self, nan, posinf, neginf)
    @runtime_error_check ccall((:atg_nan_to_num, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble, Cdouble), arg1, self, nan, posinf, neginf)
end

function atg_nan_to_num_(arg1, self, nan, posinf, neginf)
    @runtime_error_check ccall((:atg_nan_to_num_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble, Cdouble), arg1, self, nan, posinf, neginf)
end

function atg_nan_to_num_out(arg1, out, self, nan, posinf, neginf)
    @runtime_error_check ccall((:atg_nan_to_num_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble, Cdouble), arg1, out, self, nan, posinf, neginf)
end

function atg_nanmean(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_nanmean, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_nanmean_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_nanmean_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_nanmedian(arg1, self)
    @runtime_error_check ccall((:atg_nanmedian, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_nanmedian_dim(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_nanmedian_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_nanmedian_dim_values(arg1, values, indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_nanmedian_dim_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, keepdim)
end

function atg_nanquantile(arg1, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_nanquantile, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, self, q, dim, keepdim)
end

function atg_nanquantile_new(arg1, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_nanquantile_new, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Ptr{Cchar}), arg1, self, q, dim, keepdim, interpolation)
end

function atg_nanquantile_new_out(arg1, out, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_nanquantile_new_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint, Ptr{Cchar}), arg1, out, self, q, dim, keepdim, interpolation)
end

function atg_nanquantile_new_scalar(arg1, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_nanquantile_new_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint, Ptr{Cchar}), arg1, self, q, dim, keepdim, interpolation)
end

function atg_nanquantile_new_scalar_out(arg1, out, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_nanquantile_new_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64, Cint, Ptr{Cchar}), arg1, out, self, q, dim, keepdim, interpolation)
end

function atg_nanquantile_out(arg1, out, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_nanquantile_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, out, self, q, dim, keepdim)
end

function atg_nanquantile_scalar(arg1, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_nanquantile_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint), arg1, self, q, dim, keepdim)
end

function atg_nanquantile_scalar_out(arg1, out, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_nanquantile_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64, Cint), arg1, out, self, q, dim, keepdim)
end

function atg_nansum(arg1, self, dtype)
    @runtime_error_check ccall((:atg_nansum, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_nansum_dim_intlist(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_nansum_dim_intlist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_nansum_intlist_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_nansum_intlist_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_narrow(arg1, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dim, start, length)
end

function atg_narrow_copy(arg1, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow_copy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dim, start, length)
end

function atg_narrow_copy_out(arg1, out, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow_copy_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Int64), arg1, out, self, dim, start, length)
end

function atg_narrow_tensor(arg1, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, Int64), arg1, self, dim, start, length)
end

function atg_native_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, momentum, eps)
    @runtime_error_check ccall((:atg_native_batch_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, momentum, eps)
end

function atg_native_batch_norm_out(arg1, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps)
    @runtime_error_check ccall((:atg_native_batch_norm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps)
end

function atg_native_group_norm(arg1, input, weight, bias, n, C, HxW, group, eps)
    @runtime_error_check ccall((:atg_native_group_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Int64, Int64, Cdouble), arg1, input, weight, bias, n, C, HxW, group, eps)
end

function atg_native_layer_norm(arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps)
    @runtime_error_check ccall((:atg_native_layer_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, tensor, tensor, Cdouble), arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps)
end

function atg_native_norm(arg1, self)
    @runtime_error_check ccall((:atg_native_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_native_norm_scalaropt_dim_dtype(arg1, self, p, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_native_norm_scalaropt_dim_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Int64}, Cint, Cint, Cint), arg1, self, p, dim_data, dim_len, keepdim, dtype)
end

function atg_ne(arg1, self, other)
    @runtime_error_check ccall((:atg_ne, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ne_(arg1, self, other)
    @runtime_error_check ccall((:atg_ne_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ne_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ne_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_ne_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_ne_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ne_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_ne_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ne_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ne_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_neg(arg1, self)
    @runtime_error_check ccall((:atg_neg, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_neg_(arg1, self)
    @runtime_error_check ccall((:atg_neg_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_neg_out(arg1, out, self)
    @runtime_error_check ccall((:atg_neg_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_negative(arg1, self)
    @runtime_error_check ccall((:atg_negative, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_negative_(arg1, self)
    @runtime_error_check ccall((:atg_negative_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_negative_out(arg1, out, self)
    @runtime_error_check ccall((:atg_negative_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_new_empty(arg1, self, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_empty, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, options_kind, options_device)
end

function atg_new_empty_strided(arg1, self, size_data, size_len, stride_data, stride_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_empty_strided, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, stride_data, stride_len, options_kind, options_device)
end

function atg_new_full(arg1, self, size_data, size_len, fill_value, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_full, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, scalar, Cint, Cint), arg1, self, size_data, size_len, fill_value, options_kind, options_device)
end

function atg_new_ones(arg1, self, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_ones, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, options_kind, options_device)
end

function atg_new_zeros(arg1, self, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_zeros, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, options_kind, options_device)
end

function atg_nextafter(arg1, self, other)
    @runtime_error_check ccall((:atg_nextafter, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_nextafter_(arg1, self, other)
    @runtime_error_check ccall((:atg_nextafter_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_nextafter_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_nextafter_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_nll_loss(arg1, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss2d(arg1, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss2d_backward(arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss2d_backward_grad_input(arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss2d_out(arg1, out, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64), arg1, out, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss_backward(arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss_nd(arg1, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss_nd, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss_out(arg1, out, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64), arg1, out, self, target, weight, reduction, ignore_index)
end

function atg_nonzero(arg1, self)
    @runtime_error_check ccall((:atg_nonzero, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_nonzero_numpy(arg1, self)
    @runtime_error_check ccall((:atg_nonzero_numpy, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_nonzero_out(arg1, out, self)
    @runtime_error_check ccall((:atg_nonzero_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_norm(arg1, self)
    @runtime_error_check ccall((:atg_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_norm_dtype_out(arg1, out, self, p, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_norm_dtype_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, p, dim_data, dim_len, keepdim, dtype)
end

function atg_norm_except_dim(arg1, v, pow, dim)
    @runtime_error_check ccall((:atg_norm_except_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, v, pow, dim)
end

function atg_norm_out(arg1, out, self, p, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_norm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, Ptr{Int64}, Cint, Cint), arg1, out, self, p, dim_data, dim_len, keepdim)
end

function atg_norm_scalaropt_dim(arg1, self, p, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_norm_scalaropt_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Int64}, Cint, Cint), arg1, self, p, dim_data, dim_len, keepdim)
end

function atg_norm_scalaropt_dim_dtype(arg1, self, p, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_norm_scalaropt_dim_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Int64}, Cint, Cint, Cint), arg1, self, p, dim_data, dim_len, keepdim, dtype)
end

function atg_norm_scalaropt_dtype(arg1, self, p, dtype)
    @runtime_error_check ccall((:atg_norm_scalaropt_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Cint), arg1, self, p, dtype)
end

function atg_normal(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, out, mean, std)
end

function atg_normal_(arg1, self, mean, std)
    @runtime_error_check ccall((:atg_normal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, mean, std)
end

function atg_normal_float_float_out(arg1, out, mean, std, size_data, size_len)
    @runtime_error_check ccall((:atg_normal_float_float_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble, Ptr{Int64}, Cint), arg1, out, mean, std, size_data, size_len)
end

function atg_normal_float_tensor_out(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal_float_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, tensor), arg1, out, mean, std)
end

function atg_normal_tensor_tensor_out(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal_tensor_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, mean, std)
end

function atg_not_equal(arg1, self, other)
    @runtime_error_check ccall((:atg_not_equal, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_not_equal_(arg1, self, other)
    @runtime_error_check ccall((:atg_not_equal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_not_equal_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_not_equal_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_not_equal_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_not_equal_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_not_equal_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_not_equal_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_not_equal_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_not_equal_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_nuclear_norm(arg1, self, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, keepdim)
end

function atg_nuclear_norm_dim(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_nuclear_norm_dim_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm_dim_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_nuclear_norm_out(arg1, out, self, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, keepdim)
end

function atg_numpy_t(arg1, self)
    @runtime_error_check ccall((:atg_numpy_t, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_one_hot(arg1, self, num_classes)
    @runtime_error_check ccall((:atg_one_hot, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, num_classes)
end

function atg_ones(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_ones, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_ones_like(arg1, self)
    @runtime_error_check ccall((:atg_ones_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ones_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_ones_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_orgqr(arg1, self, input2)
    @runtime_error_check ccall((:atg_orgqr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, input2)
end

function atg_orgqr_out(arg1, out, self, input2)
    @runtime_error_check ccall((:atg_orgqr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, input2)
end

function atg_ormqr(arg1, self, input2, input3, left, transpose)
    @runtime_error_check ccall((:atg_ormqr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, self, input2, input3, left, transpose)
end

function atg_ormqr_out(arg1, out, self, input2, input3, left, transpose)
    @runtime_error_check ccall((:atg_ormqr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint), arg1, out, self, input2, input3, left, transpose)
end

function atg_outer(arg1, self, vec2)
    @runtime_error_check ccall((:atg_outer, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, vec2)
end

function atg_outer_out(arg1, out, self, vec2)
    @runtime_error_check ccall((:atg_outer_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, vec2)
end

function atg_pad_sequence(arg1, sequences_data, sequences_len, batch_first, padding_value)
    @runtime_error_check ccall((:atg_pad_sequence, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Cint, Cdouble), arg1, sequences_data, sequences_len, batch_first, padding_value)
end

function atg_pairwise_distance(arg1, x1, x2, p, eps, keepdim)
    @runtime_error_check ccall((:atg_pairwise_distance, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble, Cint), arg1, x1, x2, p, eps, keepdim)
end

function atg_pdist(arg1, self, p)
    @runtime_error_check ccall((:atg_pdist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_permute(arg1, self, dims_data, dims_len)
    @runtime_error_check ccall((:atg_permute, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dims_data, dims_len)
end

function atg_pin_memory(arg1, self, device)
    @runtime_error_check ccall((:atg_pin_memory, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, device)
end

function atg_pinverse(arg1, self, rcond)
    @runtime_error_check ccall((:atg_pinverse, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, rcond)
end

function atg_pixel_shuffle(arg1, self, upscale_factor)
    @runtime_error_check ccall((:atg_pixel_shuffle, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, upscale_factor)
end

function atg_pixel_unshuffle(arg1, self, downscale_factor)
    @runtime_error_check ccall((:atg_pixel_unshuffle, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, downscale_factor)
end

function atg_poisson(arg1, self)
    @runtime_error_check ccall((:atg_poisson, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_poisson_nll_loss(arg1, input, target, log_input, full, eps, reduction)
    @runtime_error_check ccall((:atg_poisson_nll_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint, Cdouble, Int64), arg1, input, target, log_input, full, eps, reduction)
end

function atg_polar(arg1, abs, angle)
    @runtime_error_check ccall((:atg_polar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, abs, angle)
end

function atg_polar_out(arg1, out, abs, angle)
    @runtime_error_check ccall((:atg_polar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, abs, angle)
end

function atg_polygamma(arg1, n, self)
    @runtime_error_check ccall((:atg_polygamma, libtorch_c_api), Cint, (Ptr{tensor}, Int64, tensor), arg1, n, self)
end

function atg_polygamma_(arg1, self, n)
    @runtime_error_check ccall((:atg_polygamma_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, n)
end

function atg_polygamma_out(arg1, out, n, self)
    @runtime_error_check ccall((:atg_polygamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor), arg1, out, n, self)
end

function atg_positive(arg1, self)
    @runtime_error_check ccall((:atg_positive, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_pow(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_pow_(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_pow_scalar(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, exponent)
end

function atg_pow_scalar_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, exponent)
end

function atg_pow_tensor_(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_pow_tensor_scalar(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_tensor_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_pow_tensor_scalar_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_tensor_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, exponent)
end

function atg_pow_tensor_tensor_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_tensor_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, exponent)
end

function atg_prelu(arg1, self, weight)
    @runtime_error_check ccall((:atg_prelu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, weight)
end

function atg_prelu_backward(arg1, grad_output, self, weight)
    @runtime_error_check ccall((:atg_prelu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, weight)
end

function atg_prod(arg1, self, dtype)
    @runtime_error_check ccall((:atg_prod, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_prod_dim_int(arg1, self, dim, keepdim, dtype)
    @runtime_error_check ccall((:atg_prod_dim_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, dim, keepdim, dtype)
end

function atg_prod_int_out(arg1, out, self, dim, keepdim, dtype)
    @runtime_error_check ccall((:atg_prod_int_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Cint), arg1, out, self, dim, keepdim, dtype)
end

function atg_put(arg1, self, index, source, accumulate)
    @runtime_error_check ccall((:atg_put, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, self, index, source, accumulate)
end

function atg_put_(arg1, self, index, source, accumulate)
    @runtime_error_check ccall((:atg_put_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, self, index, source, accumulate)
end

function atg_q_per_channel_scales(arg1, self)
    @runtime_error_check ccall((:atg_q_per_channel_scales, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_q_per_channel_zero_points(arg1, self)
    @runtime_error_check ccall((:atg_q_per_channel_zero_points, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_qr(arg1, self, some)
    @runtime_error_check ccall((:atg_qr, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, some)
end

function atg_qr_q(arg1, Q, R, self, some)
    @runtime_error_check ccall((:atg_qr_q, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, Q, R, self, some)
end

function atg_quantile(arg1, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_quantile, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, self, q, dim, keepdim)
end

function atg_quantile_new(arg1, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_quantile_new, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Ptr{Cchar}), arg1, self, q, dim, keepdim, interpolation)
end

function atg_quantile_new_out(arg1, out, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_quantile_new_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint, Ptr{Cchar}), arg1, out, self, q, dim, keepdim, interpolation)
end

function atg_quantile_new_scalar(arg1, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_quantile_new_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint, Ptr{Cchar}), arg1, self, q, dim, keepdim, interpolation)
end

function atg_quantile_new_scalar_out(arg1, out, self, q, dim, keepdim, interpolation)
    @runtime_error_check ccall((:atg_quantile_new_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64, Cint, Ptr{Cchar}), arg1, out, self, q, dim, keepdim, interpolation)
end

function atg_quantile_out(arg1, out, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_quantile_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, out, self, q, dim, keepdim)
end

function atg_quantile_scalar(arg1, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_quantile_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint), arg1, self, q, dim, keepdim)
end

function atg_quantile_scalar_out(arg1, out, self, q, dim, keepdim)
    @runtime_error_check ccall((:atg_quantile_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64, Cint), arg1, out, self, q, dim, keepdim)
end

function atg_quantize_per_channel(arg1, self, scales, zero_points, axis, dtype)
    @runtime_error_check ccall((:atg_quantize_per_channel, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, self, scales, zero_points, axis, dtype)
end

function atg_quantize_per_tensor(arg1, self, scale, zero_point, dtype)
    @runtime_error_check ccall((:atg_quantize_per_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint), arg1, self, scale, zero_point, dtype)
end

function atg_quantize_per_tensor_tensor_qparams(arg1, self, scale, zero_point, dtype)
    @runtime_error_check ccall((:atg_quantize_per_tensor_tensor_qparams, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, self, scale, zero_point, dtype)
end

function atg_quantize_per_tensor_tensors(arg1, tensors_data, tensors_len, scales, zero_points, dtype)
    @runtime_error_check ccall((:atg_quantize_per_tensor_tensors, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, tensor, tensor, Cint), arg1, tensors_data, tensors_len, scales, zero_points, dtype)
end

function atg_quantized_batch_norm(arg1, input, weight, bias, mean, var, eps, output_scale, output_zero_point)
    @runtime_error_check ccall((:atg_quantized_batch_norm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble, Cdouble, Int64), arg1, input, weight, bias, mean, var, eps, output_scale, output_zero_point)
end

function atg_quantized_gru_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_gru_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_lstm_cell(arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_lstm_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_max_pool1d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_quantized_max_pool1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_quantized_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_quantized_max_pool2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_quantized_rnn_relu_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_rnn_relu_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_rnn_tanh_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_rnn_tanh_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_rad2deg(arg1, self)
    @runtime_error_check ccall((:atg_rad2deg, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rad2deg_(arg1, self)
    @runtime_error_check ccall((:atg_rad2deg_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rad2deg_out(arg1, out, self)
    @runtime_error_check ccall((:atg_rad2deg_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_rand(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_rand, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_rand_like(arg1, self)
    @runtime_error_check ccall((:atg_rand_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rand_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_rand_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_randint(arg1, high, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Ptr{Int64}, Cint, Cint, Cint), arg1, high, size_data, size_len, options_kind, options_device)
end

function atg_randint_like(arg1, self, high)
    @runtime_error_check ccall((:atg_randint_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, high)
end

function atg_randint_like_low_dtype(arg1, self, low, high)
    @runtime_error_check ccall((:atg_randint_like_low_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, low, high)
end

function atg_randint_low(arg1, low, high, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint_low, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Ptr{Int64}, Cint, Cint, Cint), arg1, low, high, size_data, size_len, options_kind, options_device)
end

function atg_randint_low_out(arg1, out, low, high, size_data, size_len)
    @runtime_error_check ccall((:atg_randint_low_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Int64}, Cint), arg1, out, low, high, size_data, size_len)
end

function atg_randint_out(arg1, out, high, size_data, size_len)
    @runtime_error_check ccall((:atg_randint_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Int64}, Cint), arg1, out, high, size_data, size_len)
end

function atg_randn(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randn, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_randn_like(arg1, self)
    @runtime_error_check ccall((:atg_randn_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_randn_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_randn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_random_(arg1, self)
    @runtime_error_check ccall((:atg_random_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_random_from_(arg1, self, from, to)
    @runtime_error_check ccall((:atg_random_from_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, from, to)
end

function atg_random_to_(arg1, self, to)
    @runtime_error_check ccall((:atg_random_to_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, to)
end

function atg_randperm(arg1, n, options_kind, options_device)
    @runtime_error_check ccall((:atg_randperm, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, n, options_kind, options_device)
end

function atg_randperm_out(arg1, out, n)
    @runtime_error_check ccall((:atg_randperm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, out, n)
end

function atg_range(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_range, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_range_out(arg1, out, start, _end)
    @runtime_error_check ccall((:atg_range_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, out, start, _end)
end

function atg_range_step(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_range_step, libtorch_c_api), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_ravel(arg1, self)
    @runtime_error_check ccall((:atg_ravel, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_real(arg1, self)
    @runtime_error_check ccall((:atg_real, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_reciprocal(arg1, self)
    @runtime_error_check ccall((:atg_reciprocal, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_reciprocal_(arg1, self)
    @runtime_error_check ccall((:atg_reciprocal_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_reciprocal_out(arg1, out, self)
    @runtime_error_check ccall((:atg_reciprocal_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_reflection_pad1d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_reflection_pad1d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad1d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad1d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_reflection_pad2d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_reflection_pad2d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad2d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad2d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_reflection_pad3d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_reflection_pad3d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad3d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad3d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_relu(arg1, self)
    @runtime_error_check ccall((:atg_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_relu6(arg1, self)
    @runtime_error_check ccall((:atg_relu6, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_relu6_(arg1, self)
    @runtime_error_check ccall((:atg_relu6_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_relu_(arg1, self)
    @runtime_error_check ccall((:atg_relu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_remainder(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_remainder_(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_remainder_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_remainder_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_remainder_scalar_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_scalar_tensor, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_remainder_tensor(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_remainder_tensor_(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_remainder_tensor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_remainder_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_renorm(arg1, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Int64, scalar), arg1, self, p, dim, maxnorm)
end

function atg_renorm_(arg1, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Int64, scalar), arg1, self, p, dim, maxnorm)
end

function atg_renorm_out(arg1, out, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, Int64, scalar), arg1, out, self, p, dim, maxnorm)
end

function atg_repeat(arg1, self, repeats_data, repeats_len)
    @runtime_error_check ccall((:atg_repeat, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, repeats_data, repeats_len)
end

function atg_repeat_interleave(arg1, repeats, output_size)
    @runtime_error_check ccall((:atg_repeat_interleave, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, repeats, output_size)
end

function atg_repeat_interleave_self_int(arg1, self, repeats, dim, output_size)
    @runtime_error_check ccall((:atg_repeat_interleave_self_int, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, repeats, dim, output_size)
end

function atg_repeat_interleave_self_tensor(arg1, self, repeats, dim, output_size)
    @runtime_error_check ccall((:atg_repeat_interleave_self_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64), arg1, self, repeats, dim, output_size)
end

function atg_replication_pad1d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad1d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad1d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad1d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_replication_pad2d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad2d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad2d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad2d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_replication_pad3d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad3d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad3d_backward_grad_input(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad3d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_requires_grad_(arg1, self, requires_grad)
    @runtime_error_check ccall((:atg_requires_grad_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, requires_grad)
end

function atg_reshape(arg1, self, shape_data, shape_len)
    @runtime_error_check ccall((:atg_reshape, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, shape_data, shape_len)
end

function atg_reshape_as(arg1, self, other)
    @runtime_error_check ccall((:atg_reshape_as, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_resize_(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_resize_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_resize_as_(arg1, self, the_template)
    @runtime_error_check ccall((:atg_resize_as_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, the_template)
end

function atg_resize_as_sparse_(arg1, self, the_template)
    @runtime_error_check ccall((:atg_resize_as_sparse_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, the_template)
end

function atg_resolve_conj(arg1, self)
    @runtime_error_check ccall((:atg_resolve_conj, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_resolve_neg(arg1, self)
    @runtime_error_check ccall((:atg_resolve_neg, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rnn_relu(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_rnn_relu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_rnn_relu_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_rnn_relu_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_rnn_relu_data(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_rnn_relu_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_rnn_tanh(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_rnn_tanh, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_rnn_tanh_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_rnn_tanh_cell, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_rnn_tanh_data(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_rnn_tanh_data, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_roll(arg1, self, shifts_data, shifts_len, dims_data, dims_len)
    @runtime_error_check ccall((:atg_roll, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, shifts_data, shifts_len, dims_data, dims_len)
end

function atg_rot90(arg1, self, k, dims_data, dims_len)
    @runtime_error_check ccall((:atg_rot90, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Int64}, Cint), arg1, self, k, dims_data, dims_len)
end

function atg_round(arg1, self)
    @runtime_error_check ccall((:atg_round, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_round_(arg1, self)
    @runtime_error_check ccall((:atg_round_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_round_out(arg1, out, self)
    @runtime_error_check ccall((:atg_round_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_row_stack(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_row_stack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_row_stack_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_row_stack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_rrelu(arg1, self, training)
    @runtime_error_check ccall((:atg_rrelu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, training)
end

function atg_rrelu_(arg1, self, training)
    @runtime_error_check ccall((:atg_rrelu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, training)
end

function atg_rrelu_with_noise(arg1, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, noise, training)
end

function atg_rrelu_with_noise_(arg1, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, noise, training)
end

function atg_rrelu_with_noise_backward(arg1, grad_output, self, noise, lower, upper, training, self_is_result)
    @runtime_error_check ccall((:atg_rrelu_with_noise_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, Cint, Cint), arg1, grad_output, self, noise, lower, upper, training, self_is_result)
end

function atg_rrelu_with_noise_out(arg1, out, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, self, noise, training)
end

function atg_rsqrt(arg1, self)
    @runtime_error_check ccall((:atg_rsqrt, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rsqrt_(arg1, self)
    @runtime_error_check ccall((:atg_rsqrt_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rsqrt_out(arg1, out, self)
    @runtime_error_check ccall((:atg_rsqrt_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_rsub(arg1, self, other)
    @runtime_error_check ccall((:atg_rsub, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_rsub_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_rsub_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_scalar_tensor(arg1, s, options_kind, options_device)
    @runtime_error_check ccall((:atg_scalar_tensor, libtorch_c_api), Cint, (Ptr{tensor}, scalar, Cint, Cint), arg1, s, options_kind, options_device)
end

function atg_scatter(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_add(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_add, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_add_(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_add_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_add_out(arg1, out, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_add_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, tensor), arg1, out, self, dim, index, src)
end

function atg_scatter_reduce(arg1, self, dim, index, src, reduce)
    @runtime_error_check ccall((:atg_scatter_reduce, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, Ptr{Cchar}), arg1, self, dim, index, src, reduce)
end

function atg_scatter_reduce_(arg1, self, dim, index, src, reduce)
    @runtime_error_check ccall((:atg_scatter_reduce_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, Ptr{Cchar}), arg1, self, dim, index, src, reduce)
end

function atg_scatter_reduce_out(arg1, out, self, dim, index, src, reduce)
    @runtime_error_check ccall((:atg_scatter_reduce_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, tensor, Ptr{Cchar}), arg1, out, self, dim, index, src, reduce)
end

function atg_scatter_src_out(arg1, out, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_src_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, tensor), arg1, out, self, dim, index, src)
end

function atg_scatter_value(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_scatter_value, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_scatter_value_(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_scatter_value_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_scatter_value_out(arg1, out, self, dim, index, value)
    @runtime_error_check ccall((:atg_scatter_value_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, scalar), arg1, out, self, dim, index, value)
end

function atg_scatter_value_reduce(arg1, self, dim, index, value, reduce)
    @runtime_error_check ccall((:atg_scatter_value_reduce, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar, Ptr{Cchar}), arg1, self, dim, index, value, reduce)
end

function atg_scatter_value_reduce_(arg1, self, dim, index, value, reduce)
    @runtime_error_check ccall((:atg_scatter_value_reduce_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar, Ptr{Cchar}), arg1, self, dim, index, value, reduce)
end

function atg_scatter_value_reduce_out(arg1, out, self, dim, index, value, reduce)
    @runtime_error_check ccall((:atg_scatter_value_reduce_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, scalar, Ptr{Cchar}), arg1, out, self, dim, index, value, reduce)
end

function atg_searchsorted(arg1, sorted_sequence, self, out_int32, right)
    @runtime_error_check ccall((:atg_searchsorted, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, sorted_sequence, self, out_int32, right)
end

function atg_searchsorted_scalar(arg1, sorted_sequence, self, out_int32, right)
    @runtime_error_check ccall((:atg_searchsorted_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, Cint, Cint), arg1, sorted_sequence, self, out_int32, right)
end

function atg_searchsorted_tensor_out(arg1, out, sorted_sequence, self, out_int32, right)
    @runtime_error_check ccall((:atg_searchsorted_tensor_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, out, sorted_sequence, self, out_int32, right)
end

function atg_segment_reduce(arg1, data, reduce, lengths, indices, axis, unsafe, initial)
    @runtime_error_check ccall((:atg_segment_reduce, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Cchar}, tensor, tensor, Int64, Cint, scalar), arg1, data, reduce, lengths, indices, axis, unsafe, initial)
end

function atg_select(arg1, self, dim, index)
    @runtime_error_check ccall((:atg_select, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim, index)
end

function atg_select_backward(arg1, grad_output, input_sizes_data, input_sizes_len, dim, index)
    @runtime_error_check ccall((:atg_select_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, grad_output, input_sizes_data, input_sizes_len, dim, index)
end

function atg_selu(arg1, self)
    @runtime_error_check ccall((:atg_selu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_selu_(arg1, self)
    @runtime_error_check ccall((:atg_selu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_set_(arg1, self)
    @runtime_error_check ccall((:atg_set_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_set_requires_grad(arg1, self, r)
    @runtime_error_check ccall((:atg_set_requires_grad, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, r)
end

function atg_set_source_tensor_(arg1, self, source)
    @runtime_error_check ccall((:atg_set_source_tensor_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, source)
end

function atg_sgn(arg1, self)
    @runtime_error_check ccall((:atg_sgn, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sgn_(arg1, self)
    @runtime_error_check ccall((:atg_sgn_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sgn_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sgn_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sigmoid(arg1, self)
    @runtime_error_check ccall((:atg_sigmoid, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sigmoid_(arg1, self)
    @runtime_error_check ccall((:atg_sigmoid_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sigmoid_backward(arg1, grad_output, output)
    @runtime_error_check ccall((:atg_sigmoid_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, output)
end

function atg_sigmoid_backward_grad_input(arg1, grad_input, grad_output, output)
    @runtime_error_check ccall((:atg_sigmoid_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, output)
end

function atg_sigmoid_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sigmoid_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sign(arg1, self)
    @runtime_error_check ccall((:atg_sign, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sign_(arg1, self)
    @runtime_error_check ccall((:atg_sign_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sign_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sign_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_signbit(arg1, self)
    @runtime_error_check ccall((:atg_signbit, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_signbit_out(arg1, out, self)
    @runtime_error_check ccall((:atg_signbit_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_silu(arg1, self)
    @runtime_error_check ccall((:atg_silu, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_silu_(arg1, self)
    @runtime_error_check ccall((:atg_silu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_silu_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_silu_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_silu_backward_grad_input(arg1, grad_input, grad_output, self)
    @runtime_error_check ccall((:atg_silu_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, self)
end

function atg_silu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_silu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sin(arg1, self)
    @runtime_error_check ccall((:atg_sin, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sin_(arg1, self)
    @runtime_error_check ccall((:atg_sin_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sin_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sin_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sinc(arg1, self)
    @runtime_error_check ccall((:atg_sinc, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinc_(arg1, self)
    @runtime_error_check ccall((:atg_sinc_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sinc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sinh(arg1, self)
    @runtime_error_check ccall((:atg_sinh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinh_(arg1, self)
    @runtime_error_check ccall((:atg_sinh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sinh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_slice(arg1, self, dim, start, _end, step)
    @runtime_error_check ccall((:atg_slice, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, self, dim, start, _end, step)
end

function atg_slice_backward(arg1, grad_output, input_sizes_data, input_sizes_len, dim, start, _end, step)
    @runtime_error_check ccall((:atg_slice_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64, Int64, Int64), arg1, grad_output, input_sizes_data, input_sizes_len, dim, start, _end, step)
end

function atg_slogdet(arg1, self)
    @runtime_error_check ccall((:atg_slogdet, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_slow_conv3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_slow_conv3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
end

function atg_slow_conv3d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_slow_conv3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
end

function atg_slow_conv_dilated2d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_dilated2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_dilated3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_dilated3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose2d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose2d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose3d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_smm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_smm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_smooth_l1_loss(arg1, self, target, reduction, beta)
    @runtime_error_check ccall((:atg_smooth_l1_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cdouble), arg1, self, target, reduction, beta)
end

function atg_smooth_l1_loss_backward(arg1, grad_output, self, target, reduction, beta)
    @runtime_error_check ccall((:atg_smooth_l1_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cdouble), arg1, grad_output, self, target, reduction, beta)
end

function atg_smooth_l1_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, reduction, beta)
    @runtime_error_check ccall((:atg_smooth_l1_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Cdouble), arg1, grad_input, grad_output, self, target, reduction, beta)
end

function atg_smooth_l1_loss_out(arg1, out, self, target, reduction, beta)
    @runtime_error_check ccall((:atg_smooth_l1_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cdouble), arg1, out, self, target, reduction, beta)
end

function atg_soft_margin_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_soft_margin_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_soft_margin_loss_backward_grad_input(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_soft_margin_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_softmax(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_softplus(arg1, self)
    @runtime_error_check ccall((:atg_softplus, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_softplus_backward(arg1, grad_output, self, beta, threshold, output)
    @runtime_error_check ccall((:atg_softplus_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar, tensor), arg1, grad_output, self, beta, threshold, output)
end

function atg_softplus_backward_grad_input(arg1, grad_input, grad_output, self, beta, threshold, output)
    @runtime_error_check ccall((:atg_softplus_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, tensor), arg1, grad_input, grad_output, self, beta, threshold, output)
end

function atg_softplus_out(arg1, out, self)
    @runtime_error_check ccall((:atg_softplus_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_softshrink(arg1, self)
    @runtime_error_check ccall((:atg_softshrink, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_softshrink_backward(arg1, grad_output, self, lambd)
    @runtime_error_check ccall((:atg_softshrink_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_output, self, lambd)
end

function atg_softshrink_backward_grad_input(arg1, grad_input, grad_output, self, lambd)
    @runtime_error_check ccall((:atg_softshrink_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, grad_input, grad_output, self, lambd)
end

function atg_softshrink_out(arg1, out, self)
    @runtime_error_check ccall((:atg_softshrink_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_solve(arg1, self, A)
    @runtime_error_check ccall((:atg_solve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, A)
end

function atg_solve_solution(arg1, solution, lu, self, A)
    @runtime_error_check ccall((:atg_solve_solution, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, solution, lu, self, A)
end

function atg_sort(arg1, self, dim, descending)
    @runtime_error_check ccall((:atg_sort, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, descending)
end

function atg_sort_stable(arg1, self, stable, dim, descending)
    @runtime_error_check ccall((:atg_sort_stable, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Int64, Cint), arg1, self, stable, dim, descending)
end

function atg_sort_values(arg1, values, indices, self, dim, descending)
    @runtime_error_check ccall((:atg_sort_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, descending)
end

function atg_sort_values_stable(arg1, values, indices, self, stable, dim, descending)
    @runtime_error_check ccall((:atg_sort_values_stable, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint), arg1, values, indices, self, stable, dim, descending)
end

function atg_sparse_coo_tensor(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_sparse_coo_tensor_indices(arg1, indices, values, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, indices, values, options_kind, options_device)
end

function atg_sparse_coo_tensor_indices_size(arg1, indices, values, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor_indices_size, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, indices, values, size_data, size_len, options_kind, options_device)
end

function atg_sparse_csr_tensor(arg1, crow_indices, col_indices, values, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_csr_tensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, crow_indices, col_indices, values, options_kind, options_device)
end

function atg_sparse_csr_tensor_crow_col_value_size(arg1, crow_indices, col_indices, values, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_csr_tensor_crow_col_value_size, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, crow_indices, col_indices, values, size_data, size_len, options_kind, options_device)
end

function atg_sparse_mask(arg1, self, mask)
    @runtime_error_check ccall((:atg_sparse_mask, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mask)
end

function atg_sparse_resize_(arg1, self, size_data, size_len, sparse_dim, dense_dim)
    @runtime_error_check ccall((:atg_sparse_resize_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, self, size_data, size_len, sparse_dim, dense_dim)
end

function atg_sparse_resize_and_clear_(arg1, self, size_data, size_len, sparse_dim, dense_dim)
    @runtime_error_check ccall((:atg_sparse_resize_and_clear_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, self, size_data, size_len, sparse_dim, dense_dim)
end

function atg_special_digamma(arg1, self)
    @runtime_error_check ccall((:atg_special_digamma, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_digamma_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_digamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_entr(arg1, self)
    @runtime_error_check ccall((:atg_special_entr, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_entr_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_entr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_erf(arg1, self)
    @runtime_error_check ccall((:atg_special_erf, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_erf_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_erf_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_erfc(arg1, self)
    @runtime_error_check ccall((:atg_special_erfc, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_erfc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_erfc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_erfcx(arg1, self)
    @runtime_error_check ccall((:atg_special_erfcx, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_erfcx_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_erfcx_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_erfinv(arg1, self)
    @runtime_error_check ccall((:atg_special_erfinv, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_erfinv_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_erfinv_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_exp2(arg1, self)
    @runtime_error_check ccall((:atg_special_exp2, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_exp2_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_exp2_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_expit(arg1, self)
    @runtime_error_check ccall((:atg_special_expit, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_expit_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_expit_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_expm1(arg1, self)
    @runtime_error_check ccall((:atg_special_expm1, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_expm1_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_expm1_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_gammainc(arg1, self, other)
    @runtime_error_check ccall((:atg_special_gammainc, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_special_gammainc_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_gammainc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_special_gammaincc(arg1, self, other)
    @runtime_error_check ccall((:atg_special_gammaincc, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_special_gammaincc_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_gammaincc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_special_gammaln(arg1, self)
    @runtime_error_check ccall((:atg_special_gammaln, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_gammaln_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_gammaln_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_i0(arg1, self)
    @runtime_error_check ccall((:atg_special_i0, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_i0_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_i0_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_i0e(arg1, self)
    @runtime_error_check ccall((:atg_special_i0e, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_i0e_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_i0e_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_i1(arg1, self)
    @runtime_error_check ccall((:atg_special_i1, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_i1_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_i1_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_i1e(arg1, self)
    @runtime_error_check ccall((:atg_special_i1e, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_i1e_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_i1e_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_log1p(arg1, self)
    @runtime_error_check ccall((:atg_special_log1p, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_log1p_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_log1p_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_log_softmax(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_special_log_softmax, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_special_logit(arg1, self, eps)
    @runtime_error_check ccall((:atg_special_logit, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, eps)
end

function atg_special_logit_out(arg1, out, self, eps)
    @runtime_error_check ccall((:atg_special_logit_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, out, self, eps)
end

function atg_special_logsumexp(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_special_logsumexp, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_special_logsumexp_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_special_logsumexp_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_special_multigammaln(arg1, self, p)
    @runtime_error_check ccall((:atg_special_multigammaln, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, p)
end

function atg_special_multigammaln_out(arg1, out, self, p)
    @runtime_error_check ccall((:atg_special_multigammaln_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, p)
end

function atg_special_ndtr(arg1, self)
    @runtime_error_check ccall((:atg_special_ndtr, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_ndtr_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_ndtr_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_ndtri(arg1, self)
    @runtime_error_check ccall((:atg_special_ndtri, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_ndtri_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_ndtri_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_polygamma(arg1, n, self)
    @runtime_error_check ccall((:atg_special_polygamma, libtorch_c_api), Cint, (Ptr{tensor}, Int64, tensor), arg1, n, self)
end

function atg_special_polygamma_out(arg1, out, n, self)
    @runtime_error_check ccall((:atg_special_polygamma_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor), arg1, out, n, self)
end

function atg_special_psi(arg1, self)
    @runtime_error_check ccall((:atg_special_psi, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_psi_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_psi_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_round(arg1, self)
    @runtime_error_check ccall((:atg_special_round, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_round_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_round_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_sinc(arg1, self)
    @runtime_error_check ccall((:atg_special_sinc, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_special_sinc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_special_sinc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_special_xlog1py(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_special_xlog1py_other_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py_other_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_special_xlog1py_other_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py_other_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_special_xlog1py_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_special_xlog1py_self_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py_self_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_special_xlog1py_self_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlog1py_self_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, other)
end

function atg_special_xlogy(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlogy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_special_xlogy_other_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlogy_other_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_special_xlogy_other_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlogy_other_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_special_xlogy_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlogy_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_special_xlogy_self_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_xlogy_self_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_special_xlogy_self_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_xlogy_self_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, other)
end

function atg_special_zeta(arg1, self, other)
    @runtime_error_check ccall((:atg_special_zeta, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_special_zeta_other_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_zeta_other_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_special_zeta_other_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_zeta_other_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_special_zeta_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_zeta_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_special_zeta_self_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_special_zeta_self_scalar, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_special_zeta_self_scalar_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_special_zeta_self_scalar_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, other)
end

function atg_split(arg1, self, split_size, dim)
    @runtime_error_check ccall((:atg_split, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, split_size, dim)
end

function atg_split_with_sizes(arg1, self, split_sizes_data, split_sizes_len, dim)
    @runtime_error_check ccall((:atg_split_with_sizes, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64), arg1, self, split_sizes_data, split_sizes_len, dim)
end

function atg_sqrt(arg1, self)
    @runtime_error_check ccall((:atg_sqrt, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sqrt_(arg1, self)
    @runtime_error_check ccall((:atg_sqrt_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sqrt_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sqrt_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_square(arg1, self)
    @runtime_error_check ccall((:atg_square, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_square_(arg1, self)
    @runtime_error_check ccall((:atg_square_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_square_out(arg1, out, self)
    @runtime_error_check ccall((:atg_square_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_squeeze(arg1, self)
    @runtime_error_check ccall((:atg_squeeze, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_squeeze_(arg1, self)
    @runtime_error_check ccall((:atg_squeeze_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_squeeze_dim(arg1, self, dim)
    @runtime_error_check ccall((:atg_squeeze_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_squeeze_dim_(arg1, self, dim)
    @runtime_error_check ccall((:atg_squeeze_dim_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_sspaddmm(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_sspaddmm, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_sspaddmm_out(arg1, out, self, mat1, mat2)
    @runtime_error_check ccall((:atg_sspaddmm_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat1, mat2)
end

function atg_stack(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_stack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg_stack_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_stack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg_std(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_std, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_std_correction(arg1, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_std_correction, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, correction, keepdim)
end

function atg_std_correction_out(arg1, out, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_std_correction_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, out, self, dim_data, dim_len, correction, keepdim)
end

function atg_std_dim(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_std_mean(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_std_mean, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_std_mean_correction(arg1, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_std_mean_correction, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, correction, keepdim)
end

function atg_std_mean_dim(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std_mean_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_std_out(arg1, out, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_stft(arg1, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex)
    @runtime_error_check ccall((:atg_stft, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, tensor, Cint, Cint, Cint), arg1, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex)
end

function atg_sub(arg1, self, other)
    @runtime_error_check ccall((:atg_sub, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_sub_(arg1, self, other)
    @runtime_error_check ccall((:atg_sub_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_sub_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_sub_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_sub_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_sub_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_sub_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_sub_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_subtract(arg1, self, other)
    @runtime_error_check ccall((:atg_subtract, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_subtract_(arg1, self, other)
    @runtime_error_check ccall((:atg_subtract_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_subtract_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_subtract_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_subtract_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_subtract_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_subtract_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_subtract_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_sum(arg1, self, dtype)
    @runtime_error_check ccall((:atg_sum, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_sum_dim_intlist(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_sum_dim_intlist, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_sum_intlist_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_sum_intlist_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_sum_to_size(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_sum_to_size, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_svd(arg1, self, some, compute_uv)
    @runtime_error_check ccall((:atg_svd, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, some, compute_uv)
end

function atg_svd_u(arg1, U, S, V, self, some, compute_uv)
    @runtime_error_check ccall((:atg_svd_u, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint), arg1, U, S, V, self, some, compute_uv)
end

function atg_swapaxes(arg1, self, axis0, axis1)
    @runtime_error_check ccall((:atg_swapaxes, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, axis0, axis1)
end

function atg_swapaxes_(arg1, self, axis0, axis1)
    @runtime_error_check ccall((:atg_swapaxes_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, axis0, axis1)
end

function atg_swapdims(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_swapdims, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_swapdims_(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_swapdims_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_symeig(arg1, self, eigenvectors, upper)
    @runtime_error_check ccall((:atg_symeig, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, eigenvectors, upper)
end

function atg_symeig_e(arg1, e, V, self, eigenvectors, upper)
    @runtime_error_check ccall((:atg_symeig_e, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, e, V, self, eigenvectors, upper)
end

function atg_t(arg1, self)
    @runtime_error_check ccall((:atg_t, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_t_(arg1, self)
    @runtime_error_check ccall((:atg_t_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_take(arg1, self, index)
    @runtime_error_check ccall((:atg_take, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, index)
end

function atg_take_along_dim(arg1, self, indices, dim)
    @runtime_error_check ccall((:atg_take_along_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, indices, dim)
end

function atg_take_along_dim_out(arg1, out, self, indices, dim)
    @runtime_error_check ccall((:atg_take_along_dim_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, indices, dim)
end

function atg_take_out(arg1, out, self, index)
    @runtime_error_check ccall((:atg_take_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, index)
end

function atg_tan(arg1, self)
    @runtime_error_check ccall((:atg_tan, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tan_(arg1, self)
    @runtime_error_check ccall((:atg_tan_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tan_out(arg1, out, self)
    @runtime_error_check ccall((:atg_tan_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_tanh(arg1, self)
    @runtime_error_check ccall((:atg_tanh, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tanh_(arg1, self)
    @runtime_error_check ccall((:atg_tanh_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tanh_backward(arg1, grad_output, output)
    @runtime_error_check ccall((:atg_tanh_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, output)
end

function atg_tanh_backward_grad_input(arg1, grad_input, grad_output, output)
    @runtime_error_check ccall((:atg_tanh_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, output)
end

function atg_tanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_tanh_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_tensor_split(arg1, self, sections, dim)
    @runtime_error_check ccall((:atg_tensor_split, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, sections, dim)
end

function atg_tensor_split_indices(arg1, self, indices_data, indices_len, dim)
    @runtime_error_check ccall((:atg_tensor_split_indices, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64), arg1, self, indices_data, indices_len, dim)
end

function atg_tensor_split_tensor_indices_or_sections(arg1, self, tensor_indices_or_sections, dim)
    @runtime_error_check ccall((:atg_tensor_split_tensor_indices_or_sections, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, tensor_indices_or_sections, dim)
end

function atg_tensordot(arg1, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
    @runtime_error_check ccall((:atg_tensordot, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
end

function atg_tensordot_out(arg1, out, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
    @runtime_error_check ccall((:atg_tensordot_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
end

function atg_threshold(arg1, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, threshold, value)
end

function atg_threshold_(arg1, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, threshold, value)
end

function atg_threshold_backward(arg1, grad_output, self, threshold)
    @runtime_error_check ccall((:atg_threshold_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_output, self, threshold)
end

function atg_threshold_backward_grad_input(arg1, grad_input, grad_output, self, threshold)
    @runtime_error_check ccall((:atg_threshold_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, grad_input, grad_output, self, threshold)
end

function atg_threshold_out(arg1, out, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, out, self, threshold, value)
end

function atg_tile(arg1, self, dims_data, dims_len)
    @runtime_error_check ccall((:atg_tile, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dims_data, dims_len)
end

function atg_to(arg1, self, device)
    @runtime_error_check ccall((:atg_to, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, device)
end

function atg_to_dense(arg1, self, dtype)
    @runtime_error_check ccall((:atg_to_dense, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_to_dense_backward(arg1, grad, input)
    @runtime_error_check ccall((:atg_to_dense_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, input)
end

function atg_to_device(arg1, self, device, dtype, non_blocking, copy)
    @runtime_error_check ccall((:atg_to_device, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint, Cint), arg1, self, device, dtype, non_blocking, copy)
end

function atg_to_dtype(arg1, self, dtype, non_blocking, copy)
    @runtime_error_check ccall((:atg_to_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint), arg1, self, dtype, non_blocking, copy)
end

function atg_to_dtype_layout(arg1, self, options_kind, options_device, non_blocking, copy)
    @runtime_error_check ccall((:atg_to_dtype_layout, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint, Cint), arg1, self, options_kind, options_device, non_blocking, copy)
end

function atg_to_mkldnn(arg1, self, dtype)
    @runtime_error_check ccall((:atg_to_mkldnn, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_to_mkldnn_backward(arg1, grad, input)
    @runtime_error_check ccall((:atg_to_mkldnn_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, input)
end

function atg_to_other(arg1, self, other, non_blocking, copy)
    @runtime_error_check ccall((:atg_to_other, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, self, other, non_blocking, copy)
end

function atg_to_sparse(arg1, self)
    @runtime_error_check ccall((:atg_to_sparse, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_to_sparse_sparse_dim(arg1, self, sparse_dim)
    @runtime_error_check ccall((:atg_to_sparse_sparse_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, sparse_dim)
end

function atg_topk(arg1, self, k, dim, largest, sorted)
    @runtime_error_check ccall((:atg_topk, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cint, Cint), arg1, self, k, dim, largest, sorted)
end

function atg_topk_values(arg1, values, indices, self, k, dim, largest, sorted)
    @runtime_error_check ccall((:atg_topk_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint, Cint), arg1, values, indices, self, k, dim, largest, sorted)
end

function atg_totype(arg1, self, scalar_type)
    @runtime_error_check ccall((:atg_totype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, scalar_type)
end

function atg_trace(arg1, self)
    @runtime_error_check ccall((:atg_trace, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_trace_backward(arg1, grad, sizes_data, sizes_len)
    @runtime_error_check ccall((:atg_trace_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, grad, sizes_data, sizes_len)
end

function atg_transpose(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_transpose, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_transpose_(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_transpose_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_trapezoid(arg1, y, dim)
    @runtime_error_check ccall((:atg_trapezoid, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, y, dim)
end

function atg_trapezoid_x(arg1, y, x, dim)
    @runtime_error_check ccall((:atg_trapezoid_x, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, y, x, dim)
end

function atg_trapz(arg1, y, x, dim)
    @runtime_error_check ccall((:atg_trapz, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, y, x, dim)
end

function atg_trapz_dx(arg1, y, dx, dim)
    @runtime_error_check ccall((:atg_trapz_dx, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Int64), arg1, y, dx, dim)
end

function atg_triangular_solve(arg1, self, A, upper, transpose, unitriangular)
    @runtime_error_check ccall((:atg_triangular_solve, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint, Cint), arg1, self, A, upper, transpose, unitriangular)
end

function atg_triangular_solve_x(arg1, X, M, self, A, upper, transpose, unitriangular)
    @runtime_error_check ccall((:atg_triangular_solve_x, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint, Cint), arg1, X, M, self, A, upper, transpose, unitriangular)
end

function atg_tril(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_tril, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_tril_(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_tril_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_tril_indices(arg1, row, col, offset, options_kind, options_device)
    @runtime_error_check ccall((:atg_tril_indices, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Int64, Cint, Cint), arg1, row, col, offset, options_kind, options_device)
end

function atg_tril_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_tril_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_triplet_margin_loss(arg1, anchor, positive, negative, margin, p, eps, swap, reduction)
    @runtime_error_check ccall((:atg_triplet_margin_loss, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Cdouble, Cdouble, Cint, Int64), arg1, anchor, positive, negative, margin, p, eps, swap, reduction)
end

function atg_triu(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_triu, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_triu_(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_triu_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_triu_indices(arg1, row, col, offset, options_kind, options_device)
    @runtime_error_check ccall((:atg_triu_indices, libtorch_c_api), Cint, (Ptr{tensor}, Int64, Int64, Int64, Cint, Cint), arg1, row, col, offset, options_kind, options_device)
end

function atg_triu_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_triu_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_true_divide(arg1, self, other)
    @runtime_error_check ccall((:atg_true_divide, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_true_divide_(arg1, self, other)
    @runtime_error_check ccall((:atg_true_divide_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_true_divide_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_true_divide_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_true_divide_scalar(arg1, self, other)
    @runtime_error_check ccall((:atg_true_divide_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_true_divide_scalar_(arg1, self, other)
    @runtime_error_check ccall((:atg_true_divide_scalar_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_trunc(arg1, self)
    @runtime_error_check ccall((:atg_trunc, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_trunc_(arg1, self)
    @runtime_error_check ccall((:atg_trunc_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_trunc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_trunc_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_type_as(arg1, self, other)
    @runtime_error_check ccall((:atg_type_as, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_unbind(arg1, self, dim)
    @runtime_error_check ccall((:atg_unbind, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_unflatten(arg1, self, dim, sizes_data, sizes_len)
    @runtime_error_check ccall((:atg_unflatten, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Int64}, Cint), arg1, self, dim, sizes_data, sizes_len)
end

function atg_unflatten_dense_tensors(arg1, flat, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_unflatten_dense_tensors, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, flat, tensors_data, tensors_len)
end

function atg_unfold(arg1, self, dimension, size, step)
    @runtime_error_check ccall((:atg_unfold, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dimension, size, step)
end

function atg_unfold_backward(arg1, grad_in, input_sizes_data, input_sizes_len, dim, size, step)
    @runtime_error_check ccall((:atg_unfold_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64, Int64), arg1, grad_in, input_sizes_data, input_sizes_len, dim, size, step)
end

function atg_uniform_(arg1, self, from, to)
    @runtime_error_check ccall((:atg_uniform_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, from, to)
end

function atg_unique_consecutive(arg1, self, return_inverse, return_counts, dim)
    @runtime_error_check ccall((:atg_unique_consecutive, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint, Cint, Int64), arg1, self, return_inverse, return_counts, dim)
end

function atg_unique_dim(arg1, self, dim, sorted, return_inverse, return_counts)
    @runtime_error_check ccall((:atg_unique_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint, Cint), arg1, self, dim, sorted, return_inverse, return_counts)
end

function atg_unique_dim_consecutive(arg1, self, dim, return_inverse, return_counts)
    @runtime_error_check ccall((:atg_unique_dim_consecutive, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, dim, return_inverse, return_counts)
end

function atg_unsafe_chunk(arg1, self, chunks, dim)
    @runtime_error_check ccall((:atg_unsafe_chunk, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, chunks, dim)
end

function atg_unsafe_split(arg1, self, split_size, dim)
    @runtime_error_check ccall((:atg_unsafe_split, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, split_size, dim)
end

function atg_unsafe_split_with_sizes(arg1, self, split_sizes_data, split_sizes_len, dim)
    @runtime_error_check ccall((:atg_unsafe_split_with_sizes, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64), arg1, self, split_sizes_data, split_sizes_len, dim)
end

function atg_unsqueeze(arg1, self, dim)
    @runtime_error_check ccall((:atg_unsqueeze, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_unsqueeze_(arg1, self, dim)
    @runtime_error_check ccall((:atg_unsqueeze_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_upsample_bicubic2d(arg1, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bicubic2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bicubic2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bicubic2d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bicubic2d_out(arg1, out, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, out, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bilinear2d(arg1, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bilinear2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bilinear2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bilinear2d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_bilinear2d_out(arg1, out, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble), arg1, out, self, output_size_data, output_size_len, align_corners, scales_h, scales_w)
end

function atg_upsample_linear1d(arg1, self, output_size_data, output_size_len, align_corners, scales)
    @runtime_error_check ccall((:atg_upsample_linear1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cdouble), arg1, self, output_size_data, output_size_len, align_corners, scales)
end

function atg_upsample_linear1d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales)
    @runtime_error_check ccall((:atg_upsample_linear1d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales)
end

function atg_upsample_linear1d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales)
    @runtime_error_check ccall((:atg_upsample_linear1d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales)
end

function atg_upsample_linear1d_out(arg1, out, self, output_size_data, output_size_len, align_corners, scales)
    @runtime_error_check ccall((:atg_upsample_linear1d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cdouble), arg1, out, self, output_size_data, output_size_len, align_corners, scales)
end

function atg_upsample_nearest1d(arg1, self, output_size_data, output_size_len, scales)
    @runtime_error_check ccall((:atg_upsample_nearest1d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cdouble), arg1, self, output_size_data, output_size_len, scales)
end

function atg_upsample_nearest1d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales)
    @runtime_error_check ccall((:atg_upsample_nearest1d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales)
end

function atg_upsample_nearest1d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales)
    @runtime_error_check ccall((:atg_upsample_nearest1d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales)
end

function atg_upsample_nearest1d_out(arg1, out, self, output_size_data, output_size_len, scales)
    @runtime_error_check ccall((:atg_upsample_nearest1d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cdouble), arg1, out, self, output_size_data, output_size_len, scales)
end

function atg_upsample_nearest2d(arg1, self, output_size_data, output_size_len, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest2d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cdouble, Cdouble), arg1, self, output_size_data, output_size_len, scales_h, scales_w)
end

function atg_upsample_nearest2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest2d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_h, scales_w)
end

function atg_upsample_nearest2d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest2d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_h, scales_w)
end

function atg_upsample_nearest2d_out(arg1, out, self, output_size_data, output_size_len, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest2d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cdouble, Cdouble), arg1, out, self, output_size_data, output_size_len, scales_h, scales_w)
end

function atg_upsample_nearest3d(arg1, self, output_size_data, output_size_len, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cdouble, Cdouble, Cdouble), arg1, self, output_size_data, output_size_len, scales_d, scales_h, scales_w)
end

function atg_upsample_nearest3d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble, Cdouble, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_d, scales_h, scales_w)
end

function atg_upsample_nearest3d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cdouble, Cdouble, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, scales_d, scales_h, scales_w)
end

function atg_upsample_nearest3d_out(arg1, out, self, output_size_data, output_size_len, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_nearest3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cdouble, Cdouble, Cdouble), arg1, out, self, output_size_data, output_size_len, scales_d, scales_h, scales_w)
end

function atg_upsample_trilinear3d(arg1, self, output_size_data, output_size_len, align_corners, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_trilinear3d, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble, Cdouble), arg1, self, output_size_data, output_size_len, align_corners, scales_d, scales_h, scales_w)
end

function atg_upsample_trilinear3d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble, Cdouble), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_d, scales_h, scales_w)
end

function atg_upsample_trilinear3d_backward_grad_input(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_backward_grad_input, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble, Cdouble), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners, scales_d, scales_h, scales_w)
end

function atg_upsample_trilinear3d_out(arg1, out, self, output_size_data, output_size_len, align_corners, scales_d, scales_h, scales_w)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cdouble, Cdouble, Cdouble), arg1, out, self, output_size_data, output_size_len, align_corners, scales_d, scales_h, scales_w)
end

function atg_value_selecting_reduction_backward(arg1, grad, dim, indices, sizes_data, sizes_len, keepdim)
    @runtime_error_check ccall((:atg_value_selecting_reduction_backward, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, tensor, Ptr{Int64}, Cint, Cint), arg1, grad, dim, indices, sizes_data, sizes_len, keepdim)
end

function atg_values(arg1, self)
    @runtime_error_check ccall((:atg_values, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_vander(arg1, x, n, increasing)
    @runtime_error_check ccall((:atg_vander, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, x, n, increasing)
end

function atg_var(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_var, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_var_correction(arg1, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_var_correction, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, correction, keepdim)
end

function atg_var_correction_out(arg1, out, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_var_correction_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, out, self, dim_data, dim_len, correction, keepdim)
end

function atg_var_dim(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_var_mean(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_var_mean, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_var_mean_correction(arg1, self, dim_data, dim_len, correction, keepdim)
    @runtime_error_check ccall((:atg_var_mean_correction, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Cint), arg1, self, dim_data, dim_len, correction, keepdim)
end

function atg_var_mean_dim(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var_mean_dim, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_var_out(arg1, out, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_vdot(arg1, self, other)
    @runtime_error_check ccall((:atg_vdot, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_vdot_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_vdot_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_view(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_view, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_view_as(arg1, self, other)
    @runtime_error_check ccall((:atg_view_as, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_view_as_complex(arg1, self)
    @runtime_error_check ccall((:atg_view_as_complex, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_view_as_real(arg1, self)
    @runtime_error_check ccall((:atg_view_as_real, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_view_dtype(arg1, self, dtype)
    @runtime_error_check ccall((:atg_view_dtype, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_vsplit(arg1, self, sections)
    @runtime_error_check ccall((:atg_vsplit, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, sections)
end

function atg_vsplit_array(arg1, self, indices_data, indices_len)
    @runtime_error_check ccall((:atg_vsplit_array, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, indices_data, indices_len)
end

function atg_vstack(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_vstack, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_vstack_out(arg1, out, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_vstack_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, out, tensors_data, tensors_len)
end

function atg_where(arg1, condition)
    @runtime_error_check ccall((:atg_where, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, condition)
end

function atg_where_scalar(arg1, condition, self, other)
    @runtime_error_check ccall((:atg_where_scalar, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, condition, self, other)
end

function atg_where_scalarother(arg1, condition, self, other)
    @runtime_error_check ccall((:atg_where_scalarother, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, condition, self, other)
end

function atg_where_scalarself(arg1, condition, self, other)
    @runtime_error_check ccall((:atg_where_scalarself, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, condition, self, other)
end

function atg_where_self(arg1, condition, self, other)
    @runtime_error_check ccall((:atg_where_self, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, condition, self, other)
end

function atg_xlogy(arg1, self, other)
    @runtime_error_check ccall((:atg_xlogy, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_xlogy_(arg1, self, other)
    @runtime_error_check ccall((:atg_xlogy_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_xlogy_outscalar_other(arg1, out, self, other)
    @runtime_error_check ccall((:atg_xlogy_outscalar_other, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_xlogy_outscalar_self(arg1, out, self, other)
    @runtime_error_check ccall((:atg_xlogy_outscalar_self, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, other)
end

function atg_xlogy_outtensor(arg1, out, self, other)
    @runtime_error_check ccall((:atg_xlogy_outtensor, libtorch_c_api), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_xlogy_scalar_other(arg1, self, other)
    @runtime_error_check ccall((:atg_xlogy_scalar_other, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_xlogy_scalar_other_(arg1, self, other)
    @runtime_error_check ccall((:atg_xlogy_scalar_other_, libtorch_c_api), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_xlogy_scalar_self(arg1, self, other)
    @runtime_error_check ccall((:atg_xlogy_scalar_self, libtorch_c_api), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, other)
end

function atg_zero_(arg1, self)
    @runtime_error_check ccall((:atg_zero_, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_zeros(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_zeros, libtorch_c_api), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_zeros_like(arg1, self)
    @runtime_error_check ccall((:atg_zeros_like, libtorch_c_api), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_zeros_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_zeros_out, libtorch_c_api), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

# exports
const PREFIXES = ["at"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
