module Wrapper

using Torch_jll
export Torch_jll

using CEnum

function get_error()
  err = cglobal((:myerr, libdoeye_caml), Cstring) |> unsafe_load
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

const ngg = Ptr{Cvoid}

function get_last_error(arg1)
    @runtime_error_check ccall((:get_last_error, libdoeye_caml), Cint, (Ptr{Cchar},), arg1)
end

# no prototype is found for this function at torch_api.h:32:5, please use with caution
function flush_error()
    @runtime_error_check ccall((:flush_error, libdoeye_caml), Cint, ())
end

function at_manual_seed(arg1)
    @runtime_error_check ccall((:at_manual_seed, libdoeye_caml), Cint, (Int64,), arg1)
end

function at_new_tensor(arg1)
    @runtime_error_check ccall((:at_new_tensor, libdoeye_caml), Cint, (Ptr{tensor},), arg1)
end

# no prototype is found for this function at torch_api.h:36:5, please use with caution
function at_empty_cache()
    @runtime_error_check ccall((:at_empty_cache, libdoeye_caml), Cint, ())
end

function at_no_grad(flag)
    @runtime_error_check ccall((:at_no_grad, libdoeye_caml), Cint, (Cint,), flag)
end

# no prototype is found for this function at torch_api.h:38:5, please use with caution
function at_sync()
    @runtime_error_check ccall((:at_sync, libdoeye_caml), Cint, ())
end

function at_from_blob(arg1, data, dims, ndims, strides, nstrides, dev)
    @runtime_error_check ccall((:at_from_blob, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Cvoid}, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, data, dims, ndims, strides, nstrides, dev)
end

function at_tensor_of_data(arg1, vs, dims, ndims, element_size_in_bytes, type)
    @runtime_error_check ccall((:at_tensor_of_data, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Cvoid}, Ptr{Int64}, Cint, Cint, Cint), arg1, vs, dims, ndims, element_size_in_bytes, type)
end

function at_copy_data(tensor_, vs, numel, element_size_in_bytes)
    @runtime_error_check ccall((:at_copy_data, libdoeye_caml), Cint, (tensor, Ptr{Cvoid}, Int64, Cint), tensor_, vs, numel, element_size_in_bytes)
end

function at_float_vec(values, value_len, type)
    @runtime_error_check ccall((:at_float_vec, libdoeye_caml), Cint, (Ptr{Cdouble}, Cint, Cint), values, value_len, type)
end

function at_int_vec(values, value_len, type)
    @runtime_error_check ccall((:at_int_vec, libdoeye_caml), Cint, (Ptr{Int64}, Cint, Cint), values, value_len, type)
end

function at_defined(i, arg2)
    @runtime_error_check ccall((:at_defined, libdoeye_caml), Cint, (Ptr{Cint}, tensor), i, arg2)
end

function at_dim(i, arg2)
    @runtime_error_check ccall((:at_dim, libdoeye_caml), Cint, (Ptr{Cint}, tensor), i, arg2)
end

function at_shape(arg1, arg2)
    @runtime_error_check ccall((:at_shape, libdoeye_caml), Cint, (tensor, Ptr{Cint}), arg1, arg2)
end

function at_scalar_type(i, arg2)
    @runtime_error_check ccall((:at_scalar_type, libdoeye_caml), Cint, (Ptr{Cint}, tensor), i, arg2)
end

function at_backward(arg1, arg2, arg3)
    @runtime_error_check ccall((:at_backward, libdoeye_caml), Cint, (tensor, Cint, Cint), arg1, arg2, arg3)
end

function at_requires_grad(i, arg2)
    @runtime_error_check ccall((:at_requires_grad, libdoeye_caml), Cint, (Ptr{Cint}, tensor), i, arg2)
end

function at_grad_set_enabled(arg1)
    @runtime_error_check ccall((:at_grad_set_enabled, libdoeye_caml), Cint, (Cint,), arg1)
end

function at_get(arg1, arg2, index)
    @runtime_error_check ccall((:at_get, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, arg2, index)
end

function at_fill_double(arg1, arg2)
    @runtime_error_check ccall((:at_fill_double, libdoeye_caml), Cint, (tensor, Cdouble), arg1, arg2)
end

function at_fill_int64(arg1, arg2)
    @runtime_error_check ccall((:at_fill_int64, libdoeye_caml), Cint, (tensor, Int64), arg1, arg2)
end

function at_double_value_at_indexes(i, arg2, indexes, indexes_len)
    @runtime_error_check ccall((:at_double_value_at_indexes, libdoeye_caml), Cint, (Ptr{Cdouble}, tensor, Ptr{Cint}, Cint), i, arg2, indexes, indexes_len)
end

function at_int64_value_at_indexes(i, arg2, indexes, indexes_len)
    @runtime_error_check ccall((:at_int64_value_at_indexes, libdoeye_caml), Cint, (Ptr{Cdouble}, tensor, Ptr{Cint}, Cint), i, arg2, indexes, indexes_len)
end

function at_set_double_value_at_indexes(arg1, indexes, indexes_len, v)
    @runtime_error_check ccall((:at_set_double_value_at_indexes, libdoeye_caml), Cint, (tensor, Ptr{Cint}, Cint, Cdouble), arg1, indexes, indexes_len, v)
end

function at_set_int64_value_at_indexes(arg1, indexes, indexes_len, v)
    @runtime_error_check ccall((:at_set_int64_value_at_indexes, libdoeye_caml), Cint, (tensor, Ptr{Cint}, Cint, Int64), arg1, indexes, indexes_len, v)
end

function at_copy_(dst, src)
    @runtime_error_check ccall((:at_copy_, libdoeye_caml), Cint, (tensor, tensor), dst, src)
end

function at_print(arg1)
    @runtime_error_check ccall((:at_print, libdoeye_caml), Cint, (tensor,), arg1)
end

function at_save(arg1, filename)
    @runtime_error_check ccall((:at_save, libdoeye_caml), Cint, (tensor, Ptr{Cchar}), arg1, filename)
end

function at_load(filename)
    @runtime_error_check ccall((:at_load, libdoeye_caml), tensor, (Ptr{Cchar},), filename)
end

function at_save_multi(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_save_multi, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_multi(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_load_multi, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_multi_(tensors, tensor_names, ntensors, filename)
    @runtime_error_check ccall((:at_load_multi_, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Ptr{Cchar}}, Cint, Ptr{Cchar}), tensors, tensor_names, ntensors, filename)
end

function at_load_callback(filename, f)
    @runtime_error_check ccall((:at_load_callback, libdoeye_caml), Cint, (Ptr{Cchar}, Ptr{Cvoid}), filename, f)
end

function at_free(arg1)
    @runtime_error_check ccall((:at_free, libdoeye_caml), Cint, (tensor,), arg1)
end

function at_run_backward(tensors, ntensors, inputs, ninputs, outputs, keep_graph, create_graph)
    @runtime_error_check ccall((:at_run_backward, libdoeye_caml), Cint, (Ptr{tensor}, Cint, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint), tensors, ntensors, inputs, ninputs, outputs, keep_graph, create_graph)
end

function ato_adam(arg1, learning_rate, beta1, beta2, weight_decay)
    @runtime_error_check ccall((:ato_adam, libdoeye_caml), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble), arg1, learning_rate, beta1, beta2, weight_decay)
end

function ato_rmsprop(arg1, learning_rate, alpha, eps, weight_decay, momentum, centered)
    @runtime_error_check ccall((:ato_rmsprop, libdoeye_caml), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint), arg1, learning_rate, alpha, eps, weight_decay, momentum, centered)
end

function ato_sgd(arg1, learning_rate, momentum, dampening, weight_decay, nesterov)
    @runtime_error_check ccall((:ato_sgd, libdoeye_caml), Cint, (Ptr{optimizer}, Cdouble, Cdouble, Cdouble, Cdouble, Cint), arg1, learning_rate, momentum, dampening, weight_decay, nesterov)
end

function ato_add_parameters(arg1, arg2, ntensors)
    @runtime_error_check ccall((:ato_add_parameters, libdoeye_caml), Cint, (optimizer, Ptr{tensor}, Cint), arg1, arg2, ntensors)
end

function ato_set_learning_rate(arg1, learning_rate)
    @runtime_error_check ccall((:ato_set_learning_rate, libdoeye_caml), Cint, (optimizer, Cdouble), arg1, learning_rate)
end

function ato_set_momentum(arg1, momentum)
    @runtime_error_check ccall((:ato_set_momentum, libdoeye_caml), Cint, (optimizer, Cdouble), arg1, momentum)
end

function ato_zero_grad(arg1)
    @runtime_error_check ccall((:ato_zero_grad, libdoeye_caml), Cint, (optimizer,), arg1)
end

function ato_step(arg1)
    @runtime_error_check ccall((:ato_step, libdoeye_caml), Cint, (optimizer,), arg1)
end

function ato_free(arg1)
    @runtime_error_check ccall((:ato_free, libdoeye_caml), Cint, (optimizer,), arg1)
end

function ats_int(arg1, arg2)
    @runtime_error_check ccall((:ats_int, libdoeye_caml), Cint, (Ptr{scalar}, Int64), arg1, arg2)
end

function ats_float(arg1, arg2)
    @runtime_error_check ccall((:ats_float, libdoeye_caml), Cint, (Ptr{scalar}, Cdouble), arg1, arg2)
end

function ats_free(arg1)
    @runtime_error_check ccall((:ats_free, libdoeye_caml), Cint, (scalar,), arg1)
end

function atc_cuda_device_count(arg1)
    @runtime_error_check ccall((:atc_cuda_device_count, libdoeye_caml), Cint, (Ptr{Cint},), arg1)
end

function atc_cuda_is_available(arg1)
    @runtime_error_check ccall((:atc_cuda_is_available, libdoeye_caml), Cint, (Ptr{Cint},), arg1)
end

function atc_cudnn_is_available(arg1)
    @runtime_error_check ccall((:atc_cudnn_is_available, libdoeye_caml), Cint, (Ptr{Cint},), arg1)
end

function atc_set_benchmark_cudnn(b)
    @runtime_error_check ccall((:atc_set_benchmark_cudnn, libdoeye_caml), Cint, (Cint,), b)
end

function atm_load(arg1, arg2)
    @runtime_error_check ccall((:atm_load, libdoeye_caml), Cint, (Ptr{Cchar}, Ptr{_module}), arg1, arg2)
end

function atm_forward(arg1, arg2, tensors, ntensors)
    @runtime_error_check ccall((:atm_forward, libdoeye_caml), Cint, (Ptr{tensor}, _module, Ptr{tensor}, Cint), arg1, arg2, tensors, ntensors)
end

function atm_forward_(arg1, arg2, ivalues, nivalues)
    @runtime_error_check ccall((:atm_forward_, libdoeye_caml), Cint, (Ptr{ivalue}, _module, Ptr{ivalue}, Cint), arg1, arg2, ivalues, nivalues)
end

function atm_free(arg1)
    @runtime_error_check ccall((:atm_free, libdoeye_caml), Cint, (_module,), arg1)
end

function ati_tensor(arg1, arg2)
    @runtime_error_check ccall((:ati_tensor, libdoeye_caml), Cint, (Ptr{ivalue}, tensor), arg1, arg2)
end

function ati_int(arg1, arg2)
    @runtime_error_check ccall((:ati_int, libdoeye_caml), Cint, (Ptr{ivalue}, Int64), arg1, arg2)
end

function ati_double(arg1, arg2)
    @runtime_error_check ccall((:ati_double, libdoeye_caml), Cint, (Ptr{ivalue}, Cdouble), arg1, arg2)
end

function ati_tuple(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_tuple, libdoeye_caml), Cint, (Ptr{ivalue}, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_to_tensor(arg1, arg2)
    @runtime_error_check ccall((:ati_to_tensor, libdoeye_caml), Cint, (Ptr{tensor}, ivalue), arg1, arg2)
end

function ati_to_int(arg1, arg2)
    @runtime_error_check ccall((:ati_to_int, libdoeye_caml), Cint, (Ptr{Int64}, ivalue), arg1, arg2)
end

function ati_to_double(arg1, arg2)
    @runtime_error_check ccall((:ati_to_double, libdoeye_caml), Cint, (Ptr{Cdouble}, ivalue), arg1, arg2)
end

function ati_tuple_length(arg1, arg2)
    @runtime_error_check ccall((:ati_tuple_length, libdoeye_caml), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_to_tuple(arg1, arg2, arg3)
    @runtime_error_check ccall((:ati_to_tuple, libdoeye_caml), Cint, (ivalue, Ptr{ivalue}, Cint), arg1, arg2, arg3)
end

function ati_tag(arg1, arg2)
    @runtime_error_check ccall((:ati_tag, libdoeye_caml), Cint, (Ptr{Cint}, ivalue), arg1, arg2)
end

function ati_free(arg1)
    @runtime_error_check ccall((:ati_free, libdoeye_caml), Cint, (ivalue,), arg1)
end

function atg_abs(arg1, self)
    @runtime_error_check ccall((:atg_abs, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_abs_(arg1, self)
    @runtime_error_check ccall((:atg_abs_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_abs_out(arg1, out, self)
    @runtime_error_check ccall((:atg_abs_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_acos(arg1, self)
    @runtime_error_check ccall((:atg_acos, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acos_(arg1, self)
    @runtime_error_check ccall((:atg_acos_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_acos_out(arg1, out, self)
    @runtime_error_check ccall((:atg_acos_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_adaptive_avg_pool1d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool2d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_avg_pool3d_backward(arg1, grad_output, self)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, self)
end

function atg_adaptive_avg_pool3d_backward_out(arg1, grad_input, grad_output, self)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, self)
end

function atg_adaptive_avg_pool3d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_avg_pool3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool1d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool2d_backward(arg1, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, indices)
end

function atg_adaptive_max_pool2d_backward_out(arg1, grad_input, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, indices)
end

function atg_adaptive_max_pool2d_out(arg1, out, indices, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, indices, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_adaptive_max_pool3d_backward(arg1, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, indices)
end

function atg_adaptive_max_pool3d_backward_out(arg1, grad_input, grad_output, self, indices)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, indices)
end

function atg_adaptive_max_pool3d_out(arg1, out, indices, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_adaptive_max_pool3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, indices, self, output_size_data, output_size_len)
end

function atg_add(arg1, self, other)
    @runtime_error_check ccall((:atg_add, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_add1(arg1, self, other)
    @runtime_error_check ccall((:atg_add1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_add_(arg1, self, other)
    @runtime_error_check ccall((:atg_add_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_add_1(arg1, self, other)
    @runtime_error_check ccall((:atg_add_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_add_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_add_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_addbmm(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_addbmm_(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_addbmm_out(arg1, out, self, batch1, batch2)
    @runtime_error_check ccall((:atg_addbmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, batch1, batch2)
end

function atg_addcdiv(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcdiv_(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcdiv_out(arg1, out, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcdiv_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, tensor1, tensor2)
end

function atg_addcmul(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcmul_(arg1, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, tensor1, tensor2)
end

function atg_addcmul_out(arg1, out, self, tensor1, tensor2)
    @runtime_error_check ccall((:atg_addcmul_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, tensor1, tensor2)
end

function atg_addmm(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_addmm_(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_addmm_out(arg1, out, self, mat1, mat2)
    @runtime_error_check ccall((:atg_addmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat1, mat2)
end

function atg_addmv(arg1, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat, vec)
end

function atg_addmv_(arg1, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat, vec)
end

function atg_addmv_out(arg1, out, self, mat, vec)
    @runtime_error_check ccall((:atg_addmv_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat, vec)
end

function atg_addr(arg1, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, vec1, vec2)
end

function atg_addr_(arg1, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, vec1, vec2)
end

function atg_addr_out(arg1, out, self, vec1, vec2)
    @runtime_error_check ccall((:atg_addr_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, vec1, vec2)
end

function atg_affine_grid_generator(arg1, theta, size_data, size_len, align_corners)
    @runtime_error_check ccall((:atg_affine_grid_generator, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, theta, size_data, size_len, align_corners)
end

function atg_affine_grid_generator_backward(arg1, grad, size_data, size_len, align_corners)
    @runtime_error_check ccall((:atg_affine_grid_generator_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, grad, size_data, size_len, align_corners)
end

function atg_alias(arg1, self)
    @runtime_error_check ccall((:atg_alias, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_align_as(arg1, self, other)
    @runtime_error_check ccall((:atg_align_as, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_align_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_align_tensors, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_all(arg1, self)
    @runtime_error_check ccall((:atg_all, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_all1(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_all1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_all_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_all_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_alpha_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_alpha_dropout, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_alpha_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_alpha_dropout_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_angle(arg1, self)
    @runtime_error_check ccall((:atg_angle, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_angle_out(arg1, out, self)
    @runtime_error_check ccall((:atg_angle_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_any(arg1, self)
    @runtime_error_check ccall((:atg_any, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_any1(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_any1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_any_out(arg1, out, self, dim, keepdim)
    @runtime_error_check ccall((:atg_any_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, keepdim)
end

function atg_arange(arg1, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange, libdoeye_caml), Cint, (Ptr{tensor}, scalar, Cint, Cint), arg1, _end, options_kind, options_device)
end

function atg_arange1(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange1, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_arange2(arg1, start, _end, step, options_kind, options_device)
    @runtime_error_check ccall((:atg_arange2, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, scalar, Cint, Cint), arg1, start, _end, step, options_kind, options_device)
end

function atg_arange_out(arg1, out, _end)
    @runtime_error_check ccall((:atg_arange_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, out, _end)
end

function atg_arange_out1(arg1, out, start, _end)
    @runtime_error_check ccall((:atg_arange_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, out, start, _end)
end

function atg_argmax(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmax, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_argmin(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_argmin, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_argsort(arg1, self, dim, descending)
    @runtime_error_check ccall((:atg_argsort, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, descending)
end

function atg_as_strided(arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
    @runtime_error_check ccall((:atg_as_strided, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
end

function atg_as_strided_(arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
    @runtime_error_check ccall((:atg_as_strided_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, size_data, size_len, stride_data, stride_len, storage_offset)
end

function atg_asin(arg1, self)
    @runtime_error_check ccall((:atg_asin, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asin_(arg1, self)
    @runtime_error_check ccall((:atg_asin_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_asin_out(arg1, out, self)
    @runtime_error_check ccall((:atg_asin_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_atan(arg1, self)
    @runtime_error_check ccall((:atg_atan, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atan2(arg1, self, other)
    @runtime_error_check ccall((:atg_atan2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_atan2_(arg1, self, other)
    @runtime_error_check ccall((:atg_atan2_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_atan2_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_atan2_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_atan_(arg1, self)
    @runtime_error_check ccall((:atg_atan_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_atan_out(arg1, out, self)
    @runtime_error_check ccall((:atg_atan_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_avg_pool1d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad)
    @runtime_error_check ccall((:atg_avg_pool1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad)
end

function atg_avg_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool2d_out(arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_avg_pool3d_out(arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
    @runtime_error_check ccall((:atg_avg_pool3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint, Int64), arg1, out, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, ceil_mode, count_include_pad, divisor_override)
end

function atg_baddbmm(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_baddbmm_(arg1, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, batch1, batch2)
end

function atg_baddbmm_out(arg1, out, self, batch1, batch2)
    @runtime_error_check ccall((:atg_baddbmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, batch1, batch2)
end

function atg_bartlett_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_bartlett_window, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_bartlett_window1(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_bartlett_window1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_batch_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble, Cint), arg1, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
end

function atg_batch_norm_backward_elemt(arg1, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu)
    @runtime_error_check ccall((:atg_batch_norm_backward_elemt, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor), arg1, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu)
end

function atg_batch_norm_backward_reduce(arg1, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g)
    @runtime_error_check ccall((:atg_batch_norm_backward_reduce, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cint, Cint), arg1, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g)
end

function atg_batch_norm_elemt(arg1, input, weight, bias, mean, invstd, eps)
    @runtime_error_check ccall((:atg_batch_norm_elemt, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, input, weight, bias, mean, invstd, eps)
end

function atg_batch_norm_elemt_out(arg1, out, input, weight, bias, mean, invstd, eps)
    @runtime_error_check ccall((:atg_batch_norm_elemt_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, out, input, weight, bias, mean, invstd, eps)
end

function atg_batch_norm_gather_stats(arg1, input, mean, invstd, running_mean, running_var, momentum, eps, count)
    @runtime_error_check ccall((:atg_batch_norm_gather_stats, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble, Cdouble, Int64), arg1, input, mean, invstd, running_mean, running_var, momentum, eps, count)
end

function atg_batch_norm_gather_stats_with_counts(arg1, input, mean, invstd, running_mean, running_var, momentum, eps, counts_data, counts_len)
    @runtime_error_check ccall((:atg_batch_norm_gather_stats_with_counts, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cdouble, Cdouble, Ptr{Int64}, Cint), arg1, input, mean, invstd, running_mean, running_var, momentum, eps, counts_data, counts_len)
end

function atg_batch_norm_stats(arg1, input, eps)
    @runtime_error_check ccall((:atg_batch_norm_stats, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, input, eps)
end

function atg_batch_norm_update_stats(arg1, input, running_mean, running_var, momentum)
    @runtime_error_check ccall((:atg_batch_norm_update_stats, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble), arg1, input, running_mean, running_var, momentum)
end

function atg_bernoulli(arg1, self)
    @runtime_error_check ccall((:atg_bernoulli, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bernoulli1(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_bernoulli_(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, p)
end

function atg_bernoulli_1(arg1, self, p)
    @runtime_error_check ccall((:atg_bernoulli_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_bernoulli_out(arg1, out, self)
    @runtime_error_check ccall((:atg_bernoulli_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_bilinear(arg1, input1, input2, weight, bias)
    @runtime_error_check ccall((:atg_bilinear, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, input1, input2, weight, bias)
end

function atg_binary_cross_entropy(arg1, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, self, target, weight, reduction)
end

function atg_binary_cross_entropy_backward(arg1, grad_output, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, weight, reduction)
end

function atg_binary_cross_entropy_backward_out(arg1, grad_input, grad_output, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, weight, reduction)
end

function atg_binary_cross_entropy_out(arg1, out, self, target, weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, out, self, target, weight, reduction)
end

function atg_binary_cross_entropy_with_logits(arg1, self, target, weight, pos_weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_with_logits, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, self, target, weight, pos_weight, reduction)
end

function atg_binary_cross_entropy_with_logits_backward(arg1, grad_output, self, target, weight, pos_weight, reduction)
    @runtime_error_check ccall((:atg_binary_cross_entropy_with_logits_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, weight, pos_weight, reduction)
end

function atg_bincount(arg1, self, weights, minlength)
    @runtime_error_check ccall((:atg_bincount, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, weights, minlength)
end

function atg_bitwise_not(arg1, self)
    @runtime_error_check ccall((:atg_bitwise_not, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bitwise_not_(arg1, self)
    @runtime_error_check ccall((:atg_bitwise_not_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_bitwise_not_out(arg1, out, self)
    @runtime_error_check ccall((:atg_bitwise_not_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_bitwise_xor(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_xor1(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_xor_(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_bitwise_xor_1(arg1, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_bitwise_xor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_bitwise_xor_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_bitwise_xor_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_blackman_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_blackman_window, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_blackman_window1(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_blackman_window1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_bmm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_bmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_bmm_out(arg1, out, self, mat2)
    @runtime_error_check ccall((:atg_bmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mat2)
end

function atg_broadcast_tensors(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_broadcast_tensors, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_cartesian_prod(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_cartesian_prod, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_cat(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_cat, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg_cat_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_cat_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg_cauchy_(arg1, self, median, sigma)
    @runtime_error_check ccall((:atg_cauchy_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, median, sigma)
end

function atg_cdist(arg1, x1, x2, p, compute_mode)
    @runtime_error_check ccall((:atg_cdist, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64), arg1, x1, x2, p, compute_mode)
end

function atg_ceil(arg1, self)
    @runtime_error_check ccall((:atg_ceil, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ceil_(arg1, self)
    @runtime_error_check ccall((:atg_ceil_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ceil_out(arg1, out, self)
    @runtime_error_check ccall((:atg_ceil_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_celu(arg1, self)
    @runtime_error_check ccall((:atg_celu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_celu_(arg1, self)
    @runtime_error_check ccall((:atg_celu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_chain_matmul(arg1, matrices_data, matrices_len)
    @runtime_error_check ccall((:atg_chain_matmul, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, matrices_data, matrices_len)
end

function atg_cholesky(arg1, self, upper)
    @runtime_error_check ccall((:atg_cholesky, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, upper)
end

function atg_cholesky_inverse(arg1, self, upper)
    @runtime_error_check ccall((:atg_cholesky_inverse, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, upper)
end

function atg_cholesky_inverse_out(arg1, out, self, upper)
    @runtime_error_check ccall((:atg_cholesky_inverse_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, upper)
end

function atg_cholesky_out(arg1, out, self, upper)
    @runtime_error_check ccall((:atg_cholesky_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, upper)
end

function atg_cholesky_solve(arg1, self, input2, upper)
    @runtime_error_check ccall((:atg_cholesky_solve, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, input2, upper)
end

function atg_cholesky_solve_out(arg1, out, self, input2, upper)
    @runtime_error_check ccall((:atg_cholesky_solve_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, self, input2, upper)
end

function atg_chunk(arg1, self, chunks, dim)
    @runtime_error_check ccall((:atg_chunk, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, chunks, dim)
end

function atg_clamp(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clamp_(arg1, self, min, max)
    @runtime_error_check ccall((:atg_clamp_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, min, max)
end

function atg_clamp_max(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, max)
end

function atg_clamp_max_(arg1, self, max)
    @runtime_error_check ccall((:atg_clamp_max_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, max)
end

function atg_clamp_max_out(arg1, out, self, max)
    @runtime_error_check ccall((:atg_clamp_max_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, max)
end

function atg_clamp_min(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, min)
end

function atg_clamp_min_(arg1, self, min)
    @runtime_error_check ccall((:atg_clamp_min_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, min)
end

function atg_clamp_min_out(arg1, out, self, min)
    @runtime_error_check ccall((:atg_clamp_min_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, min)
end

function atg_clamp_out(arg1, out, self, min, max)
    @runtime_error_check ccall((:atg_clamp_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, out, self, min, max)
end

function atg_clone(arg1, self)
    @runtime_error_check ccall((:atg_clone, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_coalesce(arg1, self)
    @runtime_error_check ccall((:atg_coalesce, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_col2im(arg1, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_backward(arg1, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_backward_out(arg1, grad_input, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_col2im_out(arg1, out, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_col2im_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_combinations(arg1, self, r, with_replacement)
    @runtime_error_check ccall((:atg_combinations, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, r, with_replacement)
end

function atg_conj(arg1, self)
    @runtime_error_check ccall((:atg_conj, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conj_out(arg1, out, self)
    @runtime_error_check ccall((:atg_conj_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_constant_pad_nd(arg1, self, pad_data, pad_len)
    @runtime_error_check ccall((:atg_constant_pad_nd, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, pad_data, pad_len)
end

function atg_contiguous(arg1, self)
    @runtime_error_check ccall((:atg_contiguous, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_conv1d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv2d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv3d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_conv3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, groups)
end

function atg_conv_tbc(arg1, self, weight, bias, pad)
    @runtime_error_check ccall((:atg_conv_tbc, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, self, weight, bias, pad)
end

function atg_conv_tbc_backward(arg1, self, input, weight, bias, pad)
    @runtime_error_check ccall((:atg_conv_tbc_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, self, input, weight, bias, pad)
end

function atg_conv_transpose1d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_conv_transpose2d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_conv_transpose3d(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_conv_transpose3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Ptr{Int64}, Cint), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, groups, dilation_data, dilation_len)
end

function atg_convolution(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
    @runtime_error_check ccall((:atg_convolution, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
end

function atg_convolution_overrideable(arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
    @runtime_error_check ccall((:atg_convolution_overrideable, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Ptr{Int64}, Cint, Int64), arg1, input, weight, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, transposed, output_padding_data, output_padding_len, groups)
end

function atg_copy_sparse_to_sparse_(arg1, self, src, non_blocking)
    @runtime_error_check ccall((:atg_copy_sparse_to_sparse_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, src, non_blocking)
end

function atg_cos(arg1, self)
    @runtime_error_check ccall((:atg_cos, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cos_(arg1, self)
    @runtime_error_check ccall((:atg_cos_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cos_out(arg1, out, self)
    @runtime_error_check ccall((:atg_cos_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_cosh(arg1, self)
    @runtime_error_check ccall((:atg_cosh, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cosh_(arg1, self)
    @runtime_error_check ccall((:atg_cosh_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_cosh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_cosh_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_cosine_embedding_loss(arg1, input1, input2, target, margin, reduction)
    @runtime_error_check ccall((:atg_cosine_embedding_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Int64), arg1, input1, input2, target, margin, reduction)
end

function atg_cosine_similarity(arg1, x1, x2, dim, eps)
    @runtime_error_check ccall((:atg_cosine_similarity, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cdouble), arg1, x1, x2, dim, eps)
end

function atg_cross(arg1, self, other, dim)
    @runtime_error_check ccall((:atg_cross, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, other, dim)
end

function atg_cross_out(arg1, out, self, other, dim)
    @runtime_error_check ccall((:atg_cross_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, other, dim)
end

function atg_ctc_loss(arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, reduction, zero_infinity)
    @runtime_error_check ccall((:atg_ctc_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Int64, Cint), arg1, log_probs, targets, input_lengths_data, input_lengths_len, target_lengths_data, target_lengths_len, blank, reduction, zero_infinity)
end

function atg_ctc_loss1(arg1, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
    @runtime_error_check ccall((:atg_ctc_loss1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, Cint), arg1, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
end

function atg_cudnn_affine_grid_generator(arg1, theta, n, C, H, W)
    @runtime_error_check ccall((:atg_cudnn_affine_grid_generator, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, theta, n, C, H, W)
end

function atg_cudnn_affine_grid_generator_backward(arg1, grad, n, C, H, W)
    @runtime_error_check ccall((:atg_cudnn_affine_grid_generator_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, grad, n, C, H, W)
end

function atg_cudnn_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    @runtime_error_check ccall((:atg_cudnn_batch_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
end

function atg_cudnn_batch_norm_backward(arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace)
    @runtime_error_check ccall((:atg_cudnn_batch_norm_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble, tensor), arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace)
end

function atg_cudnn_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_backward_bias(arg1, grad_output)
    @runtime_error_check ccall((:atg_cudnn_convolution_backward_bias, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, grad_output)
end

function atg_cudnn_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_backward_weight, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_transpose(arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_transpose_backward_bias(arg1, grad_output)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_backward_bias, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, grad_output)
end

function atg_cudnn_convolution_transpose_backward_input(arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_convolution_transpose_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_cudnn_convolution_transpose_backward_weight, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_cudnn_grid_sampler(arg1, self, grid)
    @runtime_error_check ccall((:atg_cudnn_grid_sampler, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, grid)
end

function atg_cudnn_grid_sampler_backward(arg1, self, grid, grad_output)
    @runtime_error_check ccall((:atg_cudnn_grid_sampler_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, grid, grad_output)
end

function atg_cumprod(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumprod, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumprod_out(arg1, out, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumprod_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, dtype)
end

function atg_cumsum(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumsum, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_cumsum_out(arg1, out, self, dim, dtype)
    @runtime_error_check ccall((:atg_cumsum_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, dim, dtype)
end

function atg_data(arg1, self)
    @runtime_error_check ccall((:atg_data, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_dequantize(arg1, self)
    @runtime_error_check ccall((:atg_dequantize, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_det(arg1, self)
    @runtime_error_check ccall((:atg_det, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_detach(arg1, self)
    @runtime_error_check ccall((:atg_detach, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_detach_(arg1, self)
    @runtime_error_check ccall((:atg_detach_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_diag(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_diag, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_diag_embed(arg1, self, offset, dim1, dim2)
    @runtime_error_check ccall((:atg_diag_embed, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, offset, dim1, dim2)
end

function atg_diag_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_diag_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_diagflat(arg1, self, offset)
    @runtime_error_check ccall((:atg_diagflat, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, offset)
end

function atg_diagonal(arg1, self, offset, dim1, dim2)
    @runtime_error_check ccall((:atg_diagonal, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, offset, dim1, dim2)
end

function atg_digamma(arg1, self)
    @runtime_error_check ccall((:atg_digamma, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_digamma_(arg1, self)
    @runtime_error_check ccall((:atg_digamma_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_digamma_out(arg1, out, self)
    @runtime_error_check ccall((:atg_digamma_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_dist(arg1, self, other)
    @runtime_error_check ccall((:atg_dist, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div(arg1, self, other)
    @runtime_error_check ccall((:atg_div, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div1(arg1, self, other)
    @runtime_error_check ccall((:atg_div1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_div_(arg1, self, other)
    @runtime_error_check ccall((:atg_div_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_div_1(arg1, self, other)
    @runtime_error_check ccall((:atg_div_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_div_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_div_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_dot(arg1, self, tensor_)
    @runtime_error_check ccall((:atg_dot, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, tensor_)
end

function atg_dot_out(arg1, out, self, tensor_)
    @runtime_error_check ccall((:atg_dot_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, tensor_)
end

function atg_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_dropout, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_dropout_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_eig(arg1, self, eigenvectors)
    @runtime_error_check ccall((:atg_eig, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, eigenvectors)
end

function atg_eig_out(arg1, e, v, self, eigenvectors)
    @runtime_error_check ccall((:atg_eig_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, e, v, self, eigenvectors)
end

function atg_elu(arg1, self)
    @runtime_error_check ccall((:atg_elu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_elu_(arg1, self)
    @runtime_error_check ccall((:atg_elu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_elu_backward(arg1, grad_output, alpha, scale, input_scale, output)
    @runtime_error_check ccall((:atg_elu_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar, scalar, tensor), arg1, grad_output, alpha, scale, input_scale, output)
end

function atg_elu_backward_out(arg1, grad_input, grad_output, alpha, scale, input_scale, output)
    @runtime_error_check ccall((:atg_elu_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar, scalar, tensor), arg1, grad_input, grad_output, alpha, scale, input_scale, output)
end

function atg_elu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_elu_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_embedding(arg1, weight, indices, padding_idx, scale_grad_by_freq, sparse)
    @runtime_error_check ccall((:atg_embedding, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Cint), arg1, weight, indices, padding_idx, scale_grad_by_freq, sparse)
end

function atg_embedding_backward(arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
    @runtime_error_check ccall((:atg_embedding_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint, Cint), arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
end

function atg_embedding_bag(arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights)
    @runtime_error_check ccall((:atg_embedding_bag, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Int64, Cint, tensor), arg1, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights)
end

function atg_embedding_dense_backward(arg1, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
    @runtime_error_check ccall((:atg_embedding_dense_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
end

function atg_embedding_renorm_(arg1, self, indices, max_norm, norm_type)
    @runtime_error_check ccall((:atg_embedding_renorm_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble), arg1, self, indices, max_norm, norm_type)
end

function atg_embedding_sparse_backward(arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq)
    @runtime_error_check ccall((:atg_embedding_sparse_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, grad, indices, num_weights, padding_idx, scale_grad_by_freq)
end

function atg_empty(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_empty_like(arg1, self)
    @runtime_error_check ccall((:atg_empty_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_empty_like1(arg1, self, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, options_kind, options_device)
end

function atg_empty_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_empty_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_empty_strided(arg1, size_data, size_len, stride_data, stride_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_empty_strided, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, stride_data, stride_len, options_kind, options_device)
end

function atg_eq(arg1, self, other)
    @runtime_error_check ccall((:atg_eq, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_eq1(arg1, self, other)
    @runtime_error_check ccall((:atg_eq1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_eq_(arg1, self, other)
    @runtime_error_check ccall((:atg_eq_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_eq_1(arg1, self, other)
    @runtime_error_check ccall((:atg_eq_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_eq_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_eq_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_eq_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_eq_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_erf(arg1, self)
    @runtime_error_check ccall((:atg_erf, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erf_(arg1, self)
    @runtime_error_check ccall((:atg_erf_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erf_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erf_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_erfc(arg1, self)
    @runtime_error_check ccall((:atg_erfc, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfc_(arg1, self)
    @runtime_error_check ccall((:atg_erfc_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erfc_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_erfinv(arg1, self)
    @runtime_error_check ccall((:atg_erfinv, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfinv_(arg1, self)
    @runtime_error_check ccall((:atg_erfinv_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_erfinv_out(arg1, out, self)
    @runtime_error_check ccall((:atg_erfinv_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_exp(arg1, self)
    @runtime_error_check ccall((:atg_exp, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp_(arg1, self)
    @runtime_error_check ccall((:atg_exp_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_exp_out(arg1, out, self)
    @runtime_error_check ccall((:atg_exp_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_expand(arg1, self, size_data, size_len, implicit)
    @runtime_error_check ccall((:atg_expand, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, size_data, size_len, implicit)
end

function atg_expand_as(arg1, self, other)
    @runtime_error_check ccall((:atg_expand_as, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_expm1(arg1, self)
    @runtime_error_check ccall((:atg_expm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_expm1_(arg1, self)
    @runtime_error_check ccall((:atg_expm1_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_expm1_out(arg1, out, self)
    @runtime_error_check ccall((:atg_expm1_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_exponential_(arg1, self, lambd)
    @runtime_error_check ccall((:atg_exponential_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, lambd)
end

function atg_eye(arg1, n, options_kind, options_device)
    @runtime_error_check ccall((:atg_eye, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, n, options_kind, options_device)
end

function atg_eye1(arg1, n, m, options_kind, options_device)
    @runtime_error_check ccall((:atg_eye1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Int64, Cint, Cint), arg1, n, m, options_kind, options_device)
end

function atg_eye_out(arg1, out, n)
    @runtime_error_check ccall((:atg_eye_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, out, n)
end

function atg_eye_out1(arg1, out, n, m)
    @runtime_error_check ccall((:atg_eye_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, out, n, m)
end

function atg_fake_quantize_per_channel_affine(arg1, self, scale, zero_point, axis, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_channel_affine, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Int64), arg1, self, scale, zero_point, axis, quant_min, quant_max)
end

function atg_fake_quantize_per_channel_affine_backward(arg1, grad, self, scale, zero_point, axis, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_channel_affine_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, Int64), arg1, grad, self, scale, zero_point, axis, quant_min, quant_max)
end

function atg_fake_quantize_per_tensor_affine(arg1, self, scale, zero_point, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Int64, Int64), arg1, self, scale, zero_point, quant_min, quant_max)
end

function atg_fake_quantize_per_tensor_affine_backward(arg1, grad, self, scale, zero_point, quant_min, quant_max)
    @runtime_error_check ccall((:atg_fake_quantize_per_tensor_affine_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64, Int64, Int64), arg1, grad, self, scale, zero_point, quant_min, quant_max)
end

function atg_fbgemm_linear_fp16_weight(arg1, input, packed_weight, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_fp16_weight, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, packed_weight, bias)
end

function atg_fbgemm_linear_fp16_weight_fp32_activation(arg1, input, packed_weight, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_fp16_weight_fp32_activation, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, packed_weight, bias)
end

function atg_fbgemm_linear_int8_weight(arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_int8_weight, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor), arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
end

function atg_fbgemm_linear_int8_weight_fp32_activation(arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
    @runtime_error_check ccall((:atg_fbgemm_linear_int8_weight_fp32_activation, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor), arg1, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias)
end

function atg_fbgemm_pack_gemm_matrix_fp16(arg1, input)
    @runtime_error_check ccall((:atg_fbgemm_pack_gemm_matrix_fp16, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, input)
end

function atg_fbgemm_pack_quantized_matrix(arg1, input)
    @runtime_error_check ccall((:atg_fbgemm_pack_quantized_matrix, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, input)
end

function atg_fbgemm_pack_quantized_matrix1(arg1, input, K, n)
    @runtime_error_check ccall((:atg_fbgemm_pack_quantized_matrix1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, input, K, n)
end

function atg_feature_alpha_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_feature_alpha_dropout, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_feature_alpha_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_feature_alpha_dropout_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_feature_dropout(arg1, input, p, train)
    @runtime_error_check ccall((:atg_feature_dropout, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, input, p, train)
end

function atg_feature_dropout_(arg1, self, p, train)
    @runtime_error_check ccall((:atg_feature_dropout_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, p, train)
end

function atg_fft(arg1, self, signal_ndim, normalized)
    @runtime_error_check ccall((:atg_fft, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, signal_ndim, normalized)
end

function atg_fill_(arg1, self, value)
    @runtime_error_check ccall((:atg_fill_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, value)
end

function atg_fill_1(arg1, self, value)
    @runtime_error_check ccall((:atg_fill_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, value)
end

function atg_fill_diagonal_(arg1, self, fill_value, wrap)
    @runtime_error_check ccall((:atg_fill_diagonal_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Cint), arg1, self, fill_value, wrap)
end

function atg_flatten(arg1, self, start_dim, end_dim)
    @runtime_error_check ccall((:atg_flatten, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, start_dim, end_dim)
end

function atg_flip(arg1, self, dims_data, dims_len)
    @runtime_error_check ccall((:atg_flip, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dims_data, dims_len)
end

function atg_floor(arg1, self)
    @runtime_error_check ccall((:atg_floor, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_floor_(arg1, self)
    @runtime_error_check ccall((:atg_floor_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_floor_out(arg1, out, self)
    @runtime_error_check ccall((:atg_floor_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_fmod(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_fmod1(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmod_(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_fmod_1(arg1, self, other)
    @runtime_error_check ccall((:atg_fmod_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_fmod_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmod_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_fmod_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_fmod_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_frac(arg1, self)
    @runtime_error_check ccall((:atg_frac, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frac_(arg1, self)
    @runtime_error_check ccall((:atg_frac_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frac_out(arg1, out, self)
    @runtime_error_check ccall((:atg_frac_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_fractional_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool2d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool2d_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool2d_out(arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool3d(arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_fractional_max_pool3d_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool3d_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, indices)
end

function atg_fractional_max_pool3d_out(arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
    @runtime_error_check ccall((:atg_fractional_max_pool3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, tensor), arg1, output, indices, self, kernel_size_data, kernel_size_len, output_size_data, output_size_len, random_samples)
end

function atg_frobenius_norm(arg1, self)
    @runtime_error_check ccall((:atg_frobenius_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_frobenius_norm1(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_frobenius_norm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_frobenius_norm_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_frobenius_norm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_full(arg1, size_data, size_len, fill_value, options_kind, options_device)
    @runtime_error_check ccall((:atg_full, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, scalar, Cint, Cint), arg1, size_data, size_len, fill_value, options_kind, options_device)
end

function atg_full_like(arg1, self, fill_value)
    @runtime_error_check ccall((:atg_full_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, fill_value)
end

function atg_full_like1(arg1, self, fill_value, options_kind, options_device)
    @runtime_error_check ccall((:atg_full_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Cint, Cint), arg1, self, fill_value, options_kind, options_device)
end

function atg_full_out(arg1, out, size_data, size_len, fill_value)
    @runtime_error_check ccall((:atg_full_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, scalar), arg1, out, size_data, size_len, fill_value)
end

function atg_gather(arg1, self, dim, index, sparse_grad)
    @runtime_error_check ccall((:atg_gather, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, Cint), arg1, self, dim, index, sparse_grad)
end

function atg_gather_out(arg1, out, self, dim, index, sparse_grad)
    @runtime_error_check ccall((:atg_gather_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor, Cint), arg1, out, self, dim, index, sparse_grad)
end

function atg_ge(arg1, self, other)
    @runtime_error_check ccall((:atg_ge, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ge1(arg1, self, other)
    @runtime_error_check ccall((:atg_ge1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ge_(arg1, self, other)
    @runtime_error_check ccall((:atg_ge_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ge_1(arg1, self, other)
    @runtime_error_check ccall((:atg_ge_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ge_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ge_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_ge_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ge_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_gelu(arg1, self)
    @runtime_error_check ccall((:atg_gelu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_gelu_backward(arg1, grad, self)
    @runtime_error_check ccall((:atg_gelu_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, self)
end

function atg_geometric_(arg1, self, p)
    @runtime_error_check ccall((:atg_geometric_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_geqrf(arg1, self)
    @runtime_error_check ccall((:atg_geqrf, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_geqrf_out(arg1, a, tau, self)
    @runtime_error_check ccall((:atg_geqrf_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, a, tau, self)
end

function atg_ger(arg1, self, vec2)
    @runtime_error_check ccall((:atg_ger, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, vec2)
end

function atg_ger_out(arg1, out, self, vec2)
    @runtime_error_check ccall((:atg_ger_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, vec2)
end

function atg_glu(arg1, self, dim)
    @runtime_error_check ccall((:atg_glu, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_glu_backward(arg1, grad_output, self, dim)
    @runtime_error_check ccall((:atg_glu_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, grad_output, self, dim)
end

function atg_glu_backward_out(arg1, grad_input, grad_output, self, dim)
    @runtime_error_check ccall((:atg_glu_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, dim)
end

function atg_glu_out(arg1, out, self, dim)
    @runtime_error_check ccall((:atg_glu_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, dim)
end

function atg_grad(arg1, self)
    @runtime_error_check ccall((:atg_grad, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_grid_sampler(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_2d(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_2d_backward(arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_3d(arg1, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Int64, Cint), arg1, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_grid_sampler_3d_backward(arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
    @runtime_error_check ccall((:atg_grid_sampler_3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, grad_output, input, grid, interpolation_mode, padding_mode, align_corners)
end

function atg_group_norm(arg1, input, num_groups, weight, bias, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_group_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor, Cdouble, Cint), arg1, input, num_groups, weight, bias, eps, cudnn_enabled)
end

function atg_gru(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_gru, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_gru1(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_gru1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_gru_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_gru_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_gt(arg1, self, other)
    @runtime_error_check ccall((:atg_gt, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_gt1(arg1, self, other)
    @runtime_error_check ccall((:atg_gt1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gt_(arg1, self, other)
    @runtime_error_check ccall((:atg_gt_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_gt_1(arg1, self, other)
    @runtime_error_check ccall((:atg_gt_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_gt_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_gt_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_gt_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_gt_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_hamming_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_hamming_window1(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_hamming_window2(arg1, window_length, periodic, alpha, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window2, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cdouble, Cint, Cint), arg1, window_length, periodic, alpha, options_kind, options_device)
end

function atg_hamming_window3(arg1, window_length, periodic, alpha, beta, options_kind, options_device)
    @runtime_error_check ccall((:atg_hamming_window3, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cdouble, Cdouble, Cint, Cint), arg1, window_length, periodic, alpha, beta, options_kind, options_device)
end

function atg_hann_window(arg1, window_length, options_kind, options_device)
    @runtime_error_check ccall((:atg_hann_window, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, window_length, options_kind, options_device)
end

function atg_hann_window1(arg1, window_length, periodic, options_kind, options_device)
    @runtime_error_check ccall((:atg_hann_window1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint, Cint), arg1, window_length, periodic, options_kind, options_device)
end

function atg_hardshrink(arg1, self)
    @runtime_error_check ccall((:atg_hardshrink, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardshrink_backward(arg1, grad_out, self, lambd)
    @runtime_error_check ccall((:atg_hardshrink_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_out, self, lambd)
end

function atg_hardtanh(arg1, self)
    @runtime_error_check ccall((:atg_hardtanh, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardtanh_(arg1, self)
    @runtime_error_check ccall((:atg_hardtanh_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_hardtanh_backward(arg1, grad_output, self, min_val, max_val)
    @runtime_error_check ccall((:atg_hardtanh_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, grad_output, self, min_val, max_val)
end

function atg_hardtanh_backward_out(arg1, grad_input, grad_output, self, min_val, max_val)
    @runtime_error_check ccall((:atg_hardtanh_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar), arg1, grad_input, grad_output, self, min_val, max_val)
end

function atg_hardtanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_hardtanh_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_hinge_embedding_loss(arg1, self, target, margin, reduction)
    @runtime_error_check ccall((:atg_hinge_embedding_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Int64), arg1, self, target, margin, reduction)
end

function atg_histc(arg1, self, bins)
    @runtime_error_check ccall((:atg_histc, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, bins)
end

function atg_histc_out(arg1, out, self, bins)
    @runtime_error_check ccall((:atg_histc_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, bins)
end

function atg_hspmm(arg1, mat1, mat2)
    @runtime_error_check ccall((:atg_hspmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, mat1, mat2)
end

function atg_hspmm_out(arg1, out, mat1, mat2)
    @runtime_error_check ccall((:atg_hspmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, mat1, mat2)
end

function atg_ifft(arg1, self, signal_ndim, normalized)
    @runtime_error_check ccall((:atg_ifft, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, signal_ndim, normalized)
end

function atg_im2col(arg1, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_backward(arg1, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_backward_out(arg1, grad_input, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, input_size_data, input_size_len, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_im2col_out(arg1, out, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
    @runtime_error_check ccall((:atg_im2col_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, kernel_size_data, kernel_size_len, dilation_data, dilation_len, padding_data, padding_len, stride_data, stride_len)
end

function atg_imag(arg1, self)
    @runtime_error_check ccall((:atg_imag, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_imag_out(arg1, out, self)
    @runtime_error_check ccall((:atg_imag_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_index(arg1, self, indices_data, indices_len)
    @runtime_error_check ccall((:atg_index, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint), arg1, self, indices_data, indices_len)
end

function atg_index_add(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_add, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_add_(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_add_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_copy(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_copy, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_copy_(arg1, self, dim, index, source)
    @runtime_error_check ccall((:atg_index_copy_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, source)
end

function atg_index_fill(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_index_fill1(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, value)
end

function atg_index_fill_(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_index_fill_1(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_index_fill_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, value)
end

function atg_index_put(arg1, self, indices_data, indices_len, values, accumulate)
    @runtime_error_check ccall((:atg_index_put, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, Cint), arg1, self, indices_data, indices_len, values, accumulate)
end

function atg_index_put_(arg1, self, indices_data, indices_len, values, accumulate)
    @runtime_error_check ccall((:atg_index_put_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, Cint), arg1, self, indices_data, indices_len, values, accumulate)
end

function atg_index_select(arg1, self, dim, index)
    @runtime_error_check ccall((:atg_index_select, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor), arg1, self, dim, index)
end

function atg_index_select_out(arg1, out, self, dim, index)
    @runtime_error_check ccall((:atg_index_select_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, tensor), arg1, out, self, dim, index)
end

function atg_indices(arg1, self)
    @runtime_error_check ccall((:atg_indices, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_instance_norm(arg1, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
    @runtime_error_check ccall((:atg_instance_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble, Cint), arg1, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
end

function atg_int_repr(arg1, self)
    @runtime_error_check ccall((:atg_int_repr, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_inverse(arg1, self)
    @runtime_error_check ccall((:atg_inverse, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_inverse_out(arg1, out, self)
    @runtime_error_check ccall((:atg_inverse_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_irfft(arg1, self, signal_ndim, normalized, onesided, signal_sizes_data, signal_sizes_len)
    @runtime_error_check ccall((:atg_irfft, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint, Ptr{Int64}, Cint), arg1, self, signal_ndim, normalized, onesided, signal_sizes_data, signal_sizes_len)
end

function atg_isclose(arg1, self, other, rtol, atol, equal_nan)
    @runtime_error_check ccall((:atg_isclose, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble, Cint), arg1, self, other, rtol, atol, equal_nan)
end

function atg_isfinite(arg1, self)
    @runtime_error_check ccall((:atg_isfinite, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_isnan(arg1, self)
    @runtime_error_check ccall((:atg_isnan, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_kl_div(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_kl_div, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_kl_div_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_kl_div_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_kthvalue(arg1, self, k, dim, keepdim)
    @runtime_error_check ccall((:atg_kthvalue, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cint), arg1, self, k, dim, keepdim)
end

function atg_kthvalue_out(arg1, values, indices, self, k, dim, keepdim)
    @runtime_error_check ccall((:atg_kthvalue_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint), arg1, values, indices, self, k, dim, keepdim)
end

function atg_l1_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_l1_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_l1_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_l1_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_l1_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_layer_norm(arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps, cudnn_enable)
    @runtime_error_check ccall((:atg_layer_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, tensor, tensor, Cdouble, Cint), arg1, input, normalized_shape_data, normalized_shape_len, weight, bias, eps, cudnn_enable)
end

function atg_le(arg1, self, other)
    @runtime_error_check ccall((:atg_le, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_le1(arg1, self, other)
    @runtime_error_check ccall((:atg_le1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_le_(arg1, self, other)
    @runtime_error_check ccall((:atg_le_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_le_1(arg1, self, other)
    @runtime_error_check ccall((:atg_le_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_le_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_le_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_le_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_le_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_leaky_relu(arg1, self)
    @runtime_error_check ccall((:atg_leaky_relu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_leaky_relu_(arg1, self)
    @runtime_error_check ccall((:atg_leaky_relu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_leaky_relu_backward(arg1, grad_output, self, negative_slope)
    @runtime_error_check ccall((:atg_leaky_relu_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_output, self, negative_slope)
end

function atg_leaky_relu_backward_out(arg1, grad_input, grad_output, self, negative_slope)
    @runtime_error_check ccall((:atg_leaky_relu_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, grad_input, grad_output, self, negative_slope)
end

function atg_leaky_relu_out(arg1, out, self)
    @runtime_error_check ccall((:atg_leaky_relu_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_lerp(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, _end, weight)
end

function atg_lerp1(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, _end, weight)
end

function atg_lerp_(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, _end, weight)
end

function atg_lerp_1(arg1, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, _end, weight)
end

function atg_lerp_out(arg1, out, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, out, self, _end, weight)
end

function atg_lerp_out1(arg1, out, self, _end, weight)
    @runtime_error_check ccall((:atg_lerp_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, _end, weight)
end

function atg_lgamma(arg1, self)
    @runtime_error_check ccall((:atg_lgamma, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_lgamma_(arg1, self)
    @runtime_error_check ccall((:atg_lgamma_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_lgamma_out(arg1, out, self)
    @runtime_error_check ccall((:atg_lgamma_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_linear(arg1, input, weight, bias)
    @runtime_error_check ccall((:atg_linear, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, weight, bias)
end

function atg_linspace(arg1, start, _end, steps, options_kind, options_device)
    @runtime_error_check ccall((:atg_linspace, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, Int64, Cint, Cint), arg1, start, _end, steps, options_kind, options_device)
end

function atg_linspace_out(arg1, out, start, _end, steps)
    @runtime_error_check ccall((:atg_linspace_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar, Int64), arg1, out, start, _end, steps)
end

function atg_log(arg1, self)
    @runtime_error_check ccall((:atg_log, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10(arg1, self)
    @runtime_error_check ccall((:atg_log10, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10_(arg1, self)
    @runtime_error_check ccall((:atg_log10_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log10_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log10_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log1p(arg1, self)
    @runtime_error_check ccall((:atg_log1p, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log1p_(arg1, self)
    @runtime_error_check ccall((:atg_log1p_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log1p_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log1p_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log2(arg1, self)
    @runtime_error_check ccall((:atg_log2, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log2_(arg1, self)
    @runtime_error_check ccall((:atg_log2_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log2_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log2_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_(arg1, self)
    @runtime_error_check ccall((:atg_log_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log_normal_(arg1, self, mean, std)
    @runtime_error_check ccall((:atg_log_normal_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, mean, std)
end

function atg_log_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_sigmoid(arg1, self)
    @runtime_error_check ccall((:atg_log_sigmoid, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_log_sigmoid_backward(arg1, grad_output, self, buffer)
    @runtime_error_check ccall((:atg_log_sigmoid_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, buffer)
end

function atg_log_sigmoid_backward_out(arg1, grad_input, grad_output, self, buffer)
    @runtime_error_check ccall((:atg_log_sigmoid_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, grad_input, grad_output, self, buffer)
end

function atg_log_sigmoid_out(arg1, out, self)
    @runtime_error_check ccall((:atg_log_sigmoid_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_log_softmax(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_log_softmax, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_logdet(arg1, self)
    @runtime_error_check ccall((:atg_logdet, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_not(arg1, self)
    @runtime_error_check ccall((:atg_logical_not, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_not_(arg1, self)
    @runtime_error_check ccall((:atg_logical_not_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_logical_not_out(arg1, out, self)
    @runtime_error_check ccall((:atg_logical_not_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_logical_xor(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_xor, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_xor_(arg1, self, other)
    @runtime_error_check ccall((:atg_logical_xor_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_logical_xor_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_logical_xor_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_logspace(arg1, start, _end, steps, base, options_kind, options_device)
    @runtime_error_check ccall((:atg_logspace, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, Int64, Cdouble, Cint, Cint), arg1, start, _end, steps, base, options_kind, options_device)
end

function atg_logspace_out(arg1, out, start, _end, steps, base)
    @runtime_error_check ccall((:atg_logspace_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar, Int64, Cdouble), arg1, out, start, _end, steps, base)
end

function atg_logsumexp(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_logsumexp, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_logsumexp_out(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_logsumexp_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_lstm(arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_lstm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_lstm1(arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_lstm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_lstm_cell(arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_lstm_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, tensor, tensor, tensor), arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh)
end

function atg_lstsq(arg1, self, A)
    @runtime_error_check ccall((:atg_lstsq, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, A)
end

function atg_lstsq_out(arg1, X, qr, self, A)
    @runtime_error_check ccall((:atg_lstsq_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, X, qr, self, A)
end

function atg_lt(arg1, self, other)
    @runtime_error_check ccall((:atg_lt, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_lt1(arg1, self, other)
    @runtime_error_check ccall((:atg_lt1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lt_(arg1, self, other)
    @runtime_error_check ccall((:atg_lt_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_lt_1(arg1, self, other)
    @runtime_error_check ccall((:atg_lt_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_lt_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_lt_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_lt_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_lt_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_lu_solve(arg1, self, LU_data, LU_pivots)
    @runtime_error_check ccall((:atg_lu_solve, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, LU_data, LU_pivots)
end

function atg_lu_solve_out(arg1, out, self, LU_data, LU_pivots)
    @runtime_error_check ccall((:atg_lu_solve_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, LU_data, LU_pivots)
end

function atg_margin_ranking_loss(arg1, input1, input2, target, margin, reduction)
    @runtime_error_check ccall((:atg_margin_ranking_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Int64), arg1, input1, input2, target, margin, reduction)
end

function atg_masked_fill(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, mask, value)
end

function atg_masked_fill1(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, value)
end

function atg_masked_fill_(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, self, mask, value)
end

function atg_masked_fill_1(arg1, self, mask, value)
    @runtime_error_check ccall((:atg_masked_fill_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, value)
end

function atg_masked_scatter(arg1, self, mask, source)
    @runtime_error_check ccall((:atg_masked_scatter, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, source)
end

function atg_masked_scatter_(arg1, self, mask, source)
    @runtime_error_check ccall((:atg_masked_scatter_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mask, source)
end

function atg_masked_select(arg1, self, mask)
    @runtime_error_check ccall((:atg_masked_select, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mask)
end

function atg_masked_select_out(arg1, out, self, mask)
    @runtime_error_check ccall((:atg_masked_select_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mask)
end

function atg_matmul(arg1, self, other)
    @runtime_error_check ccall((:atg_matmul, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_matmul_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_matmul_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_matrix_power(arg1, self, n)
    @runtime_error_check ccall((:atg_matrix_power, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, n)
end

function atg_matrix_rank(arg1, self, symmetric)
    @runtime_error_check ccall((:atg_matrix_rank, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, symmetric)
end

function atg_matrix_rank1(arg1, self, tol, symmetric)
    @runtime_error_check ccall((:atg_matrix_rank1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cint), arg1, self, tol, symmetric)
end

function atg_max(arg1, self)
    @runtime_error_check ccall((:atg_max, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_max1(arg1, self, other)
    @runtime_error_check ccall((:atg_max1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_max2(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_max2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_max_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_max_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_max_out1(arg1, max, max_values, self, dim, keepdim)
    @runtime_error_check ccall((:atg_max_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, max, max_values, self, dim, keepdim)
end

function atg_max_pool1d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool1d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool1d_with_indices, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool2d_with_indices_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool2d_with_indices_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool2d_with_indices_out(arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool2d_with_indices_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d_with_indices(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_pool3d_with_indices_backward(arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool3d_with_indices_backward_out(arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint, tensor), arg1, grad_input, grad_output, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode, indices)
end

function atg_max_pool3d_with_indices_out(arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_max_pool3d_with_indices_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, out, indices, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_max_unpool2d(arg1, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_backward(arg1, grad_output, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_backward_out(arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool2d_out(arg1, out, self, indices, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_max_unpool2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, indices, output_size_data, output_size_len)
end

function atg_max_unpool3d(arg1, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_backward(arg1, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_backward_out(arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_unpool3d_out(arg1, out, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_max_unpool3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, indices, output_size_data, output_size_len, stride_data, stride_len, padding_data, padding_len)
end

function atg_max_values(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_max_values, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_mean(arg1, self, dtype)
    @runtime_error_check ccall((:atg_mean, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_mean1(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_mean1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_mean_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_mean_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_median(arg1, self)
    @runtime_error_check ccall((:atg_median, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_median1(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_median1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_median_out(arg1, values, indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_median_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, keepdim)
end

function atg_meshgrid(arg1, tensors_data, tensors_len)
    @runtime_error_check ccall((:atg_meshgrid, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint), arg1, tensors_data, tensors_len)
end

function atg_min(arg1, self)
    @runtime_error_check ccall((:atg_min, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_min1(arg1, self, other)
    @runtime_error_check ccall((:atg_min1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_min2(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_min2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_min_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_min_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_min_out1(arg1, min, min_indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_min_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, min, min_indices, self, dim, keepdim)
end

function atg_min_values(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_min_values, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_miopen_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    @runtime_error_check ccall((:atg_miopen_batch_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
end

function atg_miopen_batch_norm_backward(arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon)
    @runtime_error_check ccall((:atg_miopen_batch_norm_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, Cdouble), arg1, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon)
end

function atg_miopen_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_backward_bias(arg1, grad_output)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_bias, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, grad_output)
end

function atg_miopen_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_backward_weight, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose(arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, output_padding_data, output_padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose_backward_input(arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_convolution_transpose_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_convolution_transpose_backward_weight, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_depthwise_convolution_backward_weight(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
    @runtime_error_check ccall((:atg_miopen_depthwise_convolution_backward_weight, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, benchmark, deterministic)
end

function atg_miopen_rnn(arg1, input, weight_data, weight_len, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
    @runtime_error_check ccall((:atg_miopen_rnn, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64, tensor, tensor, Int64, Int64, Int64, Cint, Cdouble, Cint, Cint, Ptr{Int64}, Cint, tensor), arg1, input, weight_data, weight_len, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes_data, batch_sizes_len, dropout_state)
end

function atg_mkldnn_adaptive_avg_pool2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_mkldnn_adaptive_avg_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_mkldnn_convolution(arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_mkldnn_convolution, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, weight, bias, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
end

function atg_mkldnn_convolution_backward_input(arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
    @runtime_error_check ccall((:atg_mkldnn_convolution_backward_input, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint), arg1, self_size_data, self_size_len, grad_output, weight, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
end

function atg_mkldnn_convolution_backward_weights(arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
    @runtime_error_check ccall((:atg_mkldnn_convolution_backward_weights, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64, Cint), arg1, weight_size_data, weight_size_len, grad_output, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups, bias_defined)
end

function atg_mkldnn_linear(arg1, input, weight, bias)
    @runtime_error_check ccall((:atg_mkldnn_linear, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, input, weight, bias)
end

function atg_mkldnn_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_mkldnn_max_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_mkldnn_reorder_conv2d_weight(arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
    @runtime_error_check ccall((:atg_mkldnn_reorder_conv2d_weight, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Int64), arg1, self, padding_data, padding_len, stride_data, stride_len, dilation_data, dilation_len, groups)
end

function atg_mm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_mm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_mm_out(arg1, out, self, mat2)
    @runtime_error_check ccall((:atg_mm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, mat2)
end

function atg_mode(arg1, self, dim, keepdim)
    @runtime_error_check ccall((:atg_mode, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, keepdim)
end

function atg_mode_out(arg1, values, indices, self, dim, keepdim)
    @runtime_error_check ccall((:atg_mode_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, keepdim)
end

function atg_mse_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_mse_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_mse_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_mse_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_mse_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_mul(arg1, self, other)
    @runtime_error_check ccall((:atg_mul, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_mul1(arg1, self, other)
    @runtime_error_check ccall((:atg_mul1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_mul_(arg1, self, other)
    @runtime_error_check ccall((:atg_mul_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_mul_1(arg1, self, other)
    @runtime_error_check ccall((:atg_mul_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_mul_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_mul_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_multi_margin_loss_backward(arg1, grad_output, self, target, p, margin, weight, reduction)
    @runtime_error_check ccall((:atg_multi_margin_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, tensor, Int64), arg1, grad_output, self, target, p, margin, weight, reduction)
end

function atg_multi_margin_loss_backward_out(arg1, grad_input, grad_output, self, target, p, margin, weight, reduction)
    @runtime_error_check ccall((:atg_multi_margin_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, tensor, Int64), arg1, grad_input, grad_output, self, target, p, margin, weight, reduction)
end

function atg_multilabel_margin_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_multilabel_margin_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_multilabel_margin_loss_backward(arg1, grad_output, self, target, reduction, is_target)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, tensor), arg1, grad_output, self, target, reduction, is_target)
end

function atg_multilabel_margin_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction, is_target)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, tensor), arg1, grad_input, grad_output, self, target, reduction, is_target)
end

function atg_multilabel_margin_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_multilabel_margin_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_multinomial(arg1, self, num_samples, replacement)
    @runtime_error_check ccall((:atg_multinomial, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, num_samples, replacement)
end

function atg_multinomial_out(arg1, out, self, num_samples, replacement)
    @runtime_error_check ccall((:atg_multinomial_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint), arg1, out, self, num_samples, replacement)
end

function atg_mv(arg1, self, vec)
    @runtime_error_check ccall((:atg_mv, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, vec)
end

function atg_mv_out(arg1, out, self, vec)
    @runtime_error_check ccall((:atg_mv_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, vec)
end

function atg_mvlgamma(arg1, self, p)
    @runtime_error_check ccall((:atg_mvlgamma, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, p)
end

function atg_mvlgamma_(arg1, self, p)
    @runtime_error_check ccall((:atg_mvlgamma_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, p)
end

function atg_narrow(arg1, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dim, start, length)
end

function atg_narrow_copy(arg1, self, dim, start, length)
    @runtime_error_check ccall((:atg_narrow_copy, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dim, start, length)
end

function atg_native_batch_norm(arg1, input, weight, bias, running_mean, running_var, training, momentum, eps)
    @runtime_error_check ccall((:atg_native_batch_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Cint, Cdouble, Cdouble), arg1, input, weight, bias, running_mean, running_var, training, momentum, eps)
end

function atg_native_layer_norm(arg1, input, weight, bias, M, n, eps)
    @runtime_error_check ccall((:atg_native_layer_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cdouble), arg1, input, weight, bias, M, n, eps)
end

function atg_native_norm(arg1, self)
    @runtime_error_check ccall((:atg_native_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ne(arg1, self, other)
    @runtime_error_check ccall((:atg_ne, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ne1(arg1, self, other)
    @runtime_error_check ccall((:atg_ne1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ne_(arg1, self, other)
    @runtime_error_check ccall((:atg_ne_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_ne_1(arg1, self, other)
    @runtime_error_check ccall((:atg_ne_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_ne_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ne_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_ne_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_ne_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_neg(arg1, self)
    @runtime_error_check ccall((:atg_neg, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_neg_(arg1, self)
    @runtime_error_check ccall((:atg_neg_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_neg_out(arg1, out, self)
    @runtime_error_check ccall((:atg_neg_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_new_empty(arg1, self, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_empty, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, options_kind, options_device)
end

function atg_new_full(arg1, self, size_data, size_len, fill_value, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_full, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, scalar, Cint, Cint), arg1, self, size_data, size_len, fill_value, options_kind, options_device)
end

function atg_new_zeros(arg1, self, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_new_zeros, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, size_data, size_len, options_kind, options_device)
end

function atg_nll_loss(arg1, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss2d(arg1, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64), arg1, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss2d_backward(arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss2d_backward_out(arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss2d_out(arg1, out, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64), arg1, out, self, target, weight, reduction, ignore_index)
end

function atg_nll_loss_backward(arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss_backward_out(arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    @runtime_error_check ccall((:atg_nll_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, Int64, Int64, tensor), arg1, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
end

function atg_nll_loss_out(arg1, out, self, target, weight, reduction, ignore_index)
    @runtime_error_check ccall((:atg_nll_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64, Int64), arg1, out, self, target, weight, reduction, ignore_index)
end

function atg_nonzero(arg1, self)
    @runtime_error_check ccall((:atg_nonzero, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_nonzero_numpy(arg1, self)
    @runtime_error_check ccall((:atg_nonzero_numpy, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_nonzero_out(arg1, out, self)
    @runtime_error_check ccall((:atg_nonzero_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_norm(arg1, self)
    @runtime_error_check ccall((:atg_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_norm1(arg1, self, p, dtype)
    @runtime_error_check ccall((:atg_norm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Cint), arg1, self, p, dtype)
end

function atg_norm2(arg1, self, p, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_norm2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Int64}, Cint, Cint), arg1, self, p, dim_data, dim_len, keepdim)
end

function atg_norm3(arg1, self, p, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_norm3, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Ptr{Int64}, Cint, Cint, Cint), arg1, self, p, dim_data, dim_len, keepdim, dtype)
end

function atg_norm_except_dim(arg1, v, pow, dim)
    @runtime_error_check ccall((:atg_norm_except_dim, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, v, pow, dim)
end

function atg_norm_out(arg1, out, self, p, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_norm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, Ptr{Int64}, Cint, Cint), arg1, out, self, p, dim_data, dim_len, keepdim)
end

function atg_norm_out1(arg1, out, self, p, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_norm_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, p, dim_data, dim_len, keepdim, dtype)
end

function atg_normal_(arg1, self, mean, std)
    @runtime_error_check ccall((:atg_normal_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, mean, std)
end

function atg_normal_out(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble), arg1, out, mean, std)
end

function atg_normal_out1(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, tensor), arg1, out, mean, std)
end

function atg_normal_out2(arg1, out, mean, std)
    @runtime_error_check ccall((:atg_normal_out2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, mean, std)
end

function atg_normal_out3(arg1, out, mean, std, size_data, size_len)
    @runtime_error_check ccall((:atg_normal_out3, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble, Ptr{Int64}, Cint), arg1, out, mean, std, size_data, size_len)
end

function atg_nuclear_norm(arg1, self, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, keepdim)
end

function atg_nuclear_norm1(arg1, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, dim_data, dim_len, keepdim)
end

function atg_nuclear_norm_out(arg1, out, self, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, out, self, keepdim)
end

function atg_nuclear_norm_out1(arg1, out, self, dim_data, dim_len, keepdim)
    @runtime_error_check ccall((:atg_nuclear_norm_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim)
end

function atg_numpy_t(arg1, self)
    @runtime_error_check ccall((:atg_numpy_t, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_one_hot(arg1, self, num_classes)
    @runtime_error_check ccall((:atg_one_hot, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, num_classes)
end

function atg_ones(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_ones, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_ones_like(arg1, self)
    @runtime_error_check ccall((:atg_ones_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_ones_like1(arg1, self, options_kind, options_device)
    @runtime_error_check ccall((:atg_ones_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, options_kind, options_device)
end

function atg_ones_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_ones_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_orgqr(arg1, self, input2)
    @runtime_error_check ccall((:atg_orgqr, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, input2)
end

function atg_orgqr_out(arg1, out, self, input2)
    @runtime_error_check ccall((:atg_orgqr_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, input2)
end

function atg_ormqr(arg1, self, input2, input3, left, transpose)
    @runtime_error_check ccall((:atg_ormqr, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, self, input2, input3, left, transpose)
end

function atg_ormqr_out(arg1, out, self, input2, input3, left, transpose)
    @runtime_error_check ccall((:atg_ormqr_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint), arg1, out, self, input2, input3, left, transpose)
end

function atg_pairwise_distance(arg1, x1, x2, p, eps, keepdim)
    @runtime_error_check ccall((:atg_pairwise_distance, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cdouble, Cdouble, Cint), arg1, x1, x2, p, eps, keepdim)
end

function atg_pdist(arg1, self, p)
    @runtime_error_check ccall((:atg_pdist, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, p)
end

function atg_permute(arg1, self, dims_data, dims_len)
    @runtime_error_check ccall((:atg_permute, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, dims_data, dims_len)
end

function atg_pin_memory(arg1, self)
    @runtime_error_check ccall((:atg_pin_memory, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_pinverse(arg1, self, rcond)
    @runtime_error_check ccall((:atg_pinverse, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble), arg1, self, rcond)
end

function atg_pixel_shuffle(arg1, self, upscale_factor)
    @runtime_error_check ccall((:atg_pixel_shuffle, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, upscale_factor)
end

function atg_poisson(arg1, self)
    @runtime_error_check ccall((:atg_poisson, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_poisson_nll_loss(arg1, input, target, log_input, full, eps, reduction)
    @runtime_error_check ccall((:atg_poisson_nll_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint, Cdouble, Int64), arg1, input, target, log_input, full, eps, reduction)
end

function atg_polygamma(arg1, n, self)
    @runtime_error_check ccall((:atg_polygamma, libdoeye_caml), Cint, (Ptr{tensor}, Int64, tensor), arg1, n, self)
end

function atg_polygamma_(arg1, self, n)
    @runtime_error_check ccall((:atg_polygamma_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, n)
end

function atg_polygamma_out(arg1, out, n, self)
    @runtime_error_check ccall((:atg_polygamma_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor), arg1, out, n, self)
end

function atg_pow(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_pow1(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_pow2(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow2, libdoeye_caml), Cint, (Ptr{tensor}, scalar, tensor), arg1, self, exponent)
end

function atg_pow_(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, exponent)
end

function atg_pow_1(arg1, self, exponent)
    @runtime_error_check ccall((:atg_pow_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, exponent)
end

function atg_pow_out(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, exponent)
end

function atg_pow_out1(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, exponent)
end

function atg_pow_out2(arg1, out, self, exponent)
    @runtime_error_check ccall((:atg_pow_out2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, tensor), arg1, out, self, exponent)
end

function atg_prelu(arg1, self, weight)
    @runtime_error_check ccall((:atg_prelu, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, weight)
end

function atg_prelu_backward(arg1, grad_output, self, weight)
    @runtime_error_check ccall((:atg_prelu_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_output, self, weight)
end

function atg_prod(arg1, self, dtype)
    @runtime_error_check ccall((:atg_prod, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_prod1(arg1, self, dim, keepdim, dtype)
    @runtime_error_check ccall((:atg_prod1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, dim, keepdim, dtype)
end

function atg_prod_out(arg1, out, self, dim, keepdim, dtype)
    @runtime_error_check ccall((:atg_prod_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64, Cint, Cint), arg1, out, self, dim, keepdim, dtype)
end

function atg_put_(arg1, self, index, source, accumulate)
    @runtime_error_check ccall((:atg_put_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, self, index, source, accumulate)
end

function atg_q_per_channel_scales(arg1, self)
    @runtime_error_check ccall((:atg_q_per_channel_scales, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_q_per_channel_zero_points(arg1, self)
    @runtime_error_check ccall((:atg_q_per_channel_zero_points, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_qr(arg1, self, some)
    @runtime_error_check ccall((:atg_qr, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, some)
end

function atg_qr_out(arg1, Q, R, self, some)
    @runtime_error_check ccall((:atg_qr_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, Q, R, self, some)
end

function atg_quantize_per_channel(arg1, self, scales, zero_points, axis, dtype)
    @runtime_error_check ccall((:atg_quantize_per_channel, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, self, scales, zero_points, axis, dtype)
end

function atg_quantize_per_tensor(arg1, self, scale, zero_point, dtype)
    @runtime_error_check ccall((:atg_quantize_per_tensor, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Int64, Cint), arg1, self, scale, zero_point, dtype)
end

function atg_quantized_gru(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_quantized_gru, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_quantized_gru1(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_quantized_gru1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_quantized_gru_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_gru_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_lstm(arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first, dtype, use_dynamic)
    @runtime_error_check ccall((:atg_quantized_lstm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint, Cint, Cint), arg1, input, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first, dtype, use_dynamic)
end

function atg_quantized_lstm1(arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic)
    @runtime_error_check ccall((:atg_quantized_lstm1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint, Cint), arg1, data, batch_sizes, hx_data, hx_len, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic)
end

function atg_quantized_lstm_cell(arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_lstm_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx_data, hx_len, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_max_pool2d(arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
    @runtime_error_check ccall((:atg_quantized_max_pool2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, self, kernel_size_data, kernel_size_len, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len, ceil_mode)
end

function atg_quantized_rnn_relu_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_rnn_relu_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_quantized_rnn_tanh_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
    @runtime_error_check ccall((:atg_quantized_rnn_tanh_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, scalar, scalar, scalar, scalar), arg1, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)
end

function atg_rand(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_rand, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_rand_like(arg1, self)
    @runtime_error_check ccall((:atg_rand_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rand_like1(arg1, self, options_kind, options_device)
    @runtime_error_check ccall((:atg_rand_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, options_kind, options_device)
end

function atg_rand_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_rand_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_randint(arg1, high, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Ptr{Int64}, Cint, Cint, Cint), arg1, high, size_data, size_len, options_kind, options_device)
end

function atg_randint1(arg1, low, high, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint1, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Int64, Ptr{Int64}, Cint, Cint, Cint), arg1, low, high, size_data, size_len, options_kind, options_device)
end

function atg_randint_like(arg1, self, high)
    @runtime_error_check ccall((:atg_randint_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, high)
end

function atg_randint_like1(arg1, self, low, high)
    @runtime_error_check ccall((:atg_randint_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, low, high)
end

function atg_randint_like2(arg1, self, high, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint_like2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, high, options_kind, options_device)
end

function atg_randint_like3(arg1, self, low, high, options_kind, options_device)
    @runtime_error_check ccall((:atg_randint_like3, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cint, Cint), arg1, self, low, high, options_kind, options_device)
end

function atg_randint_out(arg1, out, high, size_data, size_len)
    @runtime_error_check ccall((:atg_randint_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Int64}, Cint), arg1, out, high, size_data, size_len)
end

function atg_randint_out1(arg1, out, low, high, size_data, size_len)
    @runtime_error_check ccall((:atg_randint_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Ptr{Int64}, Cint), arg1, out, low, high, size_data, size_len)
end

function atg_randn(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_randn, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_randn_like(arg1, self)
    @runtime_error_check ccall((:atg_randn_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_randn_like1(arg1, self, options_kind, options_device)
    @runtime_error_check ccall((:atg_randn_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, options_kind, options_device)
end

function atg_randn_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_randn_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

function atg_random_(arg1, self)
    @runtime_error_check ccall((:atg_random_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_random_1(arg1, self, to)
    @runtime_error_check ccall((:atg_random_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, to)
end

function atg_random_2(arg1, self, from, to)
    @runtime_error_check ccall((:atg_random_2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, from, to)
end

function atg_randperm(arg1, n, options_kind, options_device)
    @runtime_error_check ccall((:atg_randperm, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Cint, Cint), arg1, n, options_kind, options_device)
end

function atg_randperm_out(arg1, out, n)
    @runtime_error_check ccall((:atg_randperm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, out, n)
end

function atg_range(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_range, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_range1(arg1, start, _end, options_kind, options_device)
    @runtime_error_check ccall((:atg_range1, libdoeye_caml), Cint, (Ptr{tensor}, scalar, scalar, Cint, Cint), arg1, start, _end, options_kind, options_device)
end

function atg_range_out(arg1, out, start, _end)
    @runtime_error_check ccall((:atg_range_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, out, start, _end)
end

function atg_real(arg1, self)
    @runtime_error_check ccall((:atg_real, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_real_out(arg1, out, self)
    @runtime_error_check ccall((:atg_real_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_reciprocal(arg1, self)
    @runtime_error_check ccall((:atg_reciprocal, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_reciprocal_(arg1, self)
    @runtime_error_check ccall((:atg_reciprocal_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_reciprocal_out(arg1, out, self)
    @runtime_error_check ccall((:atg_reciprocal_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_reflection_pad1d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_reflection_pad1d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad1d_backward_out(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad1d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad1d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_reflection_pad2d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_reflection_pad2d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad2d_backward_out(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_reflection_pad2d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_reflection_pad2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_relu(arg1, self)
    @runtime_error_check ccall((:atg_relu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_relu_(arg1, self)
    @runtime_error_check ccall((:atg_relu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_remainder(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_remainder1(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_remainder_(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_remainder_1(arg1, self, other)
    @runtime_error_check ccall((:atg_remainder_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_remainder_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_remainder_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, out, self, other)
end

function atg_remainder_out1(arg1, out, self, other)
    @runtime_error_check ccall((:atg_remainder_out1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_renorm(arg1, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Int64, scalar), arg1, self, p, dim, maxnorm)
end

function atg_renorm_(arg1, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, Int64, scalar), arg1, self, p, dim, maxnorm)
end

function atg_renorm_out(arg1, out, self, p, dim, maxnorm)
    @runtime_error_check ccall((:atg_renorm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, Int64, scalar), arg1, out, self, p, dim, maxnorm)
end

function atg_repeat(arg1, self, repeats_data, repeats_len)
    @runtime_error_check ccall((:atg_repeat, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, repeats_data, repeats_len)
end

function atg_repeat_interleave(arg1, repeats)
    @runtime_error_check ccall((:atg_repeat_interleave, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, repeats)
end

function atg_repeat_interleave1(arg1, self, repeats, dim)
    @runtime_error_check ccall((:atg_repeat_interleave1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, repeats, dim)
end

function atg_repeat_interleave2(arg1, self, repeats, dim)
    @runtime_error_check ccall((:atg_repeat_interleave2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, repeats, dim)
end

function atg_replication_pad1d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad1d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad1d_backward_out(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad1d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad1d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_replication_pad2d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad2d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad2d_backward_out(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad2d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_replication_pad3d(arg1, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, padding_data, padding_len)
end

function atg_replication_pad3d_backward(arg1, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad3d_backward_out(arg1, grad_input, grad_output, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint), arg1, grad_input, grad_output, self, padding_data, padding_len)
end

function atg_replication_pad3d_out(arg1, out, self, padding_data, padding_len)
    @runtime_error_check ccall((:atg_replication_pad3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, padding_data, padding_len)
end

function atg_requires_grad_(arg1, self, _requires_grad)
    @runtime_error_check ccall((:atg_requires_grad_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, _requires_grad)
end

function atg_reshape(arg1, self, shape_data, shape_len)
    @runtime_error_check ccall((:atg_reshape, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, shape_data, shape_len)
end

function atg_reshape_as(arg1, self, other)
    @runtime_error_check ccall((:atg_reshape_as, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_resize_(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_resize_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_resize_as_(arg1, self, the_template)
    @runtime_error_check ccall((:atg_resize_as_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, the_template)
end

function atg_rfft(arg1, self, signal_ndim, normalized, onesided)
    @runtime_error_check ccall((:atg_rfft, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, signal_ndim, normalized, onesided)
end

function atg_rnn_relu(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_rnn_relu, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_rnn_relu1(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_rnn_relu1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_rnn_relu_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_rnn_relu_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_rnn_tanh(arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    @runtime_error_check ccall((:atg_rnn_tanh, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint, Cint), arg1, input, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional, batch_first)
end

function atg_rnn_tanh1(arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
    @runtime_error_check ccall((:atg_rnn_tanh1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{tensor}, Cint, Cint, Int64, Cdouble, Cint, Cint), arg1, data, batch_sizes, hx, params_data, params_len, has_biases, num_layers, dropout, train, bidirectional)
end

function atg_rnn_tanh_cell(arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
    @runtime_error_check ccall((:atg_rnn_tanh_cell, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, tensor, tensor), arg1, input, hx, w_ih, w_hh, b_ih, b_hh)
end

function atg_roll(arg1, self, shifts_data, shifts_len, dims_data, dims_len)
    @runtime_error_check ccall((:atg_roll, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, shifts_data, shifts_len, dims_data, dims_len)
end

function atg_rot90(arg1, self, k, dims_data, dims_len)
    @runtime_error_check ccall((:atg_rot90, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Ptr{Int64}, Cint), arg1, self, k, dims_data, dims_len)
end

function atg_round(arg1, self)
    @runtime_error_check ccall((:atg_round, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_round_(arg1, self)
    @runtime_error_check ccall((:atg_round_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_round_out(arg1, out, self)
    @runtime_error_check ccall((:atg_round_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_rrelu(arg1, self, training)
    @runtime_error_check ccall((:atg_rrelu, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, training)
end

function atg_rrelu_(arg1, self, training)
    @runtime_error_check ccall((:atg_rrelu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, training)
end

function atg_rrelu_with_noise(arg1, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, noise, training)
end

function atg_rrelu_with_noise_(arg1, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint), arg1, self, noise, training)
end

function atg_rrelu_with_noise_backward(arg1, grad_output, self, noise, lower, upper, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, Cint), arg1, grad_output, self, noise, lower, upper, training)
end

function atg_rrelu_with_noise_backward_out(arg1, grad_input, grad_output, self, noise, lower, upper, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, scalar, scalar, Cint), arg1, grad_input, grad_output, self, noise, lower, upper, training)
end

function atg_rrelu_with_noise_out(arg1, out, self, noise, training)
    @runtime_error_check ccall((:atg_rrelu_with_noise_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint), arg1, out, self, noise, training)
end

function atg_rsqrt(arg1, self)
    @runtime_error_check ccall((:atg_rsqrt, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rsqrt_(arg1, self)
    @runtime_error_check ccall((:atg_rsqrt_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_rsqrt_out(arg1, out, self)
    @runtime_error_check ccall((:atg_rsqrt_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_rsub(arg1, self, other)
    @runtime_error_check ccall((:atg_rsub, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_rsub1(arg1, self, other)
    @runtime_error_check ccall((:atg_rsub1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_scalar_tensor(arg1, s, options_kind, options_device)
    @runtime_error_check ccall((:atg_scalar_tensor, libdoeye_caml), Cint, (Ptr{tensor}, scalar, Cint, Cint), arg1, s, options_kind, options_device)
end

function atg_scatter(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter1(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_scatter1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_scatter_(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_1(arg1, self, dim, index, value)
    @runtime_error_check ccall((:atg_scatter_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, scalar), arg1, self, dim, index, value)
end

function atg_scatter_add(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_add, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_scatter_add_(arg1, self, dim, index, src)
    @runtime_error_check ccall((:atg_scatter_add_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, tensor, tensor), arg1, self, dim, index, src)
end

function atg_select(arg1, self, dim, index)
    @runtime_error_check ccall((:atg_select, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim, index)
end

function atg_selu(arg1, self)
    @runtime_error_check ccall((:atg_selu, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_selu_(arg1, self)
    @runtime_error_check ccall((:atg_selu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_set_(arg1, self)
    @runtime_error_check ccall((:atg_set_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_set_1(arg1, self, source)
    @runtime_error_check ccall((:atg_set_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, source)
end

function atg_set_requires_grad(arg1, self, r)
    @runtime_error_check ccall((:atg_set_requires_grad, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, r)
end

function atg_sigmoid(arg1, self)
    @runtime_error_check ccall((:atg_sigmoid, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sigmoid_(arg1, self)
    @runtime_error_check ccall((:atg_sigmoid_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sigmoid_backward(arg1, grad_output, output)
    @runtime_error_check ccall((:atg_sigmoid_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, output)
end

function atg_sigmoid_backward_out(arg1, grad_input, grad_output, output)
    @runtime_error_check ccall((:atg_sigmoid_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, output)
end

function atg_sigmoid_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sigmoid_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sign(arg1, self)
    @runtime_error_check ccall((:atg_sign, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sign_(arg1, self)
    @runtime_error_check ccall((:atg_sign_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sign_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sign_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sin(arg1, self)
    @runtime_error_check ccall((:atg_sin, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sin_(arg1, self)
    @runtime_error_check ccall((:atg_sin_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sin_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sin_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_sinh(arg1, self)
    @runtime_error_check ccall((:atg_sinh, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinh_(arg1, self)
    @runtime_error_check ccall((:atg_sinh_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sinh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sinh_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_slice(arg1, self, dim, start, _end, step)
    @runtime_error_check ccall((:atg_slice, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, Int64), arg1, self, dim, start, _end, step)
end

function atg_slogdet(arg1, self)
    @runtime_error_check ccall((:atg_slogdet, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_slow_conv3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_slow_conv3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
end

function atg_slow_conv3d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
    @runtime_error_check ccall((:atg_slow_conv3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len)
end

function atg_slow_conv_dilated2d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_dilated2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_dilated3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_dilated3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose2d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose2d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose3d(arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_slow_conv_transpose3d_out(arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
    @runtime_error_check ccall((:atg_slow_conv_transpose3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Ptr{Int64}, Cint, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, out, self, weight, kernel_size_data, kernel_size_len, bias, stride_data, stride_len, padding_data, padding_len, output_padding_data, output_padding_len, dilation_data, dilation_len)
end

function atg_smm(arg1, self, mat2)
    @runtime_error_check ccall((:atg_smm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mat2)
end

function atg_smooth_l1_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_smooth_l1_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_smooth_l1_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_smooth_l1_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_smooth_l1_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_smooth_l1_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_smooth_l1_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_smooth_l1_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_soft_margin_loss(arg1, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, self, target, reduction)
end

function atg_soft_margin_loss_backward(arg1, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, grad_output, self, target, reduction)
end

function atg_soft_margin_loss_backward_out(arg1, grad_input, grad_output, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Int64), arg1, grad_input, grad_output, self, target, reduction)
end

function atg_soft_margin_loss_out(arg1, out, self, target, reduction)
    @runtime_error_check ccall((:atg_soft_margin_loss_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64), arg1, out, self, target, reduction)
end

function atg_softmax(arg1, self, dim, dtype)
    @runtime_error_check ccall((:atg_softmax, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, dtype)
end

function atg_softplus(arg1, self)
    @runtime_error_check ccall((:atg_softplus, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_softplus_backward(arg1, grad_output, self, beta, threshold, output)
    @runtime_error_check ccall((:atg_softplus_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar, tensor), arg1, grad_output, self, beta, threshold, output)
end

function atg_softplus_backward_out(arg1, grad_input, grad_output, self, beta, threshold, output)
    @runtime_error_check ccall((:atg_softplus_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar, scalar, tensor), arg1, grad_input, grad_output, self, beta, threshold, output)
end

function atg_softplus_out(arg1, out, self)
    @runtime_error_check ccall((:atg_softplus_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_softshrink(arg1, self)
    @runtime_error_check ccall((:atg_softshrink, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_softshrink_backward(arg1, grad_output, self, lambd)
    @runtime_error_check ccall((:atg_softshrink_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_output, self, lambd)
end

function atg_softshrink_backward_out(arg1, grad_input, grad_output, self, lambd)
    @runtime_error_check ccall((:atg_softshrink_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, scalar), arg1, grad_input, grad_output, self, lambd)
end

function atg_softshrink_out(arg1, out, self)
    @runtime_error_check ccall((:atg_softshrink_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_solve(arg1, self, A)
    @runtime_error_check ccall((:atg_solve, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, A)
end

function atg_solve_out(arg1, solution, lu, self, A)
    @runtime_error_check ccall((:atg_solve_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, solution, lu, self, A)
end

function atg_sort(arg1, self, dim, descending)
    @runtime_error_check ccall((:atg_sort, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint), arg1, self, dim, descending)
end

function atg_sort_out(arg1, values, indices, self, dim, descending)
    @runtime_error_check ccall((:atg_sort_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Cint), arg1, values, indices, self, dim, descending)
end

function atg_sparse_coo_tensor(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_sparse_coo_tensor1(arg1, indices, values, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, indices, values, options_kind, options_device)
end

function atg_sparse_coo_tensor2(arg1, indices, values, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_sparse_coo_tensor2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, indices, values, size_data, size_len, options_kind, options_device)
end

function atg_sparse_mask(arg1, self, mask)
    @runtime_error_check ccall((:atg_sparse_mask, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, mask)
end

function atg_sparse_resize_(arg1, self, size_data, size_len, sparse_dim, dense_dim)
    @runtime_error_check ccall((:atg_sparse_resize_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, self, size_data, size_len, sparse_dim, dense_dim)
end

function atg_sparse_resize_and_clear_(arg1, self, size_data, size_len, sparse_dim, dense_dim)
    @runtime_error_check ccall((:atg_sparse_resize_and_clear_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64, Int64), arg1, self, size_data, size_len, sparse_dim, dense_dim)
end

function atg_split(arg1, self, split_size, dim)
    @runtime_error_check ccall((:atg_split, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, split_size, dim)
end

function atg_split_with_sizes(arg1, self, split_sizes_data, split_sizes_len, dim)
    @runtime_error_check ccall((:atg_split_with_sizes, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Int64), arg1, self, split_sizes_data, split_sizes_len, dim)
end

function atg_sqrt(arg1, self)
    @runtime_error_check ccall((:atg_sqrt, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sqrt_(arg1, self)
    @runtime_error_check ccall((:atg_sqrt_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_sqrt_out(arg1, out, self)
    @runtime_error_check ccall((:atg_sqrt_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_squeeze(arg1, self)
    @runtime_error_check ccall((:atg_squeeze, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_squeeze1(arg1, self, dim)
    @runtime_error_check ccall((:atg_squeeze1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_squeeze_(arg1, self)
    @runtime_error_check ccall((:atg_squeeze_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_squeeze_1(arg1, self, dim)
    @runtime_error_check ccall((:atg_squeeze_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_sspaddmm(arg1, self, mat1, mat2)
    @runtime_error_check ccall((:atg_sspaddmm, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, self, mat1, mat2)
end

function atg_sspaddmm_out(arg1, out, self, mat1, mat2)
    @runtime_error_check ccall((:atg_sspaddmm_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor), arg1, out, self, mat1, mat2)
end

function atg_stack(arg1, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_stack, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{tensor}, Cint, Int64), arg1, tensors_data, tensors_len, dim)
end

function atg_stack_out(arg1, out, tensors_data, tensors_len, dim)
    @runtime_error_check ccall((:atg_stack_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{tensor}, Cint, Int64), arg1, out, tensors_data, tensors_len, dim)
end

function atg_std(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_std, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_std1(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_std_mean(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_std_mean, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_std_mean1(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std_mean1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_std_out(arg1, out, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_std_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_stft(arg1, self, n_fft, hop_length, win_length, window, normalized, onesided)
    @runtime_error_check ccall((:atg_stft, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64, tensor, Cint, Cint), arg1, self, n_fft, hop_length, win_length, window, normalized, onesided)
end

function atg_sub(arg1, self, other)
    @runtime_error_check ccall((:atg_sub, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_sub1(arg1, self, other)
    @runtime_error_check ccall((:atg_sub1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_sub_(arg1, self, other)
    @runtime_error_check ccall((:atg_sub_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_sub_1(arg1, self, other)
    @runtime_error_check ccall((:atg_sub_1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar), arg1, self, other)
end

function atg_sub_out(arg1, out, self, other)
    @runtime_error_check ccall((:atg_sub_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, other)
end

function atg_sum(arg1, self, dtype)
    @runtime_error_check ccall((:atg_sum, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, dtype)
end

function atg_sum1(arg1, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_sum1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, keepdim, dtype)
end

function atg_sum_out(arg1, out, self, dim_data, dim_len, keepdim, dtype)
    @runtime_error_check ccall((:atg_sum_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, keepdim, dtype)
end

function atg_sum_to_size(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_sum_to_size, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_svd(arg1, self, some, compute_uv)
    @runtime_error_check ccall((:atg_svd, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, some, compute_uv)
end

function atg_svd_out(arg1, U, S, V, self, some, compute_uv)
    @runtime_error_check ccall((:atg_svd_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint), arg1, U, S, V, self, some, compute_uv)
end

function atg_symeig(arg1, self, eigenvectors, upper)
    @runtime_error_check ccall((:atg_symeig, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, eigenvectors, upper)
end

function atg_symeig_out(arg1, e, V, self, eigenvectors, upper)
    @runtime_error_check ccall((:atg_symeig_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cint, Cint), arg1, e, V, self, eigenvectors, upper)
end

function atg_t(arg1, self)
    @runtime_error_check ccall((:atg_t, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_t_(arg1, self)
    @runtime_error_check ccall((:atg_t_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_take(arg1, self, index)
    @runtime_error_check ccall((:atg_take, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, index)
end

function atg_take_out(arg1, out, self, index)
    @runtime_error_check ccall((:atg_take_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, out, self, index)
end

function atg_tan(arg1, self)
    @runtime_error_check ccall((:atg_tan, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tan_(arg1, self)
    @runtime_error_check ccall((:atg_tan_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tan_out(arg1, out, self)
    @runtime_error_check ccall((:atg_tan_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_tanh(arg1, self)
    @runtime_error_check ccall((:atg_tanh, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tanh_(arg1, self)
    @runtime_error_check ccall((:atg_tanh_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_tanh_backward(arg1, grad_output, output)
    @runtime_error_check ccall((:atg_tanh_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad_output, output)
end

function atg_tanh_backward_out(arg1, grad_input, grad_output, output)
    @runtime_error_check ccall((:atg_tanh_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, grad_input, grad_output, output)
end

function atg_tanh_out(arg1, out, self)
    @runtime_error_check ccall((:atg_tanh_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_tensordot(arg1, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
    @runtime_error_check ccall((:atg_tensordot, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, self, other, dims_self_data, dims_self_len, dims_other_data, dims_other_len)
end

function atg_threshold(arg1, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, threshold, value)
end

function atg_threshold_(arg1, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, scalar, scalar), arg1, self, threshold, value)
end

function atg_threshold_backward(arg1, grad_output, self, threshold)
    @runtime_error_check ccall((:atg_threshold_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar), arg1, grad_output, self, threshold)
end

function atg_threshold_out(arg1, out, self, threshold, value)
    @runtime_error_check ccall((:atg_threshold_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, scalar, scalar), arg1, out, self, threshold, value)
end

function atg_to(arg1, self, device)
    @runtime_error_check ccall((:atg_to, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, device)
end

function atg_to1(arg1, self, options_kind, options_device, non_blocking, copy)
    @runtime_error_check ccall((:atg_to1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint, Cint), arg1, self, options_kind, options_device, non_blocking, copy)
end

function atg_to2(arg1, self, dtype, non_blocking, copy)
    @runtime_error_check ccall((:atg_to2, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint), arg1, self, dtype, non_blocking, copy)
end

function atg_to3(arg1, self, other, non_blocking, copy)
    @runtime_error_check ccall((:atg_to3, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint), arg1, self, other, non_blocking, copy)
end

function atg_to4(arg1, self, device, dtype, non_blocking, copy)
    @runtime_error_check ccall((:atg_to4, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint, Cint, Cint), arg1, self, device, dtype, non_blocking, copy)
end

function atg_to_dense(arg1, self)
    @runtime_error_check ccall((:atg_to_dense, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_to_dense_backward(arg1, grad, input)
    @runtime_error_check ccall((:atg_to_dense_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, input)
end

function atg_to_mkldnn(arg1, self)
    @runtime_error_check ccall((:atg_to_mkldnn, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_to_mkldnn_backward(arg1, grad, input)
    @runtime_error_check ccall((:atg_to_mkldnn_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, grad, input)
end

function atg_to_sparse(arg1, self)
    @runtime_error_check ccall((:atg_to_sparse, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_to_sparse1(arg1, self, sparse_dim)
    @runtime_error_check ccall((:atg_to_sparse1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, sparse_dim)
end

function atg_topk(arg1, self, k, dim, largest, sorted)
    @runtime_error_check ccall((:atg_topk, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Cint, Cint), arg1, self, k, dim, largest, sorted)
end

function atg_topk_out(arg1, values, indices, self, k, dim, largest, sorted)
    @runtime_error_check ccall((:atg_topk_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Int64, Int64, Cint, Cint), arg1, values, indices, self, k, dim, largest, sorted)
end

function atg_totype(arg1, self, scalar_type)
    @runtime_error_check ccall((:atg_totype, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, scalar_type)
end

function atg_trace(arg1, self)
    @runtime_error_check ccall((:atg_trace, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_transpose(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_transpose, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_transpose_(arg1, self, dim0, dim1)
    @runtime_error_check ccall((:atg_transpose_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64), arg1, self, dim0, dim1)
end

function atg_trapz(arg1, y, x, dim)
    @runtime_error_check ccall((:atg_trapz, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, y, x, dim)
end

function atg_trapz1(arg1, y, dx, dim)
    @runtime_error_check ccall((:atg_trapz1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Int64), arg1, y, dx, dim)
end

function atg_triangular_solve(arg1, self, A, upper, transpose, unitriangular)
    @runtime_error_check ccall((:atg_triangular_solve, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Cint, Cint, Cint), arg1, self, A, upper, transpose, unitriangular)
end

function atg_triangular_solve_out(arg1, X, M, self, A, upper, transpose, unitriangular)
    @runtime_error_check ccall((:atg_triangular_solve_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, tensor, Cint, Cint, Cint), arg1, X, M, self, A, upper, transpose, unitriangular)
end

function atg_tril(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_tril, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_tril_(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_tril_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_tril_indices(arg1, row, col, offset, options_kind, options_device)
    @runtime_error_check ccall((:atg_tril_indices, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Int64, Int64, Cint, Cint), arg1, row, col, offset, options_kind, options_device)
end

function atg_tril_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_tril_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_triplet_margin_loss(arg1, anchor, positive, negative, margin, p, eps, swap, reduction)
    @runtime_error_check ccall((:atg_triplet_margin_loss, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor, Cdouble, Cdouble, Cdouble, Cint, Int64), arg1, anchor, positive, negative, margin, p, eps, swap, reduction)
end

function atg_triu(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_triu, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_triu_(arg1, self, diagonal)
    @runtime_error_check ccall((:atg_triu_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, diagonal)
end

function atg_triu_indices(arg1, row, col, offset, options_kind, options_device)
    @runtime_error_check ccall((:atg_triu_indices, libdoeye_caml), Cint, (Ptr{tensor}, Int64, Int64, Int64, Cint, Cint), arg1, row, col, offset, options_kind, options_device)
end

function atg_triu_out(arg1, out, self, diagonal)
    @runtime_error_check ccall((:atg_triu_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Int64), arg1, out, self, diagonal)
end

function atg_trunc(arg1, self)
    @runtime_error_check ccall((:atg_trunc, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_trunc_(arg1, self)
    @runtime_error_check ccall((:atg_trunc_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_trunc_out(arg1, out, self)
    @runtime_error_check ccall((:atg_trunc_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, out, self)
end

function atg_type_as(arg1, self, other)
    @runtime_error_check ccall((:atg_type_as, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_unbind(arg1, self, dim)
    @runtime_error_check ccall((:atg_unbind, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_unfold(arg1, self, dimension, size, step)
    @runtime_error_check ccall((:atg_unfold, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Int64, Int64), arg1, self, dimension, size, step)
end

function atg_uniform_(arg1, self, from, to)
    @runtime_error_check ccall((:atg_uniform_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cdouble, Cdouble), arg1, self, from, to)
end

function atg_unique_consecutive(arg1, self, return_inverse, return_counts, dim)
    @runtime_error_check ccall((:atg_unique_consecutive, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint, Int64), arg1, self, return_inverse, return_counts, dim)
end

function atg_unique_dim(arg1, self, dim, sorted, return_inverse, return_counts)
    @runtime_error_check ccall((:atg_unique_dim, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint, Cint), arg1, self, dim, sorted, return_inverse, return_counts)
end

function atg_unique_dim_consecutive(arg1, self, dim, return_inverse, return_counts)
    @runtime_error_check ccall((:atg_unique_dim_consecutive, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64, Cint, Cint), arg1, self, dim, return_inverse, return_counts)
end

function atg_unsqueeze(arg1, self, dim)
    @runtime_error_check ccall((:atg_unsqueeze, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_unsqueeze_(arg1, self, dim)
    @runtime_error_check ccall((:atg_unsqueeze_, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Int64), arg1, self, dim)
end

function atg_upsample_bicubic2d(arg1, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bicubic2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_bicubic2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_bicubic2d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_bicubic2d_out(arg1, out, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bicubic2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_bilinear2d(arg1, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bilinear2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_bilinear2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_bilinear2d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_bilinear2d_out(arg1, out, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_bilinear2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_linear1d(arg1, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_linear1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_linear1d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_linear1d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_linear1d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_linear1d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_linear1d_out(arg1, out, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_linear1d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_nearest1d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest1d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_upsample_nearest1d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest1d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest1d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest1d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest1d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest1d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_upsample_nearest2d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest2d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_upsample_nearest2d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest2d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest2d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest2d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest2d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest2d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_upsample_nearest3d(arg1, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, output_size_data, output_size_len)
end

function atg_upsample_nearest3d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest3d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len)
end

function atg_upsample_nearest3d_out(arg1, out, self, output_size_data, output_size_len)
    @runtime_error_check ccall((:atg_upsample_nearest3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint), arg1, out, self, output_size_data, output_size_len)
end

function atg_upsample_trilinear3d(arg1, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_trilinear3d, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint), arg1, self, output_size_data, output_size_len, align_corners)
end

function atg_upsample_trilinear3d_backward(arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_backward, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_trilinear3d_backward_out(arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_backward_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Ptr{Int64}, Cint, Cint), arg1, grad_input, grad_output, output_size_data, output_size_len, input_size_data, input_size_len, align_corners)
end

function atg_upsample_trilinear3d_out(arg1, out, self, output_size_data, output_size_len, align_corners)
    @runtime_error_check ccall((:atg_upsample_trilinear3d_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint), arg1, out, self, output_size_data, output_size_len, align_corners)
end

function atg_values(arg1, self)
    @runtime_error_check ccall((:atg_values, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_var(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_var, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_var1(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_var_mean(arg1, self, unbiased)
    @runtime_error_check ccall((:atg_var_mean, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint), arg1, self, unbiased)
end

function atg_var_mean1(arg1, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var_mean1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_var_out(arg1, out, self, dim_data, dim_len, unbiased, keepdim)
    @runtime_error_check ccall((:atg_var_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, Ptr{Int64}, Cint, Cint, Cint), arg1, out, self, dim_data, dim_len, unbiased, keepdim)
end

function atg_view(arg1, self, size_data, size_len)
    @runtime_error_check ccall((:atg_view, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, self, size_data, size_len)
end

function atg_view_as(arg1, self, other)
    @runtime_error_check ccall((:atg_view_as, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor), arg1, self, other)
end

function atg_where(arg1, condition)
    @runtime_error_check ccall((:atg_where, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, condition)
end

function atg_where1(arg1, condition, self, other)
    @runtime_error_check ccall((:atg_where1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, tensor, tensor), arg1, condition, self, other)
end

function atg_zero_(arg1, self)
    @runtime_error_check ccall((:atg_zero_, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_zeros(arg1, size_data, size_len, options_kind, options_device)
    @runtime_error_check ccall((:atg_zeros, libdoeye_caml), Cint, (Ptr{tensor}, Ptr{Int64}, Cint, Cint, Cint), arg1, size_data, size_len, options_kind, options_device)
end

function atg_zeros_like(arg1, self)
    @runtime_error_check ccall((:atg_zeros_like, libdoeye_caml), Cint, (Ptr{tensor}, tensor), arg1, self)
end

function atg_zeros_like1(arg1, self, options_kind, options_device)
    @runtime_error_check ccall((:atg_zeros_like1, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Cint, Cint), arg1, self, options_kind, options_device)
end

function atg_zeros_out(arg1, out, size_data, size_len)
    @runtime_error_check ccall((:atg_zeros_out, libdoeye_caml), Cint, (Ptr{tensor}, tensor, Ptr{Int64}, Cint), arg1, out, size_data, size_len)
end

# exports
const PREFIXES = ["at"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
