#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef WIN32
#define C_API extern "C" __declspec(dllexport)
#else
#define C_API extern "C"
#endif

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
typedef torch::jit::script::Module *module;
typedef torch::jit::IValue *ivalue;
typedef torch::NoGradGuard *ngg;
char* myerr = "";
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    myerr = strdup(e.what()); \
    /* jl_error(strdup(e.what())); */ \
    /* throw(e.what()); */ \
  }
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *scalar;
typedef void *module;
typedef void *ivalue;
typedef void *ngg;
#endif

C_API int get_last_error(char *);
C_API int flush_error();

C_API int at_manual_seed(int64_t);
C_API int at_new_tensor(tensor *);
C_API int at_empty_cache();
C_API int at_no_grad(int flag);
C_API int at_sync();
C_API int at_from_blob(tensor *, void *data, int64_t *dims, int ndims, int64_t *strides, int nstrides, int dev);
C_API int at_tensor_of_data(tensor *, void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type);
C_API int at_copy_data(tensor tensor, void *vs, int64_t numel, int element_size_in_bytes);
C_API int at_float_vec(tensor* tensor, double *values, int value_len, int type);
C_API int at_int_vec(tensor* tensor, int64_t *values, int value_len, int type);

C_API int at_defined(int *i, tensor);
C_API int at_dim(int *i, tensor);
C_API int at_shape(tensor, int *);
C_API int at_scalar_type(int *i, tensor);

C_API int at_backward(tensor, int, int);
C_API int at_requires_grad(int *i, tensor);
C_API int at_grad_set_enabled(int);

C_API int at_get(tensor *, tensor, int index);
C_API int at_fill_double(tensor, double);
C_API int at_fill_int64(tensor, int64_t);

C_API int at_double_value_at_indexes(double *i, tensor, int *indexes, int indexes_len);
C_API int at_int64_value_at_indexes(int64_t *i, tensor, int *indexes, int indexes_len);
C_API int at_set_double_value_at_indexes(tensor, int *indexes, int indexes_len, double v);
C_API int at_set_int64_value_at_indexes(tensor, int *indexes, int indexes_len, int64_t v);

C_API int at_copy_(tensor dst, tensor src);

C_API int at_print(tensor);
// char *at_to_string(tensor, int line_size);
C_API int at_save(tensor, char *filename);
C_API int at_load(char *filename, tensor *tensor);

C_API int at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
C_API int at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
C_API int at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

C_API int at_load_callback(char *filename, void (*f)(char *, tensor));

C_API int at_free(tensor);

C_API int at_run_backward(tensor *tensors,
                          int ntensors,
                          tensor *inputs,
                          int ninputs,
                          tensor *outputs,
                          int keep_graph,
                          int create_graph);

C_API int ato_adam(optimizer *, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay);
C_API int ato_rmsprop(optimizer *, double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered);
C_API int ato_sgd(optimizer *, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
C_API int ato_add_parameters(optimizer, tensor *, int ntensors);
C_API int ato_set_learning_rate(optimizer, double learning_rate);
C_API int ato_set_momentum(optimizer, double momentum);
C_API int ato_zero_grad(optimizer);
C_API int ato_step(optimizer);
C_API int ato_free(optimizer);

C_API int ats_int(scalar *, int64_t);
C_API int ats_float(scalar *, double);
C_API int ats_free(scalar);

C_API int atc_cuda_device_count(int *);
C_API int atc_cuda_is_available(int *);
C_API int atc_cudnn_is_available(int *);
C_API int atc_set_benchmark_cudnn(int b);

C_API int atm_load(char *, module *);
C_API int atm_forward(tensor *, module, tensor *tensors, int ntensors);
C_API int atm_forward_(ivalue *, module,
                       ivalue *ivalues,
                       int nivalues);
C_API int atm_free(module);

C_API int ati_tensor(ivalue *, tensor);
C_API int ati_int(ivalue *, int64_t);
C_API int ati_double(ivalue *, double);
C_API int ati_tuple(ivalue *, ivalue *, int);

C_API int ati_to_tensor(tensor *, ivalue);
C_API int ati_to_int(int64_t *, ivalue);
C_API int ati_to_double(double *, ivalue);
C_API int ati_tuple_length(int *, ivalue);
C_API int ati_to_tuple(ivalue, ivalue *, int);

C_API int ati_tag(int *, ivalue);

C_API int ati_free(ivalue);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif
