#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
typedef torch::jit::script::Module *module;
typedef torch::jit::IValue *ivalue;
char const *myerr = "";
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
#endif

int get_last_error(char *);
int flush_error();

int at_manual_seed(int64_t);
int at_new_tensor(tensor *);
int at_empty_cache();
int at_no_grad(int flag);
int at_sync();
int at_from_blob(tensor *, void *data, int64_t *dims, int ndims, int64_t *strides, int nstrides, int dev);
int at_tensor_of_data(tensor *, void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type);
int at_copy_data(tensor tensor, void *vs, int64_t numel, int element_size_in_bytes);
int at_float_vec(tensor *, double *values, int value_len, int type);
int at_int_vec(tensor *, int64_t *values, int value_len, int type);

int at_defined(int *, tensor);
int at_is_sparse(int *, tensor);
int at_device(int *, tensor);
int at_dim(int *, tensor);
int at_shape(tensor, int *);
int at_stride(tensor, int *);
int at_scalar_type(int *, tensor);

int at_autocast_clear_cache();
int at_autocast_decrement_nesting(int *);
int at_autocast_increment_nesting(int *);
int at_autocast_is_enabled(int *);
int at_autocast_set_enabled(int *, int b);

int at_backward(tensor, int, int);
int at_requires_grad(int *, tensor);
int at_grad_set_enabled(int *, int);

int at_get(tensor *, tensor, int index);
int at_fill_double(tensor, double);
int at_fill_int64(tensor, int64_t);

int at_double_value_at_indexes(double *, tensor, int *indexes, int indexes_len);
int at_int64_value_at_indexes(int64_t *, tensor, int *indexes, int indexes_len);
int at_set_double_value_at_indexes(tensor, int *indexes, int indexes_len, double v);
int at_set_int64_value_at_indexes(tensor, int *indexes, int indexes_len, int64_t v);

int at_copy_(tensor dst, tensor src);

int at_print(tensor);
int at_to_string(char **, tensor, int line_size);
int at_save(tensor, char *filename);
int at_load(tensor *, char *filename);

int at_get_num_threads(int *);
int at_set_num_threads(int n_threads);

int at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
int at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
int at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

int at_load_callback(char *filename, void (*f)(char *, tensor));

int at_free(tensor);

int at_run_backward(tensor *tensors,
                      int ntensors,
                      tensor *inputs,
                      int ninputs,
                      tensor *outputs,
                      int keep_graph,
                      int create_graph);

int ato_adam(optimizer *, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps);
int ato_rmsprop(optimizer *, double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered);
int ato_sgd(optimizer *, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
int ato_add_parameters(optimizer, tensor *, int ntensors);
int ato_set_learning_rate(optimizer, double learning_rate);
int ato_set_momentum(optimizer, double momentum);
int ato_zero_grad(optimizer);
int ato_step(optimizer);
int ato_free(optimizer);

int ats_int(scalar *, int64_t);
int ats_float(scalar *, double);
int ats_free(scalar);

int atc_cuda_device_count(int *);
int atc_cuda_is_available(int *);
int atc_cudnn_is_available(int *);
int atc_set_benchmark_cudnn(int b);

int atm_load(module *, char *);
int atm_forward(tensor *, module, tensor *tensors, int ntensors);
int atm_forward_(ivalue *, module,
                    ivalue *ivalues,
                    int nivalues);
int atm_free(module);

int ati_none(ivalue *);
int ati_tensor(ivalue *, tensor);
int ati_bool(ivalue *, int);
int ati_int(ivalue *, int64_t);
int ati_double(ivalue *, double);
int ati_tuple(ivalue *, ivalue *, int);
int ati_string(ivalue *, char *);
int ati_tuple(ivalue *, ivalue *, int);
int ati_generic_list(ivalue *, ivalue *, int);
int ati_generic_dict(ivalue *, ivalue *, int);
int ati_int_list(ivalue *, int64_t *, int);
int ati_double_list(ivalue *, double *, int);
int ati_bool_list(ivalue *, char *, int);
int ati_string_list(ivalue *, char **, int);
int ati_tensor_list(ivalue *, tensor *, int);

int ati_to_tensor(tensor *, ivalue);
int ati_to_int(int64_t *, ivalue);
int ati_to_double(double *, ivalue);
int ati_to_string(char **, ivalue);
int ati_to_bool(int *, ivalue);
int ati_length(int *, ivalue);
int ati_tuple_length(int *, ivalue);
int ati_to_tuple(ivalue, ivalue *, int);
int ati_to_generic_list(ivalue, ivalue *, int);
int ati_to_generic_dict(ivalue, ivalue *, int);
int ati_to_int_list(ivalue, int64_t *, int);
int ati_to_double_list(ivalue, double *, int);
int ati_to_bool_list(ivalue, char *, int);
int ati_to_tensor_list(ivalue, tensor *, int);

int ati_tag(int *, ivalue);

int ati_free(ivalue);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif
