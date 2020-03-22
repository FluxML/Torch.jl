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
typedef torch::NoGradGuard *ngg;
char* myerr = "";
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    myerr = strdup(e.what()); \
  }
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *scalar;
typedef void *module;
typedef void *ivalue;
typedef void *ngg;
#endif

char* get_last_error();
void flush_error();

void at_manual_seed(int64_t);
void at_new_tensor(tensor *);
void at_empty_cache();
void at_no_grad(int flag);
void at_sync();
void at_from_blob(tensor *, void *data, int64_t *dims, int ndims, int64_t *strides, int nstrides, int dev);
int at_tensor_of_data(tensor *, void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type);
int at_copy_data(tensor tensor, void *vs, int64_t numel, int element_size_in_bytes);
void at_float_vec(double *values, int value_len, int type);
void at_int_vec(int64_t *values, int value_len, int type);

void at_defined(int *i, tensor);
void at_dim(int *i, tensor);
void at_shape(tensor, int *);
void at_scalar_type(int *i, tensor);

void at_backward(tensor, int, int);
void at_requires_grad(int *i, tensor);
void at_grad_set_enabled(int);

void at_get(tensor *, tensor, int index);
void at_fill_double(tensor, double);
void at_fill_int64(tensor, int64_t);

void at_double_value_at_indexes(double *i, tensor, int *indexes, int indexes_len);
void at_int64_value_at_indexes(double *i, tensor, int *indexes, int indexes_len);
void at_set_double_value_at_indexes(tensor, int *indexes, int indexes_len, double v);
void at_set_int64_value_at_indexes(tensor, int *indexes, int indexes_len, int64_t v);

void at_copy_(tensor dst, tensor src);

void at_print(tensor);
char *at_to_string(tensor, int line_size);
void at_save(tensor, char *filename);
tensor at_load(char *filename);

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

void at_load_callback(char *filename, void (*f)(char *, tensor));

void at_free(tensor);

void at_run_backward(tensor *tensors,
                      int ntensors,
                      tensor *inputs,
                      int ninputs,
                      tensor *outputs,
                      int keep_graph,
                      int create_graph);

void ato_adam(optimizer *, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay);
void ato_rmsprop(optimizer *, double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered);
void ato_sgd(optimizer *, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
void ato_add_parameters(optimizer, tensor *, int ntensors);
void ato_set_learning_rate(optimizer, double learning_rate);
void ato_set_momentum(optimizer, double momentum);
void ato_zero_grad(optimizer);
void ato_step(optimizer);
void ato_free(optimizer);

void ats_int(scalar *, int64_t);
void ats_float(scalar *, double);
void ats_free(scalar);

void atc_cuda_device_count(int *);
void atc_cuda_is_available(int *);
void atc_cudnn_is_available(int *);
void atc_set_benchmark_cudnn(int b);

void atm_load(char *, module *);
int atm_forward(tensor *, module, tensor *tensors, int ntensors);
void atm_forward_(ivalue *, module,
                    ivalue *ivalues,
                    int nivalues);
void atm_free(module);

void ati_tensor(ivalue *, tensor);
void ati_int(ivalue *, int64_t);
void ati_double(ivalue *, double);
void ati_tuple(ivalue *, ivalue *, int);

void ati_to_tensor(tensor *, ivalue);
void ati_to_int(int64_t *, ivalue);
void ati_to_double(double *, ivalue);
void ati_tuple_length(int *, ivalue);
int ati_to_tuple(ivalue, ivalue *, int);

void ati_tag(int *, ivalue);

void ati_free(ivalue);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif
