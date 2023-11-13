#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<ATen/autocast_mode.h>
#include<torch/script.h>
#ifdef USE_CUDA
#include<c10/cuda/CUDACachingAllocator.h>
#include<c10/cuda/CUDAStream.h>
#endif
#include<vector>
#include "torch_api.h"

#define caml_invalid_argument printf

using namespace std;

int get_last_error(char *err) {
   int len = strlen(myerr);
   for (int i = 0; i < len; ++i) err[i] = myerr[i];
   err[len] = '\0'; 
   return 0;
}

int flush_error() {
  PROTECT(
    myerr = "";
    return 0;
  )
  return 1;
}

int at_manual_seed(int64_t seed) {
  PROTECT(
    torch::manual_seed(seed);
    return 0;
  )
  return 1;
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs, int len) {
  vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(vs[i] != nullptr ? c10::optional<torch::Tensor>(*(vs[i])) : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

at::Device device_of_int(int d) {
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

int at_from_blob(tensor *out__, void *data, int64_t *dims, int ndims, int64_t *strides, int nstrides, int dev) {
  PROTECT(
    auto options = torch::TensorOptions().device(torch::kCUDA, dev).requires_grad(false);
    torch::Tensor tens = torch::from_blob(data, torch::IntArrayRef(dims, ndims), torch::IntArrayRef(strides, nstrides), options);
    out__[0] = new torch::Tensor(tens);
    return 0;
  )
  return 1;
}

int at_new_tensor(tensor *out__) {
  PROTECT(
    out__[0] = new torch::Tensor();
    return 0;
  )
  return 1;
}

int at_empty_cache() {
  PROTECT(
#if defined(USE_CUDA)
    c10::cuda::CUDACachingAllocator::emptyCache();
    return 0;
#else
    myerr = strdup("CUDA is disabled.");
    return 1;
#endif
  )
  return 1;
}

int at_no_grad(int flag) {
  PROTECT(
    torch::GradMode::set_enabled((bool)flag);
    return 0;
  )
  return 1;
}

int at_sync() {
  PROTECT(
#ifdef USE_CUDA
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
#else
    myerr = strdup("CUDA is disabled.");
    return 1;
#endif
  )
  // torch::cuda::synchronize();
  return 1;
}

int at_tensor_of_data(tensor *out__, void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if (element_size_in_bytes != tensor.element_size()) {
      myerr = strdup("incoherent element sizes in bytes");
      return 1;
    }
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_copy_data(tensor tensor, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
    if (elt_size_in_bytes != tensor->element_size()) {
      myerr = strdup("incoherent element sizes in bytes");
      return 1;
    }
    if (numel != tensor->numel()) {
      myerr = strdup("incoherent number of elements");
      return 1;
    }
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      torch::Tensor tmp_tensor = tensor->contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    return 0;
  )
  return 1;
}

int at_float_vec(tensor *out__, double *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_int_vec(tensor *out__, int64_t *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_defined(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->defined();
    return 0;
  )
  return 1;
}

int at_is_sparse(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->is_sparse();
    return 0;
  )
  return 1;
}

int at_dim(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->dim();
    return 0;
  )
  return 1;
}

int at_shape(tensor t, int *dims) {
  PROTECT(
    int i = 0;
    for (int dim : t->sizes()) dims[i++] = dim;
    return 0;
  )
  return 1;
}

int at_stride(tensor t, int64_t *dims) {
  PROTECT(
    int i = 0;
    for (int64_t dim: t->strides()) dims[i++] = dim;
    return 0;
  )
  return 1;
}

int at_scalar_type(int *out__, tensor t) {
  PROTECT(
    out__[0] = static_cast<int>(t->scalar_type());
    return 0;
  )
  return 1;
}

int at_autocast_clear_cache() {
  PROTECT(
    at::autocast::clear_cache();
    return 0;
  )
  return 1;
}

int at_autocast_decrement_nesting(int *out__) {
  PROTECT(
    out__[0] = at::autocast::decrement_nesting();
    return 0;
  )
  return 1;
}

int at_autocast_increment_nesting(int *out__) {
  PROTECT(
    out__[0] = at::autocast::increment_nesting();
    return 0;
  )
  return 1;
}

int at_autocast_is_enabled(int *out__) {
  PROTECT(
    out__[0] = at::autocast::is_enabled();
    return 0;
  )
  return 1;
}

int at_autocast_set_enabled(int *out__, int b) {
  PROTECT(
    bool is_enabled = at::autocast::is_enabled();
    at::autocast::set_enabled(b);
    out__[0] = is_enabled;
  )
  return 1;
}

int at_device(int *out__, tensor tensor) {
  PROTECT (
    auto device = tensor->device();
    if (device.is_cpu()) out__[0] = -1;
    out__[0] = device.index();
    return 0;
  )
  return 1;
}

int at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(
    t->backward({}, keep_graph, create_graph);
    return 0;
  )
  return 1;
}

int at_requires_grad(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->requires_grad();
    return 0;
  )
  return 1;
}

int at_grad_set_enabled(int *out__, int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    out__[0] = is_enabled;
    return 0;
  )
  return 1;
}

int at_get(tensor *out__, tensor t, int index) {
  PROTECT(
    out__[0] = new torch::Tensor((*t)[index]);
    return 0;
  )
  return 1;
}

template<typename T>
T at_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    return tensor.item<T>();
  )
  return T();
}

int at_double_value_at_indexes(double *out__, tensor t, int *indexes, int indexes_len) {
  PROTECT(
    out__[0] = at_value_at_indexes<double>(t, indexes, indexes_len);
    return 0;
  )
  return 1;
}

int at_int64_value_at_indexes(int64_t *out__, tensor t, int *indexes, int indexes_len) {
  PROTECT(
    out__[0] = at_value_at_indexes<int64_t>(t, indexes, indexes_len);
    return 0;
  )
  return 1;
}

template<typename T>
int at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    tensor.fill_(v);
    return 0;
  )
  return 1;
}

int at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len, double v) {
  return at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

int at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len, int64_t v) {
  return at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

int at_fill_double(tensor t, double v) {
  PROTECT(
    t->fill_(v);
    return 0;
  )
  return 1;
}

int at_fill_int64(tensor t, int64_t v) {
  PROTECT(
    t->fill_(v);
    return 0;
  )
  return 1;
}

int at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
    return 0;
  )
  return 1;
}

int at_to_string(char **out__, tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    out__[0] = strdup(oss.str().c_str());
    return 0;
  )
  return 1;
}

int at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
    return 0;
  )
  return 1;
}

int at_save(tensor t, char *filename) {
  PROTECT(
    torch::save(*t, filename);
    return 0;
  )
  return 1;
}

int at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
    return 0;
  )
  return 1;
}

int at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    vector<torch::Tensor> ts(ntensors);
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), ts[i]);
    // Only allocate the new tensor now so that if there is an exception raised during
    // [read], no memory has to be freed.
    for (int i = 0; i < ntensors; ++i)
      tensors[i] = new torch::Tensor(ts[i]);
    return 0;
  )
  return 1;
}

int at_load_callback(char *filename, void (*f)(char *, tensor)) {
  PROTECT(
    auto module = torch::jit::load(filename);
    for (const auto &p : module.named_parameters()) {
      auto v = p.value;
      f((char*)p.name.c_str(), new torch::Tensor(v));
    }
    return 0;
  )
  return 1;
}

int at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i) {
      if (tensors[i]->device().type() == at::kCPU)
        archive.read(std::string(tensor_names[i]), *(tensors[i]));
      else {
        torch::Tensor tmp_tensor = torch::empty_like(*(tensors[i]), at::device(at::kCPU));
        archive.read(std::string(tensor_names[i]), tmp_tensor);
        tensors[i]->copy_(tmp_tensor);
      }
    }
    return 0;
  )
  return 1;
}

int at_load(tensor *out__, char *filename) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_get_num_interop_threads(int *out__) {
  PROTECT(
    out__[0] = at::get_num_interop_threads();
    return 0;
  )
  return 1;
}

int at_get_num_threads(int *out__) {
  PROTECT(
    out__[0] = at::get_num_threads();
    return 0;
  )
  return 1;
}

int at_set_num_interop_threads(int n_threads) {
  PROTECT(
    at::set_num_interop_threads(n_threads);
    return 0;
  )
  return 1;
}

int at_set_num_threads(int n_threads) {
  PROTECT(
    at::set_num_threads(n_threads);
    return 0;
  )
  return 1;
}

int at_free(tensor t) {
  PROTECT(
    delete(t);
    return 0;
  )
  return 1;
}

int at_run_backward(tensor *tensors,
                     int ntensors,
                     tensor *inputs,
                     int ninputs,
                     tensor *outputs,
                     int keep_graph,
                     int create_graph) {
  PROTECT(
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::impl::gradient_edge(*tensors[i]));

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i) {
      if (!inputs[i]->requires_grad())
        caml_invalid_argument("one of the input tensor does not use set_requires_grad");
      inputs_.push_back(torch::autograd::impl::gradient_edge(*inputs[i]));
    }

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, false, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
    return 0;
  )
  return 1;
}

int ato_adam(optimizer *out__, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .betas(std::tuple<double, double>(beta1, beta2))
        .weight_decay(weight_decay)
        .eps(eps);
    out__[0] = new torch::optim::Adam(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_rmsprop(optimizer *out__, double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered) {
  PROTECT(
    auto options =
      torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered != 0);
    out__[0] = new torch::optim::RMSprop(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_sgd(optimizer *out__, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov) {
  PROTECT(
    auto options =
      torch::optim::SGDOptions(learning_rate)
      .momentum(momentum)
      .dampening(dampening)
      .weight_decay(weight_decay)
      .nesterov(nesterov);
    out__[0] = new torch::optim::SGD(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_add_parameters(optimizer t, tensor *tensors, int ntensors) {
  PROTECT(
    for (int i = 0; i < ntensors; ++i)
      t->param_groups()[0].params().push_back(*(tensors[i]));
    return 0;
  )
  return 1;
}

int ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    torch::optim::OptimizerOptions* d = &(t->defaults());
    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
      adam->lr(learning_rate);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions*>(d)) {
              adam2->lr(learning_rate);
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
      rms->lr(learning_rate);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
              rms2->lr(learning_rate);
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
      sgd->lr(learning_rate);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions*>(d)) {
              sgd2->lr(learning_rate);
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else
      caml_invalid_argument("unexpected optimizer");
    return 0;
  )
  return 1;
}

int ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    torch::optim::OptimizerOptions* d = &(t->defaults());
    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
      auto betas = adam->betas();
      adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions*>(d)) {
              adam2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
              rms2->momentum(momentum);
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
      sgd->momentum(momentum);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions*>(d)) {
              sgd2->momentum(momentum);
          }
          else caml_invalid_argument("unexpected param group type");
      }
    }
    else
      caml_invalid_argument("unexpected optimizer");
    return 0;
  )
  return 1;
}

int ato_zero_grad(optimizer t) {
  PROTECT(
    t->zero_grad();
    return 0;
  )
  return 1;
}

int ato_step(optimizer t) {
  PROTECT(
    t->step();
    return 0;
  )
  return 1;
}

int ato_free(optimizer t) {
  PROTECT(
    delete(t);
    return 0;
  )
  return 1;
}

int ats_int(scalar *out__, int64_t v) {
  PROTECT(
    out__[0] = new torch::Scalar(v);
    return 0;
  )
  return 1;
}

int ats_float(scalar *out__, double v) {
  PROTECT(
    out__[0] = new torch::Scalar(v);
    return 0;
  )
  return 1;
}

int ats_to_int(int64_t *out__, scalar s) {
  PROTECT(
    out__[0] = s->toLong();
    return 0;
  )
  return -1;
}

int ats_to_float(double *out__, scalar s) {
  PROTECT(
    out__[0] = s->toDouble();
    return 0;
  )
  return 1;
}

int ats_to_string(char **out__, scalar s) {
  PROTECT(
    using namespace at;
    std::ostringstream oss;
    oss << (*s);
    out__[0] = strdup(oss.str().c_str());
    return 0;
  )
  return 1;
}

int ats_free(scalar s) {
  PROTECT(
    delete(s);
    return 0;
  )
  return 1;
}

int atc_cuda_device_count(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::device_count();
    return 0;
  )
  return 1;
}

int atc_cuda_is_available(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::is_available();
    return 0;
  )
  return 1;
}

int atc_cudnn_is_available(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::cudnn_is_available();
    return 0;
  )
  return 1;
}

int atc_set_benchmark_cudnn(int b) {
  PROTECT(
    at::globalContext().setBenchmarkCuDNN(b);
    return 0;
  )
  return 1;
}

int atm_load(module *out__, char *filename) {
  PROTECT(
    out__[0] = new torch::jit::script::Module(torch::jit::load(filename));
    return 0;
  )
  return 1;
}

int atm_load_str(module *out__, char *data, size_t sz) {
  PROTECT(
    std::istringstream stream(std::string(data, sz));
    out__[0] = new torch::jit::script::Module(torch::jit::load(stream));
    return 0;
  )
  return 1;
}

int atm_forward(tensor *out__, module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->forward(inputs);
    if (!output.isTensor()) {
      myerr = strdup("forward did not return a tensor");
      return 1;
    }
    out__[0] = new torch::Tensor(output.toTensor());
    return 0;
  )
  return 1;
}

int atm_forward_(ivalue *out__, module m,
                    ivalue *ivalues,
                    int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = m->forward(inputs);
    out__[0] = new torch::jit::IValue(output);
    return 0;
  )
  return 1;
}

int atm_free(module m) {
  PROTECT(
    delete(m);
    return 0;
  )
  return 1;
}

int atm_to(module m, int device, int dtype, bool non_blocking) {
  PROTECT(
    m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);
    return 0;
  )
  return 1;
}

int ati_tensor(ivalue *out__, tensor t) {
  PROTECT(
    out__[0] = new torch::jit::IValue(*t);
    return 0;
  )
  return 1;
}

int ati_int(ivalue *out__, int64_t i) {
  PROTECT(
    out__[0] = new torch::jit::IValue(i);
    return 0;
  )
  return 1;
}

int ati_double(ivalue *out__, double d) {
  PROTECT(
    out__[0] = new torch::jit::IValue(d);
    return 0;
  )
  return 1;
}

int ati_bool(ivalue *out__, int i) {
  PROTECT(
    out__[0] = new torch::jit::IValue((bool)i);
    return 0;
  )
  return 1;
}

int ati_string(ivalue *out__, char *s) {
  PROTECT(
    string str(s);
    out__[0] = new torch::jit::IValue(str);
    return 0;
  )
  return 1;
}

int ati_none(ivalue *out__) {
  PROTECT(
    out__[0] = new torch::jit::IValue();
    return 0;
  )
  return 1;
}

int ati_tuple(ivalue *out__, ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(torch::ivalue::Tuple::create(vec));
    return 0;
  )
  return 1;
}

int ati_generic_list(ivalue *out__, ivalue *is, int nvalues) {
  PROTECT(
    c10::List<torch::jit::IValue> vec(c10::AnyType::get());
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));
    return 0;
  )
  return 1;
}

int ati_generic_dict(ivalue *out__, ivalue *is, int nvalues) {
  c10::Dict<torch::jit::IValue, torch::jit::IValue> dict(c10::AnyType::get(), c10::AnyType::get());
  PROTECT(
    for (int i = 0; i < nvalues; ++i) dict.insert(*(is[2*i]), *(is[2*i+1]));
    out__[0] = new torch::jit::IValue(dict);
    return 0;
  )
  return 1;
}

int ati_int_list(ivalue *out__, int64_t *is, int nvalues) {
  PROTECT(
    c10::List<int64_t> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_double_list(ivalue *out__, double *is, int nvalues) {
  PROTECT(
    c10::List<double> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_bool_list(ivalue *out__, char *is, int nvalues) {
  PROTECT(
    c10::List<bool> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_string_list(ivalue *out__, char **is, int nvalues) {
  PROTECT(
    c10::List<string> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(string(is[i]));
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_tensor_list(ivalue *out__, tensor *is, int nvalues) {
  PROTECT(
    c10::List<at::Tensor> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_tag(int *out__, ivalue i) {
  PROTECT(
    if (i->isNone()) {
      out__[0] = 0;
      return 0;
    }
    else if (i->isTensor()) {
      out__[0] = 1;
      return 0;
    }
    else if (i->isDouble()) {
      out__[0] = 2;
      return 0;
    }
    else if (i->isInt()) {
      out__[0] = 3;
      return 0;
    }
    else if (i->isBool()) {
      out__[0] = 4;
      return 0;
    }
    else if (i->isTuple()) {
      out__[0] = 5;
      return 0;
    }
    else if (i->isIntList()) {
      out__[0] = 6;
      return 0;
    }
    else if (i->isDoubleList()) {
      out__[0] = 7;
      return 0;
    }
    else if (i->isBoolList()) {
      out__[0] = 8;
      return 0;
    }
    else if (i->isString()) {
      out__[0] = 9;
      return 0;
    }
    else if (i->isTensorList()) {
      out__[0] = 10;
      return 0;
    }
    else if (i->isList()) {
      out__[0] = 12;
      return 0;
    }
    else if (i->isGenericDict()) {
      out__[0] = 13;
      return 0;
    }
    myerr = strdup(("unsupported tag" + i->tagKind()).c_str());
    return 1;
  )
  return 1;
}

int ati_to_int(int64_t *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toInt();
    return 0;
  )
  return 1;
}

int ati_to_double(double *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toDouble();
    return 0;
  )
  return 1;
}

int ati_to_bool(int *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toBool();
    return 0;
  )
  return 1;
}

int ati_to_string(char **out__, ivalue i) {
  PROTECT(
    auto str = i->toStringRef();
    out__[0] = strdup(str.c_str());
    return 0;
  )
  return 1;
}

int ati_to_tensor(tensor *out__, ivalue i) {
  PROTECT(
    out__[0] = new torch::Tensor(i->toTensor());
    return 0;
  )
  return 1;
}

int ati_length(int *out__, ivalue i) {
  PROTECT(
    if (i->isTuple()) {
      out__[0] = i->toTuple()->elements().size();
      return 0;
    }
    else if (i->isIntList()) {
      out__[0] = i->toIntList().size();
      return 0;
    }
    else if (i->isDoubleList()) {
      out__[0] = i->toDoubleList().size();
      return 0;
    }
    else if (i->isBoolList()) {
      out__[0] = i->toBoolList().size();
      return 0;
    }
    else if (i->isString()) {
      out__[0] = i->toStringRef().size();
      return 0;
    }
    else if (i->isTensorList()) {
      out__[0] = i->toTensorList().size();
      return 0;
    }
    else if (i->isList()) {
      out__[0] = i->toList().size();
      return 0;
    }
    else if (i->isGenericDict()) {
      out__[0] = i->toGenericDict().size();
      return 0;
    }
    caml_invalid_argument("unsupported tag for this length");
    return 1;
  )
  return 1;
}

int ati_tuple_length(int *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toTuple()->elements().size();
    return 0;
  )
  return 1;
}

int ati_to_tuple(ivalue i,
                  ivalue *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toTuple()->elements();
    if (vec.size() != noutputs) {
      myerr = strdup("unexpected tuple size");
      return 1;
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
    return 0;
  )
  return 1;
}

int ati_to_generic_list(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto vec = i->toList();
    if (vec.size() != noutputs) {
      caml_invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
    return 0;
  )
  return 1;
}

int ati_to_generic_dict(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto dict = i->toGenericDict();
    if (dict.size() != noutputs) {
      caml_invalid_argument("unexpected dict size");
    }
    int k = 0;
    for (auto it = dict.begin(); it != dict.end(); ++it) {
      outputs[k++] = new torch::jit::IValue(it->key());
      outputs[k++] = new torch::jit::IValue(it->value());
    }
    return 0;
  )
  return 1;
}

int ati_to_int_list(ivalue i,
                  int64_t *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toIntList();
    if (vec.size() != noutputs) {
      caml_invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_double_list(ivalue i,
                        double *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toDoubleList();
    if (vec.size() != noutputs) {
      caml_invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_bool_list(ivalue i,
                      char *outputs,
                      int noutputs) {
  PROTECT(
    auto vec = i->toBoolList();
    if (vec.size() != noutputs) {
      caml_invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_tensor_list(ivalue i,
                        tensor *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toTensorList();
    if (vec.size() != noutputs) {
      caml_invalid_argument("unexpected tuple size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::Tensor(vec[i]);
    return 0;
  )
  return 1;
}

int ati_free(ivalue i) {
  PROTECT(
    delete(i);
    return 0;
  )
  return 1;
}

#include "torch_api_generated.cpp.h"
