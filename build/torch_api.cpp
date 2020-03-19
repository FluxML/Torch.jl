#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<torch/script.h>
#include<c10/cuda/CUDACachingAllocator.h>
#include<c10/cuda/CUDAStream.h>
#include<vector>
#include "torch_api.h"

#define caml_invalid_argument printf
using namespace std;

char* get_last_error() {
  return myerr;
}

void flush_error() {
  myerr = "";
}

void at_manual_seed(int64_t seed) {
  torch::manual_seed(seed);
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

void at_from_blob(tensor *out__, void *data, int64_t *dims, int ndims, int64_t *strides, int nstrides, int dev) {
  PROTECT(
    auto options = torch::TensorOptions().device(torch::kCUDA, dev).requires_grad(false);
    torch::Tensor tens = torch::from_blob(data, torch::IntArrayRef(dims, ndims), torch::IntArrayRef(strides, nstrides), options);
    out__[1] = new torch::Tensor(tens);
  )
  // return nullptr;
}

void at_new_tensor(tensor *out__) {
  PROTECT(
    out__[1] = new torch::Tensor();
  )
  // return nullptr;
}

void at_empty_cache() {
  PROTECT(
    c10::cuda::CUDACachingAllocator::emptyCache();
  )
}

void at_no_grad(int flag) {
  torch::GradMode::set_enabled((bool)flag);
  // return flag;
}

void at_sync() {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  // torch::cuda::synchronize();
}

int at_tensor_of_data(tensor *out__, void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    // auto options = torch::TensorOptions().dtype(torch::ScalarType(type)).requires_grad(false);
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if (element_size_in_bytes != tensor.element_size())
      // jl_error("incoherent element sizes in bytes");
      return -1;
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    out__[0] = new torch::Tensor(tensor);
  )
  return 0;
}

int at_copy_data(tensor tensor, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
    if (elt_size_in_bytes != tensor->element_size())
      // jl_error("incoherent element sizes in bytes");
      return -1;
    if (numel != tensor->numel())
      // jl_error("incoherent number of elements");
      return -1;
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU);
      void *tensor_data = tmp_tensor.contiguous().data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      void *tensor_data = tensor->contiguous().data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
  )
  return 0;
}

void at_float_vec(tensor *out__, double *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    out__[0] = new torch::Tensor(tensor);
  )
  // return nullptr;
}

void at_int_vec(tensor *out__, int64_t *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    out__[0] = new torch::Tensor(tensor);
  )
  // return nullptr;
}

void at_defined(int *i, tensor t) {
  PROTECT(i[0] = t->defined();)
  // return -1;
}

void at_dim(int *i, tensor t) {
  PROTECT(i[0] = t->dim();)
  // return -1;
}

void at_shape(tensor t, int *dims) {
  PROTECT(
    int i = 0;
    for (int dim : t->sizes()) dims[i++] = dim;
  )
}

void at_scalar_type(int *i, tensor t) {
  PROTECT(
    i[0] = static_cast<int>(t->scalar_type());
  )
}

void at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(t->backward({}, keep_graph, create_graph);)
}

void at_requires_grad(int *i, tensor t) {
  PROTECT(i[0] = t->requires_grad();)
  // return -1;
}

void at_grad_set_enabled(int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    // return is_enabled;
  )
  // return -1;
}

void at_get(tensor *out__, tensor t, int index) {
  PROTECT(out__[0] = new torch::Tensor((*t)[index]);)
  // return nullptr;
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

void at_double_value_at_indexes(double *i, tensor t, int *indexes, int indexes_len) {
   i[0] = at_value_at_indexes<double>(t, indexes, indexes_len);
}

void at_int64_value_at_indexes(int64_t *i, tensor t, int *indexes, int indexes_len) {
  i[0] = at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template<typename T>
void at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    tensor.fill_(v);
  )
}

void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len, double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len, int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(tensor t, double v) {
  PROTECT(t->fill_(v);)
}

void at_fill_int64(tensor t, int64_t v) {
  PROTECT(t->fill_(v);)
}

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    return strdup(oss.str().c_str());
  )
  return nullptr;
}

void at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
  )
}

void at_save(tensor t, char *filename) {
  PROTECT(torch::save(*t, filename);)
}

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
  )
}

void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
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
  )
}

void at_load_callback(char *filename, void (*f)(char *, tensor)) {
  PROTECT(
    auto module = torch::jit::load(filename);
    for (const auto &p : module.named_parameters()) {
      auto v = p.value;
      f((char*)p.name.c_str(), new torch::Tensor(v));
    }
  )
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
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
  )
}

void at_load(char *filename, tensor *out__) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    out__[0] = new torch::Tensor(tensor);
  )
  // return nullptr;
}

void at_free(tensor t) {
  delete(t);
}

void at_run_backward(tensor *tensors,
                     int ntensors,
                     tensor *inputs,
                     int ninputs,
                     tensor *outputs,
                     int keep_graph,
                     int create_graph) {
  PROTECT(
    torch::autograd::Engine engine;
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::impl::gradient_edge(torch::autograd::as_variable_ref(*tensors[i])));

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i) {
      if (!inputs[i]->requires_grad())
        caml_invalid_argument("one of the input tensor does not use set_requires_grad");
      inputs_.push_back(torch::autograd::impl::gradient_edge(torch::autograd::as_variable_ref(*inputs[i])));
    }

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
  )
}

void ato_adam(optimizer *out__, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .beta1(beta1)
        .beta2(beta2)
        .weight_decay(weight_decay);
    out__[0] = new torch::optim::Adam(vector<torch::Tensor>(), options);
  )
  // return nullptr;
}

void ato_rmsprop(optimizer *out__, double learning_rate,
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
    )
  // return nullptr;
}

void ato_sgd(optimizer *out__, double learning_rate,
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
  )
  // return nullptr;
}

void ato_add_parameters(optimizer t, tensor *tensors, int ntensors) {
  PROTECT(
    t->add_parameters(of_carray_tensor(tensors, ntensors));
  )
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.learning_rate(learning_rate);
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.learning_rate(learning_rate);
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.learning_rate(learning_rate);
    else
     caml_invalid_argument("unexpected optimizer");
  )
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.beta1(momentum);
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.momentum(momentum);
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.momentum(momentum);
    else
     caml_invalid_argument("unexpected optimizer");
  )
}

void ato_zero_grad(optimizer t) {
  PROTECT(t->zero_grad();)
}

void ato_step(optimizer t) {
  PROTECT(t->step();)
}

void ato_free(optimizer t) {
  delete(t);
}

void ats_int(scalar *out__, int64_t v) {
  PROTECT(out__[0] = new torch::Scalar(v);)
  // return nullptr;
}

void ats_float(scalar *out__, double v) {
  PROTECT(out__[0] = new torch::Scalar(v);)
  // return nullptr;
}

void ats_free(scalar s) {
  delete(s);
}

void atc_cuda_device_count(int *i) {
  PROTECT(i[0] = torch::cuda::device_count();)
  // return -1;
}

void atc_cuda_is_available(int *i) {
  PROTECT(i[0] = torch::cuda::is_available();)
  // return -1;
}

void atc_cudnn_is_available(int *i) {
  PROTECT(i[0] = torch::cuda::cudnn_is_available();)
  // return -1;
}

void atc_set_benchmark_cudnn(int b) {
  at::globalContext().setBenchmarkCuDNN(b);
}

void atm_load(char *filename, module *out__) {
  PROTECT(
    out__[0] = new torch::jit::script::Module(torch::jit::load(filename));
  )
  // return nullptr;
}

int atm_forward(tensor *out__, module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->forward(inputs);
    if (!output.isTensor())
      // jl_error("forward did not return a tensor");
      return -1;
    out__[0] = new torch::Tensor(output.toTensor());
  )
  // return nullptr;
  return 0;
}

void atm_forward_(ivalue *out__, module m,
                    ivalue *ivalues,
                    int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = m->forward(inputs);
    out__[0] = new torch::jit::IValue(output);
  )
  // return nullptr;
}

void atm_free(module m) {
  delete(m);
}

void ati_tensor(ivalue *out__, tensor t) {
  PROTECT(
    out__[0] = new torch::jit::IValue(*t);
  )
  // return nullptr;
}

void ati_int(ivalue *out__, int64_t i) {
  PROTECT(
    out__[0] = new torch::jit::IValue(i);
  )
  // return nullptr;
}

void ati_double(ivalue *out__, double d) {
  PROTECT(
    out__[0] = new torch::jit::IValue(d);
  )
  // return nullptr;
}

void ati_tuple(ivalue *out__, ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(torch::ivalue::Tuple::create(vec));
  )
  // return nullptr;
}

void ati_tag(int *out__, ivalue i) {
  PROTECT(
    if (i->isTensor()) out__[0] = 0;
    else if (i->isInt()) out__[0] = 1;
    else if (i->isDouble()) out__[0] = 2;
    else if (i->isTuple()) out__[0] = 3;
    // jl_error(("unsupported tag" + i->tagKind()).c_str());
    // return -1;
  )
  // return -1;
}

void ati_to_int(int64_t *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toInt();
  )
  // return -1;
}

void ati_to_double(double *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toDouble();
  )
  // return 0.;
}

void ati_to_tensor(tensor *out__, ivalue i) {
  PROTECT(
    out__[0] = new torch::Tensor(i->toTensor());
  )
  // return nullptr;
}

void ati_tuple_length(int *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toTuple()->elements().size();
  )
  // return -1;
}

int ati_to_tuple(ivalue i,
                  ivalue *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toTuple()->elements();
    if (vec.size() != noutputs) {
      // jl_error("unexpected tuple size");
      return -1;
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
  )
  return 0;
}


void ati_free(ivalue i) {
  delete(i);
}

at::Device device_of_int(int d) {
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

#include "torch_api_generated.cpp.h"
