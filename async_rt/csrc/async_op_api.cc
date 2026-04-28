#include <optional>

#include <torch/extension.h>
#include <c10/util/ArrayRef.h>

void silu_out(const torch::Tensor & input,
              torch::Tensor & output);

void silu_backward_out(const torch::Tensor & grad_output,
                       const torch::Tensor & input,
                       torch::Tensor & grad_input);

void rmsnorm_forward_out(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& rms,
    double eps
);

void rmsnorm_backward_out(
    const torch::Tensor& input,
    const torch::Tensor& grad_output,
    const torch::Tensor& weight,
    const torch::Tensor& rms,
    torch::Tensor& grad_input,
    torch::Tensor& grad_weight
);

void add_out(const torch::Tensor & input1, const torch::Tensor & input2, torch::Tensor & output);

void mul_out(const torch::Tensor & input1, const torch::Tensor & input2, torch::Tensor & output);

void tanh_out(const torch::Tensor & input, torch::Tensor & output);

void tanh_backward_out(const torch::Tensor & grad_output, const torch::Tensor & output, torch::Tensor & grad_input);

void rope_forward_impl(const torch::Tensor & x, const torch::Tensor & freqs_cis, torch::Tensor & output);

void rope_backward_impl(const torch::Tensor & grad_output, const torch::Tensor & freqs_cis, torch::Tensor & grad_x);

void random_uniform(torch::Tensor & data, double low, double high);

void add_scalar_out(const torch::Tensor & input, torch::Tensor & output, double scalar);

TORCH_LIBRARY(async_op, m) {
    m.def("silu_out(Tensor input, Tensor(a!) output) -> ()");
    m.def("silu_backward_out(Tensor grad_output, Tensor input, Tensor(a!) grad_input) -> ()");
    m.def("add_out(Tensor input1, Tensor input2, Tensor(a!) output) -> ()");
    m.def("add_scalar_out(Tensor input, Tensor(a!) output, float scalar) -> ()");
    m.def("mul_out(Tensor input1, Tensor input2, Tensor(a!) output) -> ()");
    m.def("tanh_out(Tensor input, Tensor(a!) output) -> ()");
    m.def("tanh_backward_out(Tensor grad_output, Tensor output, Tensor(a!) grad_input) -> ()");
    m.def("rmsnorm_forward_out(Tensor input, Tensor weight, Tensor(a!) output, Tensor(b!) rms, float eps) -> ()");
    m.def("rmsnorm_backward_out(Tensor input, Tensor grad_output, Tensor weight, Tensor rms, Tensor(a!) grad_input, Tensor(b!) grad_weight) -> ()");
    m.def("rope_forward_impl(Tensor x, Tensor freqs_cis, Tensor(a!) output) -> ()");
    m.def("rope_backward_impl(Tensor grad_output, Tensor freqs_cis, Tensor(b!) grad_x) -> ()");
    m.def("random_uniform(Tensor data, float low, float high) -> ()");
}

TORCH_LIBRARY_IMPL(async_op, CPU, m) {
    m.impl("silu_out", &silu_out);
    m.impl("silu_backward_out", &silu_backward_out);
    m.impl("add_out", &add_out);
    m.impl("add_scalar_out", &add_scalar_out);
    m.impl("mul_out", &mul_out);
    m.impl("tanh_out", &tanh_out);
    m.impl("tanh_backward_out", &tanh_backward_out);
    m.impl("rmsnorm_forward_out", &rmsnorm_forward_out);
    m.impl("rmsnorm_backward_out", &rmsnorm_backward_out);
    m.impl("rope_forward_impl", &rope_forward_impl);
    m.impl("rope_backward_impl", &rope_backward_impl);
    m.impl("random_uniform", &random_uniform);
}