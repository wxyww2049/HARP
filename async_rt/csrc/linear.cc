#include <ATen/ATen.h>
#include <torch/library.h>
#include "bgemm.h"

namespace {

void matmul_out(at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
                bool transpose_a, bool transpose_b) {
    const auto left = transpose_a ? a.t() : a;
    const auto right = transpose_b ? b.t() : b;
    TORCH_CHECK(left.dim() == 2 && right.dim() == 2 && out.dim() == 2 &&
                left.size(1) == right.size(0) &&
                out.size(0) == left.size(0) && out.size(1) == right.size(1),
                "invalid linear matrix dimensions");

    const int ta = transpose_a ? XhblasTrans : XhblasNoTrans;
    const int tb = transpose_b ? XhblasTrans : XhblasNoTrans;
    const auto m = left.size(0), n = right.size(1), k = left.size(1);
    if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
        a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
        out.scalar_type() == at::kBFloat16 && m <= INT_MAX && n <= INT_MAX && k <= INT_MAX &&
        bgemm_supports(m, n, k, ta, tb)) {
        const int status = bgemm_kernel(XhblasRowMajor, ta, tb, m, n, k, 1,
            reinterpret_cast<const __bf16*>(a.data_ptr<at::BFloat16>()), a.size(1),
            reinterpret_cast<const __bf16*>(b.data_ptr<at::BFloat16>()), b.size(1), 0,
            reinterpret_cast<__bf16*>(out.data_ptr<at::BFloat16>()), n, nullptr, 0);
        TORCH_CHECK(status == 0, xhblas_gemm_status_message(status));
    } else {
        at::mm_out(out, left, right);
    }
}

void linear_forward(const at::Tensor& input, const at::Tensor& weight,
                    c10::optional<at::Tensor> bias, at::Tensor& out) {
    matmul_out(out, input, weight, false, true);
    if (bias) out.add_(*bias);
}

void linear_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
                     at::Tensor& grad_input, at::Tensor& grad_weight, c10::optional<at::Tensor> grad_bias) {
    matmul_out(grad_input, grad, weight, false, false);
    matmul_out(grad_weight, grad, input, true, false);
    if (grad_bias) grad_bias->copy_(grad.sum(0));
}

} // namespace

TORCH_LIBRARY(xhpops, m) {
    m.def("kp_linear_forward(Tensor input, Tensor weight, Tensor? bias, Tensor(a!) out) -> ()");
    m.def("kp_linear_backward(Tensor grad, Tensor input, Tensor weight, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!)? grad_bias) -> ()");
}

TORCH_LIBRARY_IMPL(xhpops, CPU, m) {
    m.impl("kp_linear_forward", &linear_forward);
    m.impl("kp_linear_backward", &linear_backward);
}
