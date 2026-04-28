#include <ATen/ATen.h>
// #include <Python.h>
#include <c10/util/BFloat16.h>
#include <torch/library.h>
#include <tuple>

void ring_attention_forward(__bf16* Q, __bf16* K, __bf16* V, float* out,
                            int t_num, int h_num, int h_len, int sp_size,
                            __bf16* m_i, float* l_i, float scale);

void ring_attention_backward(__bf16* Q, __bf16* grad_Q,
                             __bf16* K, __bf16* dK,
                             __bf16* V, __bf16* dV, __bf16* grad_out,
                             int t_num, int h_num, int h_len, int sp_size,
                             float* m_i, float* l_i, float* D, float scale);

namespace {

// class ConditionalGILRelease {
// public:
//     ConditionalGILRelease() {
//         if (Py_IsInitialized() && PyGILState_Check()) {
//             state_ = PyEval_SaveThread();
//         }
//     }

//     ~ConditionalGILRelease() {
//         if (state_ != nullptr) {
//             PyEval_RestoreThread(state_);
//         }
//     }

//     ConditionalGILRelease(const ConditionalGILRelease&) = delete;
//     ConditionalGILRelease& operator=(const ConditionalGILRelease&) = delete;

// private:
//     PyThreadState* state_ = nullptr;
// };

void check_token_major_view(const at::Tensor& x, const char* name) {
    TORCH_CHECK(
        x.dim() == 4,
        name,
        " must be a [B, H, T, D] tensor");

    const int64_t h_num = x.size(1);
    const int64_t t_num = x.size(2);
    const int64_t h_len = x.size(3);
    const int64_t expected_stride_0 = h_num * t_num * h_len;
    const int64_t expected_stride_1 = h_len;
    const int64_t expected_stride_2 = h_num * h_len;

    TORCH_CHECK(
        x.stride(0) == expected_stride_0 && x.stride(1) == expected_stride_1 &&
            x.stride(2) == expected_stride_2 && x.stride(3) == 1,
        name,
        " must be a [B, H, T, D] view backed by contiguous [B, T, H, D] "
        "storage (equivalently [B, T, H * D] physical layout); got strides [",
        x.stride(0),
        ", ",
        x.stride(1),
        ", ",
        x.stride(2),
        ", ",
        x.stride(3),
        "]");
}

void check_forward_tensors(const at::Tensor& q,
                           const at::Tensor& k,
                           const at::Tensor& v,
                           const at::Tensor& out,
                           const at::Tensor& m_i,
                           const at::Tensor& l_i) {
    TORCH_CHECK(q.device().is_cpu(), "q must be on CPU");
    TORCH_CHECK(k.device().is_cpu(), "k must be on CPU");
    TORCH_CHECK(v.device().is_cpu(), "v must be on CPU");
    TORCH_CHECK(out.device().is_cpu(), "out must be on CPU");
    TORCH_CHECK(m_i.device().is_cpu(), "m_i must be on CPU");
    TORCH_CHECK(l_i.device().is_cpu(), "l_i must be on CPU");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(m_i.scalar_type() == at::kBFloat16, "m_i must be bfloat16");
    TORCH_CHECK(l_i.scalar_type() == at::kFloat, "l_i must be float32");

    TORCH_CHECK(q.dim() == 4, "q must be [B, H, T, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must have the same shape as q");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");

    TORCH_CHECK(m_i.dim() == 4, "m_i must be [B, H, T, 1]");
    TORCH_CHECK(l_i.dim() == 4, "l_i must be [B, H, T, 1]");
    TORCH_CHECK(m_i.size(0) == q.size(0) && m_i.size(1) == q.size(1) &&
                    m_i.size(2) == q.size(2) && m_i.size(3) == 1,
                "m_i must match q on [B, H, T] and have last dim 1");
    TORCH_CHECK(l_i.size(0) == q.size(0) && l_i.size(1) == q.size(1) &&
                    l_i.size(2) == q.size(2) && l_i.size(3) == 1,
                "l_i must match q on [B, H, T] and have last dim 1");

    check_token_major_view(q, "q");
    check_token_major_view(k, "k");
    check_token_major_view(v, "v");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_i.is_contiguous(), "m_i must be contiguous");
    TORCH_CHECK(l_i.is_contiguous(), "l_i must be contiguous");

    TORCH_CHECK(q.size(0) == 1,
                "current ring_attention_forward wrapper expects batch size 1");
}

void check_backward_tensors(const at::Tensor& q,
                            const at::Tensor& grad_q,
                            const at::Tensor& k,
                            const at::Tensor& dK,
                            const at::Tensor& v,
                            const at::Tensor& dV,
                            const at::Tensor& grad_out,
                            const at::Tensor& m_i,
                            const at::Tensor& l_i,
                            const at::Tensor& D) {
    TORCH_CHECK(q.device().is_cpu(), "q must be on CPU");
    TORCH_CHECK(grad_q.device().is_cpu(), "grad_q must be on CPU");
    TORCH_CHECK(k.device().is_cpu(), "k must be on CPU");
    TORCH_CHECK(dK.device().is_cpu(), "dK must be on CPU");
    TORCH_CHECK(v.device().is_cpu(), "v must be on CPU");
    TORCH_CHECK(dV.device().is_cpu(), "dV must be on CPU");
    TORCH_CHECK(grad_out.device().is_cpu(), "grad_out must be on CPU");
    TORCH_CHECK(m_i.device().is_cpu(), "m_i must be on CPU");
    TORCH_CHECK(l_i.device().is_cpu(), "l_i must be on CPU");
    TORCH_CHECK(D.device().is_cpu(), "D must be on CPU");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(grad_q.scalar_type() == at::kBFloat16, "grad_q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(dK.scalar_type() == at::kBFloat16, "dK must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(dV.scalar_type() == at::kBFloat16, "dV must be bfloat16");
    TORCH_CHECK(grad_out.scalar_type() == at::kBFloat16, "grad_out must be bfloat16");
    TORCH_CHECK(m_i.scalar_type() == at::kFloat, "m_i must be float32");
    TORCH_CHECK(l_i.scalar_type() == at::kFloat, "l_i must be float32");
    TORCH_CHECK(D.scalar_type() == at::kFloat, "D must be float32");

    TORCH_CHECK(grad_q.sizes() == q.sizes(), "grad_q must have the same shape as q");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q");
    TORCH_CHECK(dK.sizes() == q.sizes(), "dK must have the same shape as q");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must have the same shape as q");
    TORCH_CHECK(dV.sizes() == q.sizes(), "dV must have the same shape as q");
    TORCH_CHECK(grad_out.sizes() == q.sizes(),
                "grad_out must have the same logical shape as q");

    TORCH_CHECK(m_i.dim() == 4 && l_i.dim() == 4 && D.dim() == 4,
                "m_i, l_i, D must be [B, H, T, 1]");
    TORCH_CHECK(m_i.sizes() == l_i.sizes() && m_i.sizes() == D.sizes(),
                "m_i, l_i, D must have the same shape");
    TORCH_CHECK(m_i.size(0) == q.size(0) && m_i.size(1) == q.size(1) &&
                    m_i.size(2) == q.size(2) && m_i.size(3) == 1,
                "m_i, l_i, D must match q on [B, H, T] and have last dim 1");

    check_token_major_view(q, "q");
    check_token_major_view(grad_q, "grad_q");
    check_token_major_view(k, "k");
    check_token_major_view(dK, "dK");
    check_token_major_view(v, "v");
    check_token_major_view(dV, "dV");
    TORCH_CHECK(grad_out.is_contiguous(),
                "grad_out must be contiguous [B, H, T, D]");
    TORCH_CHECK(m_i.is_contiguous(), "m_i must be contiguous");
    TORCH_CHECK(l_i.is_contiguous(), "l_i must be contiguous");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");
    TORCH_CHECK(q.size(0) == 1,
                "current ring attention wrappers expect batch size 1");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ring_attention_forward_op(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor out,
    at::Tensor m_i,
    at::Tensor l_i,
    int64_t sp_size,
    double scale) {
    // check_forward_tensors(q, k, v, out, m_i, l_i);
    // ConditionalGILRelease release;
    const int h_num = static_cast<int>(q.size(1));
    const int t_num = static_cast<int>(q.size(2));
    const int h_len = static_cast<int>(q.size(3));
    // TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    at::Tensor m_i_bf16 =
        (m_i.scalar_type() == at::kBFloat16)
            ? m_i
            : at::empty_like(m_i, m_i.options().dtype(at::kBFloat16));

    ring_attention_forward(
        reinterpret_cast<__bf16*>(q.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(k.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(v.data_ptr<c10::BFloat16>()),
        out.data_ptr<float>(),
        t_num,
        h_num,
        h_len,
        static_cast<int>(sp_size),
        reinterpret_cast<__bf16*>(m_i_bf16.data_ptr<c10::BFloat16>()),
        l_i.data_ptr<float>(),
        static_cast<float>(scale));

    if (m_i.scalar_type() == at::kFloat) {
        m_i.copy_(m_i_bf16.to(at::kFloat));
    }

    return std::make_tuple(out, m_i, l_i);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ring_attention_backward_op(
    const at::Tensor& q,
    at::Tensor grad_q,
    const at::Tensor& k,
    at::Tensor dK,
    const at::Tensor& v,
    at::Tensor dV,
    const at::Tensor& grad_out,
    const at::Tensor& m_i,
    const at::Tensor& l_i,
    const at::Tensor& D,
    int64_t sp_size,
    double scale) {
    // check_backward_tensors(q, grad_q, k, dK, v, dV, grad_out, m_i, l_i, D);
    // ConditionalGILRelease release;
    const int h_num = static_cast<int>(q.size(1));
    const int t_num = static_cast<int>(q.size(2));
    const int h_len = static_cast<int>(q.size(3));
    at::Tensor m_i_float =
        (m_i.scalar_type() == at::kFloat) ? m_i : m_i.to(at::kFloat);

    ring_attention_backward(
        reinterpret_cast<__bf16*>(q.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(grad_q.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(k.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(dK.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(v.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(dV.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__bf16*>(grad_out.data_ptr<c10::BFloat16>()),
        t_num,
        h_num,
        h_len,
        static_cast<int>(sp_size),
        const_cast<float*>(m_i_float.data_ptr<float>()),
        const_cast<float*>(l_i.data_ptr<float>()),
        const_cast<float*>(D.data_ptr<float>()),
        static_cast<float>(scale));

    return std::make_tuple(grad_q, dK, dV);
}

}  // namespace

TORCH_LIBRARY(ring_attention_ops, m) {
    m.def("ring_attention_forward(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor(b!) m_i, Tensor(c!) l_i, int sp_size, float scale) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
    m.def("ring_attention_backward(Tensor q, Tensor(a!) grad_q, Tensor k, Tensor(b!) dK, Tensor v, Tensor(c!) dV, Tensor grad_out, Tensor m_i, Tensor l_i, Tensor D, int sp_size, float scale) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
}

TORCH_LIBRARY_IMPL(ring_attention_ops, CPU, m) {
    m.impl("ring_attention_forward", &ring_attention_forward_op);
    m.impl("ring_attention_backward", &ring_attention_backward_op);
}
