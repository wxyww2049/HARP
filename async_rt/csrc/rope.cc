#include <torch/torch.h>
#include <arm_sve.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <async_op_math.h>

static inline void rope_row(
    const uint16_t* x_row,
    const float* freq_row,
    uint16_t* out_row,
    std::size_t D,
    std::size_t svLen,
    std::size_t svLen2
) {
    for (std::size_t i = 0; i < D; i += svLen2) {
        svbool_t pred16 = svwhilelt_b16(i, D);
        svbool_t pred_lo = svwhilelt_b32(i, D);
        std::size_t hi_start = i + svLen;
        svbool_t pred_hi = svwhilelt_b32(hi_start, D);

        if (i + svLen2 < D) {
            svprfh(pred16, x_row + i + svLen2, SV_PLDL1KEEP);
            svprfw(svwhilelt_b32(i + svLen2, D), freq_row + i + svLen2, SV_PLDL1KEEP);
        }

        svfloat32_t x_lo, x_hi;
        bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, x_row, i, x_lo, x_hi);

        svfloat32_t f_lo = svld1_f32(pred_lo, freq_row + i);
        svfloat32_t f_hi = svld1_f32(pred_hi, freq_row + hi_start);

        svfloat32_t zero = svdup_n_f32(0.0f);
        svfloat32_t acc_lo = svcmla_f32_m(pred_lo, zero, x_lo, f_lo, 0);
        acc_lo = svcmla_f32_m(pred_lo, acc_lo, x_lo, f_lo, 90);

        svfloat32_t acc_hi = svcmla_f32_m(pred_hi, zero, x_hi, f_hi, 0);
        acc_hi = svcmla_f32_m(pred_hi, acc_hi, x_hi, f_hi, 90);

        fp32x2_store_as_bf16(pred16, pred_lo, pred_hi, out_row, i, acc_lo, acc_hi);
    }
}


void ropeForwardKernelBF16(
    const uint16_t* x,
    const float* freqs,
    uint16_t* output,
    std::size_t B,
    std::size_t H,
    std::size_t S,
    std::size_t D,
    std::ptrdiff_t x_stride_b,
    std::ptrdiff_t x_stride_h,
    std::ptrdiff_t x_stride_s,
    std::ptrdiff_t out_stride_b,
    std::ptrdiff_t out_stride_h,
    std::ptrdiff_t out_stride_s,
    std::ptrdiff_t f_stride_b,
    std::ptrdiff_t f_stride_s
) {
    std::size_t BH = B * H;

    at::parallel_for(0, (int64_t)BH, 1, [&](int64_t bh_start, int64_t bh_end) {
        std::size_t svLen = svcntw();
        std::size_t svLen2 = svLen * 2;

        for (int64_t bh = bh_start; bh < bh_end; ++bh) {
            std::size_t b = bh / H;
            std::size_t h = bh % H;

            const uint16_t* x_bh = x + b * x_stride_b + h * x_stride_h;
            uint16_t* out_bh = output + b * out_stride_b + h * out_stride_h;

            const float* freq_b = freqs + b * f_stride_b;

            for (std::size_t s = 0; s < S; ++s) {
                const uint16_t* x_row = x_bh + s * x_stride_s;
                uint16_t* out_row = out_bh + s * out_stride_s;
                const float* freq_row = freq_b + s * f_stride_s;

                rope_row(x_row, freq_row, out_row, D, svLen, svLen2);
            }
        }
    });
}

static inline void rope_backward_row(
    const uint16_t* grad_row,
    const float* freq_row,
    uint16_t* out_row,
    std::size_t D,
    std::size_t svLen,
    std::size_t svLen2 )
{
    for (std::size_t i = 0; i < D; i += svLen2) {
        svbool_t pred16 = svwhilelt_b16(i, D);
        svbool_t pred_lo = svwhilelt_b32(i, D);
        std::size_t hi_start = i + svLen;
        svbool_t pred_hi = svwhilelt_b32(hi_start, D);

        if (i + svLen2 < D) {
            svprfh(pred16, grad_row + i + svLen2, SV_PLDL1KEEP);
            svprfw(svwhilelt_b32(i + svLen2, D), freq_row + i + svLen2, SV_PLDL1KEEP);
        }

        svfloat32_t g_lo, g_hi;
        bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, grad_row, i, g_lo, g_hi);

        svfloat32_t f_lo = svld1_f32(pred_lo, freq_row + i);
        svfloat32_t f_hi = svld1_f32(pred_hi, freq_row + hi_start);

        svfloat32_t gr_lo = svuzp1_f32(g_lo, g_lo);
        svfloat32_t gi_lo = svuzp2_f32(g_lo, g_lo);
        svfloat32_t fr_lo = svuzp1_f32(f_lo, f_lo);
        svfloat32_t fi_lo = svuzp2_f32(f_lo, f_lo);

        svfloat32_t gr_hi = svuzp1_f32(g_hi, g_hi);
        svfloat32_t gi_hi = svuzp2_f32(g_hi, g_hi);
        svfloat32_t fr_hi = svuzp1_f32(f_hi, f_hi);
        svfloat32_t fi_hi = svuzp2_f32(f_hi, f_hi);

        svfloat32_t out_r_lo = svmul_f32_m(pred_lo, gr_lo, fr_lo);
        out_r_lo = svmla_f32_m(pred_lo, out_r_lo, gi_lo, fi_lo);

        svfloat32_t out_i_lo = svmul_f32_m(pred_lo, gi_lo, fr_lo);
        out_i_lo = svmls_f32_m(pred_lo, out_i_lo, gr_lo, fi_lo);

        svfloat32_t out_r_hi = svmul_f32_m(pred_hi, gr_hi, fr_hi);
        out_r_hi = svmla_f32_m(pred_hi, out_r_hi, gi_hi, fi_hi);

        svfloat32_t out_i_hi = svmul_f32_m(pred_hi, gi_hi, fr_hi);
        out_i_hi = svmls_f32_m(pred_hi, out_i_hi, gr_hi, fi_hi);

        svfloat32_t out_lo = svzip1_f32(out_r_lo, out_i_lo);
        svfloat32_t out_hi = svzip1_f32(out_r_hi, out_i_hi);

        // fp32 -> bf16
        fp32x2_store_as_bf16(pred16, pred_lo, pred_hi, out_row, i, out_lo, out_hi);
    }
}

void ropeBackwardKernelBF16(
    const uint16_t* grad_output,
    const float* freqs,
    uint16_t* grad_x,
    std::size_t B,
    std::size_t H,
    std::size_t S,
    std::size_t D,
    std::ptrdiff_t go_stride_b,
    std::ptrdiff_t go_stride_h,
    std::ptrdiff_t go_stride_s,
    std::ptrdiff_t gx_stride_b,
    std::ptrdiff_t gx_stride_h,
    std::ptrdiff_t gx_stride_s,
    std::ptrdiff_t f_stride_b,
    std::ptrdiff_t f_stride_s
) {
    std::size_t BH = B * H;

    at::parallel_for(0, (int64_t)BH, 1, [&](int64_t bh_start, int64_t bh_end) {
        std::size_t svLen = svcntw();
        std::size_t svLen2 = svLen * 2;

        for (int64_t bh = bh_start; bh < bh_end; ++bh) {
            std::size_t b = bh / H;
            std::size_t h = bh % H;

            const uint16_t* go_bh = grad_output + b * go_stride_b + h * go_stride_h;
            uint16_t* gx_bh = grad_x + b * gx_stride_b + h * gx_stride_h;
            const float* freq_b = freqs + b * f_stride_b;

            for (std::size_t s = 0; s < S; ++s) {
                const uint16_t* go_row = go_bh + s * go_stride_s;
                uint16_t* gx_row = gx_bh + s * gx_stride_s;
                const float* freq_row = freq_b + s * f_stride_s;

                rope_backward_row(go_row, freq_row, gx_row, D, svLen, svLen2);
            }
        }
    });
}

void rope_forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cis,
    torch::Tensor& output
) {

    auto B = (std::size_t)x.size(0);
    auto H = (std::size_t)x.size(1);
    auto S = (std::size_t)x.size(2);
    auto D = (std::size_t)x.size(3);

    auto freqs_real = torch::view_as_real(freqs_cis).flatten(-2).contiguous();

    int f_ndim = freqs_real.dim();

    std::ptrdiff_t f_stride_b = (f_ndim == 3) ? freqs_real.stride(0) : 0;
    std::ptrdiff_t f_stride_s = freqs_real.stride(-2);

    auto* x_ptr = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
    auto* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>());
    auto* freq_ptr = freqs_real.data_ptr<float>();

    ropeForwardKernelBF16(
        x_ptr, freq_ptr, out_ptr,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        f_stride_b, f_stride_s
    );

    return ;
}

void rope_backward_impl(
    const torch::Tensor& grad_output,
    const torch::Tensor& freqs_cis,
    torch::Tensor& grad_x
) {

    auto B = (std::size_t)grad_output.size(0);
    auto H = (std::size_t)grad_output.size(1);
    auto S = (std::size_t)grad_output.size(2);
    auto D = (std::size_t)grad_output.size(3);

    auto freqs_real = torch::view_as_real(freqs_cis).flatten(-2).contiguous();

    int f_ndim = freqs_real.dim();

    std::ptrdiff_t f_stride_b = (f_ndim == 3) ? freqs_real.stride(0) : 0;
    std::ptrdiff_t f_stride_s = freqs_real.stride(-2);

    auto* go_ptr = reinterpret_cast<const uint16_t*>(grad_output.data_ptr<at::BFloat16>());
    auto* freq_ptr = freqs_real.data_ptr<float>();
    auto* gx_ptr = reinterpret_cast<uint16_t*>(grad_x.data_ptr<at::BFloat16>());

    ropeBackwardKernelBF16(
        go_ptr,
        freq_ptr,
        gx_ptr,
        B, H, S, D,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
        f_stride_b, f_stride_s
    );
}