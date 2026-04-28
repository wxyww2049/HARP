#include <cmath>

#include <torch/python.h>
#include <ATen/ops/empty_like.h>

#include <arm_sve.h>
#include <async_op_math.h>


void _sve_silu_bfloat16(__bf16* input, __bf16* output, size_t N) {
    svfloat32_t one = svdup_f32(1.f);

#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x_lo, x_hi;
        uint16_t* input_u16 = reinterpret_cast<uint16_t*>(input);
        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output);

        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input_u16, i, x_lo, x_hi);

        svfloat32_t x_lo_ori = x_lo;
        svfloat32_t x_hi_ori = x_hi;

        // lo
        svfloat32_t neg_x_lo = svneg_f32_x(pg_lo, x_lo_ori);
        svfloat32_t exp_neg_x_lo = svexp_f32_z(pg_lo, neg_x_lo);
        svfloat32_t denom_lo = svadd_f32_x(pg_lo, one, exp_neg_x_lo);
        svfloat32_t result_lo = svdiv_f32_x(pg_lo, x_lo_ori, denom_lo);

        // hi
        svfloat32_t neg_x_hi = svneg_f32_x(pg_hi, x_hi_ori);
        svfloat32_t exp_neg_x_hi = svexp_f32_z(pg_hi, neg_x_hi);
        svfloat32_t denom_hi = svadd_f32_x(pg_hi, one, exp_neg_x_hi);
        svfloat32_t result_hi = svdiv_f32_x(pg_hi, x_hi_ori, denom_hi);

        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, output_u16, i, result_lo, result_hi);
    }
}


void _sve_silu_backward_bfloat16(__bf16* grad_output, __bf16* input, __bf16* grad_input, size_t N) {

    svfloat32_t one = svdup_f32(1.f);
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x_lo, x_hi;
        uint16_t* input_u16 = reinterpret_cast<uint16_t*>(input);
        uint16_t* grad_input_u16 = reinterpret_cast<uint16_t*>(grad_input);
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input_u16, i, x_lo, x_hi);

        svfloat32_t grad_y_lo, grad_y_hi;
        uint16_t* grad_output_u16 = reinterpret_cast<uint16_t*>(grad_output);
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, grad_output_u16, i, grad_y_lo, grad_y_hi);

        svfloat32_t neg_x_lo = svneg_f32_x(pg_lo, x_lo);
        svfloat32_t exp_neg_x_lo = svexp_f32_z(pg_lo, neg_x_lo);
        svfloat32_t denom_lo = svadd_f32_x(pg_lo, one, exp_neg_x_lo);
        svfloat32_t sigmoid_lo = svdiv_f32_x(pg_lo, one, denom_lo);

        svfloat32_t neg_x_hi = svneg_f32_x(pg_hi, x_hi);
        svfloat32_t exp_neg_x_hi = svexp_f32_z(pg_hi, neg_x_hi);
        svfloat32_t denom_hi = svadd_f32_x(pg_hi, one, exp_neg_x_hi);
        svfloat32_t sigmoid_hi = svdiv_f32_x(pg_hi, one, denom_hi);
        svfloat32_t grad_lo = svmul_f32_z(pg_lo, sigmoid_lo,
                                       svadd_f32_z(pg_lo, one,
                                       svmul_f32_z(pg_lo, x_lo, svsub_f32_z(pg_lo, one, sigmoid_lo))));
        svfloat32_t result_lo = svmul_f32_x(pg_lo, grad_y_lo, grad_lo);

        svfloat32_t grad_hi = svmul_f32_z(pg_hi, sigmoid_hi,
                                       svadd_f32_z(pg_hi, one,
                                       svmul_f32_z(pg_hi, x_hi, svsub_f32_z(pg_hi, one, sigmoid_hi))));
        svfloat32_t result_hi = svmul_f32_x(pg_hi, grad_y_hi, grad_hi);

        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, grad_input_u16, i, result_lo, result_hi);
    }
}


void silu_out(const torch::Tensor & input, torch::Tensor & output) {
    __bf16* input_ptr = reinterpret_cast<__bf16*>(input.data_ptr());
    __bf16* output_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    _sve_silu_bfloat16(input_ptr, output_ptr, input.numel());
    return ;
}

void silu_backward_out(const torch::Tensor & grad_output,
                                const torch::Tensor & input,
                                torch::Tensor & grad_input) {
    __bf16* grad_output_ptr = reinterpret_cast<__bf16*>(grad_output.data_ptr());
    __bf16* input_ptr = reinterpret_cast<__bf16*>(input.data_ptr());
    __bf16* grad_input_ptr = reinterpret_cast<__bf16*>(grad_input.data_ptr());
    _sve_silu_backward_bfloat16(grad_output_ptr, input_ptr, grad_input_ptr, input.numel());
    return ;
}
