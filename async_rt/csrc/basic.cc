#include <cmath>

#include <torch/python.h>
#include <ATen/ops/empty_like.h>

#include <arm_sve.h>
#include <async_op_math.h>


void _sve_add_bfloat16(__bf16* input1, __bf16* input2, __bf16* output, size_t N) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x1_lo, x1_hi;
        svfloat32_t x2_lo, x2_hi;
        uint16_t* input1_u16 = reinterpret_cast<uint16_t*>(input1);
        uint16_t* input2_u16 = reinterpret_cast<uint16_t*>(input2);
        
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input1_u16, i, x1_lo, x1_hi);
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input2_u16, i, x2_lo, x2_hi);

        svfloat32_t o_lo = svadd_f32_x(pg_lo, x1_lo, x2_lo);
        svfloat32_t o_hi = svadd_f32_x(pg_hi, x1_hi, x2_hi);

        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output);
        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, output_u16, i, o_lo, o_hi);
    }
}

void _sve_add_one_bfloat16(__bf16* input, __bf16* output, size_t N, double scalar) {
    svfloat32_t scalar_f32 = svdup_f32(static_cast<float>(scalar));
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x1_lo, x1_hi;
        uint16_t* input_u16 = reinterpret_cast<uint16_t*>(input);
        
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input_u16, i, x1_lo, x1_hi);

        svfloat32_t o_lo = svadd_f32_x(pg_lo, x1_lo, scalar_f32);
        svfloat32_t o_hi = svadd_f32_x(pg_hi, x1_hi, scalar_f32);

        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output);
        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, output_u16, i, o_lo, o_hi);
    }
}

void _sve_mul_bfloat16(__bf16* input1, __bf16* input2, __bf16* output, size_t N) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x1_lo, x1_hi;
        svfloat32_t x2_lo, x2_hi;
        uint16_t* input1_u16 = reinterpret_cast<uint16_t*>(input1);
        uint16_t* input2_u16 = reinterpret_cast<uint16_t*>(input2);
        
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input1_u16, i, x1_lo, x1_hi);
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input2_u16, i, x2_lo, x2_hi);

        svfloat32_t o_lo = svmul_f32_x(pg_lo, x1_lo, x2_lo);
        svfloat32_t o_hi = svmul_f32_x(pg_hi, x1_hi, x2_hi);

        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output);
        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, output_u16, i, o_lo, o_hi);
    }
}

void _sve_tanh_bfloat16(__bf16* input, __bf16* output, size_t N) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        svfloat32_t x_lo, x_hi;
        uint16_t* input_u16 = reinterpret_cast<uint16_t*>(input);
        
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, input_u16, i, x_lo, x_hi);

        svfloat32_t o_lo = svtanh_f32_z(pg_lo, x_lo);
        svfloat32_t o_hi = svtanh_f32_z(pg_hi, x_hi);

        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output);
        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, output_u16, i, o_lo, o_hi);
    }
}

void _sve_tanh_backward_bfloat16(
    __bf16* grad_output,
    __bf16* output,
    __bf16* grad_input,
    size_t N) 
{
#pragma omp parallel for
    for (size_t i = 0; i < N; i += 2 * svcntw()) {
        svbool_t pg16 = svwhilelt_b16(i, N);
        svbool_t pg_lo = svwhilelt_b32(i, N);
        std::size_t hi_start = i + svcntw();
        svbool_t pg_hi = svwhilelt_b32(hi_start, N);

        uint16_t* go_u16 = reinterpret_cast<uint16_t*>(grad_output);
        uint16_t* out_u16 = reinterpret_cast<uint16_t*>(output);
        uint16_t* gi_u16 = reinterpret_cast<uint16_t*>(grad_input);

        svfloat32_t go_lo, go_hi;
        svfloat32_t out_lo, out_hi;

        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, go_u16, i, go_lo, go_hi);
        bf16_load_to_2xfp32(pg16, pg_lo, pg_hi, out_u16, i, out_lo, out_hi);

        // grad_input = grad_output * (1 - output * output)
        svfloat32_t one_lo = svdup_f32(1.0f);
        svfloat32_t one_hi = svdup_f32(1.0f);

        svfloat32_t out2_lo = svmul_f32_z(pg_lo, out_lo, out_lo);
        svfloat32_t out2_hi = svmul_f32_z(pg_hi, out_hi, out_hi);

        svfloat32_t deriv_lo = svsub_f32_z(pg_lo, one_lo, out2_lo);
        svfloat32_t deriv_hi = svsub_f32_z(pg_hi, one_hi, out2_hi);

        svfloat32_t gi_lo = svmul_f32_z(pg_lo, go_lo, deriv_lo);
        svfloat32_t gi_hi = svmul_f32_z(pg_hi, go_hi, deriv_hi);

        fp32x2_store_as_bf16(pg16, pg_lo, pg_hi, gi_u16, i, gi_lo, gi_hi);
    }
}

void add_out(const torch::Tensor & input1, const torch::Tensor & input2, torch::Tensor & output) {
    __bf16* input1_ptr = reinterpret_cast<__bf16*>(input1.data_ptr());
    __bf16* input2_ptr = reinterpret_cast<__bf16*>(input2.data_ptr());
    __bf16* output_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    _sve_add_bfloat16(input1_ptr, input2_ptr, output_ptr, input1.numel());
    return ;
}

void add_scalar_out(const torch::Tensor & input, torch::Tensor & output, double scalar) {
    __bf16* input_ptr = reinterpret_cast<__bf16*>(input.data_ptr());
    __bf16* output_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    _sve_add_one_bfloat16(input_ptr, output_ptr, input.numel(), scalar);
    return ;
}

void mul_out(const torch::Tensor & input1, const torch::Tensor & input2, torch::Tensor & output) {
    __bf16* input1_ptr = reinterpret_cast<__bf16*>(input1.data_ptr());
    __bf16* input2_ptr = reinterpret_cast<__bf16*>(input2.data_ptr());
    __bf16* output_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    _sve_mul_bfloat16(input1_ptr, input2_ptr, output_ptr, input1.numel());
    return ;
}

void tanh_out(const torch::Tensor & input, torch::Tensor & output) {
    __bf16* input_ptr = reinterpret_cast<__bf16*>(input.data_ptr());
    __bf16* output_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    _sve_tanh_bfloat16(input_ptr, output_ptr, input.numel());
    return ;
}

void tanh_backward_out(const torch::Tensor & grad_output, const torch::Tensor & output, torch::Tensor & grad_input) {
    __bf16* go_ptr = reinterpret_cast<__bf16*>(grad_output.data_ptr());
    __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    __bf16* gi_ptr = reinterpret_cast<__bf16*>(grad_input.data_ptr());
    _sve_tanh_backward_bfloat16(go_ptr, out_ptr, gi_ptr, grad_output.numel());
    return ;
}