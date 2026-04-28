#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <arm_sve.h>
#include <async_op_math.h>

#include <ATen/Parallel.h>
#include <ATen/ops/empty_like.h>
#include <torch/python.h>
#include <torch/torch.h>


void rmsnormForwardKernelBF16(
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t* output,
    float* rms_out,
    std::size_t rows,
    std::size_t D,
    float eps,
    const std::size_t* input_row_offsets,
    const std::size_t* output_row_offsets
) {
    at::parallel_for(0, rows, 1, [&](std::size_t row_start, std::size_t row_end) {
        std::size_t svLen = svcntw();
        std::size_t svLen2 = svLen * 2;

        std::vector<float> x_buf(D);

        for (std::size_t row = row_start; row < row_end; ++row) {
            const uint16_t* x_row = input + input_row_offsets[row];

            svfloat32_t sum_sq_lo = svdup_n_f32(0.0f);
            svfloat32_t sum_sq_hi = svdup_n_f32(0.0f);

            for (std::size_t i = 0; i < D; i += svLen2) {
                svbool_t pred16 = svwhilelt_b16(i, D);
                svbool_t pred_lo = svwhilelt_b32(i, D);
                std::size_t hi_start = i + svLen;
                svbool_t pred_hi = svwhilelt_b32(hi_start, D);

                if (i + svLen2 < D) {
                    svprfh(pred16, x_row + i + svLen2, SV_PLDL1KEEP);
                }

                svfloat32_t x_lo, x_hi;
                bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, x_row, i, x_lo, x_hi);

                svst1_f32(pred_lo, x_buf.data() + i, x_lo);
                svst1_f32(pred_hi, x_buf.data() + hi_start, x_hi);

                sum_sq_lo = svmla_f32_m(pred_lo, sum_sq_lo, x_lo, x_lo);
                sum_sq_hi = svmla_f32_m(pred_hi, sum_sq_hi, x_hi, x_hi);
            }

            svbool_t ptrue = svptrue_b32();
            float sum_sq = svaddv_f32(ptrue, sum_sq_lo) + svaddv_f32(ptrue, sum_sq_hi);
            float variance = sum_sq / (float)D;
            float rms = std::sqrt(variance + eps);
            float inv_rms = 1.0f / rms;
            rms_out[row] = rms;

            svfloat32_t sv_inv_rms = svdup_n_f32(inv_rms);
            uint16_t* out_row = output + output_row_offsets[row];

            for (std::size_t i = 0; i < D; i += svLen2) {
                svbool_t pred16 = svwhilelt_b16(i, D);
                svbool_t pred_lo = svwhilelt_b32(i, D);
                std::size_t hi_start = i + svLen;
                svbool_t pred_hi = svwhilelt_b32(hi_start, D);

                svfloat32_t x_lo = svld1_f32(pred_lo, x_buf.data() + i);
                svfloat32_t x_hi = svld1_f32(pred_hi, x_buf.data() + hi_start);

                svfloat32_t out_lo = svmul_f32_m(pred_lo, x_lo, sv_inv_rms);
                svfloat32_t out_hi = svmul_f32_m(pred_hi, x_hi, sv_inv_rms);

                if (weight) {
                    if (i + svLen2 < D) {
                        svprfh(pred16, weight + i + svLen2, SV_PLDL1KEEP);
                    }

                    svfloat32_t w_lo, w_hi;
                    bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, weight, i, w_lo, w_hi);
                    out_lo = svmul_f32_m(pred_lo, out_lo, w_lo);
                    out_hi = svmul_f32_m(pred_hi, out_hi, w_hi);
                }

                fp32x2_store_as_bf16(pred16, pred_lo, pred_hi, out_row, i, out_lo, out_hi);
            }
        }
    });
}

void rmsnormBackwardKernelBF16(
    const uint16_t* input,
    const uint16_t* grad_output,
    const uint16_t* weight,
    const float* rms_saved,
    uint16_t* grad_input,
    float* grad_weight,
    std::size_t rows,
    std::size_t D,
    float /*eps*/,
    const std::size_t* input_row_offsets,
    const std::size_t* grad_output_row_offsets,
    const std::size_t* grad_input_row_offsets
) {
    if (grad_weight) {
        std::memset(grad_weight, 0, D * sizeof(float));
    }

    int num_threads = at::get_num_threads();
    std::vector<std::vector<float>> local_grad_weights;
    if (weight) {
        local_grad_weights.resize(num_threads, std::vector<float>(D, 0.0f));
    }

    at::parallel_for(0, rows, 1, [&](std::size_t row_start, std::size_t row_end) {
        float* my_grad_weight = nullptr;
        if (weight) {
            int tid = at::get_thread_num();
            my_grad_weight = local_grad_weights[tid].data();
        }

        std::size_t svLen = svcntw();
        std::size_t svLen2 = svLen * 2;

        std::vector<float> x_buf(D);
        std::vector<float> dy_buf(D);

        for (std::size_t row = row_start; row < row_end; ++row) {
            const uint16_t* x_row = input + input_row_offsets[row];
            const uint16_t* dy_row = grad_output + grad_output_row_offsets[row];
            uint16_t* dx_row = grad_input + grad_input_row_offsets[row];

            float rms = rms_saved[row];
            float inv_rms = 1.0f / rms;
            svfloat32_t sv_inv_rms = svdup_n_f32(inv_rms);

            svfloat32_t dot_lo = svdup_n_f32(0.0f);
            svfloat32_t dot_hi = svdup_n_f32(0.0f);

            for (std::size_t i = 0; i < D; i += svLen2) {
                svbool_t pred16 = svwhilelt_b16(i, D);
                svbool_t pred_lo = svwhilelt_b32(i, D);
                std::size_t hi_start = i + svLen;
                svbool_t pred_hi = svwhilelt_b32(hi_start, D);

                if (i + svLen2 < D) {
                    svprfh(pred16, x_row + i + svLen2, SV_PLDL1KEEP);
                    svprfh(pred16, dy_row + i + svLen2, SV_PLDL1KEEP);
                }

                svfloat32_t x_lo, x_hi, dy_lo, dy_hi;
                bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, x_row, i, x_lo, x_hi);
                bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, dy_row, i, dy_lo, dy_hi);

                svst1_f32(pred_lo, x_buf.data() + i, x_lo);
                svst1_f32(pred_hi, x_buf.data() + hi_start, x_hi);
                svst1_f32(pred_lo, dy_buf.data() + i, dy_lo);
                svst1_f32(pred_hi, dy_buf.data() + hi_start, dy_hi);

                svfloat32_t gn_lo = dy_lo, gn_hi = dy_hi;
                if (weight) {
                    svfloat32_t w_lo, w_hi;
                    bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, weight, i, w_lo, w_hi);
                    gn_lo = svmul_f32_m(pred_lo, dy_lo, w_lo);
                    gn_hi = svmul_f32_m(pred_hi, dy_hi, w_hi);
                }
                svfloat32_t xn_lo = svmul_f32_m(pred_lo, x_lo, sv_inv_rms);
                svfloat32_t xn_hi = svmul_f32_m(pred_hi, x_hi, sv_inv_rms);

                dot_lo = svmla_f32_m(pred_lo, dot_lo, gn_lo, xn_lo);
                dot_hi = svmla_f32_m(pred_hi, dot_hi, gn_hi, xn_hi);
            }

            svbool_t ptrue = svptrue_b32();
            float dot_sum = svaddv_f32(ptrue, dot_lo) + svaddv_f32(ptrue, dot_hi);
            float coeff = dot_sum / (float)D;
            svfloat32_t sv_coeff = svdup_n_f32(coeff);

            for (std::size_t i = 0; i < D; i += svLen2) {
                svbool_t pred16 = svwhilelt_b16(i, D);
                svbool_t pred_lo = svwhilelt_b32(i, D);
                std::size_t hi_start = i + svLen;
                svbool_t pred_hi = svwhilelt_b32(hi_start, D);

                svfloat32_t x_lo = svld1_f32(pred_lo, x_buf.data() + i);
                svfloat32_t x_hi = svld1_f32(pred_hi, x_buf.data() + hi_start);
                svfloat32_t dy_lo = svld1_f32(pred_lo, dy_buf.data() + i);
                svfloat32_t dy_hi = svld1_f32(pred_hi, dy_buf.data() + hi_start);

                svfloat32_t xn_lo = svmul_f32_m(pred_lo, x_lo, sv_inv_rms);
                svfloat32_t xn_hi = svmul_f32_m(pred_hi, x_hi, sv_inv_rms);

                svfloat32_t gn_lo = dy_lo, gn_hi = dy_hi;
                if (weight) {
                    svfloat32_t w_lo, w_hi;
                    bf16_load_to_2xfp32(pred16, pred_lo, pred_hi, weight, i, w_lo, w_hi);
                    gn_lo = svmul_f32_m(pred_lo, dy_lo, w_lo);
                    gn_hi = svmul_f32_m(pred_hi, dy_hi, w_hi);
                }

                svfloat32_t tmp_lo = svmul_f32_m(pred_lo, xn_lo, sv_coeff);
                svfloat32_t tmp_hi = svmul_f32_m(pred_hi, xn_hi, sv_coeff);
                svfloat32_t dx_lo = svsub_f32_m(pred_lo, gn_lo, tmp_lo);
                svfloat32_t dx_hi = svsub_f32_m(pred_hi, gn_hi, tmp_hi);
                dx_lo = svmul_f32_m(pred_lo, dx_lo, sv_inv_rms);
                dx_hi = svmul_f32_m(pred_hi, dx_hi, sv_inv_rms);

                fp32x2_store_as_bf16(pred16, pred_lo, pred_hi, dx_row, i, dx_lo, dx_hi);

                if (my_grad_weight) {
                    svfloat32_t gw_lo = svld1_f32(pred_lo, my_grad_weight + i);
                    svfloat32_t gw_hi = svld1_f32(pred_hi, my_grad_weight + hi_start);
                    gw_lo = svmla_f32_m(pred_lo, gw_lo, dy_lo, xn_lo);
                    gw_hi = svmla_f32_m(pred_hi, gw_hi, dy_hi, xn_hi);
                    svst1_f32(pred_lo, my_grad_weight + i, gw_lo);
                    svst1_f32(pred_hi, my_grad_weight + hi_start, gw_hi);
                }
            }
        }
    });

    if (weight && grad_weight) {
        for (int t = 0; t < num_threads; ++t) {
            const float* local = local_grad_weights[t].data();
            std::size_t svLen = svcntw();
            for (std::size_t i = 0; i < D; i += svLen) {
                svbool_t pred = svwhilelt_b32(i, D);
                svfloat32_t gw = svld1_f32(pred, grad_weight + i);
                svfloat32_t lw = svld1_f32(pred, local + i);
                gw = svadd_f32_m(pred, gw, lw);
                svst1_f32(pred, grad_weight + i, gw);
            }
        }
    }
}

struct LayoutKey {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides_a;
    std::vector<int64_t> strides_b;
    std::vector<int64_t> strides_c;

    bool operator==(const LayoutKey& other) const {
        return sizes == other.sizes &&
               strides_a == other.strides_a &&
               strides_b == other.strides_b &&
               strides_c == other.strides_c;
    }
};

struct LayoutKeyHash {
    std::size_t operator()(const LayoutKey& k) const {
        auto hash_combine = [](std::size_t& seed, int64_t v) {
            seed ^= std::hash<int64_t>{}(v) + 0x9e3779b97f4a7c15ULL +
                    (seed << 6) + (seed >> 2);
        };

        std::size_t seed = 0;

        hash_combine(seed, -11);
        for (auto v : k.sizes) {
            hash_combine(seed, v);
        }

        hash_combine(seed, -22);
        for (auto v : k.strides_a) {
            hash_combine(seed, v);
        }

        hash_combine(seed, -33);
        for (auto v : k.strides_b) {
            hash_combine(seed, v);
        }

        hash_combine(seed, -44);
        for (auto v : k.strides_c) {
            hash_combine(seed, v);
        }

        return seed;
    }
};

struct PairedOffsets {
    std::vector<std::size_t> in_offsets;
    std::vector<std::size_t> out_offsets;
};

struct TripleOffsets {
    std::vector<std::size_t> in_offsets;
    std::vector<std::size_t> dy_offsets;
    std::vector<std::size_t> dx_offsets;
};

static LayoutKey make_paired_key(
    const torch::Tensor& in_t,
    const torch::Tensor& out_t
) {
    LayoutKey key;
    key.sizes.assign(in_t.sizes().begin(), in_t.sizes().end());
    key.strides_a.assign(in_t.strides().begin(), in_t.strides().end());
    key.strides_b.assign(out_t.strides().begin(), out_t.strides().end());
    return key;
}

static LayoutKey make_triple_key(
    const torch::Tensor& in_t,
    const torch::Tensor& dy_t,
    const torch::Tensor& dx_t
) {
    LayoutKey key;
    key.sizes.assign(in_t.sizes().begin(), in_t.sizes().end());
    key.strides_a.assign(in_t.strides().begin(), in_t.strides().end());
    key.strides_b.assign(dy_t.strides().begin(), dy_t.strides().end());
    key.strides_c.assign(dx_t.strides().begin(), dx_t.strides().end());
    return key;
}

static const PairedOffsets& compute_paired_offsets_memory_order(
    const torch::Tensor& in_t,
    const torch::Tensor& out_t
) {
    static std::unordered_map<LayoutKey, std::shared_ptr<PairedOffsets>, LayoutKeyHash> cache;

    LayoutKey key = make_paired_key(in_t, out_t);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return *(it->second);
    }

    auto sizes = in_t.sizes();
    auto in_strides = in_t.strides();
    auto out_strides = out_t.strides();
    int ndim = in_t.dim();
    std::size_t rows = in_t.numel() / sizes[ndim - 1];

    std::vector<int> dims;
    for (int d = 0; d < ndim - 1; ++d) {
        dims.push_back(d);
    }
    std::sort(dims.begin(), dims.end(),
        [&](int a, int b) { return in_strides[a] < in_strides[b]; });

    auto result = std::make_shared<PairedOffsets>();
    result->in_offsets.resize(rows);
    result->out_offsets.resize(rows);

    std::vector<std::size_t> idx(ndim - 1, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t in_off = 0;
        std::size_t out_off = 0;
        for (int d = 0; d < ndim - 1; ++d) {
            in_off += idx[d] * static_cast<std::size_t>(in_strides[d]);
            out_off += idx[d] * static_cast<std::size_t>(out_strides[d]);
        }
        result->in_offsets[r] = in_off;
        result->out_offsets[r] = out_off;

        for (std::size_t k = 0; k < dims.size(); ++k) {
            int d = dims[k];
            if (++idx[d] < static_cast<std::size_t>(sizes[d])) {
                break;
            }
            idx[d] = 0;
        }
    }

    auto* ptr = result.get();
    cache.emplace(std::move(key), std::move(result));
    return *ptr;
}

static const TripleOffsets& compute_triple_offsets_memory_order(
    const torch::Tensor& in_t,
    const torch::Tensor& dy_t,
    const torch::Tensor& dx_t
) {
    static std::unordered_map<LayoutKey, std::shared_ptr<TripleOffsets>, LayoutKeyHash> cache;

    LayoutKey key = make_triple_key(in_t, dy_t, dx_t);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return *(it->second);
    }

    auto sizes = in_t.sizes();
    auto in_strides = in_t.strides();
    auto dy_strides = dy_t.strides();
    auto dx_strides = dx_t.strides();
    int ndim = in_t.dim();
    std::size_t rows = in_t.numel() / sizes[ndim - 1];

    std::vector<int> dims;
    for (int d = 0; d < ndim - 1; ++d) {
        dims.push_back(d);
    }
    std::sort(dims.begin(), dims.end(),
        [&](int a, int b) { return in_strides[a] < in_strides[b]; });

    auto result = std::make_shared<TripleOffsets>();
    result->in_offsets.resize(rows);
    result->dy_offsets.resize(rows);
    result->dx_offsets.resize(rows);

    std::vector<std::size_t> idx(ndim - 1, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t in_off = 0;
        std::size_t dy_off = 0;
        std::size_t dx_off = 0;
        for (int d = 0; d < ndim - 1; ++d) {
            in_off += idx[d] * static_cast<std::size_t>(in_strides[d]);
            dy_off += idx[d] * static_cast<std::size_t>(dy_strides[d]);
            dx_off += idx[d] * static_cast<std::size_t>(dx_strides[d]);
        }
        result->in_offsets[r] = in_off;
        result->dy_offsets[r] = dy_off;
        result->dx_offsets[r] = dx_off;

        for (std::size_t k = 0; k < dims.size(); ++k) {
            int d = dims[k];
            if (++idx[d] < static_cast<std::size_t>(sizes[d])) {
                break;
            }
            idx[d] = 0;
        }
    }

    auto* ptr = result.get();
    cache.emplace(std::move(key), std::move(result));
    return *ptr;
}

void rmsnorm_forward_out(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& rms,
    double eps
) {
    assert(input.scalar_type() == torch::kBFloat16);
    assert(input.stride(-1) == 1);

    auto sizes = input.sizes();
    std::size_t D = sizes[sizes.size() - 1];
    std::size_t rows = input.numel() / D;

    const auto& offsets = compute_paired_offsets_memory_order(input, output);

    uint16_t* in_ptr = reinterpret_cast<uint16_t*>(input.data_ptr<at::BFloat16>());
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>());
    float* rms_ptr = rms.data_ptr<float>();

    uint16_t* w_ptr = nullptr;
    if (weight.defined() && weight.numel() > 0) {
        assert(weight.is_contiguous());
        assert(weight.scalar_type() == torch::kBFloat16);
        w_ptr = reinterpret_cast<uint16_t*>(weight.data_ptr<at::BFloat16>());
    }

    rmsnormForwardKernelBF16(
        in_ptr, w_ptr, out_ptr, rms_ptr,
        rows, D, (float)eps,
        offsets.in_offsets.data(),
        offsets.out_offsets.data()
    );

    return;
}

void rmsnorm_backward_out(
    const torch::Tensor& input,
    const torch::Tensor& grad_output,
    const torch::Tensor& weight,
    const torch::Tensor& rms,
    torch::Tensor& grad_input,
    torch::Tensor& grad_weight
) {
    assert(input.stride(-1) == 1);
    assert(grad_output.stride(-1) == 1);
    assert(rms.is_contiguous());

    auto sizes = input.sizes();
    std::size_t D = sizes[sizes.size() - 1];
    std::size_t rows = input.numel() / D;

    const auto& offsets = compute_triple_offsets_memory_order(input, grad_output, grad_input);

    uint16_t* in_ptr = reinterpret_cast<uint16_t*>(input.data_ptr<at::BFloat16>());
    uint16_t* dy_ptr = reinterpret_cast<uint16_t*>(grad_output.data_ptr<at::BFloat16>());
    const float* rms_ptr = rms.data_ptr<float>();
    uint16_t* dx_ptr = reinterpret_cast<uint16_t*>(grad_input.data_ptr<at::BFloat16>());

    uint16_t* w_ptr = nullptr;
    float* gw_ptr = nullptr;
    torch::Tensor grad_weight_fp32;
    w_ptr = reinterpret_cast<uint16_t*>(weight.data_ptr<at::BFloat16>());
    grad_weight_fp32 = torch::zeros({(long)D}, torch::dtype(torch::kFloat32));

    rmsnormBackwardKernelBF16(
        in_ptr, dy_ptr, w_ptr, rms_ptr, dx_ptr, gw_ptr,
        rows, D, 0.0f,
        offsets.in_offsets.data(),
        offsets.dy_offsets.data(),
        offsets.dx_offsets.data()
    );

    grad_weight = grad_weight_fp32.to(torch::dtype(torch::kBFloat16));

}