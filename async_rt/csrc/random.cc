#include <arm_sve.h>
#include <omp.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/python.h>
struct Philox4x32Key {
    uint32_t k0, k1;
};

struct Philox4x32Ctr {
    uint32_t c0, c1, c2, c3;
};

static inline uint32_t mulhi_u32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
}

static inline Philox4x32Ctr philox4x32_round(Philox4x32Ctr ctr, Philox4x32Key key) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;

    uint32_t hi0 = mulhi_u32(M0, ctr.c0);
    uint32_t lo0 = M0 * ctr.c0;
    uint32_t hi1 = mulhi_u32(M1, ctr.c2);
    uint32_t lo1 = M1 * ctr.c2;

    Philox4x32Ctr out;
    out.c0 = hi1 ^ ctr.c1 ^ key.k0;
    out.c1 = lo1;
    out.c2 = hi0 ^ ctr.c3 ^ key.k1;
    out.c3 = lo0;
    return out;
}

static inline Philox4x32Ctr philox4x32_10(Philox4x32Ctr ctr, Philox4x32Key key) {
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;

    for (int i = 0; i < 10; ++i) {
        ctr = philox4x32_round(ctr, key);
        key.k0 += W0;
        key.k1 += W1;
    }
    return ctr;
}

static inline Philox4x32Ctr make_counter(uint64_t block_index, uint64_t seed_hi = 0) {
    Philox4x32Ctr ctr;
    ctr.c0 = static_cast<uint32_t>(block_index);
    ctr.c1 = static_cast<uint32_t>(block_index >> 32);
    ctr.c2 = static_cast<uint32_t>(seed_hi);
    ctr.c3 = static_cast<uint32_t>(seed_hi >> 32);
    return ctr;
}

static inline Philox4x32Key make_key(uint64_t seed_lo) {
    Philox4x32Key key;
    key.k0 = static_cast<uint32_t>(seed_lo);
    key.k1 = static_cast<uint32_t>(seed_lo >> 32);
    return key;
}

static inline float u32_to_uniform01_scalar(uint32_t x) {
    uint32_t bits = (x >> 9) | 0x3f800000u; // exponent = 127
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f - 1.0f;
}

static inline uint16_t fp32_to_bf16_bits_trunc(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

static inline svfloat32_t u32_to_uniform01_sve(svbool_t pg, svuint32_t x) {
    svuint32_t mant = svlsr_n_u32_x(pg, x, 9);
    svuint32_t bits = svorr_n_u32_x(pg, mant, 0x3f800000u);
    svfloat32_t f = svreinterpret_f32_u32(bits);
    return svsub_n_f32_x(pg, f, 1.0f);
}

void uniform_fill_f32_sve_omp(
    float* out,
    size_t n,
    float from,
    float to,
    uint64_t seed_lo = 0x12345678ULL,
    uint64_t seed_hi = 0x9abcdef0ULL)
{
    if (n == 0) return;
    const float scale = to - from;
    svfloat32_t vscale = svdup_n_f32(scale);
    const Philox4x32Key key = make_key(seed_lo);

    const size_t blocks = (n + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(blocks); ++b) {
        Philox4x32Ctr ctr = make_counter(static_cast<uint64_t>(b), seed_hi);
        Philox4x32Ctr rnd = philox4x32_10(ctr, key);

        alignas(16) uint32_t tmp_u32[4] = {rnd.c0, rnd.c1, rnd.c2, rnd.c3};
        const size_t base = static_cast<size_t>(b) * 4;
        const size_t remain = std::min(static_cast<size_t>(4), n - base);

        size_t i = 0;
        while (i < remain) {
            svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(i), static_cast<uint64_t>(remain));
            svuint32_t vx = svld1_u32(pg, tmp_u32 + i);
            svfloat32_t vf = u32_to_uniform01_sve(pg, vx);
            vf = svmad_n_f32_x(pg, vf, vscale, from); // vf * scale + from
            svst1_f32(pg, out + base + i, vf);

            i += svcntw();
        }
    }
}

void uniform_fill_bf16_omp(
    uint16_t* out,
    size_t n,
    float from,
    float to,
    uint64_t seed_lo = 0x12345678ULL,
    uint64_t seed_hi = 0x9abcdef0ULL)
{
    if (n == 0) return;
    const float scale = to - from;
    const Philox4x32Key key = make_key(seed_lo);

    const size_t blocks = (n + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(blocks); ++b) {
        Philox4x32Ctr ctr = make_counter(static_cast<uint64_t>(b), seed_hi);
        Philox4x32Ctr rnd = philox4x32_10(ctr, key);

        alignas(16) uint32_t tmp_u32[4] = {rnd.c0, rnd.c1, rnd.c2, rnd.c3};
        const size_t base = static_cast<size_t>(b) * 4;
        const size_t remain = std::min(static_cast<size_t>(4), n - base);

        for (size_t i = 0; i < remain; ++i) {
            float value = u32_to_uniform01_scalar(tmp_u32[i]);
            value = value * scale + from;
            out[base + i] = fp32_to_bf16_bits_trunc(value);
        }
    }
}

void random_uniform(torch::Tensor & data, double low, double high) {
    if (data.scalar_type() == torch::kFloat32) {
        float* data_ptr = reinterpret_cast<float*>(data.data_ptr());
        uniform_fill_f32_sve_omp(data_ptr, data.numel(), float(low), float(high));
        return;
    }

    if (data.scalar_type() == torch::kBFloat16) {
        uint16_t* data_ptr = reinterpret_cast<uint16_t*>(data.data_ptr());
        uniform_fill_bf16_omp(data_ptr, data.numel(), float(low), float(high));
        return;
    }

    throw std::runtime_error("random_uniform only supports float32 and bfloat16 tensors");
}
