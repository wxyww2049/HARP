#include <arm_sve.h>
#include <limits>

static const uint32_t svexp_f32_coeff[] = {
    0x3f7ffff6,
    0x3efffedb,
    0x3e2aaf33,
    0x3d2b9f17,
    0x3c072010,
};

static inline void bf16_load_to_2xfp32(
    svbool_t pred16, svbool_t pred_lo, svbool_t pred_hi,
    const uint16_t* ptr, std::size_t offset,
    svfloat32_t& out_lo, svfloat32_t& out_hi
) {
    svuint16_t raw = svld1_u16(pred16, ptr + offset);
    svuint32_t lo_u32 = svlsl_n_u32_x(pred_lo, svunpklo_u32(raw), 16);
    svuint32_t hi_u32 = svlsl_n_u32_x(pred_hi, svunpkhi_u32(raw), 16);
    out_lo = svreinterpret_f32_u32(lo_u32);
    out_hi = svreinterpret_f32_u32(hi_u32);
}

static inline void fp32x2_store_as_bf16(
    svbool_t pred16, svbool_t pred_lo, svbool_t pred_hi,
    uint16_t* ptr, std::size_t offset,
    svfloat32_t f32_lo, svfloat32_t f32_hi
) {
    svuint32_t u32_lo = svlsr_n_u32_x(pred_lo, svreinterpret_u32_f32(f32_lo), 16);
    svuint32_t u32_hi = svlsr_n_u32_x(pred_hi, svreinterpret_u32_f32(f32_hi), 16);
    svuint16_t lo_u16 = svreinterpret_u16_u32(u32_lo);
    svuint16_t hi_u16 = svreinterpret_u16_u32(u32_hi);
    svuint16_t packed = svuzp1_u16(lo_u16, hi_u16);
    svst1_u16(pred16, ptr + offset, packed);
}

static inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x) {
    const auto c1 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[0]));
    const auto c2 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[1]));
    const auto c3 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[2]));
    const auto c4 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[3]));
    const auto c5 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[4]));

    const auto shift   = svreinterpret_f32_u32(svdup_n_u32(0x4b00007f));
    const auto inv_ln2 = svreinterpret_f32_u32(svdup_n_u32(0x3fb8aa3b));
    const auto neg_ln2_hi =
        svreinterpret_f32_u32(svdup_n_u32(0xbf317200));
    const auto neg_ln2_lo =
        svreinterpret_f32_u32(svdup_n_u32(0xb5bfbe8e));

    const auto inf       = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f);
    const auto zero      = svdup_n_f32(0.f);
    const auto min_input = svdup_n_f32(-86.64f);


    const auto z     = svmla_f32_x(pg, shift, x, inv_ln2);
    const auto n     = svsub_f32_x(pg, z, shift);
    const auto scale = svreinterpret_f32_u32(svlsl_n_u32_z(pg, svreinterpret_u32_f32(z), 23));

    const auto r_hi = svmla_f32_x(pg, x, n, neg_ln2_hi);
    const auto r    = svmla_f32_x(pg, r_hi, n, neg_ln2_lo);

    const auto r2 = svmul_f32_x(pg, r, r);

    const auto p1     = svmul_f32_x(pg, c1, r);
    const auto p23    = svmla_f32_x(pg, c2, c3, r);
    const auto p45    = svmla_f32_x(pg, c4, c5, r);
    const auto p2345  = svmla_f32_x(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_x(pg, p1, p2345, r2);

    auto poly = svmla_f32_x(pg, scale, p12345, scale);

    poly = svsel_f32(svcmplt_f32(pg, x, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, x, max_input), inf, poly);

    return poly;
}

inline svfloat32_t svtanh_f32_z(svbool_t pg, svfloat32_t val)
{
    const svfloat32_t CONST_1        = svdup_n_f32(1.f);
    const svfloat32_t CONST_2        = svdup_n_f32(2.f);
    const svfloat32_t CONST_MIN_TANH = svdup_n_f32(-10.f);
    const svfloat32_t CONST_MAX_TANH = svdup_n_f32(10.f);

    svfloat32_t x     = svmin_f32_x(pg, svmax_f32_z(pg, val, CONST_MIN_TANH), CONST_MAX_TANH);
    svfloat32_t exp2x = svexp_f32_z(pg, svmul_f32_z(pg, CONST_2, x));
    svfloat32_t num   = svsub_f32_x(pg, exp2x, CONST_1);
    svfloat32_t den   = svadd_f32_x(pg, exp2x, CONST_1);
    svfloat32_t tanh  = svdiv_f32_x(pg, num, den);
    return tanh;
}