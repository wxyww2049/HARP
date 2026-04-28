#define BGEMM
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <limits>
#include <numeric>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_acle.h>
#include <kblas.h>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <numaif.h>
#include <numa.h>
#include <sched.h>
#include <cerrno>
#include <omp.h>
#include <mutex>

// dim = h_num * h_len
// t_num = batch_size*token_num
// Q: (t_num, h_num, h_len)
// grad_Q: (t_num, h_num, h_len)
// K: (t_num, h_num, h_len)
// dK: (t_num, h_num, h_len)
// V: (t_num, h_num, h_len)
// dV: (t_num, h_num, h_len)
// out: (t_num, h_num, h_len)
// grad_out: (t_num, h_num, h_len)
// D: (h_num, t_num)
// mi: (h_num, t_num)
// li: (h_num, t_num)

namespace ring_attn_component {

constexpr int atm_magic = 16;
constexpr int atpacked_m = 2 * atm_magic;
constexpr int btn_magic = 32;
constexpr int btpacked_n = 64;
constexpr int m_magic = 32;
constexpr int m_pack = 16;
constexpr int n_magic = 32;
constexpr int n_pack = 64;

__attribute__((noinline)) void pack_A_T(__bf16* A, int k, int m, int lda, __bf16* A_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = m < atpacked_m ? m : atpacked_m;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    const int block = k * atm_magic;
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t row_vec0 = svld1_bf16(pg_tail, A + i * lda);
        svbfloat16_t row_vec1 = svld1_bf16(pg_tail, A + (i + 1) * lda);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, A_p + i * atm_magic, a1);
        svst1(pg, A_p + i * atm_magic + block, a2);
    }
}

__attribute__((noinline)) void pack_B_T(__bf16* B, int n, int k, int ldb, __bf16* B_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    const int dst_panel_stride = btn_magic * btpacked_n;
    const int dst_pair_stride = 2 * btpacked_n;
    __bf16* src_panel = B;
    __bf16* dst_panel = B_p;
    for (int kk = 0; kk < k; kk += btn_magic) {
        const int rows0 = n > btn_magic ? btn_magic : n;
        const int rows1 = n > btn_magic ? (n - btn_magic) : 0;
        __bf16* src0 = src_panel;
        __bf16* src1 = src_panel + ldb * btn_magic;
        svzero_za();
        for (int mm = 0; mm < rows0; ++mm) {
            svld1_hor_za16(0, mm, pg, src0);
            src0 += ldb;
        }
        for (int mm = 0; mm < rows1; ++mm) {
            svld1_hor_za16(1, mm, pg, src1);
            src1 += ldb;
        }
        __bf16* dst = dst_panel;
        for (int nn = 0; nn < btn_magic; nn += 2) {
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst, a1);
            svst1(pg, dst + btn_magic, a2);
            row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn);
            row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn + 1);
            a1 = svzip1(row_vec0, row_vec1);
            a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst + 2 * btn_magic, a1);
            svst1(pg, dst + 3 * btn_magic, a2);
            dst += dst_pair_stride;
        }
        src_panel += btn_magic;
        dst_panel += dst_panel_stride;
    }
}

__attribute__((noinline)) void pack_A(__bf16* A, int m, int k, int lda, __bf16* A_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    svzero_za();
    for (int kk = 0; kk < k; kk += m_magic) {
        for (int mm = 0; mm < m; ++mm) {
            svld1_hor_za16(0, mm, pg, A + mm * lda + kk);
        }
        for (int nn = 0; nn < m_magic; nn += 2) {
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svst1(pg, A_p + m_pack * nn + kk * m_pack, a1);
        }
    }
}

__attribute__((noinline)) void pack_B(__bf16* B, int k, int n, int ldb, __bf16* B_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = (n - n_magic) < n_magic ? (n - n_magic) : n_magic;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t row_vec0 = svld1_bf16(pg, B + i * ldb);
        svbfloat16_t row_vec1 = svld1_bf16(pg, B + (i + 1) * ldb);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, B_p + i * n_pack, a1);
        svst1(pg, B_p + i * n_pack + n_magic, a2);
        row_vec0 = svld1_bf16(pg_tail, B + i * ldb + n_magic);
        row_vec1 = svld1_bf16(pg_tail, B + (i + 1) * ldb + n_magic);
        a1 = svzip1(row_vec0, row_vec1);
        a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, B_p + i * n_pack + 2 * n_magic, a1);
        svst1(pg, B_p + i * n_pack + 3 * n_magic, a2);
    }
}

__attribute__((noinline)) void sme_kernel_bf16(__bf16* A, __bf16* B, __bf16* C_out,
                int ldc, int k, int m_use, int n_use,
                __bf16 alpha, __bf16 beta)
__arm_streaming __arm_inout("za")
{
    (void)n_use;
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (beta == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            svuint16_t vb16 = svld1_u16(pg_bf16, reinterpret_cast<const uint16_t*>(C_out + (size_t)h * ldc));
            svuint32_t lo32 = svunpklo_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(0, h, pg, lo32);
            lo32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(1, h, pg, lo32);

            vb16 = svld1_u16(pg_bf16, reinterpret_cast<const uint16_t*>(C_out + (size_t)h * ldc + 2 * wl));
            lo32 = svunpklo_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(2, h, pg, lo32);
            lo32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(3, h, pg, lo32);
        }
    } else if (beta == 0.0f) {
        svzero_za();
    }
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t a0 = svld1_bf16(pg_bf16, a_ptr);
        a_ptr += tl;
        svbfloat16_t b0 = svldnt1_bf16(pg_bf16, b_ptr);
        svbfloat16_t b1 = svldnt1_bf16(pg_bf16, b_ptr + tl);
        svbfloat16_t b2 = svldnt1_bf16(pg_bf16, b_ptr + 2 * tl);
        svbfloat16_t b3 = svldnt1_bf16(pg_bf16, b_ptr + 3 * tl);
        b_ptr += 4 * tl;
        svmopa_za32_bf16_m(0, pg_bf16, pg_bf16, a0, b0);
        svmopa_za32_bf16_m(1, pg_bf16, pg_bf16, a0, b1);
        svmopa_za32_bf16_m(2, pg_bf16, pg_bf16, a0, b2);
        svmopa_za32_bf16_m(3, pg_bf16, pg_bf16, a0, b3);
    }
    svbfloat16_t old_bf16 = svdup_n_bf16((__bf16)0);
    svfloat32_t old_f32 = svdup_n_f32(0.0f);
    svfloat32_t scale = svdup_n_f32((float)alpha);
    if (alpha == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            svbfloat16_t z0 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 0, 2 * h);
            svbfloat16_t z1 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 1, 2 * h);
            z1 = svuzp2_bf16(z0, z1);
            svst1_bf16(pg_bf16, C_out + h * ldc, z1);
            z0 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 0, 2 * h + 1);
            z1 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 1, 2 * h + 1);
            z1 = svuzp2_bf16(z0, z1);
            svst1_bf16(pg_bf16, C_out + h * ldc + 2 * wl, z1);
        }
    } else {
        for (int h = 0; h < m_use; ++h) {
            svfloat32_t z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 0, h);
            svfloat32_t z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 1, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svbfloat16_t u0 = svreinterpret_bf16_f32(z0);
            svbfloat16_t u1 = svreinterpret_bf16_f32(z1);
            u1 = svuzp2_bf16(u0, u1);
            svst1_bf16(pg_bf16, C_out + h * ldc, u1);

            z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 2, h);
            z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 3, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            u0 = svreinterpret_bf16_f32(z0);
            u1 = svreinterpret_bf16_f32(z1);
            u1 = svuzp2_bf16(u0, u1);
            svst1_bf16(pg_bf16, C_out + h * ldc + 2 * wl, u1);
        }
    }
}

__attribute__((noinline)) void sme_kernel_f32(__bf16* A, __bf16* B, float* C_out,
                int ldc, int k, int m_use, int n_use,
                __bf16 alpha, __bf16 beta)
__arm_streaming __arm_inout("za")
{
    (void)n_use;
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (beta == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            float* c0 = C_out + h * ldc;
            float* c1 = C_out + h * ldc + wl;
            float* c2 = C_out + h * ldc + 2 * wl;
            float* c3 = C_out + h * ldc + 3 * wl;
            svld1_hor_za32(0, h, pg, c0);
            svld1_hor_za32(1, h, pg, c1);
            svld1_hor_za32(2, h, pg, c2);
            svld1_hor_za32(3, h, pg, c3);
        }
    } else if (beta == 0.0f) {
        svzero_za();
    }
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t a0 = svld1_bf16(pg_bf16, a_ptr);
        a_ptr += tl;
        svbfloat16_t b0 = svldnt1_bf16(pg_bf16, b_ptr);
        svbfloat16_t b1 = svldnt1_bf16(pg_bf16, b_ptr + tl);
        svbfloat16_t b2 = svldnt1_bf16(pg_bf16, b_ptr + 2 * tl);
        svbfloat16_t b3 = svldnt1_bf16(pg_bf16, b_ptr + 3 * tl);
        b_ptr += 4 * tl;
        svmopa_za32_bf16_m(0, pg_bf16, pg_bf16, a0, b0);
        svmopa_za32_bf16_m(1, pg_bf16, pg_bf16, a0, b1);
        svmopa_za32_bf16_m(2, pg_bf16, pg_bf16, a0, b2);
        svmopa_za32_bf16_m(3, pg_bf16, pg_bf16, a0, b3);
    }
    svbfloat16_t old_bf16 = svdup_n_bf16((__bf16)0);
    svfloat32_t old_f32 = svdup_n_f32(0.0f);
    svfloat32_t scale = svdup_n_f32((float)alpha);
    if (alpha == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            float* c0 = C_out + h * ldc;
            float* c1 = C_out + h * ldc + wl;
            float* c2 = C_out + h * ldc + 2 * wl;
            float* c3 = C_out + h * ldc + 3 * wl;
            svst1_hor_za32(0, h, pg, c0);
            svst1_hor_za32(1, h, pg, c1);
            svst1_hor_za32(2, h, pg, c2);
            svst1_hor_za32(3, h, pg, c3);
        }
    } else {
        for (int h = 0; h < m_use; ++h) {
            svfloat32_t z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 0, h);
            svfloat32_t z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 1, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svst1_f32(pg, C_out + h * ldc, z0);
            svst1_f32(pg, C_out + h * ldc + wl, z1);
            z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 2, h);
            z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 3, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svst1_f32(pg, C_out + h * ldc + 2 * wl, z0);
            svst1_f32(pg, C_out + h * ldc + 3 * wl, z1);
        }
    }
}

constexpr int magic_n = 64;
constexpr int magic_m_nt = 16;
constexpr int magic_m_t = 32;

__arm_new("za")
__attribute__((noinline)) void pack_B_whole(const enum CBLAS_TRANSPOSE transB, int n, int k, int ldb, __bf16* b, __bf16* b_p)
__arm_streaming
{
    if (transB == CblasNoTrans) {
        for (int i = 0; i < n; i += magic_n) {
            pack_B(b + i, k, magic_n, ldb, b_p + (size_t)i * k);
        }
    } else {
        for (int i = 0; i < n; i += magic_n) {
            pack_B_T(b + (size_t)i * ldb, magic_n, k, ldb, b_p + (size_t)i * k);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void pack_A_whole(const enum CBLAS_TRANSPOSE transA, int m, int k, int lda, __bf16* a, __bf16* a_p)
__arm_streaming
{
    if (transA == CblasNoTrans) {
        for (int i = 0; i < m; i += magic_m_nt) {
            pack_A(a + (size_t)i * lda, magic_m_nt, k, lda, a_p + (size_t)i * k);
        }
    } else {
        for (int i = 0; i < m; i += magic_m_t) {
            pack_A_T(a + (size_t)i, k, magic_m_t, lda, a_p + (size_t)i * k);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void easy_bgemm(int m, int n, int k, __bf16* a, __bf16* b, __bf16* c, int ldc, __bf16 alpha, __bf16 beta)
__arm_streaming
{
    for (int i = 0; i < n; i += magic_n) {
        for (int mm = 0; mm < m; mm += magic_m_nt) {
            sme_kernel_bf16(a + (size_t)mm * k,
                       b + (size_t)i * k,
                       c + (size_t)mm * ldc + i,
                       ldc, k, magic_m_nt, magic_n,
                       alpha, beta);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void easy_sbgemm(int m, int n, int k, __bf16* a, __bf16* b, float* c, int ldc, __bf16 alpha, __bf16 beta)
__arm_streaming
{
    for (int i = 0; i < n; i += magic_n) {
        for (int mm = 0; mm < m; mm += magic_m_nt) {
            sme_kernel_f32(a + (size_t)mm * k,
                       b + (size_t)i * k,
                       c + (size_t)mm * ldc + i,
                       ldc, k, magic_m_nt, magic_n,
                       alpha, beta);
        }
    }
}

static inline bool supports_pack_a(int m, int k)
{
    return m > 0 && k > 0 &&
           (m % magic_m_nt) == 0 &&
           (k % m_magic) == 0;
}

static inline bool supports_pack_b(int n, int k)
{
    return n > 0 && k > 0 &&
           (n % magic_n) == 0 &&
           (k % m_magic) == 0;
}

static inline bool supports_gemm(int m, int n, int k)
{
    return m > 0 && n > 0 && k > 0 &&
           (m % magic_m_nt) == 0 &&
           (n % magic_n) == 0 &&
           (k % m_magic) == 0;
}

size_t pack_get_size(int kind, int m, int n, int k)
{
    if (kind == CblasA) {
        return supports_pack_a(m, k) ? (size_t)m * (size_t)k : 0;
    }
    if (kind == CblasB) {
        return supports_pack_b(n, k) ? (size_t)n * (size_t)k : 0;
    }
    return 0;
}

__attribute__((noinline)) void pack_a_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                const int m,
                                                const int k,
                                                const int lda,
                                                __bf16* a,
                                                __bf16* a_p)
{
    pack_A_whole(transA, m, k, lda, a, a_p);
}

__attribute__((noinline)) void pack_b_unchecked(const enum CBLAS_TRANSPOSE transB,
                                                const int n,
                                                const int k,
                                                const int ldb,
                                                __bf16* b,
                                                __bf16* b_p)
{
    pack_B_whole(transB, n, k, ldb, b, b_p);
}

__attribute__((noinline)) void bgemm_packed_unchecked(const int m,
                                                      const int n,
                                                      const int k,
                                                      const __bf16 alpha,
                                                      __bf16* packed_a,
                                                      __bf16* packed_b,
                                                      const __bf16 beta,
                                                      __bf16* c,
                                                      const int ldc)
{
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) void sbgemm_packed_unchecked(const int m,
                                                       const int n,
                                                       const int k,
                                                       const __bf16 alpha,
                                                       __bf16* packed_a,
                                                       __bf16* packed_b,
                                                       const __bf16 beta,
                                                       float* c,
                                                       const int ldc)
{
    easy_sbgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) bool bgemm_kernel(const enum CBLAS_ORDER order,
                  const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB,
                  const int m,
                  const int n,
                  const int k,
                  const __bf16 alpha,
                  __bf16* a,
                  const int lda,
                  __bf16* b,
                  const int ldb,
                  const __bf16 beta,
                  __bf16* c,
                  const int ldc,
                  __bf16* packed_a,
                  __bf16* packed_b)
{
    if (order != CblasRowMajor) {
        return false;
    }
    if (!supports_pack_a(m, k) ||
        !supports_pack_b(n, k) ||
        !supports_gemm(m, n, k)) {
        return false;
    }
    if (a == nullptr || b == nullptr || c == nullptr ||
        packed_a == nullptr || packed_b == nullptr) {
        return false;
    }
    if (!(beta == (__bf16)0.0f || beta == (__bf16)1.0f)) {
        return false;
    }

    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
    return true;
}

__attribute__((noinline)) void bgemm_kernel_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                      const enum CBLAS_TRANSPOSE transB,
                                                      const int m,
                                                      const int n,
                                                      const int k,
                                                      const __bf16 alpha,
                                                      __bf16* a,
                                                      const int lda,
                                                      __bf16* b,
                                                      const int ldb,
                                                      const __bf16 beta,
                                                      __bf16* c,
                                                      const int ldc,
                                                      __bf16* packed_a,
                                                      __bf16* packed_b)
{
    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) void sbgemm_kernel_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                       const enum CBLAS_TRANSPOSE transB,
                                                       const int m,
                                                       const int n,
                                                       const int k,
                                                       const __bf16 alpha,
                                                       __bf16* a,
                                                       const int lda,
                                                       __bf16* b,
                                                       const int ldb,
                                                       const __bf16 beta,
                                                       float* c,
                                                       const int ldc,
                                                       __bf16* packed_a,
                                                       __bf16* packed_b)
{
    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_sbgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

}  // namespace ring_attn_component

static inline bool use_attn_component_backward_shape(int t_num, int h_num, int h_len)
{
    return h_num == 30 && h_len == 128 && (t_num == 256 || t_num == 768);
}

static inline int forward_q_tile_rows(int t_num)
{
    return t_num == 768 ? 96 : t_num;
}

struct RingAttnCache {
    int current_t_num = -1;
    int current_h_num = -1;
    int current_h_len = -1;
    int capacity_t_num = 0;
    int capacity_h_num = 0;
    int capacity_h_len = 0;
    float scale = 0.0f;
    std::vector<__bf16> alpha;
    std::vector<__bf16> alpha1;
    std::vector<__bf16> beta;
    std::vector<CBLAS_TRANSPOSE> transA;
    std::vector<CBLAS_TRANSPOSE> transB;
    std::vector<int> t_array;
    std::vector<int> h_array;
    std::vector<int> ld_array;
    std::vector<int> group_size;
    std::vector<__bf16*> Q_array;
    std::vector<__bf16*> grad_Q_array;
    std::vector<__bf16*> K_array;
    std::vector<__bf16*> dK_array;
    std::vector<__bf16*> S_array;
    std::vector<__bf16*> dP_array;
    std::vector<__bf16*> V_array;
    std::vector<__bf16*> dV_array;
    std::vector<__bf16*> grad_out_array;
    std::vector<__bf16*> c1_array;
    std::vector<float*> out_array;

    void ensure_config(int t, int hnum, int hlen, float s) {
        if (capacity_h_num < hnum) {
            alpha.resize(hnum);
            alpha1.resize(hnum);
            beta.resize(hnum);
            transA.resize(hnum);
            transB.resize(hnum);
            t_array.resize(hnum);
            h_array.resize(hnum);
            ld_array.resize(hnum);
            group_size.resize(hnum);
            Q_array.resize(hnum);
            grad_Q_array.resize(hnum);
            K_array.resize(hnum);
            dK_array.resize(hnum);
            S_array.resize(hnum);
            dP_array.resize(hnum);
            V_array.resize(hnum);
            dV_array.resize(hnum);
            grad_out_array.resize(hnum);
            c1_array.resize(hnum);
            out_array.resize(hnum);
            capacity_h_num = hnum;
        }

        capacity_t_num = std::max(capacity_t_num, t);
        capacity_h_len = std::max(capacity_h_len, hlen);
        current_t_num = t;
        current_h_num = hnum;
        current_h_len = hlen;
        scale = s;

        std::fill_n(alpha.begin(), hnum, (__bf16)1.0f);
        std::fill_n(alpha1.begin(), hnum, (__bf16)scale);
        std::fill_n(beta.begin(), hnum, (__bf16)0.0f);
        std::fill_n(transA.begin(), hnum, CblasNoTrans);
        std::fill_n(transB.begin(), hnum, CblasTrans);
        std::fill_n(t_array.begin(), hnum, t);
        std::fill_n(h_array.begin(), hnum, hlen);
        std::fill_n(ld_array.begin(), hnum, hlen * hnum);
        std::fill_n(group_size.begin(), hnum, 1);
    }
};

const int numa_offset = 16;
const int kCoresPerNuma = 38;
const size_t page_size = 2 * 1024 * 1024;

static void *manual_hbmalloc_numa(size_t size, int numa_id)
{
    size = (size + page_size - 1) & ~(page_size - 1);
    void *ptr;
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED) {
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "mmap failed: errno=%d (%s)\n", errno, strerror(errno));
            return NULL;
        }
    }
    unsigned long msk = 1ul << (numa_id + numa_offset);
    long rc = mbind(ptr, size, MPOL_BIND, &msk, 64, 0);
    if (rc != 0) {
        munmap(ptr, size);
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "mmap failed ...\n");
            return NULL;
        }
        rc = mbind(ptr, size, MPOL_BIND, &msk, 64, 0);
        if (rc != 0) {
            fprintf(stderr, "mbind failed (numa=%d): errno=%d (%s)\n",
                    numa_id, errno, strerror(errno));
        }
    }
    memset(ptr, 0, size);
    return ptr;
}

void manual_hbmfree(void *ptr, size_t size)
{
    size = (size + page_size - 1) & ~(page_size - 1);
    if (munmap(ptr, size) == -1) {
        int core_id = sched_getcpu();
        printf("core_id: %d, free fail!\n", core_id);
    }
}

inline int detect_current_numa()
{
    const char* rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
    if (rank_env && rank_env[0] != '\0') {
        return static_cast<int>(std::strtol(rank_env, nullptr, 10) % 16);
    }
    const int current_cpu = sched_getcpu();
    return current_cpu >= 0 ? (current_cpu / kCoresPerNuma) : 0;
}

int default_numa_node()
{
    int node = detect_current_numa();
    return node;
}

const std::vector<int>& current_affinity_cpus()
{
    static std::vector<int> cpus;
    static std::once_flag once;
    std::call_once(once, []() {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
            return;
        }
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                cpus.push_back(cpu);
            }
        }
    });
    return cpus;
}

void bind_omp_thread_to_allowed_cpu(int thread_idx)
{
    const std::vector<int>& cpus = current_affinity_cpus();
    if (cpus.empty()) {
        return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpus[thread_idx % cpus.size()], &mask);
    (void)sched_setaffinity(0, sizeof(mask), &mask);
}

constexpr std::size_t kWorkspaceGrowthNumerator = 5;
constexpr std::size_t kWorkspaceGrowthDenominator = 4;

std::size_t grow_workspace_elems(std::size_t needed)
{
    const std::size_t grown =
        (needed * kWorkspaceGrowthNumerator + kWorkspaceGrowthDenominator - 1) /
        kWorkspaceGrowthDenominator;
    return std::max(needed, grown);
}

std::size_t grow_workspace_bytes(std::size_t needed_bytes)
{
    constexpr std::size_t kWorkspacePageBytes = 2ULL * 1024ULL * 1024ULL;
    const std::size_t grown =
        (needed_bytes * kWorkspaceGrowthNumerator + kWorkspaceGrowthDenominator - 1) /
        kWorkspaceGrowthDenominator;
    const std::size_t aligned =
        ((grown + kWorkspacePageBytes - 1) / kWorkspacePageBytes) * kWorkspacePageBytes;
    return std::max(needed_bytes, aligned);
}

struct BgemmWorkspace {
    __bf16 *packed_a = nullptr;
    __bf16 *packed_b = nullptr;
    __bf16 *c = nullptr;
    size_t size_a = 0;
    size_t size_b = 0;
    size_t size_c = 0;

    ~BgemmWorkspace()
    {
        if (packed_a != nullptr) {
            manual_hbmfree(packed_a, size_a * sizeof(__bf16));
        }
        if (packed_b != nullptr) {
            manual_hbmfree(packed_b, size_b * sizeof(__bf16));
        }
        if (c != nullptr) {
            manual_hbmfree(c, size_c * sizeof(__bf16));
        }
    }

    bool reserve(__bf16 *&buffer, size_t &capacity, size_t needed)
    {
        if (capacity >= needed) {
            return true;
        }

        const std::size_t alloc_elems = grow_workspace_elems(needed);
        const int numa_node = default_numa_node();
        __bf16 *new_buffer = static_cast<__bf16 *>(
            manual_hbmalloc_numa(alloc_elems * sizeof(__bf16), numa_node));
        if (new_buffer == nullptr) {
            return false;
        }

        if (buffer != nullptr) {
            manual_hbmfree(buffer, capacity * sizeof(__bf16));
        }
        buffer = new_buffer;
        capacity = alloc_elems;
        return true;
    }
};

thread_local BgemmWorkspace tls_workspace;

struct RingAttnHbmWorkspace {
    __bf16* score_primary = nullptr;
    __bf16* score_secondary = nullptr;
    __bf16* pack_primary = nullptr;
    __bf16* pack_secondary = nullptr;
    __bf16* pack_tertiary = nullptr;
    std::size_t score_primary_bytes = 0;
    std::size_t score_secondary_bytes = 0;
    std::size_t pack_primary_bytes = 0;
    std::size_t pack_secondary_bytes = 0;
    std::size_t pack_tertiary_bytes = 0;

    ~RingAttnHbmWorkspace()
    {
        release_slot(score_primary, score_primary_bytes);
        release_slot(score_secondary, score_secondary_bytes);
        release_slot(pack_primary, pack_primary_bytes);
        release_slot(pack_secondary, pack_secondary_bytes);
        release_slot(pack_tertiary, pack_tertiary_bytes);
    }

    static void release_slot(__bf16*& buffer, std::size_t& bytes)
    {
        if (buffer != nullptr) {
            manual_hbmfree(buffer, bytes);
            buffer = nullptr;
            bytes = 0;
        }
    }

    bool ensure_slot(__bf16*& buffer,
                     std::size_t& capacity_bytes,
                     std::size_t needed_bytes,
                     int numa_node)
    {
        if (capacity_bytes >= needed_bytes) {
            return true;
        }

        const std::size_t alloc_bytes = grow_workspace_bytes(needed_bytes);
        __bf16* new_buffer = static_cast<__bf16*>(
            manual_hbmalloc_numa(alloc_bytes, numa_node));
        if (new_buffer == nullptr) {
            return false;
        }

        release_slot(buffer, capacity_bytes);
        buffer = new_buffer;
        capacity_bytes = alloc_bytes;
        return true;
    }
};

RingAttnHbmWorkspace& ring_attn_hbm_workspace()
{
    static RingAttnHbmWorkspace workspace;
    return workspace;
}

constexpr int kMaxAbnormalPrints = 8;
constexpr float kAbnormalValueThreshold = 1.0e5f;

inline float debug_to_float(__bf16 value)
{
    return static_cast<float>(value);
}

inline float debug_to_float(float value)
{
    return value;
}

inline bool is_abnormal_value(float value)
{
    return !std::isfinite(value) || std::fabs(value) > kAbnormalValueThreshold;
}

template <typename T>
void report_abnormal_tensor_2d(const char *name,
                               const T *data,
                               int dim0,
                               int dim1)
{
    int printed = 0;
    const int total = dim0 * dim1;
    for (int idx = 0; idx < total; ++idx) {
        const float value = debug_to_float(data[idx]);
        if (!is_abnormal_value(value)) {
            continue;
        }
        if (printed == 0) {
            std::fprintf(stderr,
                         "[rank0][abnormal] %s contains abnormal values (non-finite or |x| > 1e10)\n",
                         name);
        }
        const int i0 = idx / dim1;
        const int i1 = idx % dim1;
        std::fprintf(stderr,
                     "[rank0][abnormal] %s[%d][%d] = %.9g\n",
                     name, i0, i1, value);
        ++printed;
        if (printed >= kMaxAbnormalPrints) {
            break;
        }
    }
    if (printed > 0) {
        std::fprintf(stderr,
                     "[rank0][abnormal] %s printed %d entries (max %d)\n",
                     name, printed, kMaxAbnormalPrints);
        std::fflush(stderr);
    }
}

template <typename T>
void report_abnormal_tensor_3d(const char *name,
                               const T *data,
                               int dim0,
                               int dim1,
                               int dim2)
{
    int printed = 0;
    const int plane = dim1 * dim2;
    const int total = dim0 * plane;
    for (int idx = 0; idx < total; ++idx) {
        const float value = debug_to_float(data[idx]);
        if (!is_abnormal_value(value)) {
            continue;
        }
        if (printed == 0) {
            std::fprintf(stderr,
                         "[rank0][abnormal] %s contains abnormal values (non-finite or |x| > 1e10)\n",
                         name);
        }
        const int i0 = idx / plane;
        const int rem = idx % plane;
        const int i1 = rem / dim2;
        const int i2 = rem % dim2;
        std::fprintf(stderr,
                     "[rank0][abnormal] %s[%d][%d][%d] = %.9g\n",
                     name, i0, i1, i2, value);
        ++printed;
        if (printed >= kMaxAbnormalPrints) {
            break;
        }
    }
    if (printed > 0) {
        std::fprintf(stderr,
                     "[rank0][abnormal] %s printed %d entries (max %d)\n",
                     name, printed, kMaxAbnormalPrints);
        std::fflush(stderr);
    }
}

bool bgemm_kernel(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE transA,
                      const enum CBLAS_TRANSPOSE transB,
                      const BLASINT m,
                      const BLASINT n,
                      const BLASINT k,
                      const __bf16 alpha,
                      __bf16 *a,
                      const BLASINT lda,
                      __bf16 *b,
                      const BLASINT ldb,
                      const __bf16 beta,
                      __bf16 *c,
                      const BLASINT ldc)
{
    if (order != CblasRowMajor) {
        std::fprintf(stderr, "bgemm_kernel only supports row-major order on cpu %d\n", sched_getcpu());
        return false;
    }

    const size_t component_packed_a_size = ring_attn_component::pack_get_size(CblasA, m, n, k);
    const size_t component_packed_b_size = ring_attn_component::pack_get_size(CblasB, m, n, k);
    if (component_packed_a_size == 0 || component_packed_b_size == 0) {
        std::fprintf(stderr,
                     "bgemm_kernel unsupported shape m=%d n=%d k=%d transA=%d transB=%d on cpu %d\n",
                     static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                     static_cast<int>(transA), static_cast<int>(transB), sched_getcpu());
        return false;
    }

    if (!tls_workspace.reserve(tls_workspace.packed_a, tls_workspace.size_a, component_packed_a_size)) {
        std::fprintf(stderr, "component packed_a allocation failed on cpu %d\n", sched_getcpu());
        return false;
    }
    if (!tls_workspace.reserve(tls_workspace.packed_b, tls_workspace.size_b, component_packed_b_size)) {
        std::fprintf(stderr, "component packed_b allocation failed on cpu %d\n", sched_getcpu());
        return false;
    }

    return ring_attn_component::bgemm_kernel(order, transA, transB,
                                             m, n, k,
                                             alpha, a, lda,
                                             b, ldb, beta, c, ldc,
                                             tls_workspace.packed_a,
                                             tls_workspace.packed_b);
}

bool sbgemm_kernel(const enum CBLAS_ORDER order,
                      const enum CBLAS_TRANSPOSE transA,
                      const enum CBLAS_TRANSPOSE transB,
                      const BLASINT m,
                      const BLASINT n,
                      const BLASINT k,
                      const __bf16 alpha,
                      __bf16 *a,
                      const BLASINT lda,
                      __bf16 *b,
                      const BLASINT ldb,
                      const __bf16 beta,
                      float *c,
                      const BLASINT ldc)
{
    if (order != CblasRowMajor) {
        std::fprintf(stderr, "sbgemm_kernel only supports row-major order on cpu %d\n", sched_getcpu());
        return false;
    }

    const size_t component_packed_a_size = ring_attn_component::pack_get_size(CblasA, m, n, k);
    const size_t component_packed_b_size = ring_attn_component::pack_get_size(CblasB, m, n, k);
    if (component_packed_a_size == 0 || component_packed_b_size == 0) {
        std::fprintf(stderr,
                     "sbgemm_kernel unsupported shape m=%d n=%d k=%d transA=%d transB=%d on cpu %d\n",
                     static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                     static_cast<int>(transA), static_cast<int>(transB), sched_getcpu());
        return false;
    }

    if (!tls_workspace.reserve(tls_workspace.packed_a, tls_workspace.size_a, component_packed_a_size)) {
        std::fprintf(stderr, "component packed_a allocation failed on cpu %d\n", sched_getcpu());
        return false;
    }
    if (!tls_workspace.reserve(tls_workspace.packed_b, tls_workspace.size_b, component_packed_b_size)) {
        std::fprintf(stderr, "component packed_b allocation failed on cpu %d\n", sched_getcpu());
        return false;
    }

    ring_attn_component::sbgemm_kernel_unchecked(transA, transB,
                                                 m, n, k,
                                                 alpha, a, lda,
                                                 b, ldb, beta, c, ldc,
                                                 tls_workspace.packed_a,
                                                 tls_workspace.packed_b);
    return true;
}

inline __bf16 quick_max(__bf16 *C, int t_num) {
    int step = svcnth();
    int half = svcntw();
    int i = 0;
    svfloat32_t vmax = svdup_f32(-INFINITY);
    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_b16();
    int t_new = t_num / step * step;
    int t_tail = t_num - t_new;
    constexpr int kPrefetchBlocks = 8;
    #pragma unroll 4
    for(i=0;i<t_new;i+=step){
        int pf_i = i + kPrefetchBlocks * step;
        if (pf_i < t_new) {
            __builtin_prefetch(C + pf_i, 0, 3);
            __builtin_prefetch(C + pf_i + step, 0, 3);
        }
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint32_t lo32 = svunpklo_u32(vb16);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        vmax = svmax_f32_x(pg32, vmax, v0);
        svuint32_t hi32 = svunpkhi_u32(vb16);
        hi32 = svlsl_n_u32_x(pg32, hi32, 16);
        svfloat32_t v1 = svreinterpret_f32_u32(hi32);
        vmax = svmax_f32_m(pg32, vmax, v1);
    }
    if(t_tail>half){
        svbool_t pg32_1 = svwhilelt_b32((uint64_t)0, (uint64_t)(t_tail-half));
        pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)t_tail);
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint32_t lo32 = svunpklo_u32(vb16);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        vmax = svmax_f32_m(pg32, vmax, v0);
        svuint32_t hi32 = svunpkhi_u32(vb16);
        hi32 = svlsl_n_u32_x(pg32_1, hi32, 16);
        svfloat32_t v1 = svreinterpret_f32_u32(hi32);
        vmax = svmax_f32_m(pg32_1, vmax, v1);
    }else if(t_tail>0){
        pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)t_tail);
        pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)t_tail);
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint32_t lo32 = svunpklo_u32(vb16);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        vmax = svmax_f32_m(pg32, vmax, v0);
    }
    __bf16 ans = (__bf16)svmaxv_f32(svptrue_b32(), vmax);
    return ans;
}

inline float sve_exp(__bf16* C, float temp, int t_num, int rank) {
    int i = 0;
    int step = svcnth();
    svfloat32_t vacc = svdup_n_f32(0.0f);
    svfloat32_t temp1 = svdup_n_f32(temp);
    svfloat32_t log2e = svdup_n_f32(1.4426950408889634f);
    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_b16();
    int t_new = t_num / step * step;
    int t_tail = t_num - t_new;

    #pragma unroll 6
    for (; i < t_new; i += step) {
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint16_t vb_sign = svuzp1_u16(vb16, vb16);
        svuint32_t lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        svfloat32_t va = svrintz_f32_x(pg32, v0);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        svuint32_t vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        vacc = svadd_f32_x(pg32, vacc, v0);
        svbfloat16_t out = svcvt_bf16_x(pg32, v0);

        vb_sign = svuzp2_u16(vb16, vb16);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        va = svrintz_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        vacc = svadd_f32_x(pg32, vacc, v0);
        out = svcvtnt_bf16_x(out, pg32, v0);
        svst1_bf16(pg16, &C[i], out);

    }
    if(t_tail!=0){
        pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)t_tail/2);
        pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)t_tail);
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint16_t vb_sign = svuzp1_u16(vb16, vb16);
        svuint32_t lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        svfloat32_t va = svcvt_f32_u32_x(pg32, lo32);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        svuint32_t vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        vacc = svadd_f32_m(pg32, vacc, v0);
        svbfloat16_t out = svcvt_bf16_x(pg32, v0);

        vb_sign = svuzp2_u16(vb16, vb16);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        va = svcvt_f32_u32_x(pg32, lo32);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        vacc = svadd_f32_m(pg32, vacc, v0);
        out = svcvtnt_bf16_m(out, pg32, v0);
        svst1_bf16(pg16, &C[i], out);
    }
    return svaddv_f32(svptrue_b32(), vacc);
}

inline void sve_out(int h_len, float *out1, float correction){
    int half = svcntw();
    int h_new = h_len / half * half;
    int h_tail = h_len - h_new;
    int i;
    svbool_t pg32 = svptrue_b32();
    #pragma unroll 6
    for(i=0;i<h_new;i+=half){
        svfloat32_t vout = svld1_f32(pg32, &out1[i]);
        vout = svmul_n_f32_x(pg32, vout, correction);
        svst1_f32(pg32, &out1[i], vout);
    }
    if(h_tail!=0){
        pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)h_tail);
        svfloat32_t vout = svld1_f32(pg32, &out1[i]);
        vout = svmul_n_f32_x(pg32, vout, correction);
        svst1_f32(pg32, &out1[i], vout);
    }
}

__attribute__((noinline)) void sve_exp_li(__bf16* C, __bf16* dp, float temp, float li, int t_num, float D, float scale) {
    int i = 0;
    int step = svcnth();
    svfloat32_t temp1 = svdup_n_f32(temp);
    svfloat32_t log2e = svdup_n_f32(1.4426950408889634f);
    svfloat32_t li1 = svdup_n_f32(li);
    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_b16();
    int t_new = t_num / step * step;
    int t_tail = t_num - t_new;

    #pragma unroll 4
    for (; i < t_new; i += step) {
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint16_t vbdp = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&dp[i]));
        svuint16_t vb_sign = svuzp1_u16(vb16, vb16);
        svuint32_t lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        svfloat32_t va = svrintz_f32_x(pg32, v0);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        svuint32_t vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        v0 = svdiv_f32_x(pg32, v0, li1);

        vb_sign = svuzp1_u16(vbdp, vbdp);
        lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t vdp = svreinterpret_f32_u32(lo32);
        vdp = svadd_n_f32_x(pg32, vdp, -D);
        vdp = svmul_f32_x(pg32, v0, vdp);
        vdp = svmul_n_f32_x(pg32, vdp, scale);

        svbfloat16_t out = svcvt_bf16_x(pg32, v0);
        svbfloat16_t out_vdp = svcvt_bf16_x(pg32, vdp);

        vb_sign = svuzp2_u16(vb16, vb16);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        va = svrintz_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        v0 = svdiv_f32_x(pg32, v0, li1);
        out = svcvtnt_bf16_m(out, pg32, v0);

        vb_sign = svuzp2_u16(vbdp, vbdp);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        vdp = svreinterpret_f32_u32(lo32);
        vdp = svsub_n_f32_x(pg32, vdp, D);
        vdp = svmul_f32_x(pg32, v0, vdp);
        vdp = svmul_n_f32_x(pg32, vdp, scale);
        out_vdp = svcvtnt_bf16_m(out_vdp, pg32, vdp);
        

        svst1_bf16(pg16, &C[i], out);
        svst1_bf16(pg16, &dp[i], out_vdp);
    }
    if(t_tail!=0){
        pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)t_tail/2);
        pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)t_tail);
        svuint16_t vb16 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&C[i]));
        svuint16_t vbdp = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&dp[i]));
        svuint16_t vb_sign = svuzp1_u16(vb16, vb16);
        svuint32_t lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);

        svfloat32_t va = svrintz_f32_x(pg32, v0);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        svuint32_t vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        v0 = svdiv_f32_x(pg32, v0, li1);

        vb_sign = svuzp1_u16(vbdp, vbdp);
        lo32 = svunpklo_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        svfloat32_t vdp = svreinterpret_f32_u32(lo32);
        vdp = svadd_n_f32_x(pg32, vdp, -D);
        vdp = svmul_f32_x(pg32, v0, vdp);
        vdp = svmul_n_f32_x(pg32, vdp, scale);

        svbfloat16_t out = svcvt_bf16_x(pg32, v0);
        svbfloat16_t out_vdp = svcvt_bf16_x(pg32, vdp);

        vb_sign = svuzp2_u16(vb16, vb16);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        v0 = svreinterpret_f32_u32(lo32);
        v0 = svsub_f32_x(pg32, v0, temp1);
        v0 = svmul_f32_x(pg32, v0, log2e);
        v0 = svadd_n_f32_x(pg32, v0, 127.0f);
        lo32 = svcvt_u32_f32_x(pg32, v0);
        va = svrintz_f32_x(pg32, v0);
        v0 = svsub_f32_x(pg32, v0, va);
        v0 = svmul_n_f32_x(pg32, v0, 64.0f);
        v0 = svadd_n_f32_x(pg32, v0, 0.5f);
        vj = svcvt_u32_f32_x(pg32, v0);
        vj = svadd_u32_x(pg32, svlsl_n_u32_x(pg32, lo32, 6), vj);
        v0 = svexpa_f32(vj);
        v0 = svdiv_f32_x(pg32, v0, li1);
        out = svcvtnt_bf16_m(out, pg32, v0);

        vb_sign = svuzp2_u16(vbdp, vbdp);
        lo32 = svunpkhi_u32(vb_sign);
        lo32 = svlsl_n_u32_x(pg32, lo32, 16);
        vdp = svreinterpret_f32_u32(lo32);
        vdp = svadd_n_f32_x(pg32, vdp, -D);
        vdp = svmul_f32_x(pg32, v0, vdp);
        vdp = svmul_n_f32_x(pg32, vdp, scale);
        out_vdp = svcvtnt_bf16_m(out_vdp, pg32, vdp);
        

        svst1_bf16(pg16, &C[i], out);
        svst1_bf16(pg16, &dp[i], out_vdp);
    }
}

void destroy_ring_die_shared_memory();

namespace {

constexpr int kMaxRanksPerDie = 8;
constexpr int kMaxChunksPerTensor = 8;
constexpr std::size_t kHugePageBytes = 2ULL * 1024ULL * 1024ULL;
constexpr std::size_t kSharedChunkBytes = 32ULL * 1024ULL * 1024ULL;
constexpr std::size_t kSharedTensorBytes =
    kSharedChunkBytes * kMaxChunksPerTensor;
constexpr std::size_t kSharedTensorBf16Count =
    kSharedTensorBytes / sizeof(__bf16);
constexpr std::size_t kSharedChunkBf16Count =
    kSharedChunkBytes / sizeof(__bf16);
constexpr unsigned int kSharedMagic = 0x52494E47U;

enum SharedTensorLayout : int {
    kSharedLayoutRawBf16 = 0,
    kSharedLayoutBgemmPackB = 1,
    kSharedLayoutSbgemmPackB = 2,
};

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

struct SharedHeader {
    unsigned int magic;
    int ready;
    int die_id;
    int sub_size;
    int using_huge_pages;
    int chunk_count;
    int tensor_t_num;
    int tensor_h_num;
    int tensor_h_len;
    int out_head_contiguous;
    int k_layout;
    int v_layout;
    std::size_t mapping_bytes;
    std::size_t k_offset_bytes;
    std::size_t v_offset_bytes;
    std::size_t k_capacity_bytes;
    std::size_t v_capacity_bytes;
    std::size_t chunk_capacity_bytes;
    std::size_t k_capacity_count;
    std::size_t v_capacity_count;
    std::size_t chunk_capacity_bf16;
    std::size_t raw_tensor_bytes;
    std::size_t raw_head_bytes;
    std::size_t q_head_bytes;
    std::size_t q_tensor_bytes;
    std::size_t k_head_bytes;
    std::size_t v_head_bytes;
    std::size_t k_chunk_bytes;
    std::size_t v_chunk_bytes;
    std::size_t bwd_grad_out_bytes;
    std::size_t bwd_stats_bytes;
    std::size_t bwd_m_offset_bytes;
    std::size_t bwd_l_offset_bytes;
    std::size_t bwd_d_offset_bytes;
    std::size_t bwd_q_pack_a_head_bytes;
    std::size_t bwd_q_pack_b_head_bytes;
    std::size_t bwd_q_pack_a_tensor_bytes;
    std::size_t bwd_q_pack_b_tensor_bytes;
    std::size_t bwd_q_pack_b_offset_bytes;
    std::size_t bwd_grad_pack_a_head_bytes;
    std::size_t bwd_grad_pack_b_head_bytes;
    std::size_t bwd_grad_pack_a_tensor_bytes;
    std::size_t bwd_grad_pack_b_tensor_bytes;
    std::size_t bwd_grad_pack_b_offset_bytes;
    int pack_epoch;
    int chunk_ready[kMaxChunksPerTensor];
    int bwd_epoch;
    int bwd_partial_ready[kMaxChunksPerTensor][kMaxRanksPerDie];
    int bwd_reduce_done[kMaxChunksPerTensor];
    char shm_name[32];
};

constexpr std::size_t kControlMappingBytes = align_up(sizeof(SharedHeader), 4096);
struct RingDieShmHandle {
    MPI_Comm sub_comm = MPI_COMM_NULL;
    int world_rank = -1;
    int sub_rank = -1;
    int sub_size = 0;
    int die_id = -1;
    int fd = -1;
    void* base = nullptr;
    SharedHeader* header = nullptr;
    __bf16* k = nullptr;
    __bf16* v = nullptr;
    int k_chunk_ids[kMaxChunksPerTensor] = {};
    int v_chunk_ids[kMaxChunksPerTensor] = {};
    void* k_chunk_bases[kMaxChunksPerTensor] = {};
    void* v_chunk_bases[kMaxChunksPerTensor] = {};
    __bf16* k_chunks[kMaxChunksPerTensor] = {};
    __bf16* v_chunks[kMaxChunksPerTensor] = {};
    bool owner = false;
    bool initialized = false;
    bool using_huge_pages = false;
    char shm_name[32] = {};
};

struct RingDieBwdScratchHandle {
    MPI_Win dq_win = MPI_WIN_NULL;
    __bf16* dq_slots[kMaxRanksPerDie] = {};
    std::size_t dq_slot_bytes = 0;
    int sub_size = 0;
    bool initialized = false;
};

[[noreturn]] void abort_with_errno(const char* where, MPI_Comm comm) {
    std::perror(where);
    MPI_Abort(comm, 1);
    std::exit(1);
}

[[noreturn]] void abort_with_message(const char* message, MPI_Comm comm) {
    std::cerr << message << std::endl;
    MPI_Abort(comm, 1);
    std::exit(1);
}

RingDieShmHandle& ring_die_shm_handle() {
    static RingDieShmHandle handle;
    return handle;
}

RingDieBwdScratchHandle& ring_die_bwd_scratch_handle() {
    static RingDieBwdScratchHandle handle;
    return handle;
}

void bind_control_view(RingDieShmHandle& handle) {
    handle.header = static_cast<SharedHeader*>(handle.base);
}

void* map_file_region(int fd,
                      std::size_t bytes,
                      bool prefer_huge_pages,
                      bool* using_huge_pages) {
    *using_huge_pages = false;

#if defined(MAP_HUGETLB)
    int huge_flags = MAP_SHARED | MAP_HUGETLB;
#if defined(MAP_HUGE_2MB)
    huge_flags |= MAP_HUGE_2MB;
#endif
    if (prefer_huge_pages) {
        void* huge_addr = mmap(
            nullptr,
            bytes,
            PROT_READ | PROT_WRITE,
            huge_flags,
            fd,
            0
        );
        if (huge_addr != MAP_FAILED) {
            *using_huge_pages = true;
            return huge_addr;
        }
    }
#endif

    void* addr = mmap(
        nullptr,
        bytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    );
    if (addr == MAP_FAILED) {
        return nullptr;
    }

#if defined(MADV_HUGEPAGE)
    if (prefer_huge_pages) {
        (void)madvise(addr, bytes, MADV_HUGEPAGE);
    }
#endif
    return addr;
}

void bind_shared_region_to_hbm(void* addr,
                               std::size_t bytes,
                               int numa_id) {
    const unsigned long mask =
        1ul << static_cast<unsigned long>(numa_id + numa_offset);
    if (mbind(addr, bytes, MPOL_BIND, &mask, 64, 0) != 0) {
        std::fprintf(stderr,
                     "shared mbind failed (numa=%d): errno=%d (%s)\n",
                     numa_id,
                     errno,
                     std::strerror(errno));
    }
}

std::uint32_t shared_job_key_component() {
    static const char* kJobIdEnvNames[] = {
        "SLURM_JOB_ID",
        "CCS_JOB_ID",
        "BATCH_JOB_ID"
    };
    for (const char* env_name : kJobIdEnvNames) {
        const char* job_id = std::getenv(env_name);
        if (job_id != nullptr && job_id[0] != '\0') {
            return static_cast<std::uint32_t>(std::strtoul(job_id, nullptr, 10));
        }
    }
    return static_cast<std::uint32_t>(getuid());
}

void format_control_shm_name(char* dst, std::size_t dst_size, int die_id) {
    std::snprintf(dst,
                  dst_size,
                  "/ring_%u_die%d_ctrl",
                  shared_job_key_component(),
                  die_id);
}

key_t make_chunk_shm_key(int die_id, char kind, int chunk_idx) {
    std::uint32_t key = 0x4a000001u;
    key ^= shared_job_key_component() * 2654435761u;
    key ^= static_cast<std::uint32_t>(die_id + 1) * 2246822519u;
    key ^= static_cast<std::uint32_t>((kind == 'k') ? 0x13579bdu : 0x2468aceu);
    key ^= static_cast<std::uint32_t>(chunk_idx + 1) * 3266489917u;
    key &= 0x7fffffffu;
    if (key == static_cast<std::uint32_t>(IPC_PRIVATE)) {
        key += 1u;
    }
    return static_cast<key_t>(key);
}

int chunk_shm_flags() {
    int flags = 0666;
#ifdef SHM_HUGETLB
    flags |= SHM_HUGETLB;
#endif
#ifdef SHM_HUGE_2MB
    flags |= SHM_HUGE_2MB;
#endif
    return flags;
}

int chunk_shm_fallback_flags() {
    return 0666;
}

int create_or_open_chunk_segment(key_t key,
                                 std::size_t bytes,
                                 bool create,
                                 const char* what,
                                 MPI_Comm comm) {
    const int flag_candidates[2] = {
        chunk_shm_flags(),
        chunk_shm_fallback_flags()
    };
    const int num_candidates =
        (flag_candidates[0] == flag_candidates[1]) ? 1 : 2;

    int last_errno = 0;
    for (int idx = 0; idx < num_candidates; ++idx) {
        const int flags = flag_candidates[idx];
        if (create) {
            int shmid = shmget(key,
                               bytes,
                               IPC_CREAT | IPC_EXCL | flags);
            if (shmid != -1) {
                return shmid;
            }
            if (errno != EEXIST) {
                last_errno = errno;
                continue;
            }
        }

        int shmid = shmget(key, bytes, flags);
        if (shmid != -1) {
            return shmid;
        }
        last_errno = errno;
    }

    errno = last_errno;
    abort_with_errno(what, comm);
}

void* attach_chunk_segment(int shmid, const char* what, MPI_Comm comm) {
    void* addr = shmat(shmid, nullptr, 0);
    if (addr == reinterpret_cast<void*>(-1)) {
        abort_with_errno(what, comm);
    }
    return addr;
}

void initialize_ring_die_shared_memory(int requested_sub_size = 0);

void validate_shared_sub_size(int requested_sub_size, MPI_Comm comm) {
    if (requested_sub_size != 2 &&
        requested_sub_size != 4 &&
        requested_sub_size != 8) {
        abort_with_message("sp_size must be one of 2, 4, or 8", comm);
    }
}

__bf16* get_chunk_ptr(__bf16* base, int chunk_idx, MPI_Comm comm) {
    RingDieShmHandle& handle = ring_die_shm_handle();
    const int chunk_count =
        (handle.header != nullptr) ? handle.header->chunk_count
                                   : kMaxChunksPerTensor;
    if (chunk_idx < 0 || chunk_idx >= chunk_count) {
        abort_with_message("chunk index out of range", comm);
    }
    return base + static_cast<std::size_t>(chunk_idx) * kSharedChunkBf16Count;
}

char* get_chunk_byte_ptr(void* base, int chunk_idx, MPI_Comm comm) {
    RingDieShmHandle& handle = ring_die_shm_handle();
    const int chunk_count =
        (handle.header != nullptr) ? handle.header->chunk_count
                                   : kMaxChunksPerTensor;
    if (chunk_idx < 0 || chunk_idx >= chunk_count) {
        abort_with_message("chunk index out of range", comm);
    }
    return static_cast<char*>(base) +
           static_cast<std::size_t>(chunk_idx) * kSharedChunkBytes;
}

int next_shared_pack_epoch(int current_epoch);
int current_shared_pack_epoch(const SharedHeader* header);
void mark_shared_chunk_ready(SharedHeader* header,
                             int chunk_idx,
                             int epoch,
                             MPI_Comm comm);
void wait_for_shared_chunk_ready(const SharedHeader* header,
                                 int chunk_idx,
                                 int epoch,
                                 MPI_Comm comm);
void mark_shared_bwd_partial_ready(SharedHeader* header,
                                   int chunk_idx,
                                   int rank,
                                   int epoch,
                                   MPI_Comm comm);
void wait_for_all_shared_bwd_partials(const SharedHeader* header,
                                      int chunk_idx,
                                      int sub_size,
                                      int epoch,
                                      MPI_Comm comm);
void mark_shared_bwd_chunk_done(SharedHeader* header,
                                int chunk_idx,
                                int epoch,
                                MPI_Comm comm);
void wait_for_shared_bwd_chunk_done(const SharedHeader* header,
                                    int chunk_idx,
                                    int epoch,
                                    MPI_Comm comm);

void configure_shared_packed_layout(int t_num, int h_num, int h_len) {
    RingDieShmHandle& handle = ring_die_shm_handle();
    SharedHeader* header = handle.header;
    const std::size_t raw_head_bytes =
        static_cast<std::size_t>(t_num) * h_len * sizeof(__bf16);
    const std::size_t raw_tensor_bytes = raw_head_bytes * h_num;
    const std::size_t q_head_bytes =
        ring_attn_component::pack_get_size(CblasA, t_num, t_num, h_len) *
        sizeof(__bf16);
    const std::size_t k_head_bytes =
        ring_attn_component::pack_get_size(CblasB, t_num, t_num, h_len) *
        sizeof(__bf16);
    const std::size_t v_head_bytes =
        ring_attn_component::pack_get_size(CblasB, t_num, h_len, t_num) *
        sizeof(__bf16);
    const std::size_t q_tensor_bytes = q_head_bytes * h_num;
    const std::size_t k_chunk_bytes = k_head_bytes * h_num;
    const std::size_t v_chunk_bytes = v_head_bytes * h_num;

    if (handle.sub_rank == 0) {
        header->tensor_t_num = t_num;
        header->tensor_h_num = h_num;
        header->tensor_h_len = h_len;
        header->out_head_contiguous = 1;
        header->k_layout = kSharedLayoutBgemmPackB;
        header->v_layout = kSharedLayoutSbgemmPackB;
        header->raw_tensor_bytes = raw_tensor_bytes;
        header->raw_head_bytes = raw_head_bytes;
        header->q_head_bytes = q_head_bytes;
        header->q_tensor_bytes = q_tensor_bytes;
        header->k_head_bytes = k_head_bytes;
        header->v_head_bytes = v_head_bytes;
        header->k_chunk_bytes = k_chunk_bytes;
        header->v_chunk_bytes = v_chunk_bytes;
        header->pack_epoch = next_shared_pack_epoch(header->pack_epoch);
        for (int i = 0; i < header->chunk_count; ++i) {
            header->chunk_ready[i] = 0;
        }
    }

    MPI_Barrier(handle.sub_comm);
    // if (header->tensor_t_num != t_num) {
    //     abort_with_message("shared header packed tensor metadata mismatch", handle.sub_comm);
    // }
    if (header->k_chunk_bytes > header->chunk_capacity_bytes ||
        header->v_chunk_bytes > header->chunk_capacity_bytes) {
        abort_with_message("packed K/V chunk bytes exceed shared chunk capacity",
                           handle.sub_comm);
    }
}

void configure_shared_raw_layout(int t_num, int h_num, int h_len) {
    RingDieShmHandle& handle = ring_die_shm_handle();
    SharedHeader* header = handle.header;
    const std::size_t raw_head_bytes =
        static_cast<std::size_t>(t_num) * h_len * sizeof(__bf16);
    const std::size_t raw_tensor_bytes = raw_head_bytes * h_num;
    const std::size_t stats_bytes =
        static_cast<std::size_t>(t_num) * h_num * sizeof(float);
    const std::size_t q_pack_a_head_bytes =
        cblas_bgemm_pack_get_size(CblasA, t_num, t_num, h_len);
    const std::size_t q_pack_b_head_bytes =
        cblas_bgemm_pack_get_size(CblasB, t_num, h_len, t_num);
    const std::size_t grad_pack_a_head_bytes =
        cblas_bgemm_pack_get_size(CblasA, t_num, t_num, h_len);
    const std::size_t grad_pack_b_head_bytes =
        cblas_bgemm_pack_get_size(CblasB, t_num, h_len, t_num);
    const std::size_t q_pack_a_tensor_bytes =
        q_pack_a_head_bytes * static_cast<std::size_t>(h_num);
    const std::size_t q_pack_b_tensor_bytes =
        q_pack_b_head_bytes * static_cast<std::size_t>(h_num);
    const std::size_t q_pack_b_offset_bytes =
        align_up(q_pack_a_tensor_bytes, 64);
    const std::size_t q_chunk_bytes =
        q_pack_b_offset_bytes + q_pack_b_tensor_bytes;
    const std::size_t grad_pack_a_tensor_bytes =
        grad_pack_a_head_bytes * static_cast<std::size_t>(h_num);
    const std::size_t grad_pack_b_tensor_bytes =
        grad_pack_b_head_bytes * static_cast<std::size_t>(h_num);
    const std::size_t grad_pack_b_offset_bytes =
        align_up(grad_pack_a_tensor_bytes, 64);
    const std::size_t m_offset_bytes =
        align_up(grad_pack_b_offset_bytes + grad_pack_b_tensor_bytes, 64);
    const std::size_t l_offset_bytes = align_up(m_offset_bytes + stats_bytes, 64);
    const std::size_t d_offset_bytes = align_up(l_offset_bytes + stats_bytes, 64);
    const std::size_t grad_out_chunk_bytes = d_offset_bytes + stats_bytes;

    if (handle.sub_rank == 0) {
        header->tensor_t_num = t_num;
        header->tensor_h_num = h_num;
        header->tensor_h_len = h_len;
        header->out_head_contiguous = 1;
        header->k_layout = kSharedLayoutRawBf16;
        header->v_layout = kSharedLayoutRawBf16;
        header->raw_tensor_bytes = raw_tensor_bytes;
        header->raw_head_bytes = raw_head_bytes;
        header->q_head_bytes = q_pack_a_head_bytes;
        header->q_tensor_bytes = q_pack_a_tensor_bytes;
        header->k_head_bytes = raw_head_bytes;
        header->v_head_bytes = raw_head_bytes;
        header->k_chunk_bytes = q_chunk_bytes;
        header->v_chunk_bytes = grad_out_chunk_bytes;
        header->bwd_grad_out_bytes = raw_tensor_bytes;
        header->bwd_stats_bytes = stats_bytes;
        header->bwd_m_offset_bytes = m_offset_bytes;
        header->bwd_l_offset_bytes = l_offset_bytes;
        header->bwd_d_offset_bytes = d_offset_bytes;
        header->bwd_q_pack_a_head_bytes = q_pack_a_head_bytes;
        header->bwd_q_pack_b_head_bytes = q_pack_b_head_bytes;
        header->bwd_q_pack_a_tensor_bytes = q_pack_a_tensor_bytes;
        header->bwd_q_pack_b_tensor_bytes = q_pack_b_tensor_bytes;
        header->bwd_q_pack_b_offset_bytes = q_pack_b_offset_bytes;
        header->bwd_grad_pack_a_head_bytes = grad_pack_a_head_bytes;
        header->bwd_grad_pack_b_head_bytes = grad_pack_b_head_bytes;
        header->bwd_grad_pack_a_tensor_bytes = grad_pack_a_tensor_bytes;
        header->bwd_grad_pack_b_tensor_bytes = grad_pack_b_tensor_bytes;
        header->bwd_grad_pack_b_offset_bytes = grad_pack_b_offset_bytes;
        header->pack_epoch = next_shared_pack_epoch(header->pack_epoch);
        header->bwd_epoch = next_shared_pack_epoch(header->bwd_epoch);
        for (int i = 0; i < header->chunk_count; ++i) {
            header->chunk_ready[i] = 0;
            header->bwd_reduce_done[i] = 0;
            for (int rank = 0; rank < header->sub_size; ++rank) {
                header->bwd_partial_ready[i][rank] = 0;
            }
        }
    }

    MPI_Barrier(handle.sub_comm);
    // MPI_Win_sync(header->tensor_t_num);
    // if (header->tensor_t_num != t_num) {
    //     abort_with_message("shared header raw tensor metadata mismatch", handle.sub_comm);
    // }
    if (header->k_chunk_bytes > header->chunk_capacity_bytes ||
        header->v_chunk_bytes > header->chunk_capacity_bytes) {
        abort_with_message("query-side shared chunk bytes exceed shared chunk capacity",
                           handle.sub_comm);
    }
}

char* get_packed_head_ptr(__bf16* chunk_base,
                          std::size_t head_bytes,
                          int head_idx,
                          int h_num,
                          MPI_Comm comm) {
    if (head_idx < 0 || head_idx >= h_num) {
        abort_with_message("head index out of range", comm);
    }
    return reinterpret_cast<char*>(chunk_base) +
           static_cast<std::size_t>(head_idx) * head_bytes;
}

int next_shared_pack_epoch(int current_epoch) {
    if (current_epoch < 0 || current_epoch == std::numeric_limits<int>::max()) {
        return 1;
    }
    return current_epoch + 1;
}

int current_shared_pack_epoch(const SharedHeader* header) {
    return __atomic_load_n(&header->pack_epoch, __ATOMIC_ACQUIRE);
}

void mark_shared_chunk_ready(SharedHeader* header,
                             int chunk_idx,
                             int epoch,
                             MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count) {
        abort_with_message("ready chunk index out of range", comm);
    }
    __atomic_store_n(&header->chunk_ready[chunk_idx], epoch, __ATOMIC_RELEASE);
}

void wait_for_shared_chunk_ready(const SharedHeader* header,
                                 int chunk_idx,
                                 int epoch,
                                 MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count) {
        abort_with_message("wait chunk index out of range", comm);
    }

    int spin = 0;
    while (__atomic_load_n(&header->chunk_ready[chunk_idx], __ATOMIC_ACQUIRE) !=
           epoch) {
        if ((++spin & 0x3ff) == 0) {
            sched_yield();
        }
    }
}

void mark_shared_bwd_partial_ready(SharedHeader* header,
                                   int chunk_idx,
                                   int rank,
                                   int epoch,
                                   MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count ||
        rank < 0 || rank >= header->sub_size) {
        abort_with_message("backward ready index out of range", comm);
    }
    __atomic_store_n(&header->bwd_partial_ready[chunk_idx][rank],
                     epoch,
                     __ATOMIC_RELEASE);
}

void wait_for_all_shared_bwd_partials(const SharedHeader* header,
                                      int chunk_idx,
                                      int sub_size,
                                      int epoch,
                                      MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count) {
        abort_with_message("backward wait chunk index out of range", comm);
    }
    for (int rank = 0; rank < sub_size; ++rank) {
        int spin = 0;
        while (__atomic_load_n(&header->bwd_partial_ready[chunk_idx][rank],
                               __ATOMIC_ACQUIRE) != epoch) {
            if ((++spin & 0x3ff) == 0) {
                sched_yield();
            }
        }
    }
}

void mark_shared_bwd_chunk_done(SharedHeader* header,
                                int chunk_idx,
                                int epoch,
                                MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count) {
        abort_with_message("backward done chunk index out of range", comm);
    }
    __atomic_store_n(&header->bwd_reduce_done[chunk_idx], epoch, __ATOMIC_RELEASE);
}

void wait_for_shared_bwd_chunk_done(const SharedHeader* header,
                                    int chunk_idx,
                                    int epoch,
                                    MPI_Comm comm) {
    if (chunk_idx < 0 || chunk_idx >= header->chunk_count) {
        abort_with_message("backward done wait chunk index out of range", comm);
    }
    int spin = 0;
    while (__atomic_load_n(&header->bwd_reduce_done[chunk_idx], __ATOMIC_ACQUIRE) !=
           epoch) {
        if ((++spin & 0x3ff) == 0) {
            sched_yield();
        }
    }
}

void destroy_ring_die_backward_scratch() {
    RingDieBwdScratchHandle& handle = ring_die_bwd_scratch_handle();
    if (handle.dq_win != MPI_WIN_NULL) {
        MPI_Win_free(&handle.dq_win);
    }
    handle = RingDieBwdScratchHandle{};
}

void initialize_ring_die_backward_scratch(std::size_t dq_slot_bytes,
                                          MPI_Comm sub_comm) {
    RingDieBwdScratchHandle& handle = ring_die_bwd_scratch_handle();
    if (handle.initialized && handle.dq_slot_bytes >= dq_slot_bytes) {
        return;
    }

    if (handle.initialized) {
        destroy_ring_die_backward_scratch();
    }

    MPI_Comm_size(sub_comm, &handle.sub_size);
    if (handle.sub_size > kMaxRanksPerDie) {
        abort_with_message("backward scratch sub_size exceeds supported maximum",
                           sub_comm);
    }
    void* local_dq = nullptr;
    MPI_Win_allocate_shared(static_cast<MPI_Aint>(dq_slot_bytes), 1, MPI_INFO_NULL,
                            sub_comm, &local_dq, &handle.dq_win);

    for (int rank = 0; rank < handle.sub_size; ++rank) {
        MPI_Aint size = 0;
        int disp_unit = 0;
        void* base = nullptr;
        MPI_Win_shared_query(handle.dq_win, rank, &size, &disp_unit, &base);
        handle.dq_slots[rank] = static_cast<__bf16*>(base);
    }

    handle.dq_slot_bytes = dq_slot_bytes;
    handle.initialized = true;
}

void reduce_shared_bwd_slot_to_owner(__bf16* dst,
                                     __bf16* const* slots,
                                     int num_slots,
                                     std::size_t elem_count) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < elem_count; ++i) {
        float sum = 0.0f;
        for (int rank = 0; rank < num_slots; ++rank) {
            sum += static_cast<float>(slots[rank][i]);
        }
        dst[i] = static_cast<__bf16>(sum);
    }
}

void initialize_ring_die_shared_memory(int requested_sub_size) {
    RingDieShmHandle& handle = ring_die_shm_handle();
    if (handle.initialized) {
        if (requested_sub_size == 0 || handle.sub_size == requested_sub_size) {
            return;
        }
        destroy_ring_die_shared_memory();
    }

    if (requested_sub_size == 0) {
        abort_with_message("shared memory must be initialized with sp_size first",
                           MPI_COMM_WORLD);
    }
    validate_shared_sub_size(requested_sub_size, MPI_COMM_WORLD);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        abort_with_message(
            "MPI must be initialized before using ring_die shared memory",
            MPI_COMM_WORLD
        );
    }

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size % requested_sub_size != 0) {
        abort_with_message("world size must be divisible by sp_size",
                           MPI_COMM_WORLD);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &handle.world_rank);
    handle.die_id = handle.world_rank / requested_sub_size;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        handle.die_id,
        handle.world_rank,
        &handle.sub_comm
    );

    MPI_Comm_rank(handle.sub_comm, &handle.sub_rank);
    MPI_Comm_size(handle.sub_comm, &handle.sub_size);

    format_control_shm_name(handle.shm_name, sizeof(handle.shm_name), handle.die_id);

    if (handle.sub_rank == 0) {
        handle.owner = true;
        handle.fd = shm_open(handle.shm_name, O_CREAT | O_RDWR, 0666);
        if (handle.fd == -1) {
            abort_with_errno("shm_open(create)", MPI_COMM_WORLD);
        }

        if (ftruncate(handle.fd, static_cast<off_t>(kControlMappingBytes)) == -1) {
            abort_with_errno("ftruncate", MPI_COMM_WORLD);
        }

        handle.base = map_file_region(
            handle.fd, kControlMappingBytes, false, &handle.using_huge_pages);
        if (handle.base == nullptr) {
            abort_with_errno("mmap(create)", MPI_COMM_WORLD);
        }

        bind_control_view(handle);
        std::memset(handle.base, 0, kControlMappingBytes);

        handle.header->magic = kSharedMagic;
        handle.header->ready = 1;
        handle.header->die_id = handle.die_id;
        handle.header->sub_size = handle.sub_size;
        handle.header->using_huge_pages = handle.using_huge_pages ? 1 : 0;
        handle.header->chunk_count = handle.sub_size;
        handle.header->tensor_t_num = 0;
        handle.header->tensor_h_num = 0;
        handle.header->tensor_h_len = 0;
        handle.header->out_head_contiguous = 0;
        handle.header->k_layout = kSharedLayoutRawBf16;
        handle.header->v_layout = kSharedLayoutRawBf16;
        handle.header->mapping_bytes =
            kControlMappingBytes +
            2 * static_cast<std::size_t>(handle.sub_size) * kSharedChunkBytes;
        handle.header->k_offset_bytes = 0;
        handle.header->v_offset_bytes = 0;
        handle.header->k_capacity_bytes =
            static_cast<std::size_t>(handle.sub_size) * kSharedChunkBytes;
        handle.header->v_capacity_bytes =
            static_cast<std::size_t>(handle.sub_size) * kSharedChunkBytes;
        handle.header->chunk_capacity_bytes = kSharedChunkBytes;
        handle.header->k_capacity_count =
            static_cast<std::size_t>(handle.sub_size) * kSharedChunkBf16Count;
        handle.header->v_capacity_count =
            static_cast<std::size_t>(handle.sub_size) * kSharedChunkBf16Count;
        handle.header->chunk_capacity_bf16 = kSharedChunkBf16Count;
        handle.header->raw_tensor_bytes = 0;
        handle.header->raw_head_bytes = 0;
        handle.header->q_head_bytes = 0;
        handle.header->q_tensor_bytes = 0;
        handle.header->k_head_bytes = 0;
        handle.header->v_head_bytes = 0;
        handle.header->k_chunk_bytes = 0;
        handle.header->v_chunk_bytes = 0;
        handle.header->bwd_grad_out_bytes = 0;
        handle.header->bwd_stats_bytes = 0;
        handle.header->bwd_m_offset_bytes = 0;
        handle.header->bwd_l_offset_bytes = 0;
        handle.header->bwd_d_offset_bytes = 0;
        handle.header->bwd_q_pack_a_head_bytes = 0;
        handle.header->bwd_q_pack_b_head_bytes = 0;
        handle.header->bwd_q_pack_a_tensor_bytes = 0;
        handle.header->bwd_q_pack_b_tensor_bytes = 0;
        handle.header->bwd_q_pack_b_offset_bytes = 0;
        handle.header->bwd_grad_pack_a_head_bytes = 0;
        handle.header->bwd_grad_pack_b_head_bytes = 0;
        handle.header->bwd_grad_pack_a_tensor_bytes = 0;
        handle.header->bwd_grad_pack_b_tensor_bytes = 0;
        handle.header->bwd_grad_pack_b_offset_bytes = 0;
        handle.header->pack_epoch = 0;
        handle.header->bwd_epoch = 0;
        for (int i = 0; i < handle.header->chunk_count; ++i) {
            handle.header->chunk_ready[i] = 0;
            handle.header->bwd_reduce_done[i] = 0;
            for (int rank = 0; rank < handle.header->sub_size; ++rank) {
                handle.header->bwd_partial_ready[i][rank] = 0;
            }
        }
        std::snprintf(handle.header->shm_name,
                      sizeof(handle.header->shm_name),
                      "ring_die%d_ctrl",
                      handle.die_id);
    }

    MPI_Barrier(handle.sub_comm);

    if (handle.sub_rank != 0) {
        handle.fd = shm_open(handle.shm_name, O_RDWR, 0666);
        if (handle.fd == -1) {
            abort_with_errno("shm_open(open)", MPI_COMM_WORLD);
        }

        handle.base = map_file_region(
            handle.fd, kControlMappingBytes, false, &handle.using_huge_pages);
        if (handle.base == nullptr) {
            abort_with_errno("mmap(open)", MPI_COMM_WORLD);
        }

        bind_control_view(handle);
    }

    if (handle.header->magic != kSharedMagic || handle.header->ready == 0) {
        abort_with_message("shared memory is not initialized", MPI_COMM_WORLD);
    }

    {
        key_t chunk_key = make_chunk_shm_key(handle.die_id, 'k', handle.sub_rank);
        handle.k_chunk_ids[handle.sub_rank] = create_or_open_chunk_segment(
            chunk_key,
            kSharedChunkBytes,
            true,
            "shmget(create k chunk)",
            MPI_COMM_WORLD);
        handle.k_chunk_bases[handle.sub_rank] = attach_chunk_segment(
            handle.k_chunk_ids[handle.sub_rank],
            "shmat(create k chunk)",
            MPI_COMM_WORLD);
        bind_shared_region_to_hbm(handle.k_chunk_bases[handle.sub_rank],
                                  kSharedChunkBytes,
                                  default_numa_node());
        std::memset(handle.k_chunk_bases[handle.sub_rank], 0, kSharedChunkBytes);
        handle.k_chunks[handle.sub_rank] =
            static_cast<__bf16*>(handle.k_chunk_bases[handle.sub_rank]);
    }

    {
        key_t chunk_key = make_chunk_shm_key(handle.die_id, 'v', handle.sub_rank);
        handle.v_chunk_ids[handle.sub_rank] = create_or_open_chunk_segment(
            chunk_key,
            kSharedChunkBytes,
            true,
            "shmget(create v chunk)",
            MPI_COMM_WORLD);
        handle.v_chunk_bases[handle.sub_rank] = attach_chunk_segment(
            handle.v_chunk_ids[handle.sub_rank],
            "shmat(create v chunk)",
            MPI_COMM_WORLD);
        bind_shared_region_to_hbm(handle.v_chunk_bases[handle.sub_rank],
                                  kSharedChunkBytes,
                                  default_numa_node());
        std::memset(handle.v_chunk_bases[handle.sub_rank], 0, kSharedChunkBytes);
        handle.v_chunks[handle.sub_rank] =
            static_cast<__bf16*>(handle.v_chunk_bases[handle.sub_rank]);
    }

    MPI_Barrier(handle.sub_comm);

    for (int chunk_idx = 0; chunk_idx < handle.sub_size; ++chunk_idx) {
        if (chunk_idx == handle.sub_rank) {
            continue;
        }

        key_t chunk_key = make_chunk_shm_key(handle.die_id, 'k', chunk_idx);
        handle.k_chunk_ids[chunk_idx] = create_or_open_chunk_segment(
            chunk_key,
            kSharedChunkBytes,
            false,
            "shmget(open k chunk)",
            MPI_COMM_WORLD);
        handle.k_chunk_bases[chunk_idx] = attach_chunk_segment(
            handle.k_chunk_ids[chunk_idx],
            "shmat(open k chunk)",
            MPI_COMM_WORLD);
        handle.k_chunks[chunk_idx] =
            static_cast<__bf16*>(handle.k_chunk_bases[chunk_idx]);

        chunk_key = make_chunk_shm_key(handle.die_id, 'v', chunk_idx);
        handle.v_chunk_ids[chunk_idx] = create_or_open_chunk_segment(
            chunk_key,
            kSharedChunkBytes,
            false,
            "shmget(open v chunk)",
            MPI_COMM_WORLD);
        handle.v_chunk_bases[chunk_idx] = attach_chunk_segment(
            handle.v_chunk_ids[chunk_idx],
            "shmat(open v chunk)",
            MPI_COMM_WORLD);
        handle.v_chunks[chunk_idx] =
            static_cast<__bf16*>(handle.v_chunk_bases[chunk_idx]);
    }

    handle.k = handle.k_chunks[0];
    handle.v = handle.v_chunks[0];
    if (handle.header == nullptr || handle.k == nullptr || handle.v == nullptr) {
        abort_with_message("shared memory mapping is incomplete", MPI_COMM_WORLD);
    }

    handle.initialized = true;
    MPI_Barrier(handle.sub_comm);
}

}  // namespace

SharedHeader* get_ring_die_shared_header() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().header;
}

__bf16* get_ring_die_shared_k() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().k;
}

__bf16* get_ring_die_shared_v() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().v;
}

__bf16* get_ring_die_shared_k_chunk(int chunk_idx) {
    initialize_ring_die_shared_memory(0);
    RingDieShmHandle& handle = ring_die_shm_handle();
    if (chunk_idx < 0 || chunk_idx >= handle.header->chunk_count) {
        abort_with_message("shared k chunk index out of range", handle.sub_comm);
    }
    return handle.k_chunks[chunk_idx];
}

__bf16* get_ring_die_shared_v_chunk(int chunk_idx) {
    initialize_ring_die_shared_memory(0);
    RingDieShmHandle& handle = ring_die_shm_handle();
    if (chunk_idx < 0 || chunk_idx >= handle.header->chunk_count) {
        abort_with_message("shared v chunk index out of range", handle.sub_comm);
    }
    return handle.v_chunks[chunk_idx];
}

__bf16* get_ring_die_shared_q_chunk(int chunk_idx) {
    return get_ring_die_shared_k_chunk(chunk_idx);
}

__bf16* get_ring_die_shared_grad_out_chunk(int chunk_idx) {
    return get_ring_die_shared_v_chunk(chunk_idx);
}

__bf16* get_ring_die_shared_q_pack_b_chunk(int chunk_idx) {
    SharedHeader* header = get_ring_die_shared_header();
    char* q_base = reinterpret_cast<char*>(get_ring_die_shared_q_chunk(chunk_idx));
    return reinterpret_cast<__bf16*>(q_base + header->bwd_q_pack_b_offset_bytes);
}

__bf16* get_ring_die_shared_grad_out_pack_b_chunk(int chunk_idx) {
    SharedHeader* header = get_ring_die_shared_header();
    char* grad_base = reinterpret_cast<char*>(get_ring_die_shared_grad_out_chunk(chunk_idx));
    return reinterpret_cast<__bf16*>(grad_base + header->bwd_grad_pack_b_offset_bytes);
}

float* get_ring_die_shared_m_chunk(int chunk_idx) {
    SharedHeader* header = get_ring_die_shared_header();
    char* grad_out_base = reinterpret_cast<char*>(
        get_ring_die_shared_grad_out_chunk(chunk_idx));
    return reinterpret_cast<float*>(grad_out_base + header->bwd_m_offset_bytes);
}

float* get_ring_die_shared_l_chunk(int chunk_idx) {
    SharedHeader* header = get_ring_die_shared_header();
    char* grad_out_base = reinterpret_cast<char*>(
        get_ring_die_shared_grad_out_chunk(chunk_idx));
    return reinterpret_cast<float*>(grad_out_base + header->bwd_l_offset_bytes);
}

float* get_ring_die_shared_d_chunk(int chunk_idx) {
    SharedHeader* header = get_ring_die_shared_header();
    char* grad_out_base = reinterpret_cast<char*>(
        get_ring_die_shared_grad_out_chunk(chunk_idx));
    return reinterpret_cast<float*>(grad_out_base + header->bwd_d_offset_bytes);
}

std::size_t get_ring_die_shared_chunk_bytes() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().header->chunk_capacity_bytes;
}

std::size_t get_ring_die_shared_chunk_count() {
    initialize_ring_die_shared_memory(0);
    return static_cast<std::size_t>(ring_die_shm_handle().header->chunk_count);
}

MPI_Comm get_ring_die_comm() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().sub_comm;
}

int get_ring_die_rank() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().sub_rank;
}

int get_ring_die_id() {
    initialize_ring_die_shared_memory(0);
    return ring_die_shm_handle().die_id;
}

void ring_attention_backward_step_ref(__bf16* Q,__bf16* grad_Q,
                             __bf16* K, __bf16 *dK, 
                             __bf16* V, __bf16* dV, __bf16* grad_out,
                             int t_num, int h_num, int h_len, int step,
                             float *m_i, float* l_i, float *D, float scale);

static void ring_attention_backward_impl(
    __bf16* Q, __bf16* grad_Q,
    __bf16* K, __bf16* dK,
    __bf16* V, __bf16* dV, __bf16* grad_out,
    int t_num, int h_num, int h_len,
    bool accumulate_grad_q,
    bool accumulate_dkv,
    float* m_i, float* l_i, float* D, float scale);

void destroy_ring_die_shared_memory() {
    RingDieShmHandle& handle = ring_die_shm_handle();
    if (!handle.initialized) {
        destroy_ring_die_backward_scratch();
        return;
    }

    destroy_ring_die_backward_scratch();
    MPI_Barrier(handle.sub_comm);

    for (int chunk_idx = 0; chunk_idx < kMaxChunksPerTensor; ++chunk_idx) {
        if (handle.k_chunk_bases[chunk_idx] != nullptr) {
            if (shmdt(handle.k_chunk_bases[chunk_idx]) == -1) {
                abort_with_errno("shmdt(k chunk)", MPI_COMM_WORLD);
            }
        }
        if (handle.v_chunk_bases[chunk_idx] != nullptr) {
            if (shmdt(handle.v_chunk_bases[chunk_idx]) == -1) {
                abort_with_errno("shmdt(v chunk)", MPI_COMM_WORLD);
            }
        }
    }

    if (handle.base != nullptr) {
        if (munmap(handle.base, kControlMappingBytes) == -1) {
            abort_with_errno("munmap(control)", MPI_COMM_WORLD);
        }
    }

    if (handle.fd != -1) {
        if (close(handle.fd) == -1) {
            abort_with_errno("close(control)", MPI_COMM_WORLD);
        }
    }

    if (handle.k_chunk_ids[handle.sub_rank] != -1 &&
        shmctl(handle.k_chunk_ids[handle.sub_rank], IPC_RMID, nullptr) == -1 &&
        errno != EINVAL && errno != EIDRM) {
        abort_with_errno("shmctl(IPC_RMID k chunk)", MPI_COMM_WORLD);
    }
    if (handle.v_chunk_ids[handle.sub_rank] != -1 &&
        shmctl(handle.v_chunk_ids[handle.sub_rank], IPC_RMID, nullptr) == -1 &&
        errno != EINVAL && errno != EIDRM) {
        abort_with_errno("shmctl(IPC_RMID v chunk)", MPI_COMM_WORLD);
    }

    if (handle.owner) {
        if (shm_unlink(handle.shm_name) == -1 && errno != ENOENT) {
            abort_with_errno("shm_unlink(control)", MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(handle.sub_comm);
    MPI_Comm_free(&handle.sub_comm);
    handle = RingDieShmHandle{};
}

void ring_attention_forward(__bf16* lQ, __bf16* lK, __bf16* lV, float* out,
                                  int t_num, int h_num, int h_len, int sp_size,
                                  __bf16* m_i, float* l_i, float scale) {
    initialize_ring_die_shared_memory(sp_size);
    MPI_Comm sub_comm = get_ring_die_comm();
    SharedHeader* header = get_ring_die_shared_header();
    int sub_rank = get_ring_die_rank();
    // if (!use_attn_component_backward_shape(t_num, h_num, h_len)) {
    //     abort_with_message("ring_attention_forward only supports (t_num,h_num,h_len)=(768,30,128) or (256,30,128)",
    //                        sub_comm);
    // }
    configure_shared_packed_layout(t_num, h_num, h_len);
    if (sp_size != header->chunk_count) {
        abort_with_message("sp_size must match shared chunk count", sub_comm);
    }
    const int pack_epoch = current_shared_pack_epoch(header);
    const std::size_t q_tensor_bytes = header->q_tensor_bytes;
    const std::size_t score_tensor_bytes =
        static_cast<std::size_t>(t_num) * h_num * t_num * sizeof(__bf16);
    static RingAttnCache cache;
    cache.ensure_config(t_num, h_num, h_len, scale);
    RingAttnHbmWorkspace& hbm_workspace = ring_attn_hbm_workspace();
    const int numa_node = default_numa_node();
    if (!hbm_workspace.ensure_slot(hbm_workspace.score_primary,
                                   hbm_workspace.score_primary_bytes,
                                   score_tensor_bytes,
                                   numa_node)) {
        fprintf(stderr, "C1 allocation failed\n");
        return;
    }
    if (!hbm_workspace.ensure_slot(hbm_workspace.pack_primary,
                                   hbm_workspace.pack_primary_bytes,
                                   q_tensor_bytes,
                                   numa_node)) {
        fprintf(stderr, "hbm_Q allocation failed\n");
        return;
    }
    __bf16* C1 = hbm_workspace.score_primary;
    __bf16* hbm_Q = hbm_workspace.pack_primary;

    #pragma omp parallel num_threads(h_num)
    {
        int rank = omp_get_thread_num();
        // bind_omp_thread_to_allowed_cpu(rank);

        __bf16* local_k = lK + static_cast<std::size_t>(rank) * h_len;
        __bf16* local_v = lV + static_cast<std::size_t>(rank) * h_len;
        __bf16* local_q = lQ + static_cast<std::size_t>(rank) * h_len;
        const int input_ld = h_num * h_len;
        __bf16* k_chunk = reinterpret_cast<__bf16*>(
            get_packed_head_ptr(get_ring_die_shared_k_chunk(sub_rank),
                                header->k_head_bytes,
                                rank,
                                h_num,
                                sub_comm));
        __bf16* v_chunk = reinterpret_cast<__bf16*>(
            get_packed_head_ptr(get_ring_die_shared_v_chunk(sub_rank),
                                header->v_head_bytes,
                                rank,
                                h_num,
                                sub_comm));
        __bf16* q_chunk = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(hbm_Q) +
            static_cast<std::size_t>(rank) * header->q_head_bytes);
        ring_attn_component::pack_b_unchecked(CblasTrans,
                                              t_num, h_len, input_ld,
                                              local_k, k_chunk);
        ring_attn_component::pack_b_unchecked(CblasNoTrans,
                                              h_len, t_num, input_ld,
                                              local_v, v_chunk);
        ring_attn_component::pack_a_unchecked(CblasNoTrans,
                                              t_num, h_len, input_ld,
                                              local_q, q_chunk);
    }
    // mark_shared_chunk_ready(header, sub_rank, pack_epoch, sub_comm);
    MPI_Barrier(sub_comm);
    #pragma omp parallel num_threads(h_num)
    {
        int rank = omp_get_thread_num();
        __bf16* Q = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(hbm_Q) +
            static_cast<std::size_t>(rank) * header->q_head_bytes);
        __bf16* C = C1 + static_cast<std::size_t>(t_num) * t_num * rank;
        float* out1 = out + static_cast<std::size_t>(rank) * t_num * h_len;
        for(int step=0;step<sp_size;step+=1){
            const int chunk_idx = (step + sub_rank) % sp_size;
            // if (chunk_idx != sub_rank) {
            //     wait_for_shared_chunk_ready(header, chunk_idx, pack_epoch, sub_comm);
            // }
            __bf16* K = reinterpret_cast<__bf16*>(
                get_packed_head_ptr(get_ring_die_shared_k_chunk(chunk_idx),
                                    header->k_head_bytes,
                                    rank,
                                    h_num,
                                    sub_comm));
            __bf16* V = reinterpret_cast<__bf16*>(
                get_packed_head_ptr(get_ring_die_shared_v_chunk(chunk_idx),
                                    header->v_head_bytes,
                                    rank,
                                    h_num,
                                    sub_comm));
                ring_attn_component::bgemm_packed_unchecked(t_num, t_num, h_len,
                                                            (__bf16)scale,
                                                            Q, K,
                                                            (__bf16)0.0f, C, t_num);
            if (step == 0) {
                for (int j = 0; j < t_num; ++j) {
                    __bf16 temp = quick_max(C + j * t_num, t_num);
                    l_i[rank * t_num + j] =
                        sve_exp(C + j * t_num, (float)temp, t_num, rank);
                    m_i[rank * t_num + j] = temp;
                }
                const size_t packed_a_size =
                    ring_attn_component::pack_get_size(CblasA, t_num, h_len, t_num);
                if (!tls_workspace.reserve(tls_workspace.packed_a, tls_workspace.size_a, packed_a_size)) {
                    std::fprintf(stderr, "packed_a allocation failed on cpu %d\n", sched_getcpu());
                    abort_with_message("packed_a allocation failed in whole forward",
                                       sub_comm);
                }
                ring_attn_component::pack_a_unchecked(CblasNoTrans,
                                                      t_num, t_num, t_num,
                                                      C, tls_workspace.packed_a);
                ring_attn_component::sbgemm_packed_unchecked(t_num, h_len, t_num,
                                                             (__bf16)1.0f,
                                                             tls_workspace.packed_a, V,
                                                             (__bf16)0.0f, out1, h_len);
            } else {
                for (int j = 0; j < t_num; ++j) {
                    __bf16 temp = quick_max(C + j * t_num, t_num);
                    if(temp<m_i[rank * t_num + j]){
                        temp = m_i[rank * t_num + j];
                    }
                    float correction = std::exp((float)(m_i[rank * t_num + j] - temp));
                    l_i[rank * t_num + j] =
                        sve_exp(C + j * t_num, temp, t_num, rank) +
                        l_i[rank * t_num + j] * correction;
                    if(correction!=1){
                        sve_out(h_len, out1 + static_cast<std::size_t>(j) * h_len, correction);
                    }
                    m_i[rank * t_num + j] = temp;
                }
                const size_t packed_a_size =
                    ring_attn_component::pack_get_size(CblasA, t_num, h_len, t_num);
                if (!tls_workspace.reserve(tls_workspace.packed_a, tls_workspace.size_a, packed_a_size)) {
                    std::fprintf(stderr, "packed_a allocation failed on cpu %d\n", sched_getcpu());
                    abort_with_message("packed_a allocation failed in whole forward",
                                       sub_comm);
                }
                ring_attn_component::pack_a_unchecked(CblasNoTrans,
                                                      t_num, t_num, t_num,
                                                      C, tls_workspace.packed_a);
                ring_attn_component::sbgemm_packed_unchecked(t_num, h_len, t_num,
                                                             (__bf16)1.0f,
                                                             tls_workspace.packed_a, V,
                                                             (__bf16)1.0f, out1, h_len);
            }   
        }
        for(int j=0;j<t_num;j++){
            for(int k=0;k<h_len;k++){
                out1[static_cast<std::size_t>(j) * h_len + k] /= l_i[rank*t_num + j];
            }
        }
    }
}

void ring_attention_backward(__bf16* Q,__bf16* grad_Q,
                     __bf16* K, __bf16 *dK, 
                     __bf16* V, __bf16* dV, __bf16* grad_out,
                    int t_num, int h_num, int h_len, int sp_size,
                    float *m_i, float* l_i, float *D, float scale){
    initialize_ring_die_shared_memory(sp_size);
    MPI_Comm sub_comm = get_ring_die_comm();
    SharedHeader* header = get_ring_die_shared_header();
    const int sub_rank = get_ring_die_rank();
    configure_shared_raw_layout(t_num, h_num, h_len);
    if (sp_size != header->chunk_count) {
        abort_with_message("sp_size must match shared chunk count", sub_comm);
    }

    const int bwd_epoch = __atomic_load_n(&header->bwd_epoch, __ATOMIC_ACQUIRE);
    // if (!use_attn_component_backward_shape(t_num, h_num, h_len)) {
    //     abort_with_message("ring_attention_backward only supports (t_num,h_num,h_len)=(768,30,128) or (256,30,128)",
    //                        sub_comm);
    // }
    const std::size_t raw_tensor_bytes = header->raw_tensor_bytes;
    const std::size_t raw_tensor_elems = raw_tensor_bytes / sizeof(__bf16);
    const std::size_t stats_bytes = header->bwd_stats_bytes;
    const std::size_t packed_k_head_bytes =
        cblas_bgemm_pack_get_size(CblasB, t_num, t_num, h_len);
    const std::size_t packed_v_head_bytes =
        cblas_bgemm_pack_get_size(CblasB, t_num, t_num, h_len);
    const std::size_t packed_k_tensor_bytes =
        static_cast<std::size_t>(h_num) * packed_k_head_bytes;
    const std::size_t packed_v_tensor_bytes =
        static_cast<std::size_t>(h_num) * packed_v_head_bytes;
    const std::size_t packed_k_nt_head_bytes =
        ring_attn_component::pack_get_size(CblasB, t_num, h_len, t_num) *
        sizeof(__bf16);
    const std::size_t packed_k_nt_tensor_bytes =
        static_cast<std::size_t>(h_num) * packed_k_nt_head_bytes;
    const std::size_t packed_dp_head_bytes =
        cblas_bgemm_pack_get_size(CblasA, t_num, h_len, t_num);

    initialize_ring_die_backward_scratch(
        raw_tensor_bytes * static_cast<std::size_t>(sp_size),
        sub_comm);
    RingDieBwdScratchHandle& scratch = ring_die_bwd_scratch_handle();

    const std::size_t score_tensor_bytes =
        static_cast<std::size_t>(t_num) * h_num * t_num * sizeof(__bf16);
    RingAttnHbmWorkspace& hbm_workspace = ring_attn_hbm_workspace();
    const int numa_node = default_numa_node();
    if (!hbm_workspace.ensure_slot(hbm_workspace.pack_primary,
                                   hbm_workspace.pack_primary_bytes,
                                   packed_k_tensor_bytes,
                                   numa_node)) {
        abort_with_message("backward packed K allocation failed", sub_comm);
    }
    if (!hbm_workspace.ensure_slot(hbm_workspace.pack_secondary,
                                   hbm_workspace.pack_secondary_bytes,
                                   packed_v_tensor_bytes,
                                   numa_node)) {
        abort_with_message("backward packed V allocation failed", sub_comm);
    }
    if (!hbm_workspace.ensure_slot(hbm_workspace.pack_tertiary,
                                   hbm_workspace.pack_tertiary_bytes,
                                   packed_k_nt_tensor_bytes,
                                   numa_node)) {
        abort_with_message("backward packed K(NoTrans) allocation failed", sub_comm);
    }
    if (!hbm_workspace.ensure_slot(hbm_workspace.score_primary,
                                   hbm_workspace.score_primary_bytes,
                                   score_tensor_bytes,
                                   numa_node) ||
        !hbm_workspace.ensure_slot(hbm_workspace.score_secondary,
                                   hbm_workspace.score_secondary_bytes,
                                   score_tensor_bytes,
                                   numa_node)) {
        abort_with_message("backward whole S/dP allocation failed", sub_comm);
    }
    __bf16* packed_local_k = hbm_workspace.pack_primary;
    __bf16* packed_local_v = hbm_workspace.pack_secondary;
    __bf16* packed_local_k_nt = hbm_workspace.pack_tertiary;
    __bf16* S = hbm_workspace.score_primary;
    __bf16* dP = hbm_workspace.score_secondary;
    int packed_dp_alloc_failed = 0;

    std::memcpy(get_ring_die_shared_m_chunk(sub_rank), m_i, stats_bytes);
    std::memcpy(get_ring_die_shared_l_chunk(sub_rank), l_i, stats_bytes);
    std::memcpy(get_ring_die_shared_d_chunk(sub_rank), D, stats_bytes);

    const int input_ld = h_num * h_len;
    #pragma omp parallel num_threads(h_num)
    {
        const int rank = omp_get_thread_num();
        __bf16* local_k = K + static_cast<std::size_t>(rank) * h_len;
        __bf16* local_v = V + static_cast<std::size_t>(rank) * h_len;
        __bf16* local_q = Q + static_cast<std::size_t>(rank) * h_len;
        __bf16* local_grad_out =
            grad_out + static_cast<std::size_t>(rank) * h_len * t_num;
        __bf16* local_k_pack_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(packed_local_k) +
            static_cast<std::size_t>(rank) * packed_k_head_bytes);
        __bf16* local_k_nt_pack_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(packed_local_k_nt) +
            static_cast<std::size_t>(rank) * packed_k_nt_head_bytes);
        __bf16* local_v_pack_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(packed_local_v) +
            static_cast<std::size_t>(rank) * packed_v_head_bytes);
        __bf16* shared_q_pack_a_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(get_ring_die_shared_q_chunk(sub_rank)) +
            static_cast<std::size_t>(rank) * header->bwd_q_pack_a_head_bytes);
        __bf16* shared_q_pack_b_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(get_ring_die_shared_q_pack_b_chunk(sub_rank)) +
            static_cast<std::size_t>(rank) * header->bwd_q_pack_b_head_bytes);
        __bf16* shared_grad_pack_a_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(get_ring_die_shared_grad_out_chunk(sub_rank)) +
            static_cast<std::size_t>(rank) * header->bwd_grad_pack_a_head_bytes);
        __bf16* shared_grad_pack_b_head = reinterpret_cast<__bf16*>(
            reinterpret_cast<char*>(get_ring_die_shared_grad_out_pack_b_chunk(sub_rank)) +
            static_cast<std::size_t>(rank) * header->bwd_grad_pack_b_head_bytes);

        ring_attn_component::pack_a_unchecked(CblasNoTrans, t_num, h_len, input_ld,
                                              local_q, shared_q_pack_a_head);
        ring_attn_component::pack_b_unchecked(CblasNoTrans, h_len, t_num, input_ld,
                                              local_q, shared_q_pack_b_head);
        ring_attn_component::pack_a_unchecked(CblasNoTrans, t_num, h_len, h_len,
                                              local_grad_out, shared_grad_pack_a_head);
        ring_attn_component::pack_b_unchecked(CblasNoTrans, h_len, t_num, h_len,
                                              local_grad_out, shared_grad_pack_b_head);
        ring_attn_component::pack_b_unchecked(CblasTrans, t_num, h_len, input_ld,
                                              local_k, local_k_pack_head);
        ring_attn_component::pack_b_unchecked(CblasNoTrans, h_len, t_num, input_ld,
                                              local_k, local_k_nt_pack_head);
        ring_attn_component::pack_b_unchecked(CblasTrans, t_num, h_len, input_ld,
                                              local_v, local_v_pack_head);
        if (!tls_workspace.reserve(tls_workspace.packed_a,
                                   tls_workspace.size_a,
                                   packed_dp_head_bytes) ||
            !tls_workspace.reserve(tls_workspace.packed_b,
                                   tls_workspace.size_b,
                                   packed_dp_head_bytes)) {
            #pragma omp atomic write
            packed_dp_alloc_failed = 1;
        }
    }
    if (packed_dp_alloc_failed) {
        abort_with_message("backward packed dP allocation failed", sub_comm);
    }

    MPI_Barrier(sub_comm);
    for (int step = 0; step < sp_size; ++step) {
        const int query_idx = (sub_rank + step) % sp_size;
        __bf16* shared_q_chunk = get_ring_die_shared_q_chunk(query_idx);
        __bf16* shared_grad_out_chunk = get_ring_die_shared_grad_out_chunk(query_idx);
        __bf16* shared_q_pack_b_chunk = get_ring_die_shared_q_pack_b_chunk(query_idx);
        __bf16* shared_grad_out_pack_b_chunk =
            get_ring_die_shared_grad_out_pack_b_chunk(query_idx);
        float* shared_m = get_ring_die_shared_m_chunk(query_idx);
        float* shared_l = get_ring_die_shared_l_chunk(query_idx);
        float* shared_d = get_ring_die_shared_d_chunk(query_idx);

        #pragma omp parallel num_threads(h_num)
        {
            const int rank = omp_get_thread_num();
            __bf16* shared_q_pack_a_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(shared_q_chunk) +
                static_cast<std::size_t>(rank) * header->bwd_q_pack_a_head_bytes);
            __bf16* shared_q_pack_b_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(shared_q_pack_b_chunk) +
                static_cast<std::size_t>(rank) * header->bwd_q_pack_b_head_bytes);
            __bf16* shared_grad_out_pack_a_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(shared_grad_out_chunk) +
                static_cast<std::size_t>(rank) * header->bwd_grad_pack_a_head_bytes);
            __bf16* shared_grad_out_pack_b_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(shared_grad_out_pack_b_chunk) +
                static_cast<std::size_t>(rank) * header->bwd_grad_pack_b_head_bytes);
            __bf16* local_k_pack_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(packed_local_k) +
                static_cast<std::size_t>(rank) * packed_k_head_bytes);
            __bf16* local_k_nt_pack_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(packed_local_k_nt) +
                static_cast<std::size_t>(rank) * packed_k_nt_head_bytes);
            __bf16* local_v_pack_head = reinterpret_cast<__bf16*>(
                reinterpret_cast<char*>(packed_local_v) +
                static_cast<std::size_t>(rank) * packed_v_head_bytes);
            __bf16* dq_partial_head =
                scratch.dq_slots[sub_rank] +
                static_cast<std::size_t>(query_idx) * raw_tensor_elems +
                static_cast<std::size_t>(rank) * h_len;
            __bf16* dK_head = dK + static_cast<std::size_t>(rank) * h_len;
            __bf16* dV_head = dV + static_cast<std::size_t>(rank) * h_len;
            __bf16* S_head = S + static_cast<std::size_t>(rank) * t_num * t_num;
            __bf16* dP_head = dP + static_cast<std::size_t>(rank) * t_num * t_num;

            ring_attn_component::bgemm_packed_unchecked(t_num, t_num, h_len,
                                                        (__bf16)scale,
                                                        shared_q_pack_a_head,
                                                        local_k_pack_head,
                                                        (__bf16)0.0f,
                                                        S_head, t_num);
            ring_attn_component::bgemm_packed_unchecked(t_num, t_num, h_len,
                                                        (__bf16)1.0f,
                                                        shared_grad_out_pack_a_head,
                                                        local_v_pack_head,
                                                        (__bf16)0.0f,
                                                        dP_head, t_num);

            for (int j = 0; j < t_num; ++j) {
                sve_exp_li(S_head + static_cast<std::size_t>(j) * t_num,
                           dP_head + static_cast<std::size_t>(j) * t_num,
                           shared_m[rank * t_num + j],
                           shared_l[rank * t_num + j],
                           t_num,
                           shared_d[rank * t_num + j],
                           scale);
            }

            ring_attn_component::pack_a_unchecked(CblasNoTrans, t_num, t_num, t_num,
                                                  dP_head, tls_workspace.packed_a);
            ring_attn_component::pack_a_unchecked(CblasTrans, t_num, t_num, t_num,
                                                  dP_head, tls_workspace.packed_b);
            ring_attn_component::bgemm_packed_unchecked(t_num, h_len, t_num,
                                                        (__bf16)1.0f,
                                                        tls_workspace.packed_a,
                                                        local_k_nt_pack_head,
                                                        (__bf16)0.0f,
                                                        dq_partial_head, input_ld);
            const __bf16 dkv_beta = (step == 0) ? (__bf16)0.0f : (__bf16)1.0f;
            ring_attn_component::bgemm_packed_unchecked(t_num, h_len, t_num,
                                                        (__bf16)1.0f,
                                                        tls_workspace.packed_b,
                                                        shared_q_pack_b_head,
                                                        dkv_beta,
                                                        dK_head, input_ld);
            ring_attn_component::pack_a_unchecked(CblasTrans, t_num, t_num, t_num,
                                                  S_head, tls_workspace.packed_a);
            ring_attn_component::bgemm_packed_unchecked(t_num, h_len, t_num,
                                                        (__bf16)1.0f,
                                                        tls_workspace.packed_a,
                                                        shared_grad_out_pack_b_head,
                                                        dkv_beta,
                                                        dV_head, input_ld);
        }
    }
    MPI_Barrier(sub_comm);

    MPI_Win_sync(scratch.dq_win);
    __bf16* dq_query_slots[kMaxRanksPerDie] = {};
    for (int rank = 0; rank < scratch.sub_size; ++rank) {
        dq_query_slots[rank] =
            scratch.dq_slots[rank] +
            static_cast<std::size_t>(sub_rank) * raw_tensor_elems;
    }
    reduce_shared_bwd_slot_to_owner(
        grad_Q, dq_query_slots, scratch.sub_size, raw_tensor_elems);
    MPI_Barrier(sub_comm);
}

static void ring_attention_backward_impl(
    __bf16* Q, __bf16* grad_Q,
    __bf16* K, __bf16* dK,
    __bf16* V, __bf16* dV, __bf16* grad_out,
    int t_num, int h_num, int h_len,
    bool accumulate_grad_q,
    bool accumulate_dkv,
    float* m_i, float* l_i, float* D, float scale) {
    static RingAttnCache cache;
    BlasSetNumThreadsLocal(1);
    cache.ensure_config(t_num, h_num, h_len, scale);

    const std::size_t score_tensor_bytes =
        static_cast<std::size_t>(t_num) * h_num * t_num * sizeof(__bf16);
    RingAttnHbmWorkspace& hbm_workspace = ring_attn_hbm_workspace();
    const int numa_node = default_numa_node();
    if (!hbm_workspace.ensure_slot(hbm_workspace.score_primary,
                                   hbm_workspace.score_primary_bytes,
                                   score_tensor_bytes,
                                   numa_node) ||
        !hbm_workspace.ensure_slot(hbm_workspace.score_secondary,
                                   hbm_workspace.score_secondary_bytes,
                                   score_tensor_bytes,
                                   numa_node)) {
        fprintf(stderr, "C1 allocation failed\n");
        return;
    }
    __bf16* S = hbm_workspace.score_primary;
    __bf16* dP = hbm_workspace.score_secondary;

    #pragma omp parallel num_threads(h_num)
    {
        int rank = omp_get_thread_num();
        cache.Q_array[rank] = Q + h_len * rank;
        cache.grad_Q_array[rank] = grad_Q + h_len * rank;
        cache.K_array[rank] = K + h_len * rank;
        cache.dK_array[rank] = dK + h_len * rank;
        cache.S_array[rank] = S + t_num * t_num * rank;
        cache.dP_array[rank] = dP + t_num * t_num * rank;
        cache.V_array[rank] = V + h_len * rank;
        cache.dV_array[rank] = dV + h_len * rank;
        cache.grad_out_array[rank] = grad_out + h_len * t_num * rank;

        bgemm_kernel(CblasRowMajor, cache.transA[rank], cache.transB[rank],
        cache.t_array[rank], cache.t_array[rank], cache.h_array[rank],
        cache.alpha1[rank], cache.Q_array[rank], cache.ld_array[rank],
        cache.K_array[rank], cache.ld_array[rank], cache.beta[rank],
        cache.S_array[rank], cache.t_array[rank]);
        
        bgemm_kernel(CblasRowMajor, cache.transA[rank], cache.transB[rank],
        cache.t_array[rank], cache.t_array[rank], cache.h_array[rank],
        cache.alpha[rank], cache.grad_out_array[rank], cache.h_array[rank],
        cache.V_array[rank], cache.ld_array[rank], cache.beta[rank],
        cache.dP_array[rank], cache.t_array[rank]);

        for (int j = 0; j < t_num; ++j){
            sve_exp_li(S + rank * t_num * t_num + j * t_num, 
                dP + rank * t_num * t_num + j * t_num, 
                m_i[rank * t_num + j], 
                l_i[rank * t_num + j], t_num, D[rank * t_num + j], scale);
        }

        const float dkv_beta =
            accumulate_dkv ? cache.alpha[rank] : cache.beta[rank];
        const float grad_q_beta =
            accumulate_grad_q ? cache.alpha[rank] : cache.beta[rank];

        bgemm_kernel(CblasRowMajor, cache.transB[rank], cache.transA[rank],
        cache.t_array[rank], cache.h_array[rank], cache.t_array[rank],
        cache.alpha[rank], cache.S_array[rank], cache.t_array[rank],
        cache.grad_out_array[rank], cache.h_array[rank], dkv_beta,
        cache.dV_array[rank], cache.ld_array[rank]);

        bgemm_kernel(CblasRowMajor, cache.transA[rank], cache.transA[rank],
        cache.t_array[rank], cache.h_array[rank], cache.t_array[rank],
        cache.alpha[rank], cache.dP_array[rank], cache.t_array[rank],
        cache.K_array[rank], cache.ld_array[rank], grad_q_beta,
        cache.grad_Q_array[rank], cache.ld_array[rank]);

        bgemm_kernel(CblasRowMajor, cache.transB[rank], cache.transA[rank],
        cache.t_array[rank], cache.h_array[rank], cache.t_array[rank],
        cache.alpha[rank], cache.dP_array[rank], cache.t_array[rank],
        cache.Q_array[rank], cache.ld_array[rank], dkv_beta,
        cache.dK_array[rank], cache.ld_array[rank]);
    }
}

void ring_attention_backward_step_ref(__bf16* Q,__bf16* grad_Q,
                             __bf16* K, __bf16 *dK, 
                             __bf16* V, __bf16* dV, __bf16* grad_out,
                             int t_num, int h_num, int h_len, int step,
                             float *m_i, float* l_i, float *D, float scale){
    const bool accumulate = (step != 0);
    ring_attention_backward_impl(
        Q, grad_Q, K, dK, V, dV, grad_out,
        t_num, h_num, h_len,
        accumulate, accumulate,
        m_i, l_i, D, scale);
}
