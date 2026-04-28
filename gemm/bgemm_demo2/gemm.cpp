#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_acle.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../bgemm/shared_workspace.h"

enum XHBLAS_ORDER {
    XhblasRowMajor = 101,
    XhblasColMajor = 102
};

enum XHBLAS_TRANSPOSE {
    XhblasNoTrans = 111,
    XhblasTrans = 112,
    XhblasConjTrans = 113
};

namespace final_demo2 {

constexpr int kCoresPerNuma = 38;
static const char* kModuleName = "demo2";

static void* alloc_hbm(size_t size, int numa_id, const char* tag = "unknown")
{
    (void)numa_id;
    (void)tag;
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void free_hbm(void* ptr, size_t size, const char* tag = "unknown")
{
    (void)size;
    (void)tag;
    free(ptr);
}

static int detect_base_numa()
{
    const char* rank_env = getenv("OMPI_COMM_WORLD_RANK");
    if (rank_env && rank_env[0] != '\0') {
        return (int)(strtol(rank_env, NULL, 10) % 16);
    }
    const int current_cpu = sched_getcpu();
    return current_cpu >= 0 ? (current_cpu / kCoresPerNuma) : 0;
}

static void bind_to_cpu(int base_numa, int thread_rank)
{
    const int target_cpu = base_numa * kCoresPerNuma + (thread_rank % kCoresPerNuma);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(target_cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity failed");
        exit(1);
    }
}

static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static int max_int(int a, int b)
{
    return a > b ? a : b;
}

static int getenv_int_or(const char* name, int default_value)
{
    const char* value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    const int parsed = atoi(value);
    return parsed > 0 ? parsed : default_value;
}

static XHBLAS_TRANSPOSE normalize_transpose(XHBLAS_TRANSPOSE trans)
{
    return trans == XhblasConjTrans ? XhblasTrans : trans;
}

static inline __bf16 trunc_f32_to_bf16(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const uint16_t hi = (uint16_t)(bits >> 16);
    __bf16 out;
    memcpy(&out, &hi, sizeof(hi));
    return out;
}

static inline svfloat32_t load_bf16_as_f32(svbool_t pg32, svbool_t pg16, const __bf16* src)
{
    svbfloat16_t vals16 = svld1_bf16(pg16, src);
    svuint16_t bits16 = svreinterpret_u16_bf16(vals16);
    svuint32_t bits32 = svlsl_n_u32_z(pg32, svunpklo_u32(bits16), 16);
    return svreinterpret_f32_u32(bits32);
}

static inline void store_f32_as_bf16(svbool_t pg32, svbool_t pg16, __bf16* dst, svfloat32_t src)
{
    svuint32_t bits32 = svlsr_n_u32_z(pg32, svreinterpret_u32_f32(src), 16);
    svuint16_t bits16 = svqxtnb_u32(bits32);
    bits16 = svuzp1(bits16, bits16);
    svst1_bf16(pg16, dst, svreinterpret_bf16_u16(bits16));
}

static void writeback_float_to_bf16(const float* src, __bf16* dst, int rows, int cols, int ldc, float alpha)
{
    for (int i = 0; i < rows; ++i) {
        const float* src_row = src + (size_t)i * ldc;
        __bf16* dst_row = dst + (size_t)i * ldc;
        for (int j = 0; j < cols; ++j) {
            dst_row[j] = trunc_f32_to_bf16(src_row[j] * alpha);
        }
    }
}

static void scale_matrix(__bf16* C, int M, int N, int ldc, float factor)
{
    if (factor == 1.0f) {
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const size_t idx = (size_t)i * ldc + j;
            C[idx] = trunc_f32_to_bf16((float)C[idx] * factor);
        }
    }
}

struct GemmSpec {
    int M;
    int N;
    int K;
};

static const GemmSpec kSupportedSpecs[] = {
    {1024, 2560, 3840},
    {1024,   64, 3840},
    {1024,  768, 3840},
};

static const GemmSpec* find_spec(int M, int N, int K)
{
    for (size_t i = 0; i < sizeof(kSupportedSpecs) / sizeof(kSupportedSpecs[0]); ++i) {
        const GemmSpec& spec = kSupportedSpecs[i];
        if (spec.M == M && spec.N == N && spec.K == K) {
            return &spec;
        }
    }
    return NULL;
}

namespace demo2_nt {

constexpr int m_magic = 32;
constexpr int m_pack = 16;
constexpr int n_magic = 32;
constexpr int n_pack = 64;
constexpr int gemm_m_tile = 16;
constexpr int gemm_b_pack = 64;
constexpr int fixed_magic_k = 3840;
static int ensure_workspace(size_t a_bytes, size_t b_bytes, int numa_id, __bf16** A_p, __bf16** B_p)
{
    void* shared_a = NULL;
    void* shared_b = NULL;
    if (bgemm_shared_workspace::ensure_buffers(a_bytes, b_bytes, 0, numa_id,
                                               "demo2",
                                               &shared_a, &shared_b, NULL) != 0) {
        return -1;
    }
    *A_p = (__bf16*)shared_a;
    *B_p = (__bf16*)shared_b;
    return (*A_p && *B_p) ? 0 : -1;
}

void pack_A(__bf16* A, int m, int k, int lda, __bf16* A_p)
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

void pack_B(__bf16* B, int k, int n, int ldb, __bf16* B_p)
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

void sme_kernel(__bf16* A, __bf16* B, __bf16* C_out, int ldc, int k, int m_use, int n_use,
                const __bf16* ba)
__arm_streaming __arm_inout("za")
{
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    svbool_t pg_tail32 = svwhilelt_b32((uint64_t)0, (uint64_t)(n_use - n_magic / 2 * 3));
    svbool_t pg_tail16 = svwhilelt_b16((uint64_t)0, (uint64_t)(n_use - n_magic));
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (!ba) {
        svzero_za();
    } else {
        alignas(64) float bias_row[gemm_b_pack] = {0.0f};
        for (int j = 0; j < n_use; ++j) {
            bias_row[j] = (float)ba[j];
        }
        svzero_za();
        for (int h = 0; h < m_use; ++h) {
            svld1_hor_za32(0, h, pg, bias_row);
            svld1_hor_za32(1, h, pg, bias_row + wl);
            svld1_hor_za32(2, h, pg, bias_row + 2 * wl);
            svld1_hor_za32(3, h, pg_tail32, bias_row + 3 * wl);
        }
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

    svbfloat16_t old = svdup_n_bf16((__bf16)0);
    for (int h = 0; h < m_use; ++h) {
        svbfloat16_t z0 = svread_hor_za16_bf16_m(old, svptrue_b16(), 0, 2 * h);
        svbfloat16_t z1 = svread_hor_za16_bf16_m(old, svptrue_b16(), 1, 2 * h);
        z1 = svuzp2_bf16(z0, z1);
        svst1_bf16(pg_bf16, C_out + h * ldc, z1);
        z0 = svread_hor_za16_bf16_m(old, svptrue_b16(), 0, 2 * h + 1);
        z1 = svread_hor_za16_bf16_m(old, svptrue_b16(), 1, 2 * h + 1);
        z1 = svuzp2_bf16(z0, z1);
        svst1_bf16(pg_tail16, C_out + h * ldc + 2 * wl, z1);
    }
}

static int choose_threads(int requested, int N)
{
    const int n_tiles = N / gemm_b_pack;
    return max_int(1, min_int(requested, n_tiles));
}

__arm_new("za")
static void run_thread_work(__bf16* A, __bf16* A_p, __bf16* B, __bf16* B_p, __bf16* C,
                            int M, int N, int K, int rank, int nt, const __bf16* ba)
__arm_streaming
{
    for (int mm = rank * gemm_m_tile; mm < M; mm += gemm_m_tile * nt) {
        pack_A(A + (size_t)mm * K, gemm_m_tile, K, K, A_p + (size_t)mm * K);
    }

    #pragma omp barrier

    for (int nn = rank * gemm_b_pack; nn < N; nn += gemm_b_pack * nt) {
        __bf16* B_panel = B_p + (size_t)rank * gemm_b_pack * K;
        pack_B(B + nn, K, gemm_b_pack, N, B_panel);
        for (int mm = 0; mm < M; mm += gemm_m_tile) {
            sme_kernel(A_p + (size_t)mm * K,
                       B_panel,
                       C + (size_t)mm * N + nn,
                       N, K, gemm_m_tile, gemm_b_pack,
                       ba ? ba + nn : NULL);
        }
    }
}

static int run(__bf16* A, __bf16* B, __bf16* C, int M, int N, int K, int bench_threads, float alpha,
               const __bf16* ba)
{
    if (K != fixed_magic_k || M % gemm_m_tile != 0 || N % gemm_b_pack != 0) {
        return -6;
    }

    const int active_threads = choose_threads(bench_threads, N);
    const int M_p = (M + gemm_m_tile - 1) / gemm_m_tile * gemm_m_tile;
    const size_t A_p_elems = (size_t)M_p * K;
    const size_t B_p_elems = (size_t)active_threads * gemm_b_pack * K;
    const int base_numa = detect_base_numa();
    __bf16* A_p = NULL;
    __bf16* B_p = NULL;
    if (ensure_workspace(A_p_elems * sizeof(__bf16), B_p_elems * sizeof(__bf16), base_numa, &A_p, &B_p) != 0) {
        return -7;
    }

    #pragma omp parallel num_threads(active_threads)
    {
        const int rank = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        bind_to_cpu(base_numa, rank);
        run_thread_work(A, A_p, B, B_p, C, M, N, K, rank, nt, ba);
    }

    scale_matrix(C, M, N, N, alpha);
    return 0;
}

} // namespace demo2_nt

static int scale_only(float beta, __bf16* C, int M, int N, int ldc)
{
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * (size_t)ldc * sizeof(__bf16));
        return 0;
    }
    return -9;
}

static int xhblas_gemm_status(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                              int M, int N, int K, float alpha,
                              const __bf16* A, int lda,
                              const __bf16* B, int ldb,
                              float beta, __bf16* C, int ldc, const __bf16* ba, int sign)
{
    trans_a = normalize_transpose(trans_a);
    trans_b = normalize_transpose(trans_b);

    if (order != XhblasRowMajor) {
        return -1;
    }
    if (trans_a != XhblasNoTrans) {
        return -2;
    }
    if (trans_b != XhblasNoTrans) {
        return -3;
    }
    if (M < 0 || N < 0 || K < 0) {
        return -4;
    }
    if (M == 0 || N == 0) {
        return 0;
    }
    if (beta != 0.0f) {
        return -9;
    }
    if (K == 0 || alpha == 0.0f) {
        return scale_only(beta, C, M, N, ldc);
    }
    if (lda != K || ldb != N || ldc != N) {
        return -5;
    }
    if (!find_spec(M, N, K)) {
        return -6;
    }

    const int bench_threads = getenv_int_or("FINAL_GEMM_THREADS", 32);
    return demo2_nt::run(const_cast<__bf16*>(A), const_cast<__bf16*>(B), C, M, N, K, bench_threads, alpha,
                         sign == 1 ? ba : NULL);
}

static const char* status_message(int status)
{
    switch (status) {
        case 0: return "success";
        case -1: return "only row-major layout is supported";
        case -2: return "only NoTrans A is supported";
        case -3: return "only NoTrans B is supported";
        case -4: return "invalid dimensions";
        case -5: return "invalid leading dimensions";
        case -6: return "unsupported shape";
        case -7: return "workspace allocation failed";
        case -9: return "non-zero beta is unsupported";
        default: return "unknown error";
    }
}

} // namespace final_demo2

extern "C" int xhblas_gemm(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    return final_demo2::xhblas_gemm_status(order, trans_a, trans_b,
                                           M, N, K, alpha,
                                           A, lda, B, ldb, beta, C, ldc, ba, sign);
}

#ifdef FINAL_DEMO_TEST_MAIN
namespace final_demo2 {

static void fill_test_bf16(__bf16* data, int rows, int cols, int ld, int seed_i, int seed_j)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < ld; ++j) {
            if (j < cols) {
                data[(size_t)i * ld + j] = (__bf16)(((i * seed_i + j * seed_j) % 15) - 7);
            } else {
                data[(size_t)i * ld + j] = (__bf16)0;
            }
        }
    }
}

static inline float bf16_to_float_bits(__bf16 v)
{
    uint16_t bits16 = 0;
    memcpy(&bits16, &v, sizeof(bits16));
    uint32_t bits32 = (uint32_t)bits16 << 16;
    float out = 0.0f;
    memcpy(&out, &bits32, sizeof(out));
    return out;
}

static int compare_bf16_tiles(const __bf16* a, const __bf16* b, int rows, int cols, int ld)
{
    int bad = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const size_t idx = (size_t)i * ld + j;
            uint16_t av = 0;
            uint16_t bv = 0;
            memcpy(&av, &a[idx], sizeof(av));
            memcpy(&bv, &b[idx], sizeof(bv));
            if (av != bv) {
                if (bad < 8) {
                    printf("bf16 mismatch at (%d,%d): a=0x%04x b=0x%04x a_f=%f b_f=%f\n",
                           i, j, av, bv, bf16_to_float_bits(a[idx]), bf16_to_float_bits(b[idx]));
                }
                ++bad;
            }
        }
    }
    return bad;
}

static void reference_gemm_nt(const __bf16* A, const __bf16* B, float* C, int M, int N, int K)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += (float)A[(size_t)i * K + p] * (float)B[(size_t)p * N + j];
            }
            C[(size_t)i * N + j] = acc;
        }
    }
}

static int run_correctness_case(const GemmSpec& spec)
{
    const size_t A_elems = (size_t)spec.M * spec.K;
    const size_t B_elems = (size_t)spec.K * spec.N;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "correctness.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "correctness.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C");
    __bf16* C_ref = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C_ref");
    float* C_ref_f = (float*)alloc_hbm(C_elems * sizeof(float), base_numa, "correctness.C_ref_f");
    if (!A || !B || !C || !C_ref || !C_ref_f) {
        return -1;
    }

    fill_test_bf16(A, spec.M, spec.K, spec.K, 11, 7);
    fill_test_bf16(B, spec.K, spec.N, spec.N, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));
    memset(C_ref_f, 0, C_elems * sizeof(float));

    const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, XhblasNoTrans,
                                          spec.M, spec.N, spec.K, 1.0f,
                                          A, spec.K, B, spec.N, 0.0f, C, spec.N);
    reference_gemm_nt(A, B, C_ref_f, spec.M, spec.N, spec.K);
    writeback_float_to_bf16(C_ref_f, C_ref, spec.M, spec.N, spec.N, 1.0f);

    const int bad = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);
    printf("case M=%d N=%d K=%d status=%d bad=%d\n",
           spec.M, spec.N, spec.K, status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "correctness.A");
    free_hbm(B, B_elems * sizeof(__bf16), "correctness.B");
    free_hbm(C, C_elems * sizeof(__bf16), "correctness.C");
    free_hbm(C_ref, C_elems * sizeof(__bf16), "correctness.C_ref");
    free_hbm(C_ref_f, C_elems * sizeof(float), "correctness.C_ref_f");
    return (status == 0 && bad == 0) ? 0 : -1;
}

static int run_benchmark_case(const GemmSpec& spec, int warmup, int iters)
{
    const size_t A_elems = (size_t)spec.M * spec.K;
    const size_t B_elems = (size_t)spec.K * spec.N;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "benchmark.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "benchmark.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "benchmark.C");
    if (!A || !B || !C) {
        return -1;
    }

    fill_test_bf16(A, spec.M, spec.K, spec.K, 11, 7);
    fill_test_bf16(B, spec.K, spec.N, spec.N, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));

    for (int i = 0; i < warmup; ++i) {
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, XhblasNoTrans,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, spec.K, B, spec.N, 0.0f, C, spec.N);
        if (status != 0) {
            free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
            free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
            free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
            return status;
        }
    }

    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const double start = omp_get_wtime();
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, XhblasNoTrans,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, spec.K, B, spec.N, 0.0f, C, spec.N);
        const double end = omp_get_wtime();
        if (status != 0) {
            free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
            free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
            free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
            return status;
        }
        total += end - start;
    }

    const double avg = total / iters;
    const double gflops = (2.0 * (double)spec.M * (double)spec.N * (double)spec.K) / avg / 1.0e9;
    printf("bench M=%d N=%d K=%d avg_time=%.6f s gflops=%.6f sink=%f\n",
           spec.M, spec.N, spec.K, avg, gflops, (float)C[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
    free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
    free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
    return 0;
}

} // namespace final_demo2

int main()
{
    if (getenv("FINAL_DEMO_BENCH")) {
        const int warmup = final_demo2::getenv_int_or("FINAL_DEMO_BENCH_WARMUP", 1);
        const int iters = final_demo2::getenv_int_or("FINAL_DEMO_BENCH_ITERS", 5);
        const int bench_threads = final_demo2::getenv_int_or("FINAL_DEMO_BENCH_THREADS", 32);
        char thread_buf[32];
        snprintf(thread_buf, sizeof(thread_buf), "%d", bench_threads);
        setenv("FINAL_GEMM_THREADS", thread_buf, 1);

        int failures = 0;
        for (size_t i = 0; i < sizeof(final_demo2::kSupportedSpecs) / sizeof(final_demo2::kSupportedSpecs[0]); ++i) {
            failures += final_demo2::run_benchmark_case(final_demo2::kSupportedSpecs[i], warmup, iters) != 0;
        }
        printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;
    for (size_t i = 0; i < sizeof(final_demo2::kSupportedSpecs) / sizeof(final_demo2::kSupportedSpecs[0]); ++i) {
        failures += final_demo2::run_correctness_case(final_demo2::kSupportedSpecs[i]) != 0;
    }
    printf("overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
#endif
