#include <arm_sve.h>
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

namespace outer_demo {

constexpr int kCoresPerNuma = 38;
static const char* kModuleName = "outer";

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
};

static const GemmSpec kSupportedSpecs[] = {
    {15360,  256},
    { 3840,  256},
    { 1024,  256},
    {  256, 1024},
};
static int ensure_workspace(size_t b_f_bytes, int numa_id, float** B_f)
{
    void* shared_b = NULL;
    if (bgemm_shared_workspace::ensure_buffers(0, b_f_bytes, 0, numa_id,
                                               "outer",
                                               NULL, &shared_b, NULL) != 0) {
        return -1;
    }
    *B_f = (float*)shared_b;
    return *B_f ? 0 : -1;
}

static const GemmSpec* find_spec(int M, int N, int K)
{
    if (K != 1) {
        return NULL;
    }
    for (size_t i = 0; i < sizeof(kSupportedSpecs) / sizeof(kSupportedSpecs[0]); ++i) {
        const GemmSpec& spec = kSupportedSpecs[i];
        if (spec.M == M && spec.N == N) {
            return &spec;
        }
    }
    return NULL;
}

static void preprocess_b(const __bf16* B, float* B_f, int N, float alpha)
{
    const int wl = svcntw();
    const svfloat32_t alpha_v = svdup_n_f32(alpha);
    for (int j = 0; j < N; j += wl) {
        const int cols = min_int(wl, N - j);
        svbool_t pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)cols);
        svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)cols);
        svbfloat16_t b16 = svld1_bf16(pg16, B + j);
        svuint16_t bits16 = svreinterpret_u16_bf16(b16);
        svuint32_t bits32 = svlsl_n_u32_z(pg32, svunpklo_u32(bits16), 16);
        svfloat32_t bf = svreinterpret_f32_u32(bits32);
        svfloat32_t scaled = svmul_f32_x(pg32, bf, alpha_v);
        svst1_f32(pg32, B_f + j, scaled);
    }
}

static void outer_product_row(const float* B_f, __bf16* C_row, int N, float a_scalar, const __bf16* bias)
{
    const int wl = svcntw();
    const svfloat32_t a_v = svdup_n_f32(a_scalar);
    for (int j = 0; j < N; j += wl) {
        const int cols = min_int(wl, N - j);
        svbool_t pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)cols);
        svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)cols);
        svfloat32_t b_v = svld1_f32(pg32, B_f + j);
        svfloat32_t c_v = svmul_f32_x(pg32, a_v, b_v);
        if (bias) {
            svfloat32_t bias_v = load_bf16_as_f32(pg32, pg16, bias + j);
            c_v = svadd_f32_x(pg32, c_v, bias_v);
        }
        store_f32_as_bf16(pg32, pg16, C_row + j, c_v);
    }
}

static int run_outer_gemm(const __bf16* A, const __bf16* B, __bf16* C, int M, int N, float alpha,
                          const __bf16* bias)
{
    float* B_f = NULL;
    const int base_numa = detect_base_numa();
    if (ensure_workspace((size_t)N * sizeof(float), base_numa, &B_f) != 0) {
        return -7;
    }
    preprocess_b(B, B_f, N, alpha);

    const int requested_threads = getenv_int_or("FINAL_GEMM_THREADS", 32);
    const int active_threads = min_int(requested_threads, M);
    #pragma omp parallel num_threads(active_threads)
    {
        const int rank = omp_get_thread_num();
        bind_to_cpu(base_numa, rank);
        #pragma omp for schedule(static)
        for (int i = 0; i < M; ++i) {
            outer_product_row(B_f, C + (size_t)i * N, N, (float)A[i], bias);
        }
    }
    return 0;
}

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
    if (trans_a != XhblasTrans) {
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
    if (lda != M || ldb != N || ldc != N) {
        return -5;
    }
    if (!find_spec(M, N, K)) {
        return -6;
    }

    const int status = run_outer_gemm(A, B, C, M, N, alpha, sign == 1 ? ba : NULL);
    if (status != 0) {
        return status;
    }
    return 0;
}

static const char* status_message(int status)
{
    switch (status) {
        case 0: return "success";
        case -1: return "only row-major layout is supported";
        case -2: return "only Trans A is supported";
        case -3: return "only NoTrans B is supported";
        case -4: return "invalid dimensions";
        case -5: return "invalid leading dimensions";
        case -6: return "unsupported shape";
        case -7: return "workspace allocation failed";
        case -9: return "non-zero beta is unsupported";
        default: return "unknown error";
    }
}

} // namespace outer_demo

extern "C" int xhblas_gemm(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    return outer_demo::xhblas_gemm_status(order, trans_a, trans_b,
                                          M, N, K, alpha,
                                          A, lda, B, ldb, beta, C, ldc, ba, sign);
}

#ifdef FINAL_DEMO_TEST_MAIN
namespace outer_demo {

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

static void reference_outer_gemm(const __bf16* A, const __bf16* B, __bf16* C, int M, int N, float alpha)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const float a = (float)A[i] * alpha;
        for (int j = 0; j < N; ++j) {
            C[(size_t)i * N + j] = trunc_f32_to_bf16(a * (float)B[j]);
        }
    }
}

static int run_correctness_case(const GemmSpec& spec)
{
    const size_t A_elems = (size_t)spec.M;
    const size_t B_elems = (size_t)spec.N;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "correctness.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "correctness.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C");
    __bf16* C_ref = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C_ref");
    if (!A || !B || !C || !C_ref) {
        return -1;
    }

    fill_test_bf16(A, 1, spec.M, spec.M, 11, 7);
    fill_test_bf16(B, 1, spec.N, spec.N, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));

    const int status = xhblas_gemm_status(XhblasRowMajor, XhblasTrans, XhblasNoTrans,
                                          spec.M, spec.N, 1, 1.0f,
                                          A, spec.M, B, spec.N, 0.0f, C, spec.N);
    reference_outer_gemm(A, B, C_ref, spec.M, spec.N, 1.0f);
    const int bad = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);
    printf("case M=%d N=%d K=1 status=%d bad=%d\n", spec.M, spec.N, status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "correctness.A");
    free_hbm(B, B_elems * sizeof(__bf16), "correctness.B");
    free_hbm(C, C_elems * sizeof(__bf16), "correctness.C");
    free_hbm(C_ref, C_elems * sizeof(__bf16), "correctness.C_ref");
    return (status == 0 && bad == 0) ? 0 : -1;
}

static int run_benchmark_case(const GemmSpec& spec, int warmup, int iters)
{
    const size_t A_elems = (size_t)spec.M;
    const size_t B_elems = (size_t)spec.N;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "benchmark.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "benchmark.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "benchmark.C");
    if (!A || !B || !C) {
        return -1;
    }

    fill_test_bf16(A, 1, spec.M, spec.M, 11, 7);
    fill_test_bf16(B, 1, spec.N, spec.N, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));

    for (int i = 0; i < warmup; ++i) {
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasTrans, XhblasNoTrans,
                                              spec.M, spec.N, 1, 1.0f,
                                              A, spec.M, B, spec.N, 0.0f, C, spec.N);
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
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasTrans, XhblasNoTrans,
                                              spec.M, spec.N, 1, 1.0f,
                                              A, spec.M, B, spec.N, 0.0f, C, spec.N);
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
    const double gflops = (2.0 * (double)spec.M * (double)spec.N) / avg / 1.0e9;
    printf("bench M=%d N=%d K=1 avg_time=%.6f s gflops=%.6f sink=%f\n",
           spec.M, spec.N, avg, gflops, (float)C[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
    free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
    free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
    return 0;
}

} // namespace outer_demo

int main()
{
    if (getenv("FINAL_DEMO_BENCH")) {
        const int warmup = outer_demo::getenv_int_or("FINAL_DEMO_BENCH_WARMUP", 1);
        const int iters = outer_demo::getenv_int_or("FINAL_DEMO_BENCH_ITERS", 5);
        const int bench_threads = outer_demo::getenv_int_or("FINAL_DEMO_BENCH_THREADS", 32);
        char thread_buf[32];
        snprintf(thread_buf, sizeof(thread_buf), "%d", bench_threads);
        setenv("FINAL_GEMM_THREADS", thread_buf, 1);

        int failures = 0;
        for (size_t i = 0; i < sizeof(outer_demo::kSupportedSpecs) / sizeof(outer_demo::kSupportedSpecs[0]); ++i) {
            failures += outer_demo::run_benchmark_case(outer_demo::kSupportedSpecs[i], warmup, iters) != 0;
        }
        printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;
    for (size_t i = 0; i < sizeof(outer_demo::kSupportedSpecs) / sizeof(outer_demo::kSupportedSpecs[0]); ++i) {
        failures += outer_demo::run_correctness_case(outer_demo::kSupportedSpecs[i]) != 0;
    }
    printf("overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
#endif
