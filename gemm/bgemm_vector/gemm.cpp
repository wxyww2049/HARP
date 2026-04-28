#include <arm_sve.h>
#include <arm_acle.h>
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

namespace vector_demo {

constexpr int kCoresPerNuma = 38;
static const char* kModuleName = "vector";

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

struct GemmSpec {
    int M;
    int N;
    int K;
    XHBLAS_TRANSPOSE trans_b;
};

static const GemmSpec kSupportedSpecs[] = {
    {1,   256, 15360, XhblasNoTrans},
    {1, 15360,   256, XhblasTrans},
    {1,   256,  1024, XhblasTrans},
    {1,  1024,   256, XhblasTrans},
    {1,   256,  1024, XhblasNoTrans},
    {1,  3840,   256, XhblasTrans},
    {1,   256,  3840, XhblasNoTrans},
    {1,  1024,   256, XhblasNoTrans},
};
static int ensure_workspace(size_t a_f_bytes, int numa_id, float** A_f)
{
    void* shared_a = NULL;
    if (bgemm_shared_workspace::ensure_buffers(a_f_bytes, 0, 0, numa_id,
                                               "vector",
                                               &shared_a, NULL, NULL) != 0) {
        return -1;
    }
    *A_f = (float*)shared_a;
    return *A_f ? 0 : -1;
}

static const GemmSpec* find_spec(int M, int N, int K, XHBLAS_TRANSPOSE trans_b)
{
    for (size_t i = 0; i < sizeof(kSupportedSpecs) / sizeof(kSupportedSpecs[0]); ++i) {
        const GemmSpec& spec = kSupportedSpecs[i];
        if (spec.M == M && spec.N == N && spec.K == K && spec.trans_b == trans_b) {
            return &spec;
        }
    }
    return NULL;
}

static void preprocess_a(const __bf16* A, float* A_f, int K, float alpha)
{
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < K; ++p) {
        A_f[p] = (float)A[p] * alpha;
    }
}

static void gemv_nt_kernel(const float* A_f, const __bf16* B, __bf16* C, int N, int K, int requested_threads,
                           const __bf16* bias)
{
    const int wl = svcntw();
    const int work_items = (N + wl - 1) / wl;
    const int active_threads = max_int(1, min_int(requested_threads, work_items));
    const int base_numa = detect_base_numa();

    #pragma omp parallel num_threads(active_threads)
    {
        const int rank = omp_get_thread_num();
        bind_to_cpu(base_numa, rank);

        #pragma omp for schedule(static)
        for (int j = 0; j < N; j += wl) {
            const int cols = min_int(wl, N - j);
            svbool_t pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)cols);
            svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)cols);
            svfloat32_t acc = svdup_n_f32(0.0f);
            for (int p = 0; p < K; ++p) {
                svfloat32_t a_v = svdup_n_f32(A_f[p]);
                svfloat32_t b_v = load_bf16_as_f32(pg32, pg16, B + (size_t)p * N + j);
                acc = svmla_f32_m(pg32, acc, b_v, a_v);
            }
            if (bias) {
                svfloat32_t bias_v = load_bf16_as_f32(pg32, pg16, bias + j);
                acc = svadd_f32_x(pg32, acc, bias_v);
            }
            store_f32_as_bf16(pg32, pg16, C + j, acc);
        }
    }
}

static float dot_bf16_row_sve(const float* A_f, const __bf16* B_row, int K)
{
    const int wl = svcntw();
    svfloat32_t acc = svdup_n_f32(0.0f);
    for (int p = 0; p < K; p += wl) {
        const int cols = min_int(wl, K - p);
        svbool_t pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)cols);
        svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)cols);
        svfloat32_t a_v = svld1_f32(pg32, A_f + p);
        svfloat32_t b_v = load_bf16_as_f32(pg32, pg16, B_row + p);
        acc = svmla_f32_m(pg32, acc, a_v, b_v);
    }
    return svaddv_f32(svptrue_b32(), acc);
}

static void gemv_t_kernel(const float* A_f, const __bf16* B_t, __bf16* C, int N, int K, int requested_threads,
                          const __bf16* bias)
{
    const int block = 4;
    const int work_items = (N + block - 1) / block;
    const int active_threads = max_int(1, min_int(requested_threads, work_items));
    const int base_numa = detect_base_numa();

    #pragma omp parallel num_threads(active_threads)
    {
        const int rank = omp_get_thread_num();
        bind_to_cpu(base_numa, rank);

        #pragma omp for schedule(static)
        for (int j = 0; j < N; j += block) {
            const int cols = min_int(block, N - j);
            for (int jj = 0; jj < cols; ++jj) {
                float acc = dot_bf16_row_sve(A_f, B_t + (size_t)(j + jj) * K, K);
                if (bias) {
                    acc += (float)bias[j + jj];
                }
                C[j + jj] = trunc_f32_to_bf16(acc);
            }
        }
    }
}

static int run_vector_gemm(const __bf16* A, const __bf16* B, __bf16* C,
                           int N, int K, XHBLAS_TRANSPOSE trans_b, float alpha, const __bf16* bias)
{
    float* A_f = NULL;
    const int base_numa = detect_base_numa();
    if (ensure_workspace((size_t)K * sizeof(float), base_numa, &A_f) != 0) {
        return -7;
    }

    preprocess_a(A, A_f, K, alpha);
    const int requested_threads = getenv_int_or("FINAL_GEMM_THREADS", 32);
    if (trans_b == XhblasNoTrans) {
        gemv_nt_kernel(A_f, B, C, N, K, requested_threads, bias);
    } else {
        gemv_t_kernel(A_f, B, C, N, K, requested_threads, bias);
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
    if (trans_a != XhblasNoTrans) {
        return -2;
    }
    if (trans_b != XhblasNoTrans && trans_b != XhblasTrans) {
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
    const int a_cols = K;
    const int b_cols = trans_b == XhblasNoTrans ? N : K;
    if (lda != a_cols || ldb != b_cols || ldc != N) {
        return -5;
    }
    if (!find_spec(M, N, K, trans_b)) {
        return -6;
    }

    return run_vector_gemm(A, B, C, N, K, trans_b, alpha, sign == 1 ? ba : NULL);
}

static const char* status_message(int status)
{
    switch (status) {
        case 0: return "success";
        case -1: return "only row-major layout is supported";
        case -2: return "only NoTrans A is supported";
        case -3: return "unsupported TransB";
        case -4: return "invalid dimensions";
        case -5: return "invalid leading dimensions";
        case -6: return "unsupported shape";
        case -7: return "workspace allocation failed";
        case -9: return "non-zero beta is unsupported";
        default: return "unknown error";
    }
}

} // namespace vector_demo

extern "C" int xhblas_gemm(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    return vector_demo::xhblas_gemm_status(order, trans_a, trans_b,
                                           M, N, K, alpha,
                                           A, lda, B, ldb, beta, C, ldc, ba, sign);
}

#ifdef FINAL_DEMO_TEST_MAIN
namespace vector_demo {

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

static void reference_vec_gemm(const __bf16* A, const __bf16* B, __bf16* C,
                               int N, int K, XHBLAS_TRANSPOSE trans_b, float alpha)
{
    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int p = 0; p < K; ++p) {
            const float a = (float)A[p] * alpha;
            const float b = trans_b == XhblasNoTrans ? (float)B[(size_t)p * N + j]
                                                     : (float)B[(size_t)j * K + p];
            acc += a * b;
        }
        C[j] = trunc_f32_to_bf16(acc);
    }
}

static int run_correctness_case(const GemmSpec& spec)
{
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)spec.K;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "correctness.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "correctness.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C");
    __bf16* C_ref = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C_ref");
    if (!A || !B || !C || !C_ref) {
        return -1;
    }

    fill_test_bf16(A, 1, spec.K, spec.K, 11, 7);
    fill_test_bf16(B, b_rows, b_cols, b_cols, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));

    const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, spec.trans_b,
                                          spec.M, spec.N, spec.K, 1.0f,
                                          A, spec.K, B, b_cols, 0.0f, C, spec.N, NULL, 0);
    reference_vec_gemm(A, B, C_ref, spec.N, spec.K, spec.trans_b, 1.0f);
    const int bad = compare_bf16_tiles(C_ref, C, 1, spec.N, spec.N);
    printf("case M=%d N=%d K=%d transB=%d status=%d bad=%d\n",
           spec.M, spec.N, spec.K, (int)spec.trans_b, status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "correctness.A");
    free_hbm(B, B_elems * sizeof(__bf16), "correctness.B");
    free_hbm(C, C_elems * sizeof(__bf16), "correctness.C");
    free_hbm(C_ref, C_elems * sizeof(__bf16), "correctness.C_ref");
    return (status == 0 && bad == 0) ? 0 : -1;
}

static int run_benchmark_case(const GemmSpec& spec, int warmup, int iters)
{
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)spec.K;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.N;
    const int base_numa = detect_base_numa();
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "benchmark.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "benchmark.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), base_numa, "benchmark.C");
    if (!A || !B || !C) {
        return -1;
    }

    fill_test_bf16(A, 1, spec.K, spec.K, 11, 7);
    fill_test_bf16(B, b_rows, b_cols, b_cols, 13, 5);
    memset(C, 0, C_elems * sizeof(__bf16));

    for (int i = 0; i < warmup; ++i) {
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, spec.trans_b,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, spec.K, B, b_cols, 0.0f, C, spec.N, NULL, 0);
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
        const int status = xhblas_gemm_status(XhblasRowMajor, XhblasNoTrans, spec.trans_b,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, spec.K, B, b_cols, 0.0f, C, spec.N, NULL, 0);
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
    const double gflops = (2.0 * (double)spec.N * (double)spec.K) / avg / 1.0e9;
    printf("bench M=%d N=%d K=%d transB=%d avg_time=%.6f s gflops=%.6f sink=%f\n",
           spec.M, spec.N, spec.K, (int)spec.trans_b, avg, gflops, (float)C[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
    free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
    free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
    return 0;
}

} // namespace vector_demo

int main()
{
    if (getenv("FINAL_DEMO_BENCH")) {
        const int warmup = vector_demo::getenv_int_or("FINAL_DEMO_BENCH_WARMUP", 1);
        const int iters = vector_demo::getenv_int_or("FINAL_DEMO_BENCH_ITERS", 5);
        const int bench_threads = vector_demo::getenv_int_or("FINAL_DEMO_BENCH_THREADS", 32);
        char thread_buf[32];
        snprintf(thread_buf, sizeof(thread_buf), "%d", bench_threads);
        setenv("FINAL_GEMM_THREADS", thread_buf, 1);

        int failures = 0;
        for (size_t i = 0; i < sizeof(vector_demo::kSupportedSpecs) / sizeof(vector_demo::kSupportedSpecs[0]); ++i) {
            failures += vector_demo::run_benchmark_case(vector_demo::kSupportedSpecs[i], warmup, iters) != 0;
        }
        printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;
    for (size_t i = 0; i < sizeof(vector_demo::kSupportedSpecs) / sizeof(vector_demo::kSupportedSpecs[0]); ++i) {
        failures += vector_demo::run_correctness_case(vector_demo::kSupportedSpecs[i]) != 0;
    }
    printf("overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
#endif
