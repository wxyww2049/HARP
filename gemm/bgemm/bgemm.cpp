#include "bgemm.h"
#include "../memory_node.h"
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <numaif.h>
#include <stddef.h>
#include <sys/mman.h>
#ifdef FINAL_DEMO_TEST_MAIN
#include <mpi.h>
#endif

namespace unified_bgemm {

constexpr size_t kSharedABytes = 100ull * 1024ull * 1024ull;
constexpr size_t kSharedBBytes = 100ull * 1024ull * 1024ull;
constexpr size_t kSharedCBytes = 200ull * 1024ull * 1024ull;
constexpr int kCoresPerNuma = 38;
constexpr int kNumaOffset = 16;
constexpr unsigned long kMbindMaxNodes = 64;
constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;

inline size_t align_to_hugepage(size_t size)
{
    return (size + kHugePageSize - 1) & ~(kHugePageSize - 1);
}

static void* alloc_hbm(size_t size, int numa_id, const char* tag)
{
    const size_t mapped_size = align_to_hugepage(size);
    void* ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED) {
        ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "[shared-workspace] mmap failed: errno=%d (%s)\n",
                    errno, strerror(errno));
            return NULL;
        }
    }

    const unsigned long mask = 1ul << harp_memory_node(numa_id + 16);
    long rc = mbind(ptr, mapped_size, MPOL_BIND, &mask, kMbindMaxNodes, 0);
    if (rc != 0) {
        munmap(ptr, mapped_size);
        ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "[shared-workspace] mmap failed: errno=%d (%s)\n",
                    errno, strerror(errno));
            return NULL;
        }
        rc = mbind(ptr, mapped_size, MPOL_BIND, &mask, kMbindMaxNodes, 0);
        if (rc != 0) {
            fprintf(stderr,
                    "[shared-workspace] mbind failed (numa=%d): errno=%d (%s)\n",
                    numa_id, errno, strerror(errno));
        }
    }

    memset(ptr, 0, mapped_size);
    return ptr;
}

static void free_hbm(void* ptr, size_t size, const char* tag)
{
    if (!ptr) {
        return;
    }
    const size_t mapped_size = align_to_hugepage(size);
    if (munmap(ptr, mapped_size) != 0) {
        fprintf(stderr, "[shared-workspace] munmap failed: errno=%d (%s)\n",
                errno, strerror(errno));
    }
}

static void* alloc_host(size_t size)
{
    return alloc_hbm(size, 0, "host");
}

static void free_host(void* ptr)
{
    free_hbm(ptr, 0, "host");
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

static const char* order_name(XHBLAS_ORDER order)
{
    switch (order) {
        case XhblasRowMajor: return "RowMajor";
        case XhblasColMajor: return "ColMajor";
    }
    return "UnknownOrder";
}

static const char* transpose_name(XHBLAS_TRANSPOSE trans)
{
    switch (trans) {
        case XhblasNoTrans: return "NoTrans";
        case XhblasTrans: return "Trans";
        case XhblasConjTrans: return "ConjTrans";
    }
    return "UnknownTrans";
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

enum class ImplKind {
    Demo1,
    Demo2,
    Outer,
    Vector,
};

struct GemmSpec {
    int M;
    int N;
    int K;
    XHBLAS_TRANSPOSE trans_a;
    XHBLAS_TRANSPOSE trans_b;
    ImplKind impl;
};

static const GemmSpec kDemo1Specs[] = {
    {10240, 3840,  768, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 3840, 2560, 1024, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 3840,  768, 1024, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 3072,   64, 3840, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 3840,   64, 1024, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    {10240, 3840,  256, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 3840, 3840,  768, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 3840, 3840,  256, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
};

static const GemmSpec kDemo1ExtraSpecs[] = {
    { 768,  3840, 10240, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 768,  3840, 10240, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    { 768, 10240,  3840, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    {3840, 10240,   768, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 768, 10240,  3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    {1024,  3840,  2560, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 256,  3840, 10240, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 256, 10240,  3840, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 768,  3840,  3840, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 256,  3840, 10240, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    { 768,  3840,  3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    {1024,  3840,   768, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    {3840, 10240,   256, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    { 256, 10240,  3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    {1024,  3840,    64, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 256,  3840,  3840, XhblasNoTrans, XhblasTrans,   ImplKind::Demo1},
    { 256,  3840,  3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
    {  64,  3840,  3072, XhblasTrans,   XhblasNoTrans, ImplKind::Demo1},
    {3072,  3840,    64, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo1},
};

static const GemmSpec kDemo2Specs[] = {
    {1024, 2560, 3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo2},
    {1024,   64, 3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo2},
    {1024,  768, 3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Demo2},
};

static const GemmSpec kOuterSpecs[] = {
    {15360,  256, 1, XhblasTrans,   XhblasNoTrans, ImplKind::Outer},
    { 3840,  256, 1, XhblasTrans,   XhblasNoTrans, ImplKind::Outer},
    { 1024,  256, 1, XhblasTrans,   XhblasNoTrans, ImplKind::Outer},
    {  256, 1024, 1, XhblasTrans,   XhblasNoTrans, ImplKind::Outer},
};

static const GemmSpec kVectorSpecs[] = {
    {1,   256, 15360, XhblasNoTrans, XhblasNoTrans, ImplKind::Vector},
    {1, 15360,   256, XhblasNoTrans, XhblasTrans,   ImplKind::Vector},
    {1,   256,  1024, XhblasNoTrans, XhblasTrans,   ImplKind::Vector},
    {1,  1024,   256, XhblasNoTrans, XhblasTrans,   ImplKind::Vector},
    {1,   256,  1024, XhblasNoTrans, XhblasNoTrans, ImplKind::Vector},
    {1,  3840,   256, XhblasNoTrans, XhblasTrans,   ImplKind::Vector},
    {1,   256,  3840, XhblasNoTrans, XhblasNoTrans, ImplKind::Vector},
    {1,  1024,   256, XhblasNoTrans, XhblasNoTrans, ImplKind::Vector},
};

static const GemmSpec* find_spec_in_list(const GemmSpec* specs, size_t count,
                                         int M, int N, int K,
                                         XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b)
{
    for (size_t i = 0; i < count; ++i) {
        const GemmSpec& spec = specs[i];
        if (spec.M == M && spec.N == N && spec.K == K &&
            spec.trans_a == trans_a && spec.trans_b == trans_b) {
            return &spec;
        }
    }
    return NULL;
}

static const GemmSpec* find_spec(int M, int N, int K,
                                 XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b)
{
    const GemmSpec* spec = find_spec_in_list(kDemo1Specs, sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]),
                                             M, N, K, trans_a, trans_b);
    if (spec) {
        return spec;
    }
    spec = find_spec_in_list(kDemo1ExtraSpecs, sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]),
                             M, N, K, trans_a, trans_b);
    if (spec) {
        return spec;
    }
    spec = find_spec_in_list(kDemo2Specs, sizeof(kDemo2Specs) / sizeof(kDemo2Specs[0]),
                             M, N, K, trans_a, trans_b);
    if (spec) {
        return spec;
    }
    spec = find_spec_in_list(kOuterSpecs, sizeof(kOuterSpecs) / sizeof(kOuterSpecs[0]),
                             M, N, K, trans_a, trans_b);
    if (spec) {
        return spec;
    }
    return find_spec_in_list(kVectorSpecs, sizeof(kVectorSpecs) / sizeof(kVectorSpecs[0]),
                             M, N, K, trans_a, trans_b);
}

extern "C" int bgemm_demo1_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                int M, int N, int K, float alpha,
                                const __bf16* A, int lda,
                                const __bf16* B, int ldb,
                                float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int bgemm_demo2_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                int M, int N, int K, float alpha,
                                const __bf16* A, int lda,
                                const __bf16* B, int ldb,
                                float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int bgemm_outer_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                int M, int N, int K, float alpha,
                                const __bf16* A, int lda,
                                const __bf16* B, int ldb,
                                float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int bgemm_vector_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                 int M, int N, int K, float alpha,
                                 const __bf16* A, int lda,
                                 const __bf16* B, int ldb,
                                 float beta, __bf16* C, int ldc, __bf16* ba, int sign);

static int run_selected_impl(const GemmSpec& spec,
                             XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                             int M, int N, int K, float alpha,
                             const __bf16* A, int lda,
                             const __bf16* B, int ldb,
                             float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    switch (spec.impl) {
        case ImplKind::Demo1:
            return bgemm_demo1_impl(order, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ba, sign);
        case ImplKind::Demo2:
            return bgemm_demo2_impl(order, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ba, sign);
        case ImplKind::Outer:
            return bgemm_outer_impl(order, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ba, sign);
        case ImplKind::Vector:
            return bgemm_vector_impl(order, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ba, sign);
    }
    return -6;
}

static int scale_only(float beta, __bf16* C, int M, int N, int ldc)
{
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * (size_t)ldc * sizeof(__bf16));
        return 0;
    }
    return -8;
}

static int xhblas_gemm_status_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                   int M, int N, int K, float alpha,
                                   const __bf16* A, int lda,
                                   const __bf16* B, int ldb,
                                   float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    trans_a = normalize_transpose(trans_a);
    trans_b = normalize_transpose(trans_b);

    if (order != XhblasRowMajor) {
        return -1;
    }
    if (trans_a != XhblasNoTrans && trans_a != XhblasTrans) {
        return -2;
    }
    if (trans_b != XhblasNoTrans && trans_b != XhblasTrans) {
        return -3;
    }
    if (M < 0 || N < 0 || K < 0) {
        return -4;
    }
    const int a_cols = trans_a == XhblasNoTrans ? K : M;
    const int b_cols = trans_b == XhblasNoTrans ? N : K;
    if (lda != a_cols || ldb != b_cols || ldc != N) {
        return -5;
    }

    const GemmSpec* spec = find_spec(M, N, K, trans_a, trans_b);
    if (!spec) {
        return -6;
    }

    if (beta != 0.0f) {
        return -8;
    }
    if (K == 0 || alpha == 0.0f) {
        return scale_only(beta, C, M, N, ldc);
    }

    return run_selected_impl(*spec, order, trans_a, trans_b,
                             M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ba, sign);
}

static const char* status_message_impl(int status)
{
    switch (status) {
        case 0: return "success";
        case -1: return "only row-major layout is supported";
        case -2: return "unsupported TransA";
        case -3: return "unsupported TransB";
        case -4: return "invalid dimensions";
        case -5: return "invalid leading dimensions";
        case -6: return "unsupported shape";
        case -7: return "workspace allocation failed";
        case -8: return "non-zero beta is unsupported in the bf16-output path";
        case -9: return "backend rejected request";
        default: return "unknown error";
    }
}

} // namespace unified_bgemm

extern "C" int bgemm_supports(int m, int n, int k, int trans_a, int trans_b)
{
    return unified_bgemm::find_spec(m, n, k, (XHBLAS_TRANSPOSE)trans_a, (XHBLAS_TRANSPOSE)trans_b) != nullptr;
}

extern "C" int bgemm_kernel(int order, int trans_a, int trans_b,
                            int M, int N, int K, float alpha,
                            const __bf16* A, int lda,
                            const __bf16* B, int ldb,
                            float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    int result = unified_bgemm::xhblas_gemm_status_impl((XHBLAS_ORDER)order,
                                                        (XHBLAS_TRANSPOSE)trans_a,
                                                        (XHBLAS_TRANSPOSE)trans_b,
                                                        M, N, K, alpha,
                                                        A, lda, B, ldb, beta, C, ldc, ba, sign);
    if (result != 0) {
        if (result == -6) {
            fprintf(stderr,
                    "[bgemm] unsupported shape: order=%s(%d) M=%d N=%d K=%d transA=%s(%d) transB=%s(%d) lda=%d ldb=%d ldc=%d sign=%d\n",
                    unified_bgemm::order_name((XHBLAS_ORDER)order), order,
                    M, N, K,
                    unified_bgemm::transpose_name((XHBLAS_TRANSPOSE)trans_a), trans_a,
                    unified_bgemm::transpose_name((XHBLAS_TRANSPOSE)trans_b), trans_b,
                    lda, ldb, ldc, sign);
        } else {
            fprintf(stderr, "bgemm failed: %s\n", xhblas_gemm_status_message(result));
        }
    }
    if (result != 0 || sign == 0 || ba == NULL) {
        return result;
    }

    if (sign == 1) {
        return result;
    } else if (sign == 2) {
        #pragma omp parallel for num_threads(32)
        for (int j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[i * M + j];
            }
            ba[j] = sum;
        }
    }
    return result;
}

extern "C" int xhblas_gemm(int order, int trans_a, int trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    return bgemm_kernel(order, trans_a, trans_b,
                        M, N, K, alpha,
                        A, lda, B, ldb, beta, C, ldc, ba, sign);
}

extern "C" const char* xhblas_gemm_status_message(int status)
{
    return unified_bgemm::status_message_impl(status);
}

#ifdef FINAL_DEMO_TEST_MAIN
namespace unified_bgemm {

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

static void fill_inputs_for_mode(__bf16* A, __bf16* B, int M, int N, int K,
                                 XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b)
{
    if (trans_a == XhblasTrans) {
        fill_test_bf16(A, K, M, M, 11, 7);
    } else {
        fill_test_bf16(A, M, K, K, 11, 7);
    }
    if (trans_b == XhblasTrans) {
        fill_test_bf16(B, N, K, K, 13, 5);
    } else {
        fill_test_bf16(B, K, N, N, 13, 5);
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
            if ((av & 0x7fff) == 0 && (bv & 0x7fff) == 0) {
                continue;
            }
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

static void reference_gemm_dispatch(const __bf16* A, const __bf16* B, float* C,
                                    int M, int N, int K,
                                    XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                const float a = trans_a == XhblasTrans ? (float)A[(size_t)p * M + i]
                                                       : (float)A[(size_t)i * K + p];
                const float b = trans_b == XhblasTrans ? (float)B[(size_t)j * K + p]
                                                       : (float)B[(size_t)p * N + j];
                acc += a * b;
            }
            C[(size_t)i * N + j] = acc;
        }
    }
}

static bool same_problem(const GemmSpec& a, const GemmSpec& b)
{
    return a.M == b.M && a.N == b.N && a.K == b.K &&
           a.trans_a == b.trans_a && a.trans_b == b.trans_b;
}

static bool seen_earlier_problem(const GemmSpec& spec,
                                 const GemmSpec* list, size_t count,
                                 size_t end)
{
    const size_t stop = end < count ? end : count;
    for (size_t i = 0; i < stop; ++i) {
        if (same_problem(spec, list[i])) {
            return true;
        }
    }
    return false;
}

static bool seen_in_demo1(const GemmSpec& spec)
{
    for (size_t i = 0; i < sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]); ++i) {
        if (same_problem(spec, kDemo1Specs[i])) {
            return true;
        }
    }
    return false;
}

static const char* impl_name(ImplKind impl)
{
    switch (impl) {
        case ImplKind::Demo1: return "demo1";
        case ImplKind::Demo2: return "demo2";
        case ImplKind::Outer: return "outer";
        case ImplKind::Vector: return "vector";
    }
    return "unknown";
}

static int run_correctness_case(const GemmSpec& spec)
{
    const int a_rows = spec.trans_a == XhblasTrans ? spec.K : spec.M;
    const int a_cols = spec.trans_a == XhblasTrans ? spec.M : spec.K;
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;

    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "correctness.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "correctness.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), -16, "correctness.C");
    __bf16* C_ref = (__bf16*)alloc_host(C_elems * sizeof(__bf16));
    float* C_ref_f = (float*)alloc_host(C_elems * sizeof(float));
    if (!A || !B || !C || !C_ref || !C_ref_f) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));
    memset(C_ref_f, 0, C_elems * sizeof(float));

    const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                    spec.M, spec.N, spec.K, 1.0f,
                                    A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                    NULL, 0);
    reference_gemm_dispatch(A, B, C_ref_f, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    writeback_float_to_bf16(C_ref_f, C_ref, spec.M, spec.N, spec.N, 1.0f);
    const int bad = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);

    printf("case impl=%s M=%d N=%d K=%d transA=%d transB=%d status=%d bad=%d\n",
           impl_name(spec.impl), spec.M, spec.N, spec.K,
           (int)spec.trans_a, (int)spec.trans_b, status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "correctness.A");
    free_hbm(B, B_elems * sizeof(__bf16), "correctness.B");
    free_hbm(C, C_elems * sizeof(__bf16), "correctness.C");
    free_host(C_ref);
    free_host(C_ref_f);
    return (status == 0 && bad == 0) ? 0 : -1;
}

static int run_benchmark_case(const GemmSpec& spec, int warmup, int iters)
{
    const int a_rows = spec.trans_a == XhblasTrans ? spec.K : spec.M;
    const int a_cols = spec.trans_a == XhblasTrans ? spec.M : spec.K;
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), rank, "benchmark.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), rank, "benchmark.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), rank-16, "benchmark.C");
    if (!A || !B || !C) {
        return -1;
    }
    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));

    for (int i = 0; i < warmup; ++i) {
        const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                        spec.M, spec.N, spec.K, 1.0f,
                                        A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                        NULL, 0);
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
        const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                        spec.M, spec.N, spec.K, 1.0f,
                                        A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                        NULL, 0);
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
    printf("bench impl=%s M=%d N=%d K=%d transA=%d transB=%d avg_time=%.6f s gflops=%.6f sink=%f\n",
           impl_name(spec.impl), spec.M, spec.N, spec.K,
           (int)spec.trans_a, (int)spec.trans_b, avg, gflops, (float)C[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
    free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
    free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
    return 0;
}

static int run_list_correctness(const GemmSpec* list, size_t count)
{
    int failures = 0;
    for (size_t i = 0; i < count; ++i) {
        failures += run_correctness_case(list[i]) != 0;
    }
    return failures;
}

static int run_list_benchmark(const GemmSpec* list, size_t count, int warmup, int iters)
{
    int failures = 0;
    for (size_t i = 0; i < count; ++i) {
        failures += run_benchmark_case(list[i], warmup, iters) != 0;
    }
    return failures;
}

static bool sign_applicable(const GemmSpec& spec, int sign)
{
    if (sign == 2) {
        return spec.trans_a == XhblasTrans;
    }
    return true;
}

static int run_sign_benchmark_case(const GemmSpec& spec, int sign, int warmup, int iters)
{
    if (!sign_applicable(spec, sign)) {
        return 0;
    }

    const int a_rows = spec.trans_a == XhblasTrans ? spec.K : spec.M;
    const int a_cols = spec.trans_a == XhblasTrans ? spec.M : spec.K;
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int ba_len = sign == 1 ? spec.N : spec.M;
    const size_t BA_elems = (size_t)ba_len;

    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "sign_bench.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "sign_bench.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), 0, "sign_bench.C");
    __bf16* BA = (__bf16*)alloc_hbm(BA_elems * sizeof(__bf16), 0, "sign_bench.BA");
    if (!A || !B || !C || !BA) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(BA, 0, BA_elems * sizeof(__bf16));
    if (sign == 1) {
        fill_test_bf16(BA, 1, ba_len, ba_len, 17, 3);
    }

    for (int i = 0; i < warmup; ++i) {
        const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                        spec.M, spec.N, spec.K, 1.0f,
                                        A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                        BA, sign);
        if (status != 0) {
            free_hbm(A, A_elems * sizeof(__bf16), "sign_bench.A");
            free_hbm(B, B_elems * sizeof(__bf16), "sign_bench.B");
            free_hbm(C, C_elems * sizeof(__bf16), "sign_bench.C");
            free_hbm(BA, BA_elems * sizeof(__bf16), "sign_bench.BA");
            return status;
        }
    }

    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const double start = omp_get_wtime();
        const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                        spec.M, spec.N, spec.K, 1.0f,
                                        A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                        BA, sign);
        const double end = omp_get_wtime();
        if (status != 0) {
            free_hbm(A, A_elems * sizeof(__bf16), "sign_bench.A");
            free_hbm(B, B_elems * sizeof(__bf16), "sign_bench.B");
            free_hbm(C, C_elems * sizeof(__bf16), "sign_bench.C");
            free_hbm(BA, BA_elems * sizeof(__bf16), "sign_bench.BA");
            return status;
        }
        total += end - start;
    }

    const double avg = total / iters;
    const double gflops = (2.0 * (double)spec.M * (double)spec.N * (double)spec.K) / avg / 1.0e9;
    printf("bench sign=%d impl=%s M=%d N=%d K=%d transA=%d transB=%d avg_time=%.6f s gflops=%.6f sinkC=%f sinkBA=%f\n",
           sign, impl_name(spec.impl), spec.M, spec.N, spec.K,
           (int)spec.trans_a, (int)spec.trans_b, avg, gflops,
           (float)C[0], (float)BA[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "sign_bench.A");
    free_hbm(B, B_elems * sizeof(__bf16), "sign_bench.B");
    free_hbm(C, C_elems * sizeof(__bf16), "sign_bench.C");
    free_hbm(BA, BA_elems * sizeof(__bf16), "sign_bench.BA");
    return 0;
}

static int run_list_sign_benchmark(const GemmSpec* list, size_t count, int sign, int warmup, int iters)
{
    int failures = 0;
    for (size_t i = 0; i < count; ++i) {
        failures += run_sign_benchmark_case(list[i], sign, warmup, iters) != 0;
    }
    return failures;
}

static int run_sign_suite_benchmark(int sign, int warmup, int iters)
{
    int failures = 0;
    failures += run_list_sign_benchmark(kDemo1Specs, sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]),
                                        sign, warmup, iters);
    for (size_t i = 0; i < sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]); ++i) {
        if (seen_in_demo1(kDemo1ExtraSpecs[i]) ||
            seen_earlier_problem(kDemo1ExtraSpecs[i],
                                 kDemo1ExtraSpecs,
                                 sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]),
                                 i)) {
            continue;
        }
        failures += run_sign_benchmark_case(kDemo1ExtraSpecs[i], sign, warmup, iters) != 0;
    }
    failures += run_list_sign_benchmark(kDemo2Specs, sizeof(kDemo2Specs) / sizeof(kDemo2Specs[0]),
                                        sign, warmup, iters);
    failures += run_list_sign_benchmark(kOuterSpecs, sizeof(kOuterSpecs) / sizeof(kOuterSpecs[0]),
                                        sign, warmup, iters);
    failures += run_list_sign_benchmark(kVectorSpecs, sizeof(kVectorSpecs) / sizeof(kVectorSpecs[0]),
                                        sign, warmup, iters);
    return failures;
}

static int run_sign1_correctness_case()
{
    const GemmSpec spec = {1, 256, 1024, XhblasNoTrans, XhblasNoTrans, ImplKind::Vector};
    const int a_rows = spec.M;
    const int a_cols = spec.K;
    const int b_rows = spec.K;
    const int b_cols = spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const size_t bias_elems = (size_t)spec.N;

    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "sign1.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "sign1.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), 0, "sign1.C");
    __bf16* C_ref = (__bf16*)alloc_host(C_elems * sizeof(__bf16));
    __bf16* bias = (__bf16*)alloc_hbm(bias_elems * sizeof(__bf16), 0, "sign1.bias");
    float* C_ref_f = (float*)alloc_host(C_elems * sizeof(float));
    if (!A || !B || !C || !C_ref || !bias || !C_ref_f) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    fill_test_bf16(bias, 1, spec.N, spec.N, 17, 3);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));
    memset(C_ref_f, 0, C_elems * sizeof(float));

    const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                    spec.M, spec.N, spec.K, 1.0f,
                                    A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                    bias, 1);

    reference_gemm_dispatch(A, B, C_ref_f, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    for (int i = 0; i < spec.M; ++i) {
        for (int j = 0; j < spec.N; ++j) {
            C_ref_f[(size_t)i * spec.N + j] += (float)bias[j];
        }
    }
    writeback_float_to_bf16(C_ref_f, C_ref, spec.M, spec.N, spec.N, 1.0f);
    const int bad = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);
    printf("case sign=1 M=%d N=%d K=%d status=%d bad=%d\n", spec.M, spec.N, spec.K, status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "sign1.A");
    free_hbm(B, B_elems * sizeof(__bf16), "sign1.B");
    free_hbm(C, C_elems * sizeof(__bf16), "sign1.C");
    free_host(C_ref);
    free_hbm(bias, bias_elems * sizeof(__bf16), "sign1.bias");
    free_host(C_ref_f);
    return (status == 0 && bad == 0) ? 0 : -1;
}

static int run_sign2_correctness_case()
{
    const GemmSpec spec = {3840, 64, 1024, XhblasTrans, XhblasNoTrans, ImplKind::Demo1};
    const int a_rows = spec.K;
    const int a_cols = spec.M;
    const int b_rows = spec.K;
    const int b_cols = spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const size_t ba_elems = (size_t)spec.M;

    __bf16* A = (__bf16*)alloc_hbm(A_elems * sizeof(__bf16), -16, "sign2.A");
    __bf16* B = (__bf16*)alloc_hbm(B_elems * sizeof(__bf16), -16, "sign2.B");
    __bf16* C = (__bf16*)alloc_hbm(C_elems * sizeof(__bf16), 0, "sign2.C");
    __bf16* C_ref = (__bf16*)alloc_host(C_elems * sizeof(__bf16));
    __bf16* ba = (__bf16*)alloc_hbm(ba_elems * sizeof(__bf16), 0, "sign2.ba");
    __bf16* ba_ref = (__bf16*)alloc_host(ba_elems * sizeof(__bf16));
    float* C_ref_f = (float*)alloc_host(C_elems * sizeof(float));
    if (!A || !B || !C || !C_ref || !ba || !ba_ref || !C_ref_f) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));
    memset(ba, 0, ba_elems * sizeof(__bf16));
    memset(ba_ref, 0, ba_elems * sizeof(__bf16));
    memset(C_ref_f, 0, C_elems * sizeof(float));

    const int status = bgemm_kernel(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                    spec.M, spec.N, spec.K, 1.0f,
                                    A, a_cols, B, b_cols, 0.0f, C, spec.N,
                                    ba, 2);

    reference_gemm_dispatch(A, B, C_ref_f, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    writeback_float_to_bf16(C_ref_f, C_ref, spec.M, spec.N, spec.N, 1.0f);
    for (int j = 0; j < spec.M; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < spec.K; ++i) {
            sum += A[(size_t)i * spec.M + j];
        }
        ba_ref[j] = sum;
    }

    const int bad_c = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);
    const int bad_ba = compare_bf16_tiles(ba_ref, ba, 1, spec.M, spec.M);
    printf("case sign=2 M=%d N=%d K=%d status=%d bad_c=%d bad_ba=%d\n",
           spec.M, spec.N, spec.K, status, bad_c, bad_ba);

    free_hbm(A, A_elems * sizeof(__bf16), "sign2.A");
    free_hbm(B, B_elems * sizeof(__bf16), "sign2.B");
    free_hbm(C, C_elems * sizeof(__bf16), "sign2.C");
    free_host(C_ref);
    free_hbm(ba, ba_elems * sizeof(__bf16), "sign2.ba");
    free_host(ba_ref);
    free_host(C_ref_f);
    return (status == 0 && bad_c == 0 && bad_ba == 0) ? 0 : -1;
}

} // namespace unified_bgemm

namespace unified_bgemm {

static int run_test_driver()
{
    if (getenv("FINAL_DEMO_BENCH")) {
        const int warmup = getenv_int_or("FINAL_DEMO_BENCH_WARMUP", 1);
        const int iters = getenv_int_or("FINAL_DEMO_BENCH_ITERS", 5);
        const int bench_threads = getenv_int_or("FINAL_DEMO_BENCH_THREADS", 32);
        char thread_buf[32];
        snprintf(thread_buf, sizeof(thread_buf), "%d", bench_threads);
        setenv("FINAL_GEMM_THREADS", thread_buf, 1);

        const char* sign_env = getenv("FINAL_DEMO_BENCH_SIGN");
        if (sign_env && *sign_env) {
            int failures = 0;
            if (strcmp(sign_env, "all") == 0) {
                for (int sign = 0; sign <= 2; ++sign) {
                    printf("benchmark sign=%d begin\n", sign);
                    failures += run_sign_suite_benchmark(sign, warmup, iters);
                }
            } else {
                const int sign = atoi(sign_env);
                printf("benchmark sign=%d begin\n", sign);
                failures += run_sign_suite_benchmark(sign, warmup, iters);
            }
            printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
            return failures == 0 ? 0 : 1;
        }

        int failures = 0;
        failures += run_list_benchmark(kDemo1Specs,
                                       sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]),
                                       warmup, iters);
        
        for (size_t i = 0; i < sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]); ++i) {
            if (seen_in_demo1(kDemo1ExtraSpecs[i]) ||
                seen_earlier_problem(kDemo1ExtraSpecs[i],
                                                    kDemo1ExtraSpecs,
                                                    sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]),
                                                    i)) {
                continue;
            }
            failures += run_benchmark_case(kDemo1ExtraSpecs[i], warmup, iters) != 0;
        }
        failures += run_list_benchmark(kDemo2Specs,
                                       sizeof(kDemo2Specs) / sizeof(kDemo2Specs[0]),
                                       warmup, iters);
        failures += run_list_benchmark(kOuterSpecs,
                                       sizeof(kOuterSpecs) / sizeof(kOuterSpecs[0]),
                                       warmup, iters);
        failures += run_list_benchmark(kVectorSpecs,
                                       sizeof(kVectorSpecs) / sizeof(kVectorSpecs[0]),
                                       warmup, iters);
        printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;
    failures += run_list_correctness(kDemo1Specs,
                                     sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]));
    for (size_t i = 0; i < sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]); ++i) {
        if (seen_in_demo1(kDemo1ExtraSpecs[i]) ||
            seen_earlier_problem(kDemo1ExtraSpecs[i],
                                                kDemo1ExtraSpecs,
                                                sizeof(kDemo1ExtraSpecs) / sizeof(kDemo1ExtraSpecs[0]),
                                                i)) {
            continue;
        }
        failures += run_correctness_case(kDemo1ExtraSpecs[i]) != 0;
    }
    failures += run_list_correctness(kDemo2Specs,
                                     sizeof(kDemo2Specs) / sizeof(kDemo2Specs[0]));
    failures += run_list_correctness(kOuterSpecs,
                                     sizeof(kOuterSpecs) / sizeof(kOuterSpecs[0]));
    failures += run_list_correctness(kVectorSpecs,
                                     sizeof(kVectorSpecs) / sizeof(kVectorSpecs[0]));
    failures += run_sign1_correctness_case() != 0;
    failures += run_sign2_correctness_case() != 0;
    printf("overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

} // namespace unified_bgemm

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    const int init_status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (init_status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed: status=%d\n", init_status);
        return 1;
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("[mpi] rank=%d/%d start provided_thread_level=%d\n", rank, world_size, provided);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    const int local_status = unified_bgemm::run_test_driver();

    int global_status = 0;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[mpi] world_size=%d global_status=%d overall=%s\n",
               world_size, global_status, global_status == 0 ? "PASS" : "FAIL");
        fflush(stdout);
    }

    const int finalize_status = MPI_Finalize();
    if (finalize_status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Finalize failed: status=%d\n", finalize_status);
        return 1;
    }
    return global_status;
}
#endif
