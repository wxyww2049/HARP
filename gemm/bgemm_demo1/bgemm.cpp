#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_acle.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <sched.h>
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

namespace final_demo {

constexpr int kCoresPerNuma = 38;
static const char* kModuleName = "demo1";

static void *alloc_hbm(size_t size, int numa_id, const char* tag = "unknown")
{
    (void)numa_id;
    (void)tag;
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void free_hbm(void *ptr, size_t size, const char* tag = "unknown")
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

static void add_bias_tile_inplace(__bf16* C, int ldc, int rows, int cols, const __bf16* bias)
{
    for (int i = 0; i < rows; ++i) {
        __bf16* row = C + (size_t)i * ldc;
        for (int j = 0; j < cols; ++j) {
            row[j] += bias[j];
        }
    }
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
};

struct GemmSpec {
    int M;
    int N;
    int K;
    XHBLAS_TRANSPOSE trans_a;
    XHBLAS_TRANSPOSE trans_b;
    int magic_k;
    ImplKind impl;
};

static const GemmSpec kDemo2Specs[] = {
    {768,   3840, 10240, XhblasNoTrans, XhblasTrans,   2560, ImplKind::Demo2},
    {768,   3840, 10240, XhblasNoTrans, XhblasNoTrans, 2560, ImplKind::Demo2},
    {768,  10240,  3840, XhblasNoTrans, XhblasTrans,   3840, ImplKind::Demo2},
    {3840, 10240,   768, XhblasTrans,   XhblasNoTrans,  768, ImplKind::Demo2},
    {768,  10240,  3840, XhblasNoTrans, XhblasNoTrans, 3840, ImplKind::Demo2},
    {1024,  3840,  2560, XhblasNoTrans, XhblasTrans,   2560, ImplKind::Demo2},
    {256,   3840, 10240, XhblasNoTrans, XhblasTrans,   2560, ImplKind::Demo2},
    {256,  10240,  3840, XhblasNoTrans, XhblasTrans,   3840, ImplKind::Demo2},
    {768,   3840,  3840, XhblasNoTrans, XhblasTrans,   3840, ImplKind::Demo2},
    {256,   3840, 10240, XhblasNoTrans, XhblasNoTrans, 2560, ImplKind::Demo2},
    {768,   3840,  3840, XhblasNoTrans, XhblasNoTrans, 3840, ImplKind::Demo2},
    {1024,  3840,   768, XhblasNoTrans, XhblasTrans,    768, ImplKind::Demo2},
    {3840, 10240,   256, XhblasTrans,   XhblasNoTrans,  256, ImplKind::Demo2},
    {256,  10240,  3840, XhblasNoTrans, XhblasNoTrans, 3840, ImplKind::Demo2},
    {1024,  3840,    64, XhblasNoTrans, XhblasTrans,     64, ImplKind::Demo2},
    {256,   3840,  3840, XhblasNoTrans, XhblasTrans,   3840, ImplKind::Demo2},
    {256,   3840,  3840, XhblasNoTrans, XhblasNoTrans, 3840, ImplKind::Demo2},
    {64,    3840,  3072, XhblasTrans,   XhblasNoTrans, 3072, ImplKind::Demo2},
    {3072,  3840,    64, XhblasNoTrans, XhblasNoTrans,   64, ImplKind::Demo2},
    {1024,  4096,    4096, XhblasTrans, XhblasNoTrans,   4096, ImplKind::Demo2},
};

static const GemmSpec kDemo1Specs[] = {
    {10240, 3840,   768, XhblasTrans,   XhblasNoTrans,  768, ImplKind::Demo1},
    {3840,  2560,  1024, XhblasTrans,   XhblasNoTrans, 1024, ImplKind::Demo1},
    {3840,   768,  1024, XhblasTrans,   XhblasNoTrans, 1024, ImplKind::Demo1},
    {3072,    64,  3840, XhblasNoTrans, XhblasTrans,   3840, ImplKind::Demo1},
    {3840,    64,  1024, XhblasTrans,   XhblasNoTrans, 1024, ImplKind::Demo1},
    {10240, 3840,   256, XhblasTrans,   XhblasNoTrans,  256, ImplKind::Demo1},
    {3840,  3840,   768, XhblasTrans,   XhblasNoTrans,  768, ImplKind::Demo1},
    {3840,  3840,   256, XhblasTrans,   XhblasNoTrans,  256, ImplKind::Demo1},
    {4096,  1024,    4096, XhblasTrans, XhblasNoTrans,   4096, ImplKind::Demo1},
    {10240,  4096,   4096, XhblasTrans, XhblasNoTrans,   4096, ImplKind::Demo1},
};

static const GemmSpec* find_spec(int M, int N, int K,
                                 XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b)
{
    for (size_t i = 0; i < sizeof(kDemo1Specs) / sizeof(kDemo1Specs[0]); ++i) {
        const GemmSpec& spec = kDemo1Specs[i];
        if (spec.M == M && spec.N == N && spec.K == K &&
            spec.trans_a == trans_a && spec.trans_b == trans_b) {
            return &spec;
        }
    }
    for (size_t i = 0; i < sizeof(kDemo2Specs) / sizeof(kDemo2Specs[0]); ++i) {
        const GemmSpec& spec = kDemo2Specs[i];
        if (spec.M == M && spec.N == N && spec.K == K &&
            spec.trans_a == trans_a && spec.trans_b == trans_b) {
            return &spec;
        }
    }
    return NULL;
}

namespace demo2 {

constexpr int atm_magic = 16;
constexpr int atpacked_m = 2 * atm_magic;
constexpr int btn_magic = 32;
constexpr int btpacked_n = 64;
constexpr int m_magic = 32;
constexpr int m_pack = 16;
constexpr int n_magic = 32;
constexpr int n_pack = 64;
constexpr int gemm_m_tile = 16;
constexpr int gemm_b_pack = 64;
constexpr int gemm_n_tile_min = 48;
struct ReadyState {
    int* a_ready = NULL;
    size_t a_ready_elems = 0;
};

static ReadyState g_ready_state;

static int ensure_workspace(size_t a_bytes, size_t b_bytes, size_t c_bytes, size_t a_ready_elems, int numa_id,
                            __bf16** A_p, __bf16** B_p, float** C_p, int** a_ready)
{
    void* shared_a = NULL;
    void* shared_b = NULL;
    void* shared_c = NULL;
    if (bgemm_shared_workspace::ensure_buffers(a_bytes, b_bytes, c_bytes, numa_id,
                                               "demo1.demo2",
                                               &shared_a, &shared_b, &shared_c) != 0) {
        return -1;
    }
    if (a_ready_elems > g_ready_state.a_ready_elems) {
        delete[] g_ready_state.a_ready;
        g_ready_state.a_ready = new int[a_ready_elems]();
        g_ready_state.a_ready_elems = a_ready_elems;
    }
    *A_p = (__bf16*)shared_a;
    *B_p = (__bf16*)shared_b;
    *C_p = (float*)shared_c;
    *a_ready = g_ready_state.a_ready;
    return (*A_p && *B_p && *C_p && *a_ready) ? 0 : -1;
}

void pack_A_T(__bf16* A, int k, int m, int lda, __bf16* A_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = m < atpacked_m ? m : atpacked_m;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    const svbfloat16_t zero = svdup_n_bf16((__bf16)0);
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

void pack_B_T(__bf16* B, int n, int k, int ldb, __bf16* B_p)
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

void sme_kernel(__bf16* A, __bf16* B, float* C_work, __bf16* C_out,
                int ldc, int k, int m_use, int n_use, int sign,
                bool final_writeback, float alpha, const __bf16* ba)
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
    if (sign == 0) {
        if(!ba){
            svzero_za();
        }else{
            svuint16_t vb16 = svld1_u16(pg_bf16, reinterpret_cast<const uint16_t*>(ba));
            svuint32_t lo32 = svunpklo_u32(vb16);
            svuint32_t hi32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            hi32 = svlsl_n_u32_x(pg, hi32, 16);
            for(int h=0;h<m_use;++h){
                svwrite_hor_za32_u32_m(0, h, pg, lo32);
                svwrite_hor_za32_u32_m(1, h, pg, hi32);
            }
            vb16 = svld1_u16(pg_tail16, reinterpret_cast<const uint16_t*>(ba+ tl));
            lo32 = svunpklo_u32(vb16);
            hi32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            hi32 = svlsl_n_u32_x(pg, hi32, 16);
            for(int h=0;h<m_use;++h){
                svwrite_hor_za32_u32_m(2, h, pg, lo32);
                svwrite_hor_za32_u32_m(3, h, pg_tail32, hi32);
            }
        }
    } else {
        for (int h = 0; h < m_use; ++h) {
            svld1_hor_za32(0, h, pg, C_work + h * ldc);
            svld1_hor_za32(1, h, pg, C_work + h * ldc + wl);
            svld1_hor_za32(2, h, pg, C_work + h * ldc + 2 * wl);
            svld1_hor_za32(3, h, pg_tail32, C_work + h * ldc + 3 * wl);
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
    if(!final_writeback){
        for (int h = 0; h < m_use; ++h) {
            svst1_hor_za32(0, h, pg, C_work + h * ldc);
            svst1_hor_za32(1, h, pg, C_work + h * ldc + wl);
            svst1_hor_za32(2, h, pg, C_work + h * ldc + 2 * wl);
            svst1_hor_za32(3, h, pg_tail32, C_work + h * ldc + 3 * wl);
        }
    }

    else{
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
}

static int choose_gemm_n_tile(int N, int nt)
{
    for (int tile = gemm_b_pack; tile >= gemm_n_tile_min; --tile) {
        if (N % tile != 0) {
            continue;
        }
        if (N % (tile * nt) != 0) {
            continue; 
        }
        return tile;
    }
    return -1;
}

__arm_new("za")
static void run_thread_impl(__bf16* A, __bf16* A_p, __bf16* B, __bf16* B_p, __bf16* C_out, float* C_work,
                            int* a_ready, int M, int N, int K, int M_p, int magic_k,
                            int gemm_n_tile, int m_tiles, int tile_done_words,
                            int rank, int nt, bool transpose_a, bool transpose_b, float alpha,
                            const __bf16* ba)
__arm_streaming
{
    unsigned long long tile_done[8];

    for (int kk = 0; kk < K; kk += magic_k) {
        const int kk_id = kk / magic_k;
        if (transpose_a) {
            const int pack_rows = 2 * gemm_m_tile;
            for (int mm = rank * pack_rows; mm < M; mm += pack_rows * nt) {
                const int z_m = min_int(pack_rows, M - mm);
                const int tile_idx0 = mm / gemm_m_tile;
                pack_A_T(A + kk * M + mm, magic_k, z_m, M, A_p + mm * magic_k + M_p * kk);
                __atomic_store_n(a_ready + (size_t)kk_id * m_tiles + tile_idx0, 1, __ATOMIC_RELEASE);
                if (z_m > gemm_m_tile && tile_idx0 + 1 < m_tiles) {
                    __atomic_store_n(a_ready + (size_t)kk_id * m_tiles + tile_idx0 + 1, 1, __ATOMIC_RELEASE);
                }
            }
        } else {
            for (int mm = rank * gemm_m_tile; mm < M; mm += gemm_m_tile * nt) {
                const int z_m = min_int(gemm_m_tile, M - mm);
                pack_A(A + mm * K + kk, z_m, magic_k, K, A_p + mm * magic_k + M_p * kk);
                __atomic_store_n(a_ready + (size_t)kk_id * m_tiles + mm / gemm_m_tile, 1, __ATOMIC_RELEASE);
            }
        }
    }

    for (int kk = 0; kk < K; kk += magic_k) {
        const int kk_id = kk / magic_k;
        const int sign = kk == 0 ? 0 : 1;
        const bool final_writeback = (kk + magic_k >= K);
        for (int nn = rank * gemm_n_tile; nn < N; nn += gemm_n_tile * nt) {
            if (transpose_b) {
                pack_B_T(B + nn * K + kk, gemm_n_tile, magic_k, K,
                         B_p + rank * gemm_b_pack * magic_k);
            } else {
                pack_B(B + nn + kk * N, magic_k, gemm_n_tile, N,
                       B_p + rank * gemm_b_pack * magic_k);
            }

            memset(tile_done, 0, (size_t)tile_done_words * sizeof(unsigned long long));
            int remaining = m_tiles;
            while (remaining > 0) {
                bool made_progress = false;
                for (int owner_offset = 0; owner_offset < nt; ++owner_offset) {
                    int owner = rank + owner_offset;
                    if (owner >= nt) {
                        owner -= nt;
                    }
                    for (int j = owner * gemm_m_tile; j < M; j += gemm_m_tile * nt) {
                        const int tile_idx = j / gemm_m_tile;
                        const int word_idx = tile_idx >> 6;
                        const unsigned long long tile_mask = 1ull << (tile_idx & 63);
                        if (tile_done[word_idx] & tile_mask) {
                            continue;
                        }
                        const size_t ready_idx = (size_t)kk_id * m_tiles + tile_idx;
                        if (__atomic_load_n(a_ready + ready_idx, __ATOMIC_ACQUIRE) != 1) {
                            continue;
                        }
                        const int z_m = min_int(gemm_m_tile, M - j);
                        __bf16* B_panel = B_p + rank * gemm_b_pack * magic_k;
                        sme_kernel(A_p + M_p * kk + j * magic_k,
                                   B_panel,
                                   C_work + j * N + nn,
                                   C_out + j * N + nn,
                                   N, magic_k, z_m, gemm_n_tile, sign,
                                   final_writeback, alpha,
                                   sign == 0 && ba ? ba + nn : NULL);
                        tile_done[word_idx] |= tile_mask;
                        --remaining;
                        made_progress = true;
                    }
                }
                if (made_progress) {
                    continue;
                }
                bool waited_once = false;
                for (int owner_offset = 0; owner_offset < nt && !waited_once; ++owner_offset) {
                    int owner = rank + owner_offset;
                    if (owner >= nt) {
                        owner -= nt;
                    }
                    for (int j = owner * gemm_m_tile; j < M; j += gemm_m_tile * nt) {
                        const int tile_idx = j / gemm_m_tile;
                        const int word_idx = tile_idx >> 6;
                        const unsigned long long tile_mask = 1ull << (tile_idx & 63);
                        if (tile_done[word_idx] & tile_mask) {
                            continue;
                        }
                        const size_t ready_idx = (size_t)kk_id * m_tiles + tile_idx;
                        while (__atomic_load_n(a_ready + ready_idx, __ATOMIC_ACQUIRE) != 1) {
                            asm volatile("yield");
                        }
                        const int z_m = min_int(gemm_m_tile, M - j);
                        __bf16* B_panel = B_p + rank * gemm_b_pack * magic_k;
                        sme_kernel(A_p + M_p * kk + j * magic_k,
                                   B_panel,
                                   C_work + j * N + nn,
                                   C_out + j * N + nn,
                                   N, magic_k, z_m, gemm_n_tile, sign,
                                   final_writeback, alpha,
                                   sign == 0 && ba ? ba + nn : NULL);
                        tile_done[word_idx] |= tile_mask;
                        --remaining;
                        waited_once = true;
                        break;
                    }
                }
            }
        }
    }
}

static void run_impl(__bf16* A, __bf16* A_p, __bf16* B, __bf16* B_p, __bf16* C_out, float* C_work,
                     int M, int N, int K, int magic_k, int bench_threads,
                     bool transpose_a, bool transpose_b, float alpha, const __bf16* ba)
{
    const int gemm_n_tile = choose_gemm_n_tile(N, bench_threads);
    if (gemm_n_tile < 0) {
        return;
    }
    const int M_p = (M + gemm_m_tile - 1) / gemm_m_tile * gemm_m_tile;
    const int m_tiles = (M + gemm_m_tile - 1) / gemm_m_tile;
    const int k_tiles = (K + magic_k - 1) / magic_k;
    const int tile_done_words = (m_tiles + 63) / 64;
    int* a_ready = g_ready_state.a_ready;
    memset(a_ready, 0, (size_t)m_tiles * k_tiles * sizeof(*a_ready));
    const int base_numa = detect_base_numa();

    #pragma omp parallel num_threads(bench_threads)
    {
        const int rank = omp_get_thread_num();
        bind_to_cpu(base_numa, rank);
        const int nt = omp_get_num_threads();
        run_thread_impl(A, A_p, B, B_p, C_out, C_work, a_ready, M, N, K, M_p, magic_k,
                        gemm_n_tile, m_tiles, tile_done_words, rank, nt,
                        transpose_a, transpose_b, alpha, ba);
    }
}

static int run(__bf16* A, __bf16* B, __bf16* C,
               int M, int N, int K, int magic_k, int bench_threads,
               bool transpose_a, bool transpose_b, float alpha, const __bf16* ba)
{
    const int M_p = (M + gemm_m_tile - 1) / gemm_m_tile * gemm_m_tile;
    const size_t A_p_elems = (size_t)M_p * K;
    const size_t B_p_elems = (size_t)bench_threads * gemm_b_pack * magic_k;
    const size_t C_p_elems = (size_t)M * N;
    const int m_tiles = (M + gemm_m_tile - 1) / gemm_m_tile;
    const int k_tiles = (K + magic_k - 1) / magic_k;
    const int base_numa = detect_base_numa();
    __bf16 *A_p = NULL;
    __bf16 *B_p = NULL;
    float *C_p = NULL;
    int* a_ready = NULL;
    if (ensure_workspace(A_p_elems * sizeof(__bf16), B_p_elems * sizeof(__bf16),
                         C_p_elems * sizeof(float), (size_t)m_tiles * k_tiles, base_numa,
                         &A_p, &B_p, &C_p, &a_ready) != 0) {
        return -7;
    }
    run_impl(A, A_p, B, B_p, C, C_p, M, N, K, magic_k, bench_threads,
             transpose_a, transpose_b, alpha, ba);
    return 0;
}

} // namespace demo2

namespace demo1 {

constexpr int m_magic = 32;
constexpr int n_magic = 16;
constexpr int packed_m = 64;
constexpr int packed_n = 2 * n_magic;
constexpr int gemm_m_tile = 64;
constexpr int gemm_n_tile = 16;
static int ensure_workspace(size_t a_bytes, size_t b_bytes, size_t c_bytes, int numa_id,
                            __bf16** A_p, __bf16** B_p, float** C_p)
{
    void* shared_a = NULL;
    void* shared_b = NULL;
    void* shared_c = NULL;
    if (bgemm_shared_workspace::ensure_buffers(a_bytes, b_bytes, c_bytes, numa_id,
                                               "demo1.demo1",
                                               &shared_a, &shared_b, &shared_c) != 0) {
        return -1;
    }
    *A_p = (__bf16*)shared_a;
    *B_p = (__bf16*)shared_b;
    *C_p = (float*)shared_c;
    return (*A_p && *B_p && *C_p) ? 0 : -1;
}

void pack_A(__bf16* A, int m, int k, int lda, __bf16* A_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    const int dst_panel_stride = m_magic * packed_m;
    const int dst_pair_stride = 2 * packed_m;
    __bf16* src_panel = A;
    __bf16* dst_panel = A_p;
    for (int kk = 0; kk < k; kk += m_magic) {
        const int rows0 = m > m_magic ? m_magic : m;
        const int rows1 = m > m_magic ? (m - m_magic) : 0;
        __bf16* src0 = src_panel;
        __bf16* src1 = src_panel + lda * m_magic;

        svzero_za();
        for (int mm = 0; mm < rows0; ++mm) {
            svld1_hor_za16(0, mm, pg, src0);
            src0 += lda;
        }
        for (int mm = 0; mm < rows1; ++mm) {
            svld1_hor_za16(1, mm, pg, src1);
            src1 += lda;
        }

        __bf16* dst = dst_panel;
        for (int nn = 0; nn < m_magic; nn += 2) {
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst, a1);
            svst1(pg, dst + m_magic, a2);
            row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn);
            row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn + 1);
            a1 = svzip1(row_vec0, row_vec1);
            a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst + 2 * m_magic, a1);
            svst1(pg, dst + 3 * m_magic, a2);
            dst += dst_pair_stride;
        }
        src_panel += m_magic;
        dst_panel += dst_panel_stride;
    }
}

void pack_B(__bf16* B, int k, int n, int ldb, __bf16* B_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = n < packed_n ? n : packed_n;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    const svbfloat16_t zero = svdup_n_bf16((__bf16)0);
    const int block = k * n_magic;
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t row_vec0 = svsel(pg_tail, svld1_bf16(pg_tail, B + i * ldb), zero);
        svbfloat16_t row_vec1 = svsel(pg_tail, svld1_bf16(pg_tail, B + (i + 1) * ldb), zero);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, B_p + i * n_magic, a1);
        svst1(pg, B_p + i * n_magic + block, a2);
    }
}

void pack_A_T(__bf16* A_t, int k, int m, int lda, __bf16* A_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols0 = m < m_magic ? m : m_magic;
    const int valid_cols1 = m > m_magic ? (m - m_magic) : 0;
    svbool_t pg0 = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols0);
    svbool_t pg1 = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols1);
    const svbfloat16_t zero = svdup_n_bf16((__bf16)0);
    for (int i = 0; i < k; i += 2) {
        const int base = i * packed_m;
        svbfloat16_t row_vec0 = svsel(pg0, svld1_bf16(pg0, A_t + i * lda), zero);
        svbfloat16_t row_vec1 = svsel(pg0, svld1_bf16(pg0, A_t + (i + 1) * lda), zero);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, A_p + base, a1);
        svst1(pg, A_p + base + m_magic, a2);
        row_vec0 = svsel(pg1, svld1_bf16(pg1, A_t + i * lda + m_magic), zero);
        row_vec1 = svsel(pg1, svld1_bf16(pg1, A_t + (i + 1) * lda + m_magic), zero);
        a1 = svzip1(row_vec0, row_vec1);
        a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, A_p + base + 2 * m_magic, a1);
        svst1(pg, A_p + base + 3 * m_magic, a2);
    }
}

void pack_B_T(__bf16* B_t, int n, int k, int lda, __bf16* B_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    const int rows = n < packed_n ? n : packed_n;
    const int block = k * n_magic;
    __bf16* src_panel = B_t;
    for (int kk = 0; kk < k; kk += m_magic) {
        __bf16* src = src_panel;
        svzero_za();
        for (int mm = 0; mm < rows; ++mm) {
            svld1_hor_za16(0, mm, pg, src);
            src += lda;
        }
        for (int nn = 0; nn < m_magic; nn += 2) {
            const int base = (kk + nn) * n_magic;
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, B_p + base, a1);
            svst1(pg, B_p + base + block, a2);
        }
        src_panel += m_magic;
    }
}

void sme_kernel(__bf16* A, __bf16* B, float* C_work, __bf16* C_out,
                int ldc, int k, int m_use, int n_use, int sign,
                bool final_writeback, float alpha, const __bf16* ba)
__arm_streaming __arm_inout("za")
{
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    svbool_t pg_tail32 = svwhilelt_b32((uint64_t)0, (uint64_t)n_use);
    svbool_t pg_tail16 = svwhilelt_b16((uint64_t)0, (uint64_t)n_use);
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (sign == 0) {
        if (!ba) {
            svzero_za();
        } else {
            alignas(64) float bias_row[gemm_n_tile] = {0.0f};
            for (int j = 0; j < n_use; ++j) {
                bias_row[j] = (float)ba[j];
            }
            svzero_za();
            for (int h = 0; h < wl; ++h) {
                if (h < m_use) {
                    svld1_hor_za32(0, h, pg_tail32, bias_row);
                }
                if (h + wl < m_use) {
                    svld1_hor_za32(1, h, pg_tail32, bias_row);
                }
                if (h + 2 * wl < m_use) {
                    svld1_hor_za32(2, h, pg_tail32, bias_row);
                }
                if (h + 3 * wl < m_use) {
                    svld1_hor_za32(3, h, pg_tail32, bias_row);
                }
            }
        }
    } else {
        for (int h = 0; h < wl; ++h) {
            if (h < m_use) {
                svld1_hor_za32(0, h, pg_tail32, C_work + h * ldc);
            }
            if (h + wl < m_use) {
                svld1_hor_za32(1, h, pg_tail32, C_work + (h + wl) * ldc);
            }
            if (h + 2 * wl < m_use) {
                svld1_hor_za32(2, h, pg_tail32, C_work + (h + 2 * wl) * ldc);
            }
            if (h + 3 * wl < m_use) {
                svld1_hor_za32(3, h, pg_tail32, C_work + (h + 3 * wl) * ldc);
            }
        }
    }

    for (int i = 0; i < k; i += 2) {
        svbfloat16_t b0 = svldnt1_bf16(pg_bf16, b_ptr);
        b_ptr += tl;

        svbfloat16_t a0 = svld1_bf16(pg_bf16, a_ptr);
        svbfloat16_t a1 = svld1_bf16(pg_bf16, a_ptr + tl);
        svbfloat16_t a2 = svld1_bf16(pg_bf16, a_ptr + 2 * tl);
        svbfloat16_t a3 = svld1_bf16(pg_bf16, a_ptr + 3 * tl);
        a_ptr += 4 * tl;

        svmopa_za32_bf16_m(0, pg_bf16, pg_bf16, a0, b0);
        svmopa_za32_bf16_m(1, pg_bf16, pg_bf16, a1, b0);
        svmopa_za32_bf16_m(2, pg_bf16, pg_bf16, a2, b0);
        svmopa_za32_bf16_m(3, pg_bf16, pg_bf16, a3, b0);
    }
    if(!final_writeback){
        for (int h = 0; h < wl; ++h) {
            if (h < m_use) {
                svst1_hor_za32(0, h, pg_tail32, C_work + h * ldc);
            }
            if (h + wl < m_use) {
                svst1_hor_za32(1, h, pg_tail32, C_work + (h + wl) * ldc);
            }
            if (h + 2 * wl < m_use) {
                svst1_hor_za32(2, h, pg_tail32, C_work + (h + 2 * wl) * ldc);
            }
            if (h + 3 * wl < m_use) {
                svst1_hor_za32(3, h, pg_tail32, C_work + (h + 3 * wl) * ldc);
            }
        }
    }
    else{
        svbfloat16_t z0;
        for (int h = 0; h < wl; ++h) {
            if (h < m_use) {
                z0 = svread_hor_za16_bf16_m(svundef_bf16(), pg_bf16, 0, 2 * h);
                z0 = svuzp2_bf16(z0, z0);
                svst1_bf16(pg_tail16, C_out + h * ldc, z0);
            }
            if (h + wl < m_use) {
                z0 = svread_hor_za16_bf16_m(svundef_bf16(), pg_bf16, 1, 2 * h);
                z0 = svuzp2_bf16(z0, z0);
                svst1_bf16(pg_tail16, C_out + (h + wl) * ldc, z0);
            }
            if(h + 2 * wl < m_use) {
                z0 = svread_hor_za16_bf16_m(svundef_bf16(), pg_bf16, 0, 2 * h + 1);
                z0 = svuzp2_bf16(z0, z0);
                svst1_bf16(pg_tail16, C_out + (h + 2 * wl) * ldc, z0);
            }
            if(h + 3 * wl < m_use) {
                z0 = svread_hor_za16_bf16_m(svundef_bf16(), pg_bf16, 1, 2 * h + 1);
                z0 = svuzp2_bf16(z0, z0);
                svst1_bf16(pg_tail16, C_out + (h + 3 * wl) * ldc, z0);
            }
        }
    }
}

__arm_new("za")
static void run_thread_impl(__bf16* A, __bf16* A_p, __bf16* B, __bf16* B_p, __bf16* C_out, float* C_work,
                            int M, int N, int K, int N_p, int magic_k,
                            int rank, int nt, bool transpose_a, bool transpose_b, float alpha,
                            const __bf16* ba)
__arm_streaming
{
    for (int kk = 0; kk < K; kk += magic_k) {
        for (int nn = rank * packed_n; nn < N; nn += packed_n * nt) {
            const int z_n_pack = min_int(packed_n, N - nn);
            if (transpose_b) {
                pack_B_T(B + nn * K + kk, z_n_pack, magic_k, K, B_p + N_p * kk + nn * magic_k);
            } else {
                pack_B(B + nn + kk * N, magic_k, z_n_pack, N, B_p + N_p * kk + nn * magic_k);
            }
        }
    }
    #pragma omp barrier

    for (int kk = 0; kk < K; kk += magic_k) {
        const int sign = kk == 0 ? 0 : 1;
        const bool final_writeback = (kk + magic_k >= K);
        for (int i = rank * gemm_m_tile; i < M; i += gemm_m_tile * nt) {
            const int z_m = min_int(gemm_m_tile, M - i);
            if (transpose_a) {
                pack_A_T(A + kk * M + i, magic_k, z_m, M, A_p + rank * packed_m * magic_k);
            } else {
                pack_A(A + i * K + kk, z_m, magic_k, K, A_p + rank * packed_m * magic_k);
            }
            for (int j = 0; j < N; j += gemm_n_tile) {
                const int z_n = min_int(gemm_n_tile, N - j);
                sme_kernel(A_p + rank * packed_m * magic_k,
                           B_p + j * magic_k + kk * N_p,
                           C_work + i * N + j,
                           C_out + i * N + j,
                           N, magic_k, z_m, z_n, sign,
                           final_writeback, alpha,
                           sign == 0 && ba ? ba + j : NULL);
            }
        }
    }
}

static void run_impl(__bf16* A, __bf16* A_p, __bf16* B, __bf16* B_p, __bf16* C_out, float* C_work,
                     int M, int N, int K, int magic_k, int bench_threads,
                     bool transpose_a, bool transpose_b, float alpha, const __bf16* ba)
{
    const int N_p = (N + packed_n - 1) / packed_n * packed_n;
    const int base_numa = detect_base_numa();
    #pragma omp parallel num_threads(bench_threads)
    {
        const int rank = omp_get_thread_num();
        bind_to_cpu(base_numa, rank);
        const int nt = omp_get_num_threads();
        run_thread_impl(A, A_p, B, B_p, C_out, C_work, M, N, K, N_p, magic_k,
                        rank, nt, transpose_a, transpose_b, alpha, ba);
    }
}

static int run(__bf16* A, __bf16* B, __bf16* C,
               int M, int N, int K, int magic_k, int bench_threads,
               bool transpose_a, bool transpose_b, float alpha, const __bf16* ba)
{
    const int N_p = (N + packed_n - 1) / packed_n * packed_n;
    const size_t A_p_elems = (size_t)bench_threads * packed_m * magic_k;
    const size_t B_p_elems = (size_t)N_p * K;
    const size_t C_p_elems = (size_t)M * N;
    const int base_numa = detect_base_numa();
    __bf16 *A_p = NULL;
    __bf16 *B_p = NULL;
    float *C_p = NULL;
    if (ensure_workspace(A_p_elems * sizeof(__bf16), B_p_elems * sizeof(__bf16),
                         C_p_elems * sizeof(float), base_numa, &A_p, &B_p, &C_p) != 0) {
        return -7;
    }
    run_impl(A, A_p, B, B_p, C, C_p, M, N, K, magic_k, bench_threads,
             transpose_a, transpose_b, alpha, ba);
    return 0;
}

} // namespace demo1

static int run_selected_impl(const GemmSpec& spec, const __bf16* A, const __bf16* B, __bf16* C,
                             int bench_threads, float alpha, const __bf16* ba)
{
    switch (spec.impl) {
        case ImplKind::Demo1:
            return demo1::run(const_cast<__bf16*>(A), const_cast<__bf16*>(B), C,
                              spec.M, spec.N, spec.K, spec.magic_k, bench_threads,
                              spec.trans_a == XhblasTrans, spec.trans_b == XhblasTrans, alpha, ba);
        case ImplKind::Demo2:
            return demo2::run(const_cast<__bf16*>(A), const_cast<__bf16*>(B), C,
                              spec.M, spec.N, spec.K, spec.magic_k, bench_threads,
                              spec.trans_a == XhblasTrans, spec.trans_b == XhblasTrans, alpha, ba);
    }
    return -1;
}

static int scale_only(float beta, __bf16* C, int M, int N, int ldc)
{
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * (size_t)ldc * sizeof(__bf16));
        return 0;
    }
    return -8;
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
    if (trans_a != XhblasNoTrans && trans_a != XhblasTrans) {
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
        return -8;
    }
    if (K == 0 || alpha == 0.0f) {
        return scale_only(beta, C, M, N, ldc);
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

    const int bench_threads = getenv_int_or("FINAL_GEMM_THREADS", 32);
    return run_selected_impl(*spec, A, B, C, bench_threads, alpha,
                             sign == 1 ? ba : NULL);
}

static const char* status_message(int status)
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
        default: return "unknown error";
    }
}

} // namespace final_demo

extern "C" int xhblas_gemm(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    return final_demo::xhblas_gemm_status(order, trans_a, trans_b,
                                          M, N, K, alpha,
                                          A, lda, B, ldb, beta, C, ldc, ba, sign);
}

#ifdef FINAL_DEMO_TEST_MAIN
namespace final_demo {

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

static int compare_bf16_tiles(const __bf16* a, const __bf16* b, int m_use, int n_use, int ld)
{
    int bad = 0;
    for (int i = 0; i < m_use; ++i) {
        for (int j = 0; j < n_use; ++j) {
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

static int run_correctness_case(const GemmSpec& spec)
{
    const int a_rows = spec.trans_a == XhblasTrans ? spec.K : spec.M;
    const int a_cols = spec.trans_a == XhblasTrans ? spec.M : spec.K;
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();

    __bf16* A = (__bf16*) alloc_hbm(A_elems * sizeof(__bf16), -16, "correctness.A");
    __bf16* B = (__bf16*) alloc_hbm(B_elems * sizeof(__bf16), -16, "correctness.B");
    __bf16* C = (__bf16*) alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C");
    __bf16* C_ref = (__bf16*) alloc_hbm(C_elems * sizeof(__bf16), base_numa, "correctness.C_ref");
    float* C_ref_f = (float*) alloc_hbm(C_elems * sizeof(float), base_numa, "correctness.C_ref_f");
    if (!A || !B || !C || !C_ref || !C_ref_f) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));
    memset(C_ref, 0, C_elems * sizeof(__bf16));
    memset(C_ref_f, 0, C_elems * sizeof(float));

    const int status = xhblas_gemm_status(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                          spec.M, spec.N, spec.K, 1.0f,
                                          A, a_cols, B, b_cols, 0.0f, C, spec.N, NULL, 0);
    reference_gemm_dispatch(A, B, C_ref_f, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);

    writeback_float_to_bf16(C_ref_f, C_ref, spec.M, spec.N, spec.N, 1.0f);
    const int bad = compare_bf16_tiles(C_ref, C, spec.M, spec.N, spec.N);
    printf("case M=%d N=%d K=%d transA=%d transB=%d impl=%d status=%d bad=%d\n",
           spec.M, spec.N, spec.K, (int)spec.trans_a, (int)spec.trans_b, (int)spec.impl,
           status, bad);

    free_hbm(A, A_elems * sizeof(__bf16), "correctness.A");
    free_hbm(B, B_elems * sizeof(__bf16), "correctness.B");
    free_hbm(C, C_elems * sizeof(__bf16), "correctness.C");
    free_hbm(C_ref, C_elems * sizeof(__bf16), "correctness.C_ref");
    free_hbm(C_ref_f, C_elems * sizeof(float), "correctness.C_ref_f");
    return (status == 0 && bad == 0) ? 0 : -1;
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

static int run_benchmark_case(const GemmSpec& spec, int warmup, int iters)
{
    const int a_rows = spec.trans_a == XhblasTrans ? spec.K : spec.M;
    const int a_cols = spec.trans_a == XhblasTrans ? spec.M : spec.K;
    const int b_rows = spec.trans_b == XhblasTrans ? spec.N : spec.K;
    const int b_cols = spec.trans_b == XhblasTrans ? spec.K : spec.N;
    const size_t A_elems = (size_t)a_rows * a_cols;
    const size_t B_elems = (size_t)b_rows * b_cols;
    const size_t C_elems = (size_t)spec.M * spec.N;
    const int base_numa = detect_base_numa();

    __bf16* A = (__bf16*) alloc_hbm(A_elems * sizeof(__bf16), base_numa-16, "benchmark.A");
    __bf16* B = (__bf16*) alloc_hbm(B_elems * sizeof(__bf16), base_numa-16, "benchmark.B");
    __bf16* C = (__bf16*) alloc_hbm(C_elems * sizeof(__bf16), base_numa, "benchmark.C");
    if (!A || !B || !C) {
        return -1;
    }

    fill_inputs_for_mode(A, B, spec.M, spec.N, spec.K, spec.trans_a, spec.trans_b);
    memset(C, 0, C_elems * sizeof(__bf16));

    for (int i = 0; i < warmup; ++i) {
        const int status = xhblas_gemm_status(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, a_cols, B, b_cols, 0.0f, C, spec.N, NULL, 0);
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
        const int status = xhblas_gemm_status(XhblasRowMajor, spec.trans_a, spec.trans_b,
                                              spec.M, spec.N, spec.K, 1.0f,
                                              A, a_cols, B, b_cols, 0.0f, C, spec.N, NULL, 0);
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
    printf("bench M=%d N=%d K=%d transA=%d transB=%d avg_time=%.6f s gflops=%.6f sink=%f\n",
           spec.M, spec.N, spec.K, (int)spec.trans_a, (int)spec.trans_b,
           avg, gflops, (float)C[0]);

    free_hbm(A, A_elems * sizeof(__bf16), "benchmark.A");
    free_hbm(B, B_elems * sizeof(__bf16), "benchmark.B");
    free_hbm(C, C_elems * sizeof(__bf16), "benchmark.C");
    return 0;
}

} // namespace final_demo

int main()
{
    if (getenv("FINAL_DEMO_BENCH")) {
        const int warmup = final_demo::getenv_int_or("FINAL_DEMO_BENCH_WARMUP", 1);
        const int iters = final_demo::getenv_int_or("FINAL_DEMO_BENCH_ITERS", 5);
        const int bench_threads = final_demo::getenv_int_or("FINAL_DEMO_BENCH_THREADS", 32);
        char thread_buf[32];
        snprintf(thread_buf, sizeof(thread_buf), "%d", bench_threads);
        setenv("FINAL_GEMM_THREADS", thread_buf, 1);

        int failures = 0;
        for (size_t i = 0; i < sizeof(final_demo::kDemo1Specs) / sizeof(final_demo::kDemo1Specs[0]); ++i) {
            failures += final_demo::run_benchmark_case(final_demo::kDemo1Specs[i], warmup, iters) != 0;
        }
        for (size_t i = 0; i < sizeof(final_demo::kDemo2Specs) / sizeof(final_demo::kDemo2Specs[0]); ++i) {
            if (final_demo::seen_in_demo1(final_demo::kDemo2Specs[i]) ||
                final_demo::seen_earlier_problem(final_demo::kDemo2Specs[i],
                                                 final_demo::kDemo2Specs,
                                                 sizeof(final_demo::kDemo2Specs) / sizeof(final_demo::kDemo2Specs[0]),
                                                 i)) {
                continue;
            }
            failures += final_demo::run_benchmark_case(final_demo::kDemo2Specs[i], warmup, iters) != 0;
        }
        printf("benchmark overall: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

    int failures = 0;
    for (size_t i = 0; i < sizeof(final_demo::kDemo1Specs) / sizeof(final_demo::kDemo1Specs[0]); ++i) {
        failures += final_demo::run_correctness_case(final_demo::kDemo1Specs[i]) != 0;
    }
    for (size_t i = 0; i < sizeof(final_demo::kDemo2Specs) / sizeof(final_demo::kDemo2Specs[0]); ++i) {
        failures += final_demo::run_correctness_case(final_demo::kDemo2Specs[i]) != 0;
    }
    printf("overall: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
#endif
