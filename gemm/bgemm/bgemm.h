#pragma once

#include <arm_acle.h>

enum XHBLAS_ORDER {
    XhblasRowMajor = 101,
    XhblasColMajor = 102
};

enum XHBLAS_TRANSPOSE {
    XhblasNoTrans = 111,
    XhblasTrans = 112,
    XhblasConjTrans = 113
};

#ifdef __cplusplus
extern "C" {
#endif

int bgemm_kernel(int order, int trans_a, int trans_b,
                 int M, int N, int K, float alpha,
                 const __bf16* A, int lda,
                 const __bf16* B, int ldb,
                 float beta, __bf16* C, int ldc,
                 __bf16* ba, int sign);

const char* xhblas_gemm_status_message(int status);

#ifdef __cplusplus
}
#endif
