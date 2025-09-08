#include <emscripten/emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Basic BLAS constants
#define CBLAS_ROW_MAJOR 101
#define CBLAS_COL_MAJOR 102
#define CBLAS_NO_TRANS  111
#define CBLAS_TRANS     112
#define CBLAS_CONJ_TRANS 113

// Simplified CBLAS implementation for WASM Foundation Tier 1

// Level 3 BLAS - Matrix-matrix operations
EMSCRIPTEN_KEEPALIVE
void blas_dgemm(int order, int transA, int transB, int m, int n, int k,
                double alpha, double* A, int lda, double* B, int ldb,
                double beta, double* C, int ldc) {
    // Simple matrix multiplication C = alpha * A * B + beta * C
    int i, j, l;
    
    // Apply beta scaling to C first
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    
    // Perform matrix multiplication
    if (order == CBLAS_ROW_MAJOR) {
        if (transA == CBLAS_NO_TRANS && transB == CBLAS_NO_TRANS) {
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (l = 0; l < k; l++) {
                        sum += A[i * lda + l] * B[l * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

EMSCRIPTEN_KEEPALIVE
void blas_sgemm(int order, int transA, int transB, int m, int n, int k,
                float alpha, float* A, int lda, float* B, int ldb,
                float beta, float* C, int ldc) {
    // Simple matrix multiplication C = alpha * A * B + beta * C
    int i, j, l;
    
    // Apply beta scaling to C first
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    
    // Perform matrix multiplication
    if (order == CBLAS_ROW_MAJOR) {
        if (transA == CBLAS_NO_TRANS && transB == CBLAS_NO_TRANS) {
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (l = 0; l < k; l++) {
                        sum += A[i * lda + l] * B[l * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

// Level 1 BLAS - Vector operations
EMSCRIPTEN_KEEPALIVE
double blas_ddot(int n, double* x, int incx, double* y, int incy) {
    double result = 0.0;
    int i, ix = 0, iy = 0;
    
    for (i = 0; i < n; i++) {
        result += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    
    return result;
}

EMSCRIPTEN_KEEPALIVE
float blas_sdot(int n, float* x, int incx, float* y, int incy) {
    float result = 0.0f;
    int i, ix = 0, iy = 0;
    
    for (i = 0; i < n; i++) {
        result += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    
    return result;
}

EMSCRIPTEN_KEEPALIVE
void blas_daxpy(int n, double alpha, double* x, int incx, double* y, int incy) {
    int i, ix = 0, iy = 0;
    
    for (i = 0; i < n; i++) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}

EMSCRIPTEN_KEEPALIVE
void blas_saxpy(int n, float alpha, float* x, int incx, float* y, int incy) {
    int i, ix = 0, iy = 0;
    
    for (i = 0; i < n; i++) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
    }
}

EMSCRIPTEN_KEEPALIVE
double blas_dnrm2(int n, double* x, int incx) {
    double sum = 0.0;
    int i, ix = 0;
    
    for (i = 0; i < n; i++) {
        sum += x[ix] * x[ix];
        ix += incx;
    }
    
    return sqrt(sum);
}

EMSCRIPTEN_KEEPALIVE
float blas_snrm2(int n, float* x, int incx) {
    float sum = 0.0f;
    int i, ix = 0;
    
    for (i = 0; i < n; i++) {
        sum += x[ix] * x[ix];
        ix += incx;
    }
    
    return sqrtf(sum);
}

EMSCRIPTEN_KEEPALIVE
void blas_dscal(int n, double alpha, double* x, int incx) {
    int i, ix = 0;
    
    for (i = 0; i < n; i++) {
        x[ix] *= alpha;
        ix += incx;
    }
}

EMSCRIPTEN_KEEPALIVE
void blas_sscal(int n, float alpha, float* x, int incx) {
    int i, ix = 0;
    
    for (i = 0; i < n; i++) {
        x[ix] *= alpha;
        ix += incx;
    }
}

// Utility functions
EMSCRIPTEN_KEEPALIVE
const char* blas_get_version(void) {
    return "SimpleBLAS-1.0-wasm";
}

EMSCRIPTEN_KEEPALIVE
int blas_get_num_threads(void) {
    return 1; // Single-threaded for simplicity
}

EMSCRIPTEN_KEEPALIVE
void blas_set_num_threads(int num_threads) {
    // No-op for single-threaded implementation
}