#ifndef __ARNOLDI_CUDA_SUPPORT_H__
#define __ARNOLDI_CUDA_SUPPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Macros.h"


namespace Arnoldi
{

void check_for_nans(char message[], int Size, real *array);

bool InitCUDA(int GPU_number);
real* device_allocate_real(int Nx, int Ny);
real* device_allocate_real(int Nx, int Ny, int Nz);
void device_host_real_cpy(real* device, real* host, int Nx, int Ny);
void host_device_real_cpy(real* host, real* device, int Nx, int Ny);
void checkError(cublasStatus_t status, const char *msg);




//new start here:
void vectors_add_GPU(cublasHandle_t handle, int N, real alpha, real *x, real *y);
void matrixMultVector_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, real beta, real *res);
real vector_norm2_GPU(cublasHandle_t handle, int N, real *vec);
void normalize_vector_GPU(cublasHandle_t handle, int N, real *x);
void vector_copy_GPU(cublasHandle_t handle, int N, real *vec_source, real *vec_dest);
real vector_dot_product_GPU(cublasHandle_t handle, int N, real *vec1, real *vec2);
void set_matrix_colomn_GPU(int Row, int Col, real *mat, real *vec, int col_number);
void get_matrix_colomn_GPU(int Row, int Col, real *mat, real *vec, int col_number);
void matrixMultVector_part_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, int part_Cols, real beta, real *res);
void matrixDotVector_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, real beta, real *res);
void matrixDotVector_part_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, int part_Cols, real beta, real *res);
void set_vector_value_GPU(int N, real val, real *vec);
void set_vector_inverce_GPU(int N, real *vec);
void matrixMultMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, real *A, real alpha, real *B, real beta, real *C);
void matrixTMultMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, real *A, real alpha, real *B, real beta, real *C);
void matrixMultComplexMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, cublasComplex *A, cublasComplex *B, cublasComplex *C);

}
#endif
