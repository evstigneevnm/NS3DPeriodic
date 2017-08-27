#ifndef __ARNOLDI_H_Arnoldi_Driver_H__
#define __ARNOLDI_H_Arnoldi_Driver_H__

#include "Macros.h"
#include "cuda_supp.h"
#include "Matrix_Vector_emulator.h"
#include "memory_operations.h"
#include "Products.h"





void Arnoldi_driver(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *V_d, real *H, real *vec_f_d, int k, int m, real *vec_v_d, real *vec_w_d, real *vec_c_d, real *vec_h_d, real *vec_h);

#endif