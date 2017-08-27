#ifndef __ARNOLDI_Implicit_restart_Arnoldi_H__
#define __ARNOLDI_Implicit_restart_Arnoldi_H__

#include "Macros.h"
#include "timer.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "Arnoldi_Driver.h"
#include "Select_Shifts.h"
#include "QR_Shifts.h"
#include "Matrix_Vector_emulator.h"
#include "file_operations.h"
#include <cblas.h>

extern "C" int openblas_get_num_threads(void);
extern "C" void openblas_set_num_threads(int);

real Implicit_restart_Arnoldi_GPU_data(cublasHandle_t handle, bool verbose, int N, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], int k, int m, complex real* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real=NULL, real *eigenvectors_imag=NULL, int BLASTreads=1);

real Implicit_restart_Arnoldi_GPU_data_Matrix_Exponent(cublasHandle_t handle, bool verbose, int N,  user_map_vector Axb_exponent_invert, void *user_struct_exponent_invert, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], char which_exponent[2], int k, int m, complex real* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real_d=NULL, real *eigenvectors_imag_d=NULL, int BLASThreads=1);




#endif