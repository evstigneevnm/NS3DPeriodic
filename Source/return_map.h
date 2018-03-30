#ifndef __RETURN_MAP_H__
#define __RETURN_MAP_H__

#include <cmath>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"
#include "file_operations.h"
#include "memory_operations.h"
#include "RK_time_step.h"




void execute_return_map(int j_fixed, int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);


void execute_sections(int j_fixed, int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void execute_loaded_data(unsigned int intersecitons, int j_fixed, int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

#endif