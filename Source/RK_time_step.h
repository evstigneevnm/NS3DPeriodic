#ifndef __RK_TIME_STEP_H__
#define __RK_TIME_STEP_H__


#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"

#include "diffusion.h"
#include "divergence.h"
#include "advection_2_3.h"
#include "advection_WENO.h"


void copy_arrays(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz,  cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3);


void single_time_step(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, cudaComplex *ux_hat_d_back, cudaComplex *uy_hat_d_back, cudaComplex *uz_hat_d_back, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void single_time_step_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void single_time_step_iUV_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void RK3_SSP(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);


void RK3_SSP_UV_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);


void RK3_SSP_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1, cudaComplex *vx_hat_d_1, cudaComplex *vy_hat_d_1, cudaComplex *vz_hat_d_1, cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void return_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d);

#endif
