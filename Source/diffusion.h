#ifndef __DIFFUSION_H__
#define __DIFFUSION_H__


#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"

void diffusion_device(int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d);

void velocity_to_double(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, real* ux_d, cudaComplex *uy_hat_d, real* uy_d, cudaComplex *uz_hat_d, real* uz_d);

void solve_advection_diffusion_projection(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void solve_advection_diffusion_projection_UV(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void solve_advection_diffusion_projection_UV_RHS(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

void RHS_advection_diffusion_projection(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d);

#endif
