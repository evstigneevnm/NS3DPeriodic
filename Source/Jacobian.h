#ifndef __JACOBIAN_H__
#define __JACOBIAN_H__


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Macros.h"
#include "RK_time_step.h"
#include "file_operations.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "math_support.h"

#ifndef IJ
	#define IJ(i,j) (i)*(NJacobian)+(j)
#endif



void build_Jacobian(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *U_eps_d, cudaComplex *RHSx_plus, cudaComplex *RHSx_minus, cudaComplex *RHSy_plus, cudaComplex *RHSy_minus, cudaComplex *RHSz_plus, cudaComplex *RHSz_minus, cudaComplex *Diff_RHSx, cudaComplex *Diff_RHSy, cudaComplex *Diff_RHSz, real *Jacobian_d);


void print_Jacobian(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d);

#endif