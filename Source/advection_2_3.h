#ifndef __H_ADVECTION_H__
#define __H_ADVECTION_H__

#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"


//arrays for 2/3 padding
static cudaComplex *ux_hat_d_ext, *uy_hat_d_ext, *uz_hat_d_ext;
static cudaComplex *ux_x_hat_d_ext, *ux_y_hat_d_ext, *ux_z_hat_d_ext;
static cudaComplex *uy_x_hat_d_ext, *uy_y_hat_d_ext, *uy_z_hat_d_ext;
static cudaComplex *uz_x_hat_d_ext, *uz_y_hat_d_ext, *uz_z_hat_d_ext;
static real *ux_d_ext,*uy_d_ext,*uz_d_ext;
static real *der_x_d_ext,*der_y_d_ext,*der_z_d_ext;

//for B(u,v)=(u,\nabla)v+(v,\nabla)u
static cudaComplex *vx_hat_d_ext, *vy_hat_d_ext, *vz_hat_d_ext;
static cudaComplex *vx_x_hat_d_ext, *vx_y_hat_d_ext, *vx_z_hat_d_ext;
static cudaComplex *vy_x_hat_d_ext, *vy_y_hat_d_ext, *vy_z_hat_d_ext;
static cudaComplex *vz_x_hat_d_ext, *vz_y_hat_d_ext, *vz_z_hat_d_ext;

static real *vx_d_ext,*vy_d_ext,*vz_d_ext;


static real *Qx_d_ext,*Qy_d_ext,*Qz_d_ext;

static real *mask_2_3_d_ext;


__global__ void set_to_zero_real(int Nx, int Ny, int Nz, cudaComplex *a1, cudaComplex *a2, cudaComplex *a3);

__global__ void build_vels_device(int Nx, int Ny, int Nz, cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3);

void set_to_zero(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *a1, cudaComplex *a2, cudaComplex *a3);

void init_dealiasing(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, real *mask_2_3_d);

void clean_dealiasing();

void calculate_convolution_2p3(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d);

void calculate_convolution_2p3_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d);


#endif
