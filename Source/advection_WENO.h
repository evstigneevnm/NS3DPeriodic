#ifndef __H_ADVECTION_WENO_H__
#define __H_ADVECTION_WENO_H__

#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"
#include "advection_2_3.h"


static cudaComplex *ux_hat_d_ext_W, *uy_hat_d_ext_W, *uz_hat_d_ext_W;
static real *ux_d_ext_W,*uy_d_ext_W,*uz_d_ext_W;

static real *Qx_d_ext_W,*Qy_d_ext_W,*Qz_d_ext_W;


void WenoAdvection(dim3 dimBlock, dim3 dimGrid, dim3 dimBlock_C, dim3 dimGrid_C, int Nx, int Ny, int Nz, int Mz, int scheme, cudaComplex* ux_hat_d, cudaComplex* uy_hat_d, cudaComplex* uz_hat_d, real dx, real dy, real dz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d);


void init_WENO(int Nx, int Ny, int Nz, int Mz);
void clean_WENO();


#endif
