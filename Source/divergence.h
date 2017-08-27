#ifndef __DIVERGENCE_H__
#define __DIVERGENCE_H__


#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"


void divergence_device(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *div_hat_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, real dt, real Re);

void devergence_to_double(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *div_hat_d, real* div_d);



#endif
