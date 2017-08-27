#ifndef __H_CUDA_SUPP_H__
#define __H_CUDA_SUPP_H__

#include <stdarg.h>
#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include <cstdlib>
#include "Macros.h"
#include "min_max_reduction.h"

bool InitCUDA(int GPU_number=-1);
cudaComplex* device_allocate_complex(int Nx, int Ny, int Nz);
real* device_allocate_real(int Nx, int Ny, int Nz);
void device_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz);
void host_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz);
void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_real(int count, ...);
void device_allocate_all_complex(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_complex(int count, ...);
void calc_dt(dim3 dimBlock, dim3 dimGrid, real CFL, int Nx, int Ny, int Nz,  real dx, real dy, real dz, real* ux, real* uy, real* uz, real* cfl_in, real *cfl_out, real *dt_pointer);
void retrive_Shapiro_step(real* ret, real* cfl_out);

int check_nans_kernel(char message[], int N, real *vec);
int check_nans_kernel(char message[], int N, cudaComplex *vec);

#endif
