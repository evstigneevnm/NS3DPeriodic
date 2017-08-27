#ifndef __ARNOLDI_MEMORY_OPERATIONS_H__
#define __ARNOLDI_MEMORY_OPERATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <cublas_v2.h>
#include "Macros.h"

namespace Arnoldi
{

real* allocate_d(int Nx, int Ny, int Nz);
int* allocate_i(int Nx, int Ny, int Nz);
void allocate_real(int Nx, int Ny, int Nz, int count, ...);
void allocate_int(int Nx, int Ny, int Nz, int count, ...);
void deallocate_real(int count, ...);
void deallocate_int(int count, ...);
int* device_allocate_int(int Nx, int Ny, int Nz);
real* device_allocate_real(int Nx, int Ny, int Nz);
cublasComplex* device_allocate_complex(int Nx, int Ny, int Nz);
void to_device_from_host_int_cpy(int* device, int* host, int Nx, int Ny, int Nz);
void to_host_from_device_int_cpy(int* host, int* device, int Nx, int Ny, int Nz);
void to_device_from_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz);
void to_host_from_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz);
void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...);
void device_deallocate_all_real(int count, ...);

}


#endif
