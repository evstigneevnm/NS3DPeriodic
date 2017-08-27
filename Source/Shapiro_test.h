#ifndef __SHAPIRO_TEST_H__
#define __SHAPIRO_TEST_H__


#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "math_support.h"
#include "min_max_reduction.h"

real Shapiro_test_case(dim3 dimGrid, dim3 dimBlock, real dx, real dy, real dz, real current_time, real Re, real *ux, real *uy, real *uz, int Nx, int Ny, int Nz, real *cfl_in, real *cfl_out, real *ret);

#endif
