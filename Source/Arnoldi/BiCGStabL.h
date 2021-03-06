#ifndef __H_BICGSTABL_H__
#define __H_BICGSTABL_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "Products.h"
#include "memory_operations.h"
#include "Matrix_Vector_emulator.h"

#include <iostream> //for exaptions


int BiCGStabL(cublasHandle_t handle, int L, int N, user_map_vector Axb, void *user_struct, real *x, real* RHS, real *tol, int *Iter, bool verbose, unsigned int skip=500);
	



#endif