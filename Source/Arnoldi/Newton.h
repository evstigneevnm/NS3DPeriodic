#ifndef __H_NEWTON_H__
#define __H_NEWTON_H__


#include "Macros.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "Matrix_Vector_emulator.h"
#include "BiCGStabL.h"


int Newton(cublasHandle_t cublasHandle, user_map_vector Jacobi_Axb, void *user_struct_Jacobi,  user_map_vector RHS_Axb, void *user_struct_RHS, int N, real *x, real *tol, int *iter, real tol_linsolve, int iter_linsolve, bool verbose=false, unsigned int skip=100);



#endif
