//WARNING: declaration of BiCGStab is defined in file "Matrix_Vector_emulator.h" due to cross declaration!
//In this project this file is void

#ifndef __H_BICGSTABL_H__
#define __H_BICGSTABL_H__

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include "Macros.h"
#include "cuda_supp.h"
#include "Products.h"
#include "memory_operations.h"
#include "Matrix_Vector_emulator.h"


int BiCGStabL(int L, int N, user_map_vector Axb, void *user_struct, real *x, real* RHS, real *tol, int *Iter, bool verbose, unsigned int skip=500);
	



#endif