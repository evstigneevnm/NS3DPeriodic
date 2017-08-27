#ifndef __ARNOLDI_H_Matrix_Vector_emulator_H__
#define __ARNOLDI_H_Matrix_Vector_emulator_H__


#include "Macros.h"
#include "cuda_supp.h"
#include "Products.h"
#include "memory_operations.h"
//#include "BiCGStabL.h"

//declare user function prototype to call Vector mapping via structure object pointer "matctx" with domain in "src" and image in "dst"
typedef void (*user_map_vector) (void* matctx, void* src, void* dst);


typedef struct user_Ax_struture{ 
	int N;
} Ax_struct;


typedef struct user_Ax_struture_exponential{ 
	int N;
	real tau;
	int T;
	real shift_real;

	//for RK3 integration:
	real *vec_step0;
	real *vec_step1;
	real *vec_step2;

	//for BiCGStab(L):
	int BiCG_L;
	real BiCG_tol;
	int BiCG_Iter;
	cublasHandle_t handle;

} Ax_struct_exponential;


typedef struct user_Ax_struture1{ 
	int N;
	real *A_d;
	cublasHandle_t handle;
} Ax_struct_1;



//declare user function that will be passed with structure
void user_Ax_function(Ax_struct *SC, real * vec_f_in, real * vec_f_out);
void user_Ax_function_1(Ax_struct_1 *SC, real * vec_f_in, real * vec_f_out);
//exponential estimate and inversion with shift
void Axb_exponent_invert(Ax_struct_exponential *SC_exponential, real * vec_f_in, real * vec_f_out);



int BiCGStabL(cublasHandle_t handle, int L, int N, user_map_vector Axb, void *user_struct, real *x, real* RHS, real *tol, int *Iter, bool verbose, unsigned int skip=500);

#endif