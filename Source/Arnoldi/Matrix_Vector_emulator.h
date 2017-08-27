#ifndef __ARNOLDI_H_Matrix_Vector_emulator_H__
#define __ARNOLDI_H_Matrix_Vector_emulator_H__


#include "Macros.h"
#include "cuda_supp.h"

//declare user function prototype to call Vector mapping via structure object pointer "matctx" with domain in "src" and image in "dst"
typedef void (*user_map_vector) (void* matctx, void* src, void* dst);


typedef struct user_Ax_struture{ 
	int N;
} Ax_struct;

typedef struct user_Ax_struture1{ 
	int N;
	real *A_d;
	cublasHandle_t handle;
} Ax_struct_1;



//declare user function that will be passed with structure
void user_Ax_function(Ax_struct *SC, double * vec_f_in, double * vec_f_out);
void user_Ax_function_1(Ax_struct_1 *SC, double * vec_f_in, double * vec_f_out);


#endif