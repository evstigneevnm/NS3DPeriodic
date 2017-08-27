#include "Matrix_Vector_emulator.h"

#ifndef I1DP
	#define I1DP(i) ((i)>(N-1)?(i)-N:(i)<0?(N+(i)):(i)) 
#endif



//kernel for user definded function
__global__ void call_vector_map_kernel(int N, real *vec_source, real *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		
		vec_dest[i]=vec_source[I1DP(i+1)]-2.0*vec_source[i]+vec_source[I1DP(i-1)];
	}
	
}

//user definded fucntion example

void user_Ax_function(Ax_struct *SC, double * vec_f_in, double * vec_f_out){

	int N=SC->N;
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	call_vector_map_kernel<<< blocks, threads>>>(N, vec_f_in, vec_f_out);

}


void user_Ax_function_1(Ax_struct_1 *SC, double * vec_f_in, double * vec_f_out){

	Arnoldi::matrixMultVector_GPU(SC->handle, SC->N, SC->A_d, SC->N, 1.0, vec_f_in, 0.0, vec_f_out);

}





//DEBUG function:

void call_vector_map_GPU(cublasHandle_t handle, int N, real *A_d, real *vec_f_d, real *res_d){

	Arnoldi::matrixMultVector_GPU(handle, N, A_d, N, 1.0, vec_f_d, 0.0, res_d);


}




void call_vector_map_GPU(int N,  real *vec_f_d, real *res_d){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	call_vector_map_kernel<<< blocks, threads>>>(N, vec_f_d, res_d);
}

