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
	
	vec_dest[0]=0;
}




//user definded fucntion example

void user_Ax_function(Ax_struct *SC, real * vec_f_in, real * vec_f_out){

	int N=SC->N;
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	call_vector_map_kernel<<< blocks, threads>>>(N, vec_f_in, vec_f_out);

}


__global__  void single_euler_step(int N, real tau, real *vec_in, real *vec_out){
	//assume that vec_out contains the action of RHS on the vec_in

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){

		vec_out[i]=tau*vec_out[i]+vec_in[i];

	}


}

__global__  void copy_vectors(int N, real *vec_in, real *vec_out){
	//copy vec_in to vec_out

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){

		vec_out[i]=vec_in[i];

	}


}


__global__  void shift_vector(int N, real *vec_in, real shift, real *vec_out){
	//copy vec_in to vec_out

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){

		vec_out[i]=vec_out[i]-shift*vec_in[i];
	}


}


void single_step(dim3 threads, dim3 blocks, int N, real tau, real * vec_f_in, real * vec_f_out){

	call_vector_map_kernel<<< blocks, threads>>>(N, vec_f_in, vec_f_out);
	single_euler_step<<< blocks, threads>>>(N, tau, vec_f_in, vec_f_out);

	//TODO: implement RK3 or RK4 exponent estimate! Euler method is only for a test!

}




