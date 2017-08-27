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



//user definded fucntion example

void user_Ax_function_exponential(Ax_struct_exponential *SC_exponential, real * vec_f_in, real * vec_f_out){

	int N=SC_exponential->N;
	real tau=SC_exponential->tau;
	int T=SC_exponential->T;
	real *vec_step0=SC_exponential->vec_step0;
	real *vec_step1=SC_exponential->vec_step1;
	real *vec_step2=SC_exponential->vec_step2;
	real shift_real=SC_exponential->shift_real;
	cublasHandle_t handle=SC_exponential->handle;


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	

	copy_vectors<<< blocks, threads>>>(N, vec_f_in, vec_step0); //input vector vec_f_in -> vec_step0
	for(int t=0;t<T;t++){
		single_step(threads, blocks, N, tau, vec_step0, vec_f_out);	//signle time step
		copy_vectors<<< blocks, threads>>>(N, vec_f_out, vec_step0);			//copy vec_f_out->vec_step0
	}
	copy_vectors<<< blocks, threads>>>(N, vec_step0, vec_f_out);
	//apply shifting! vec_f_out=vec_f_out-shift_real*vec_f_in
	shift_vector<<< blocks, threads>>>(N, vec_f_in, shift_real, vec_f_out);

//	call_vector_map_kernel<<< blocks, threads>>>(N, vec_f_in, vec_f_out);
}



void Axb_exponent_invert(Ax_struct_exponential *SC_exponential, real * vec_f_in, real * vec_f_out){

	int L=SC_exponential->BiCG_L;
	int N=SC_exponential->N;
	real *tol=new real[1];
	tol[0]=SC_exponential->BiCG_tol;
	int *Iter=new int[1];
	Iter[0]=SC_exponential->BiCG_Iter;
	cublasHandle_t handle=SC_exponential->handle;

	int res_flag=BiCGStabL(handle, L, N, (user_map_vector) user_Ax_function_exponential, (Ax_struct_exponential*) SC_exponential, vec_f_out, vec_f_in, tol, Iter, false, 500); //true->false; 10->ommit!
	switch (res_flag){
		case 0: //timer_print();
				//printf("converged with: %le, and %i iterations\n", tol[0], Iter[0]);
				//printf("%.03le ",tol[0]); 
				printf("%i|",Iter[0]); 
				fflush(stdout);
				break;
		case 1: //timer_print();
				printf("not converged with: %le, and %i iterations\n", tol[0], Iter[0]);
				exit(-1);
				break;
		case -1: printf("rho is 0 with: %le, and %i iterations\n", tol[0], Iter[0]);
				exit(-1);
				break;
		case -2: printf("omega is with: %le, and %i iterations\n", tol[0], Iter[0]);
				exit(-1);
				break;
		case -3: printf("NANs with: %le, and %i iterations\n", tol[0], Iter[0]);
				exit(-1);
				break;
	}
	delete [] tol, Iter;


}



void user_Ax_function_1(Ax_struct_1 *SC, real * vec_f_in, real * vec_f_out){

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

