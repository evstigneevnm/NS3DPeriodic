#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h> //for timer
//support
#include "Macros.h"
#include "cuda_supp.h"
#include "file_operations.h"
#include "math_support.h"
#include "memory_operations.h"
//Time step
#include "RK_time_step.h"

//Shapiro test case
#include "Shapiro_test.h"

//Jacobian matrix calcualtion
#include "Jacobian.h"

//For IRA eigen problem
#include "Arnoldi/Implicit_restart_Arnoldi.h"
#include <complex.h>

//BiCGStab(L)
#include "Arnoldi/BiCGStabL.h"

//during debug!
#include "Arnoldi/cuda_supp.h"

#define DEBUG 1

//=================================================================
//=====Some validation initial conditions and exact solutions======
//=================================================================
/*

-> Taylor-Green vortex
   initial conditions for Taylor-Green vortex:
	factor=2.0d0/sqrt(3.0d0)
	u(i,j,k)=factor*sin(theta+2.0d0*pi/3.0d0)*sin(x(i))*cos(y(j))*cos(z(k))
	v(i,j,k)=factor*sin(theta-2.0d0*pi/3.0d0)*cos(x(i))*sin(y(j))*cos(z(k))
	w(i,j,k)=factor*sin(theta)*cos(x(i))*cos(y(j))*sin(z(k))


-> ABC flow.
   initial conditions, force:
	v=(0;0;0)^T
	f=( A sin( kz ) + C cos( ky ); B sin( kx ) + C cos( kz ); C sin( ky ) + B cos( kx ) )^T*k^2
	assuming A=B=C=1, k=1.
   solution:
	v = ( A sin( kz ) + C cos( ky ); B sin( kx ) + C cos( kz ); C sin( ky ) + B cos( kx ) )^T
	
-> A. Shapiro " The use of an exact solution of the Navier-Stokes equations 
	 in a validation test of a three-dimensional nonhydrostatic numerical model"
	 Monthly Weather Review vol. 121, 2420-2425, (1993).
   initial conditions:
	time(1)=0.0d0
	factor=sqrt(3.0d0)
	u(i,j,k)=-0.5*( factor*cos(x(i))*sin(y(j))*sin(z(k))+sin(x(i))*cos(y(j))*cos(z(k)) )*exp(-(factor**2)*time(1)/Re)
	v(i,j,k)=0.5*(  factor*sin(x(i))*cos(y(j))*sin(z(k))-cos(x(i))*sin(y(j))*cos(z(k)) )*exp(-(factor**2)*time(1)/Re)
	w(i,j,k)=cos(x(i))*cos(y(j))*sin(z(k))*exp(-(factor**2)*time(1)/Re)
   solution:
   	u(x,y,z,t)=-0.25*(cos(x)sin(y)sin(z)+sin(x)cos(y)cos(z))exp(-t/Re)
	v(x,y,z,t)= 0.25*(sin(x)cos(y)sin(z)-cos(x)sin(y)cos(z))exp(-t/Re)
	w(x,y,z,t)= 0.5*cos(x)cos(y)sin(z)exp(-t/Re)


*/



//Structure for passing Arnoldi call



//RK3_SSP(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);

typedef struct NS_RK3_Call_parameters_Struture{ 
		dim3 dimGrid;
		dim3 dimBlock;
		dim3 dimGrid_C;
		dim3 dimBlock_C;
		real dx;
		real dy; 
		real dz;
		real dt;
		real Re;
		int Nx;
		int Ny;
		int Nz;
		int Mz;
		cudaComplex *ux_hat_d;
		cudaComplex *uy_hat_d;
		cudaComplex *uz_hat_d;
		cudaComplex *vx_hat_d;
		cudaComplex *vy_hat_d;
		cudaComplex *vz_hat_d;
		cudaComplex *ux_hat_d_1;
		cudaComplex *uy_hat_d_1;
		cudaComplex *uz_hat_d_1;
		cudaComplex *vx_hat_d_1;
		cudaComplex *vy_hat_d_1;
		cudaComplex *vz_hat_d_1;
		cudaComplex *ux_hat_d_2;
		cudaComplex *uy_hat_d_2;
		cudaComplex *uz_hat_d_2;
		cudaComplex *ux_hat_d_3;
		cudaComplex *uy_hat_d_3;
		cudaComplex *uz_hat_d_3;
		cudaComplex *fx_hat_d;
		cudaComplex *fy_hat_d;
		cudaComplex *fz_hat_d;
		cudaComplex *Qx_hat_d;
		cudaComplex *Qy_hat_d;
		cudaComplex *Qz_hat_d;
		cudaComplex *div_hat_d;
		real* kx_nabla_d;
		real* ky_nabla_d;
		real *kz_nabla_d;
		real *din_diffusion_d;
		real *din_poisson_d;
		real *AM_11_d;
		real *AM_22_d;
		real *AM_33_d;
		real *AM_12_d;
		real *AM_13_d;
		real *AM_23_d;
		int Timesteps_period;
} struct_NS3D_RK3_Call;





typedef struct NS_RK3_inverce_Exponent_Call_parameters_Struture{ 
		dim3 dimGrid;
		dim3 dimBlock;
		dim3 dimGrid_C;
		dim3 dimBlock_C;
		real dx;
		real dy; 
		real dz;
		real dt;
		real Re;
		int Nx;
		int Ny;
		int Nz;
		int Mz;
		cudaComplex *ux_hat_d;
		cudaComplex *uy_hat_d;
		cudaComplex *uz_hat_d;
		cudaComplex *vx_hat_d;
		cudaComplex *vy_hat_d;
		cudaComplex *vz_hat_d;
		cudaComplex *ux_hat_d_1;
		cudaComplex *uy_hat_d_1;
		cudaComplex *uz_hat_d_1;
		cudaComplex *vx_hat_d_1;
		cudaComplex *vy_hat_d_1;
		cudaComplex *vz_hat_d_1;
		cudaComplex *ux_hat_d_2;
		cudaComplex *uy_hat_d_2;
		cudaComplex *uz_hat_d_2;
		cudaComplex *ux_hat_d_3;
		cudaComplex *uy_hat_d_3;
		cudaComplex *uz_hat_d_3;
		cudaComplex *fx_hat_d;
		cudaComplex *fy_hat_d;
		cudaComplex *fz_hat_d;
		cudaComplex *Qx_hat_d;
		cudaComplex *Qy_hat_d;
		cudaComplex *Qz_hat_d;
		cudaComplex *div_hat_d;
		real* kx_nabla_d;
		real* ky_nabla_d;
		real *kz_nabla_d;
		real *din_diffusion_d;
		real *din_poisson_d;
		real *AM_11_d;
		real *AM_22_d;
		real *AM_33_d;
		real *AM_12_d;
		real *AM_13_d;
		real *AM_23_d;
		int Timesteps_period;

		// real exponent shifting
		real shift_real;

		//for BiCGStab(L):
		int BiCG_L;
		real BiCG_tol;
		int BiCG_Iter;
		cublasHandle_t handle;
} struct_NS3D_RK3_iExp_Call;




//kernel to make NS arrays from Arnoldi vector
__global__ void copy_vectors_kernel_3N(int N, real *vec_source, real *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		
		vec_dest[i]=vec_source[i];
		vec_dest[i+N]=vec_source[i+N];
		vec_dest[i+2*N]=vec_source[i+2*N];
	}
	
}



//kernel to make NS arrays from Arnoldi vector
__global__ void velocities_from_A_vector_kernel(int N, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vx_hat_d[i].x=0.0;
		vx_hat_d[i].y=vec_source[i];
		vy_hat_d[i].x=0.0;
		vy_hat_d[i].y=vec_source[i+N];		
		vz_hat_d[i].x=0.0;
		vz_hat_d[i].y=vec_source[i+2*N];
	}
	
}


__global__ void velocities_from_A_vector_scaled_kernel(int N, int N_point, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vx_hat_d[i].x=0.0;
		vx_hat_d[i].y=vec_source[i]*N_point;
		vy_hat_d[i].x=0.0;
		vy_hat_d[i].y=vec_source[i+N_point]*N_point;		
		vz_hat_d[i].x=0.0;
		vz_hat_d[i].y=vec_source[i+2*N_point]*N_point;
	}
	
}


//kernel to make NS arrays from Arnoldi vector with reduced size by 3
__global__ void velocities_from_A_vector_reduced_scaled_kernel(int N, int N_point, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	//assume that size of vec_source is (N-1)*3=3*N-3. these omitted 3 elements are zero Fourier components.

	
	if(i<(N-1)){
		vx_hat_d[i+1].x=0.0;
		vx_hat_d[i+1].y=vec_source[i]*N_point;
		vy_hat_d[i+1].x=0.0;
		vy_hat_d[i+1].y=vec_source[i+(N-1)]*N_point;		
		vz_hat_d[i+1].x=0.0;
		vz_hat_d[i+1].y=vec_source[i+(N-1)+(N-1)]*N_point;
	}
	vx_hat_d[0].x=0.0;
	vx_hat_d[0].y=0.0;
	vy_hat_d[0].x=0.0;
	vy_hat_d[0].y=0.0;
	vz_hat_d[0].x=0.0;
	vz_hat_d[0].y=0.0;

}





//kernel to make NS arrays from Arnoldi vector
__global__ void velocities_from_A_vector_reduced_kernel(int N, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	//assume that size of vec_source is (N-1)*3=3*N-3. these omitted 3 elements are zero Fourier components.

//*
	if(i<(N-1)){

		vx_hat_d[i+1].x=0.0;
		vx_hat_d[i+1].y=vec_source[i];
		vy_hat_d[i+1].x=0.0;
		vy_hat_d[i+1].y=vec_source[i+(N-1)];		
		vz_hat_d[i+1].x=0.0;
		vz_hat_d[i+1].y=vec_source[i+(N-1)+(N-1)];
	}
//*/
	vx_hat_d[0].x=0.0;
	vx_hat_d[0].y=0.0;
	vy_hat_d[0].x=0.0;
	vy_hat_d[0].y=0.0;		
	vz_hat_d[0].x=0.0;
	vz_hat_d[0].y=0.0;//vec_source[i+(N-1)+(N-1)];


}


void velocities_from_A_vector(int N, int N_point, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	velocities_from_A_vector_scaled_kernel<<< blocks, threads>>>(N, N_point, vec_source, vx_hat_d, vy_hat_d, vz_hat_d);
}

void velocities_from_A_vector_reduced(int N, int N_point, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	velocities_from_A_vector_reduced_scaled_kernel<<< blocks, threads>>>(N, N_point, vec_source, vx_hat_d, vy_hat_d, vz_hat_d);
}



//kernel to make Arnoldi vector from resulting vectors of NS
__global__ void A_vector_from_velocities_kernel(int N, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real *vec_source){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	//I had a bug here! Don't use temp variables!!!
	//For Arnoldi we modify only elements with index i!=0 for each velocity field

	if((i>0)&&(i<N)){
		vec_source[i]=vx_hat_d[i].y;
		vec_source[i+N]=vy_hat_d[i].y;
		vec_source[i+2*N]=vz_hat_d[i].y;
	}

}


__global__ void A_vector_from_velocities_reduced_kernel(int N, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	//assume that size of vec_source is (N-1)*3=3*N-3. these omitted 3 elements are zero Fourier components.


	if(i<(N-1)){
		vec_dest[i]=vx_hat_d[i+1].y;
		vec_dest[i+(N-1)]=vy_hat_d[i+1].y;
		vec_dest[i+2*(N-1)]=vz_hat_d[i+1].y;
	}

}


__global__ void A_vector_from_velocities_reduced_shifted_kernel(int N, real *vec_source, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real *vec_dest, real shift_real){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	//assume that size of vec_source is (N-1)*3=3*N-3. these omitted 3 elements are zero Fourier components.

	//
	//SOME BUG HERE! 170829!!!
	//A_vector_from_velocities_reduced_shifted_kernel
	//

	if(i<(N-1)){
		vec_dest[i]=vx_hat_d[i+1].y-shift_real*vec_source[i];
		vec_dest[i+(N-1)]=vy_hat_d[i+1].y-shift_real*vec_source[i+(N-1)];
		vec_dest[i+2*(N-1)]=vz_hat_d[i+1].y-shift_real*vec_source[i+2*(N-1)];
	}

}



void NSCallMatrixVector(struct_NS3D_RK3_Call *SC, double * vec_f_in, double * vec_f_out){


	int N_Arnoldi=(SC->Nx)*(SC->Ny)*(SC->Mz);
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N_Arnoldi+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	//copy_vectors_kernel_3N<<< blocks, threads>>>(N_Arnoldi, vec_f_in, vec_f_out);

	velocities_from_A_vector_kernel<<< blocks, threads>>>(N_Arnoldi, vec_f_in, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d);

	RK3_SSP_UV_RHS(SC->dimGrid, SC->dimBlock, SC->dimGrid_C, SC->dimBlock_C, SC->dx, SC->dy, SC->dz, SC->dt, SC->Re, SC->Nx, SC->Ny, SC->Nz, SC->Mz, SC->ux_hat_d, SC->uy_hat_d, SC->uz_hat_d,  SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, SC->ux_hat_d_1, SC->uy_hat_d_1, SC->uz_hat_d_1,  SC->ux_hat_d_2, SC->uy_hat_d_2, SC->uz_hat_d_2,  SC->ux_hat_d_3, SC->uy_hat_d_3, SC->uz_hat_d_3,  SC->fx_hat_d, SC->fy_hat_d, SC->fz_hat_d, SC->Qx_hat_d, SC->Qy_hat_d, SC->Qz_hat_d, SC->div_hat_d, SC->kx_nabla_d,  SC->ky_nabla_d, SC->kz_nabla_d, SC->din_diffusion_d, SC->din_poisson_d, SC->AM_11_d, SC->AM_22_d, SC->AM_33_d,  SC->AM_12_d, SC->AM_13_d, SC->AM_23_d);
	
	A_vector_from_velocities_kernel<<< blocks, threads>>>(N_Arnoldi, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, vec_f_out);

	//check_nans_kernel("F_out", N_Arnoldi*3, vec_f_out);
}


void NSCallMatrixVector_reduced(struct_NS3D_RK3_Call *SC, double * vec_f_in, double * vec_f_out){


	int N_Arnoldi=(SC->Nx)*(SC->Ny)*(SC->Mz);
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N_Arnoldi+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);


	velocities_from_A_vector_reduced_kernel<<< blocks, threads>>>(N_Arnoldi, vec_f_in, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d);
	
	RK3_SSP_UV_RHS(SC->dimGrid, SC->dimBlock, SC->dimGrid_C, SC->dimBlock_C, SC->dx, SC->dy, SC->dz, SC->dt, SC->Re, SC->Nx, SC->Ny, SC->Nz, SC->Mz, SC->ux_hat_d, SC->uy_hat_d, SC->uz_hat_d,  SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, SC->ux_hat_d_1, SC->uy_hat_d_1, SC->uz_hat_d_1,  SC->ux_hat_d_2, SC->uy_hat_d_2, SC->uz_hat_d_2,  SC->ux_hat_d_3, SC->uy_hat_d_3, SC->uz_hat_d_3,  SC->fx_hat_d, SC->fy_hat_d, SC->fz_hat_d, SC->Qx_hat_d, SC->Qy_hat_d, SC->Qz_hat_d, SC->div_hat_d, SC->kx_nabla_d,  SC->ky_nabla_d, SC->kz_nabla_d, SC->din_diffusion_d, SC->din_poisson_d, SC->AM_11_d, SC->AM_22_d, SC->AM_33_d,  SC->AM_12_d, SC->AM_13_d, SC->AM_23_d);
	

	A_vector_from_velocities_reduced_kernel<<< blocks, threads>>>(N_Arnoldi, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, vec_f_out);
		


}



void NSCallMatrixVector_exponential(struct_NS3D_RK3_Call *SC, double * vec_f_in, double * vec_f_out){


	int N_Arnoldi=(SC->Nx)*(SC->Ny)*(SC->Mz);
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N_Arnoldi+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	velocities_from_A_vector_kernel<<< blocks, threads>>>(N_Arnoldi, vec_f_in, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d);

	int Timesteps_period=SC->Timesteps_period;

	for(int tt=0;tt<Timesteps_period;tt++){



		RK3_SSP_UV(SC->dimGrid, SC->dimBlock, SC->dimGrid_C, SC->dimBlock_C, SC->dx, SC->dy, SC->dz, SC->dt, SC->Re, SC->Nx, SC->Ny, SC->Nz, SC->Mz, SC->ux_hat_d, SC->uy_hat_d, SC->uz_hat_d,  SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, SC->ux_hat_d_1, SC->uy_hat_d_1, SC->uz_hat_d_1,  SC->vx_hat_d_1, SC->vy_hat_d_1, SC->vz_hat_d_1,  SC->ux_hat_d_2, SC->uy_hat_d_2, SC->uz_hat_d_2,  SC->ux_hat_d_3, SC->uy_hat_d_3, SC->uz_hat_d_3,  SC->fx_hat_d, SC->fy_hat_d, SC->fz_hat_d, SC->Qx_hat_d, SC->Qy_hat_d, SC->Qz_hat_d, SC->div_hat_d, SC->kx_nabla_d,  SC->ky_nabla_d, SC->kz_nabla_d, SC->din_diffusion_d, SC->din_poisson_d, SC->AM_11_d, SC->AM_22_d, SC->AM_33_d,  SC->AM_12_d, SC->AM_13_d, SC->AM_23_d);


	
	}

	A_vector_from_velocities_kernel<<< blocks, threads>>>(N_Arnoldi, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, vec_f_out);

	printf(".");
	fflush(stdout);
}


void NSCallMatrixVector_exponential_reduced(struct_NS3D_RK3_iExp_Call *SC, real * vec_f_in, real * vec_f_out){


	int N_Arnoldi=(SC->Nx)*(SC->Ny)*(SC->Mz);
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N_Arnoldi+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	velocities_from_A_vector_reduced_kernel<<< blocks, threads>>>(N_Arnoldi, vec_f_in, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d);


	int Timesteps_period=SC->Timesteps_period;

	for(int tt=0;tt<Timesteps_period;tt++){


		//RK3_SSP_UV(SC->dimGrid, SC->dimBlock, SC->dimGrid_C, SC->dimBlock_C, SC->dx, SC->dy, SC->dz, SC->dt, SC->Re, SC->Nx, SC->Ny, SC->Nz, SC->Mz, SC->ux_hat_d, SC->uy_hat_d, SC->uz_hat_d,  SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, SC->ux_hat_d_1, SC->uy_hat_d_1, SC->uz_hat_d_1,  SC->vx_hat_d_1, SC->vy_hat_d_1, SC->vz_hat_d_1,  SC->ux_hat_d_2, SC->uy_hat_d_2, SC->uz_hat_d_2,  SC->ux_hat_d_3, SC->uy_hat_d_3, SC->uz_hat_d_3,  SC->fx_hat_d, SC->fy_hat_d, SC->fz_hat_d, SC->Qx_hat_d, SC->Qy_hat_d, SC->Qz_hat_d, SC->div_hat_d, SC->kx_nabla_d,  SC->ky_nabla_d, SC->kz_nabla_d, SC->din_diffusion_d, SC->din_poisson_d, SC->AM_11_d, SC->AM_22_d, SC->AM_33_d,  SC->AM_12_d, SC->AM_13_d, SC->AM_23_d);

		single_time_step_iUV_RHS(SC->dimGrid, SC->dimBlock, SC->dimGrid_C, SC->dimBlock_C, SC->dx, SC->dy, SC->dz, SC->dt, SC->Re, SC->Nx, SC->Ny, SC->Nz, SC->Mz, SC->ux_hat_d, SC->uy_hat_d, SC->uz_hat_d,  SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d,  SC->Qx_hat_d, SC->Qy_hat_d, SC->Qz_hat_d, SC->div_hat_d, SC->kx_nabla_d,  SC->ky_nabla_d, SC->kz_nabla_d, SC->din_diffusion_d, SC->din_poisson_d, SC->AM_11_d, SC->AM_22_d, SC->AM_33_d,  SC->AM_12_d, SC->AM_13_d, SC->AM_23_d);

		
	}

	//apply shifting and store vectors back
	A_vector_from_velocities_reduced_shifted_kernel<<< blocks, threads>>>(N_Arnoldi, vec_f_in, SC->vx_hat_d, SC->vy_hat_d, SC->vz_hat_d, vec_f_out, SC->shift_real);
	

}



// inverce matrix exponent

void Axb_exponent_invert(struct_NS3D_RK3_iExp_Call *SC_exponential, real * vec_f_in, real * vec_f_out){

	int L=SC_exponential->BiCG_L;
	int N=(SC_exponential->Nx)*(SC_exponential->Ny)*(SC_exponential->Mz);

	real *tol=new real[1];
	tol[0]=SC_exponential->BiCG_tol;
	int *Iter=new int[1];
	Iter[0]=SC_exponential->BiCG_Iter;
	cublasHandle_t handle=SC_exponential->handle;
	//NSCallMatrixVector_exponential_reduced(SC_exponential, vec_f_in, vec_f_out);



	int res_flag=BiCGStabL(handle, L, 3*N-3, (user_map_vector) NSCallMatrixVector_exponential_reduced, (struct_NS3D_RK3_iExp_Call*) SC_exponential, vec_f_out, vec_f_in, tol, Iter, false, 1); //true->false; 10->ommit!
	switch (res_flag){
		case 0: //timer_print();
				//printf("converged with: %le, and %i iterations\n", tol[0], Iter[0]);
				//printf("%.03le ",tol[0]); 
				//printf("%i, %.03le|",Iter[0], tol[0]); 
				printf("%i|", Iter[0]); 
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








void init_complex_and_Fourier(int Nx, int Ny, int Nz, real* ux_d, cudaComplex *ux_hat_d, real* uy_d, cudaComplex *uy_hat_d, real* uz_d, cudaComplex *uz_hat_d, real* fx_d, cudaComplex *fx_hat_d, real* fy_d, cudaComplex *fy_hat_d, real* fz_d, cudaComplex *fz_hat_d ){


	FFTN_Device(ux_d, ux_hat_d);
	
	FFTN_Device(uy_d, uy_hat_d);
	
	FFTN_Device(uz_d, uz_hat_d);
	
	FFTN_Device(fx_d, fx_hat_d);
	
	FFTN_Device(fy_d, fy_hat_d);
	
	FFTN_Device(fz_d, fz_hat_d);
}



void Image_to_Domain(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real* ux_d, cudaComplex *ux_hat_d, real* uy_d, cudaComplex *uy_hat_d, real* uz_d, cudaComplex *uz_hat_d){

	iFFTN_Device(dimGrid, dimBlock, ux_hat_d, ux_d, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_hat_d, uy_d, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uz_hat_d, uz_d, Nx, Ny, Nz);

}


void Initial_Taylor_Green_vortex(int Nx, int Ny, int Nz, real dx, real dy, real dz, real *ux, real *uy, real *uz){
	
	int j,k,l;
	FS
		real y=(k-Ny/2)*dy;
		real x=(j-Nx/2)*dx;		
		real z=(l-Nz/2)*dz;	
		//Taylor-Green vortex initial conditions
		real factor=2.0/sqrt(3.0);
		real theta=0.1;
		ux[IN(j,k,l)]=factor*sin(theta+2.0*PI/3.0)*sin(x)*cos(y)*cos(z);
		uy[IN(j,k,l)]=factor*sin(theta-2.0*PI/3.0)*cos(x)*sin(y)*cos(z);
		uz[IN(j,k,l)]=factor*sin(theta)*cos(x)*cos(y)*sin(z);

	FE


}

void Initial_ABC_flow(int Nx, int Ny, int Nz, real dx, real dy, real dz, real *ux, real *uy, real *uz, real *fx, real *fy, real *fz, real* uxABC, real* uyABC, real* uzABC, real Re){
	
	int j,k,l;
	real cA=1.0,cB=1.0,cC=1.0,ck=1.0;

	FS
		real y=(k-Ny/2)*dy;
		real x=(j-Nx/2)*dx;		
		real z=(l-Nz/2)*dz;	
		ux[IN(j,k,l)]=sin(y)*cos(z);
		uy[IN(j,k,l)]=sin(z)*cos(x);
		uz[IN(j,k,l)]=sin(x)*cos(y);
		//force
		fx[IN(j,k,l)]=ck*ck*(cA*sin(ck*x)+cC*cos(ck*y))/Re;//sin(nn*y); 
		fy[IN(j,k,l)]=ck*ck*(cB*sin(ck*x)+cC*cos(ck*z))/Re;//0.0;
		fz[IN(j,k,l)]=ck*ck*(cC*sin(ck*y)+cB*cos(ck*x))/Re;//0.0;
		//ABC flow Analytical solution
		uxABC[IN(j,k,l)]=cA*sin(ck*z) + cC*cos(ck*y);
		uyABC[IN(j,k,l)]= cB*sin(ck*x) + cC*cos(ck*z);
		uzABC[IN(j,k,l)]=cC*sin(ck*y) + cB*cos(ck*x);
	FE
}

void Initial_Kolmogorov_flow(int Nx, int Ny, int Nz, real alpha, real beta, real nn, real dx, real dy, real dz, real *ux, real *uy, real *uz, real *fx, real *fy, real *fz){

	int j,k,l;	
	FS
		real y=(k)*dy;
		real x=(j)*dx;		
		real z=(l)*dz;	


		//Kolmogorov flow 
		//initial conditions
		
		ux[IN(j,k,l)]=1.0*sin(y*nn)*sin(z*nn)+0.5*rand()/(RAND_MAX - 1);//sin(PI*i/(Nx/3.0));  //n*alpha*
		real NSY=1.0/alpha;
		real NSZ=1.0/beta;
		uy[IN(j,k,l)]=0.5*sin(alpha*x)*sin(z)+0.5*rand()/(RAND_MAX - 1);
		uz[IN(j,k,l)]=0.5*sin(alpha*x)*sin(y)+0.5*rand()/(RAND_MAX - 1);
		//Kolmogorov flow force
		fx[IN(j,k,l)]=sin(y*nn)*sin(z*nn); 
		fy[IN(j,k,l)]=0.0*sin(z);
		fz[IN(j,k,l)]=0.0*sin(x);


	FE
}



void Fourier_Initial_Kolmogorov_flow(int Nx, int Ny, int Nz, int Mz, real alpha, real beta, real nn, real dx, real dy, real dz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im, real *fx, real *fy, real *fz, real Re){



srand ( time(NULL) );
Nz=Mz;
int j,k,l;	
int n=(int)nn;
	fFS

		ux_hat_Re[IN(j,k,l)]=0.0;
		ux_hat_Im[IN(j,k,l)]=0.00*rand()/(RAND_MAX - 1);
		uy_hat_Re[IN(j,k,l)]=0.0;
		uy_hat_Im[IN(j,k,l)]=0.00*rand()/(RAND_MAX - 1);
		uz_hat_Re[IN(j,k,l)]=0.0;
		uz_hat_Im[IN(j,k,l)]=0.00*rand()/(RAND_MAX - 1);

	FE
		ux_hat_Im[IN(0,0,0)]=0.0;
		uy_hat_Im[IN(0,0,0)]=0.0;
		uz_hat_Im[IN(0,0,0)]=0.0;
		ux_hat_Re[IN(0,0,0)]=0.0;
		uy_hat_Re[IN(0,0,0)]=0.0;
		uz_hat_Re[IN(0,0,0)]=0.0;

//Re:	
		//ux_hat_Re[IN(0,n,n)]=(Nx)*(Ny)*(Nz)*0.5;
				//ux_hat_Re[IN(0,Ny-n,Nz-n)]=-Nx*Ny*Nz*0.25;
		//ux_hat_Re[IN(0,Ny-n,n)]=-(Nx)*(Ny)*(Nz)*0.5;
				//ux_hat_Re[IN(0,n,Nz-n)]=Nx*Ny*Nz*0.25;

//Im:


		ux_hat_Im[IN(0,n,n)]=-(Nx)*(Ny)*(Nz)*0.25*0.5*Re;
				//ux_hat_Im[IN(0,Ny-n,Mz-n)]=(Nx)*(Ny)*(Nz)*0.25;
		ux_hat_Im[IN(0,Ny-n,n)]=-(Nx)*(Ny)*(Nz)*0.25*0.5*Re;
				//ux_hat_Im[IN(0,n,Nz-n)]=(Nx)*(Ny)*(Nz)*0.25;

}



void Fourier_Initial_perturbation(real Magnitude, int Nx, int Ny, int Nz, int Mz, real alpha, real beta, real nn, real dx, real dy, real dz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im){


real Scale=1.0*Nx*Ny*Nz;
srand ( time(NULL) );
Nz=Mz;
int j,k,l;	
int n=(int)nn;

	for(int j=0;j<Nx;j++)
	for(int k=0;k<Ny;k++)
	for(int l=0;l<Mz;l++){
		//ux_hat_Re[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
		ux_hat_Im[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
		//uy_hat_Re[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
		uy_hat_Im[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
		//uz_hat_Re[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
		uz_hat_Im[IN(j,k,l)]+=Magnitude*(rand()/(1.0*RAND_MAX - 1.0)-rand()/(1.0*RAND_MAX - 1.0));
	
	


	}

		ux_hat_Im[IN(0,0,0)]=0.0;
		uy_hat_Im[IN(0,0,0)]=0.0;
		uz_hat_Im[IN(0,0,0)]=0.0;
		ux_hat_Re[IN(0,0,0)]=0.0;
		uy_hat_Re[IN(0,0,0)]=0.0;
		uz_hat_Re[IN(0,0,0)]=0.0;
	
	//	ux_hat_Re[IN(0,n,n)]=(Nx)*(Ny)*(Nz)*0.5;
	//	ux_hat_Im[IN(0,Ny-n,Nz-n)]=-Nx*Ny*Nz*0.5;
	//	ux_hat_Re[IN(0,Ny-n,n)]=-(Nx)*(Ny)*(Nz)*0.5;
	//	ux_hat_Im[IN(0,n,Nz-n)]=Nx*Ny*Nz*0.5;

}


//A. Shapiro " The use of an exact solution of the Navier-Stokes equations 
//	 in a validation test of a three-dimensional nonhydrostatic numerical model"
//	 Monthly Weather Review vol. 121, 2420-2425, (1993).
void Initial_Shapiro(int Nx, int Ny, int Nz, real dx, real dy, real dz, real *ux, real *uy, real *uz, real* uxABC, real* uyABC, real* uzABC, real Re, real current_time){
	
	int j,k,l;	
	FS
		real y=(k)*dy;
		real x=(j)*dx;		
		real z=(l)*dz;	
		real initial_time=0.0;
		real factor=sqrt(3.0);
		ux[IN(j,k,l)]=-0.5*(factor*cos(x)*sin(y)*sin(z)+sin(x)*cos(y)*cos(z) )*exp(-(factor*factor)*initial_time/Re);
		uy[IN(j,k,l)]=0.5*(factor*sin(x)*cos(y)*sin(z)-cos(x)*sin(y)*cos(z) )*exp(-(factor*factor)*initial_time/Re);
		uz[IN(j,k,l)]=cos(x)*cos(y)*sin(z)*exp(-(factor*factor)*initial_time/Re);
   //solution:
		uxABC[IN(j,k,l)]=-0.5*(factor*cos(x)*sin(y)*sin(z)+sin(x)*cos(y)*cos(z) )*exp(-(factor*factor)*current_time/Re);
		uyABC[IN(j,k,l)]=0.5*(factor*sin(x)*cos(y)*sin(z)-cos(x)*sin(y)*cos(z) )*exp(-(factor*factor)*current_time/Re);
		uzABC[IN(j,k,l)]=cos(x)*cos(y)*sin(z)*exp(-(factor*factor)*current_time/Re);
	FE


}


void set_all_zero(int Nx,int Ny,int Nz,real* ux,real* uy,real* uz, real* div_pos,real* uxABC,real* uyABC,real* uzABC, real* fx, real* fy, real* fz){

	int j,k,l;

	FS
		ux[IN(j,k,l)]=0.0;
		uy[IN(j,k,l)]=0.0;	
		uz[IN(j,k,l)]=0.0;
		uxABC[IN(j,k,l)]=0.0;
		uyABC[IN(j,k,l)]=0.0;	
		uzABC[IN(j,k,l)]=0.0;
		div_pos[IN(j,k,l)]=0.0;
		fx[IN(j,k,l)]=0.0;
		fy[IN(j,k,l)]=0.0;	
		fz[IN(j,k,l)]=0.0;
	FE



}

int main (int argc, char *argv[])
{
	int j, k, l, Nx=32, Ny=32, Nz=32, Mz=Nz/2+1;


	//if(argc!=13){
	if((argc<15)||(argc>18)){
		/*  
			0  - program name
			1  - Nx
			2  - Ny
			3  - Nz
			4  - X
			5  - Y
			6  - Z
			7  - number of timesteps
			8  - number of dropping timesteps
			9  - size of a constant timestep
			10 - number of timesteps in one period
			11 - Reynolds number
			12 - n for Kolmogorov problem
			13 - Read or not control file (0 - don't, 1 - simple control file, 2 - Fourier control file)
			14 - Initial perturbations applied on Fourier harmonics
			15 - GPU number to use (Optional 1)
			16 - IRA setup: number of eigenvalues (Optional 2)
			17 - IRA setup: extension of Krylov subspace (Optional 2)

		*/
		printf("%s Nx, Ny, Nz, X, Y, Z, timesteps, timesteps_to_drop, timestep_size, period_timestep, Reynolds, n, CF, Pert, GPU_number(Optional), k_eig(Optional), m_eig(Optional).\n",argv[0]);
		printf("Parameter number/it's description:\n");
		printf("1  - Nx - number of Fourier harmonics in X direction\n");
		printf("2  - Ny - number of Fourier harmonics in Y direction\n");
		printf("3  - Nz - number of Fourier harmonics in Z direction\n");
		printf("4  - X - length in X direction, realtive to 2Pi (i.e. X=2 => X=4Pi)\n");
		printf("5  - Y - length in Y direction\n");
		printf("6  - Z - length in Z direction\n");
		printf("7  - timesteps - number of timesteps\n");
		printf("8  - timesteps_to_drop - number of dropping timesteps\n");
		printf("9  - timestep_size- size of a constant timestep\n");
		printf("10 - period_timestep - number of timesteps in one period (for IRA calculation)\n");
		printf("11 - Reynolds - Reynolds number\n");
		printf("12 - n for Kolmogorov problem\n");
		printf("13 - CF - Read or not control file (0 - don't, 1 - simple control file, 2 - Fourier control file)\n");
		printf("14 - Pert - Initial perturbations applied on Fourier harmonics\n");
	
		printf("15 - GPU_number - GPU number to use (Optional 1)\n");
		printf("16 - k_eig(Optional) - IRA setup: number of eigenvalues (Optional 2)\n");
		printf("17 - m_eig(Optional) - IRA setup: extension of Krylov subspace (Optional 2)\n");
		return 0;
	}
	//(, shift_ux, shift_uy, shift_uz,)

	int GPU_Number=-1;
	int k_A=6;// for IRA setup!
	int m_A=3;// for IRA setup!

	
	Nx=atoi(argv[1]);
	Ny=atoi(argv[2]);	
	Nz=atoi(argv[3]);
	Mz=Nz/2+1;
	real X=atof(argv[4]);
	real Y=atof(argv[5]);
	real Z=atof(argv[6]);

	int timesteps=atoi(argv[7]);
	int drop=atoi(argv[8]); //time to drop
	real dt=atof(argv[9]);
	int Timesteps_period=atoi(argv[10]);
	real Re=atof(argv[11]);
	real nn=atof(argv[12]);
	int CFile=atoi(argv[13]);	
	real Perturbation=atof(argv[14]);
	if(argc==16){
		GPU_Number=atoi(argv[15]);
	}
	
	if(argc==17){
		k_A=atoi(argv[15]);
		m_A=atoi(argv[16]);
	}
	if(argc==18){
		GPU_Number=atoi(argv[15]);		
		k_A=atoi(argv[16]);
		m_A=atoi(argv[17]);
	}
//checking parameters:	

//	printf("\nUsing %i parameters: %i %i %i %0.2f %0.2f %0.2f %i %i %f %i %0.2f %0.1f %i %e %i %i %i\n", argc-1, Nx, Ny, Nz, X,Y,Z, timesteps, drop, dt, Timesteps_period, Re, nn, CFile, Perturbation, GPU_Number, k_A, m_A);
	printf("Parameter:\n");
	printf("1  - Nx=%i\n", Nx);
	printf("2  - Ny=%i\n", Ny);
	printf("3  - Nz=%i\n", Nz);
	printf("   - Mz=%i\n", Mz);
	printf("4  - X=%0.1f\n", X);
	printf("5  - Y=%0.1f\n", Y);
	printf("6  - Z=%0.1f\n", Z);
	printf("7  - timesteps=%i\n",timesteps);
	printf("8  - timesteps_to_drop=%i\n",drop);
	printf("9  - timestep_size=%e\n",dt);
	printf("10 - period_timestep=%i\n",Timesteps_period);
	printf("11 - Reynolds=%0.4f\n", Re);
	printf("12 - n=%0.2f\n",nn);
	printf("13 - CF=%i\n",CFile);
	printf("14 - Pert=%e\n",Perturbation);
	printf("15 - GPU_number=%i\n",GPU_Number);
	printf("16 - k_eig=%i\n",k_A);
	printf("17 - m_eig=%i\n",m_A);


	fflush(stdout);

	real shift_ux=0;
	real shift_uy=0;
	real shift_uz=0;

	real alpha=Y/X;
	real beta=Y/Z;
	X=2.0*PI/alpha;
	Y=2.0*PI;
	Z=2.0*PI/beta;
	real dx=X/(real)(Nx);
	real dy=Y/(real)(Ny);	
	real dz=Z/(real)(Nz);	
	real Lx=X;
	real Ly=Y;
	real Lz=Z;
	real middleX=0.5*X;
	real middleY=0.5*Y;
	real middleZ=0.5*Z;
	//check for cuda and select device!
	if(!InitCUDA(GPU_Number)) {
		return 0;
	}
	cudaDeviceReset();	


//	real dt=0.25*min2(min2(dx,dy),min2(dx,dz));
		
	real *ux,*uy,*uz, *div_pos, *u_abs;
	real *fx, *fy, *fz;
	//Fourier modes for output and control file
	real *ux_Re, *ux_Im, *uy_Re, *uy_Im, *uz_Re, *uz_Im;

	allocate_real(Nx, Ny, Nz, 8, &ux, &uy, &uz, &div_pos, &fx, &fy, &fz, &u_abs);
	//Fourier modes (Mz - for Fourier Image)
	allocate_real(Nx, Ny, Mz, 6, &ux_Re, &ux_Im, &uy_Re, &uy_Im, &uz_Re, &uz_Im);

	//real* dtret=allocate_d(2,1); //for timestep size return. Do i need it now?
	
	struct timeval start, end;

	real *uxABC, *uyABC, *uzABC;
	allocate_real(Nx, Ny, Nz, 3, &uxABC, &uyABC, &uzABC);

	set_all_zero(Nx,Ny,Nz,ux,uy,uz,div_pos,uxABC,uyABC,uzABC,fx,fy,fz);

	//Initial_Taylor_Green_vortex(Nx, Ny, Nz, dx, dy, dz, ux, uy, uz);

	Initial_Kolmogorov_flow(Nx, Ny, Nz, alpha, beta, nn, dx, dy, dz, ux, uy, uz, fx, fy, fz);

	bool Fourier_Initial_Conditions_flag=true;
	Fourier_Initial_Kolmogorov_flow(Nx, Ny, Nz, Mz, alpha, beta, nn, dx, dy, dz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im, fx, fy, fz,Re);

	//Initial_Shapiro(Nx, Ny, Nz, dx, dy, dz, ux, uy, uz, uxABC, uyABC, uzABC, Re, 0.0);

	//write_out_file_vec_pos_interp("p_outABCVec.pos", Nx, Ny, Nz, dx, dy, dz, uxABC, uyABC, uzABC);


	//do we read a control file?
	if(DEBUG!=1)
	if(CFile==1){
		printf("reading control file...");
		read_control_file(Nx, Ny, Nz, ux, uy, uz);
		printf(" done\n");
	}else if(CFile==2){
		printf("reading Fourier control file...");
		read_control_fourier(Nx, Ny, Mz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);
		printf(" done\n");
	}

	Fourier_Initial_perturbation(Perturbation, Nx, Ny, Nz, Mz, alpha, beta, nn, dx, dy, dz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);

	if(DEBUG!=1)
		write_out_file_vec_pos_interp("p_outXXX_V.pos", Nx, Ny, Nz, dx, dy, dz, ux, uy, uz);

	//cuda starts here!!!
	

	//host arrays for working
	real *kx_laplace, *ky_laplace, *kz_laplace; //1D wave numbers Laplace operator
	real *din_poisson, *din_diffusion;	//3D arrays of wavenumbers
	real *kx_nabla, *ky_nabla, *kz_nabla; //1D wave numbers \nabla operator (pure Im!)
	//cuda arrays:
	real *din_poisson_d, *din_diffusion_d;	//3D arrays of wavenumbers
	real *kx_nabla_d, *ky_nabla_d, *kz_nabla_d; //1D wave numbers \nabla operator (pure Im!)
	//Elements of projection matrix
	real *AM_11, *AM_22, *AM_33, *AM_12, *AM_13, *AM_23;
	//mask matrix for 2/3 dealiaing advection 
	real *mask_2_3;



	//device arrays for kinetic energy and dissipation
	real *energy_d, *energy_out1_d, *energy_out2_d,*dissipation_d;

	real *fx_d, *fy_d, *fz_d;
	real *ux_d, *uy_d, *uz_d, *div_pos_d, *u_abs_d;
	cudaComplex *ux_complex_d, *ux_hat_d, *uy_complex_d, *uy_hat_d, *uz_complex_d, *uz_hat_d, *div_hat_d, *div_complex_d, *fx_complex_d, *fy_complex_d, *fz_complex_d, *fx_hat_d, *fy_hat_d, *fz_hat_d;
	//for WENO advection
	real *ux1_d, *uy1_d, *uz1_d, *ux2_d, *uy2_d, *uz2_d;
	//for RK-3
	cudaComplex *ux_hat_d_1, *uy_hat_d_1, *uz_hat_d_1, *ux_hat_d_2, *uy_hat_d_2, *uz_hat_d_2, *ux_hat_d_3, *uy_hat_d_3, *uz_hat_d_3;
	//for high wavenumber analysis
	cudaComplex *ux_red_hat_d, *uy_red_hat_d, *uz_red_hat_d, *u_temp_complex_d;
	real *ux_red_d, *uy_red_d, *uz_red_d;
	//curl components in device memory
	real *rot_x_d, *rot_y_d, *rot_z_d;

	//Elements of projection matrix in device memory
	real *AM_11_d, *AM_22_d, *AM_33_d, *AM_12_d, *AM_13_d, *AM_23_d;
	
	//mask matrix for 2/3 dealiaing advection in device mem
	real *mask_2_3_d;

	//Fourier modes for output and control file in device mem
	real *ux_Re_d, *ux_Im_d, *uy_Re_d, *uy_Im_d, *uz_Re_d, *uz_Im_d;

	//timestep retreaving
	real *cfl_in, *cfl_out;


	//velicity after LES filters
	//device:
	cudaComplex *ux_filt_hat_d, *uy_filt_hat_d, *uz_filt_hat_d;
	real *ux_filt_d, *uy_filt_d, *uz_filt_d, *u_abs_filt_d;
	//host:
	real *ux_filt, *uy_filt, *uz_filt, *u_abs_filt;


	allocate_real(Nx, Ny, Mz, 2, &din_poisson, &din_diffusion);
	allocate_real(Nx, 1, 1, 2, &kx_laplace, &kx_nabla);
	allocate_real(1, Ny, 1, 2, &ky_laplace, &ky_nabla);
	allocate_real(1, 1, Mz, 2, &kz_laplace, &kz_nabla);
	//projection matrix elements
	allocate_real(Nx, Ny, Mz, 7, &AM_11, &AM_22, &AM_33, &AM_12, &AM_13, &AM_23, &mask_2_3);
	//filtered velocity
	allocate_real(Nx, Ny, Nz, 4, &ux_filt, &uy_filt, &uz_filt, &u_abs_filt);


	device_allocate_all_real(Nx, Ny, Nz, 15, &din_poisson_d, &din_diffusion_d, &ux_d, &uy_d, &uz_d, &div_pos_d, &fx_d, &fy_d, &fz_d, &u_abs_d, &cfl_in, &cfl_out, &rot_x_d, &rot_y_d, &rot_z_d);

	//filtered device velocity
	device_allocate_all_real(Nx, Ny, Nz, 4, &ux_filt_d, &uy_filt_d, &uz_filt_d, &u_abs_filt_d);

	//for projection matrix
	device_allocate_all_real(Nx, Ny, Mz, 6, &AM_11_d, &AM_22_d, &AM_33_d, &AM_12_d, &AM_13_d, &AM_23_d);

//kinetic energy and dissipation
	device_allocate_all_real(Nx, Ny, Nz, 4, &energy_d, &energy_out1_d, &energy_out2_d, &dissipation_d);


//for WENO advection
	device_allocate_all_real(Nx, Ny, Nz, 6, &ux1_d, &uy1_d, &uz1_d, &ux2_d, &uy2_d, &uz2_d);
//Fourier modes in device methods
	device_allocate_all_real(Nx, Ny, Mz, 6, &ux_Re_d, &ux_Im_d, &uy_Re_d, &uy_Im_d, &uz_Re_d, &uz_Im_d);

	kx_nabla_d=device_allocate_real(Nx,1,1);
	ky_nabla_d=device_allocate_real(1,Ny,1);
	kz_nabla_d=device_allocate_real(1,1,Mz);

	mask_2_3_d=device_allocate_real(Nx,Ny,Mz);

	device_allocate_all_complex(Nx, Ny, Mz, 7,  &ux_hat_d,&uy_hat_d, &uz_hat_d, &div_hat_d, &fx_hat_d, &fy_hat_d, &fz_hat_d);
//for RK-3
	device_allocate_all_complex(Nx, Ny, Mz, 9, &ux_hat_d_1, &uy_hat_d_1, &uz_hat_d_1, &ux_hat_d_2, &uy_hat_d_2, &uz_hat_d_2, &ux_hat_d_3, &uy_hat_d_3, &uz_hat_d_3);

//for high wavenumber analysis
	device_allocate_all_complex(Nx,Ny,Mz, 4, &ux_red_hat_d, &uy_red_hat_d, &uz_red_hat_d, &u_temp_complex_d);
	device_allocate_all_real(Nx, Ny, Nz, 3, &ux_red_d, &uy_red_d, &uz_red_d);


//for LES filtering
	device_allocate_all_complex(Nx,Ny,Mz, 3, &ux_filt_hat_d, &uy_filt_hat_d, &uz_filt_hat_d);


	//init cufft plans

	cufftResult result;
	cufftHandle planC2R, planR2C;

	result = cufftPlan3d(&planR2C, Nx, Ny, Nz, RealToComplex);
	if (result != CUFFT_SUCCESS) { printf ("*CUFFT MakePlan R->C* failed\n"); return; }
	result = cufftPlan3d(&planC2R, Nx, Ny, Nz, ComplexToReal);
	if (result != CUFFT_SUCCESS) { printf ("*CUFFT MakePlan C->R* failed\n"); return; }

	init_fft_plans(planR2C, planC2R);


//set up unchanging wave numbers
	build_Laplace_Wavenumbers(Nx, Ny, Mz, Lx, Ly, Lz, kx_laplace, ky_laplace, kz_laplace);
	build_Laplace_and_Diffusion(Nx, Ny, Mz, din_poisson, din_diffusion, kx_laplace, ky_laplace, kz_laplace);
	build_Nabla_Wavenumbers(Nx, Ny, Mz, Lx, Ly, Lz, kx_nabla, ky_nabla, kz_nabla);

//set up projection matrix per element
	build_projection_matrix_elements(Nx, Ny, Mz, Lx, Ly, Lz, AM_11, AM_22, AM_33, AM_12, AM_13, AM_23);

//set up dealiazing mask matrix
	build_mask_matrix(Nx, Ny, Mz, Lx, Ly, Lz, mask_2_3);



//	copy all to device memory
	printf("HOST->DEVICE...");
	device_host_real_cpy(kx_nabla_d, kx_nabla, Nx, 1, 1);
	device_host_real_cpy(ky_nabla_d, ky_nabla, 1, Ny, 1);
	device_host_real_cpy(kz_nabla_d, kz_nabla, 1, 1, Mz);
	device_host_real_cpy(din_poisson_d, din_poisson, Nx, Ny, Mz);
	device_host_real_cpy(din_diffusion_d, din_diffusion, Nx, Ny, Mz);



//copy projection matrix elements into device memory
	device_host_real_cpy(AM_11_d, AM_11, Nx, Ny, Mz);	
	device_host_real_cpy(AM_22_d, AM_22, Nx, Ny, Mz);	
	device_host_real_cpy(AM_33_d, AM_33, Nx, Ny, Mz);	
	device_host_real_cpy(AM_12_d, AM_12, Nx, Ny, Mz);	
	device_host_real_cpy(AM_13_d, AM_13, Nx, Ny, Mz);	
	device_host_real_cpy(AM_23_d, AM_23, Nx, Ny, Mz);	
//copy dealiazing mask matrix
	device_host_real_cpy(mask_2_3_d, mask_2_3, Nx, Ny, Mz);


//copy Fourier modes to device
	device_host_real_cpy(ux_Re_d, ux_Re, Nx, Ny, Mz);	
	device_host_real_cpy(ux_Im_d, ux_Im, Nx, Ny, Mz);	
	device_host_real_cpy(uy_Re_d, uy_Re, Nx, Ny, Mz);	
	device_host_real_cpy(uy_Im_d, uy_Im, Nx, Ny, Mz);	
	device_host_real_cpy(uz_Re_d, uz_Re, Nx, Ny, Mz);	
	device_host_real_cpy(uz_Im_d, uz_Im, Nx, Ny, Mz);	


//copy working arrays
	device_host_real_cpy(ux_d, ux, Nx, Ny, Nz);
	device_host_real_cpy(uy_d, uy, Nx, Ny, Nz);
	device_host_real_cpy(uz_d, uz, Nx, Ny, Nz);
	device_host_real_cpy(div_pos_d, div_pos, Nx, Ny, Nz);
	
	device_host_real_cpy(u_abs_d, div_pos, Nx, Ny, Nz);

	device_host_real_cpy(fx_d, fx, Nx, Ny, Nz);
	device_host_real_cpy(fy_d, fy, Nx, Ny, Nz);
	device_host_real_cpy(fz_d, fz, Nx, Ny, Nz);

	device_host_real_cpy(rot_x_d, div_pos, Nx, Ny, Nz);
	device_host_real_cpy(rot_y_d, div_pos, Nx, Ny, Nz);
	device_host_real_cpy(rot_z_d, div_pos, Nx, Ny, Nz);

//CUDA grid setup

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//Set dim blocks for Device
	unsigned int k1, k2;
	// step 1: compute # of threads per block
	unsigned int nthreads = block_size_x * block_size_y;
	// step 2: compute # of blocks needed
	unsigned int nblocks = ( Nx*Ny*Nz + nthreads -1 )/nthreads ;
	// step 3: find grid's configuration
	float db_nblocks = (float)nblocks ;
	k1 = (unsigned int) floor( sqrt(db_nblocks) ) ;
	k2 = (unsigned int) ceil( db_nblocks/((float)k1)) ;

	dim3 dimBlock(block_size_x, block_size_y, 1);
	dim3 dimGrid( k2, k1, 1 );

	printf("\n dimGrid: %dX%dX%d, dimBlock: %dX%dX%d. \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);


	// step 2: compute # of blocks needed
	nblocks = ( Nx*Ny*Mz + nthreads -1 )/nthreads ;
	// step 3: find grid's configuration
	db_nblocks = (float)nblocks ;
	k1 = (unsigned int) floor( sqrt(db_nblocks) ) ;
	k2 = (unsigned int) ceil( db_nblocks/((float)k1)) ;

	dim3 dimBlock_C(block_size_x, block_size_y, 1);
	dim3 dimGrid_C( k2, k1, 1 );

	printf(" dimGrid_C: %dX%dX%d, dimBlock_C: %dX%dX%d. \n", dimGrid_C.x, dimGrid_C.y, dimGrid_C.z, dimBlock_C.x, dimBlock_C.y, dimBlock_C.z);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// 2/3 dealiasing and WENO initialization
	init_dealiasing(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, mask_2_3_d);
	init_WENO(Nx, Ny, Nz, Mz);

	//return of the (U,\nabla)U convolution
	cudaComplex *Qx_hat_d, *Qy_hat_d, *Qz_hat_d;
	device_allocate_all_complex(Nx, Ny, Mz, 3, &Qx_hat_d, &Qy_hat_d, &Qz_hat_d);


//init Fourier modes if we haven't read them from control_file_Fourier.dat
	//if(CFile!=2){
	init_complex_and_Fourier(Nx, Ny, Nz, ux_d,  ux_hat_d, uy_d, uy_hat_d, uz_d,  uz_hat_d, fx_d, fx_hat_d, fy_d, fy_hat_d, fz_d, fz_hat_d);
	//}
	

	if((CFile==2)||(Fourier_Initial_Conditions_flag==true)){
		printf("Using direct Fourier modes initialization.\n");
		all_double2Fourier(dimGrid_C, dimBlock_C, ux_Re_d, ux_Im_d, ux_hat_d, uy_Re_d, uy_Im_d, uy_hat_d, uz_Re_d, uz_Im_d, uz_hat_d, Nx, Ny, Mz);
	}
	if(Fourier_Initial_Conditions_flag==true){
		init_fources_fourier_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, fx_hat_d, fy_hat_d, fz_hat_d);
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	printf(" done\n");


	real CFL=0.05;
	real TotalTime=0.0;

	//write initial conditions
	Image_to_Domain(dimGrid, dimBlock, Nx, Ny, Nz, ux_d, ux_hat_d, uy_d, uy_hat_d, uz_d, uz_hat_d);
	velocity_to_abs_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d, uy_d, uz_d, u_abs_d);	
	if(DEBUG!=1)
		write_drop_files_from_device(drop, 0, Nx, Ny, Nz, ux, uy, uz, u_abs, div_pos, dx, dy, dz, ux_d, uy_d, uz_d, u_abs_d, div_pos_d);


	//droping data to a file	
	FILE *stream;
	stream=fopen("time_dependant.dat", "w" );

	//drop time dependant data
	real ux_loc,uy_loc,uz_loc, div_loc, rot_x_loc, rot_y_loc, rot_z_loc;

	//dt*=CFL; //temp with constant time step!!!


	real* Sh_ret=(real*)malloc(sizeof(real)*2);
	
	gettimeofday(&start, NULL);
	for(int t=0;t<timesteps;t++){

		//real time step starts here
	
			//!!!
		//calc_dt(dimBlock, dimGrid, one_over_n2, one_over_n3, CFL, Nx, Ny, Nz, dx, dy, dz,  ux_d,  uy_d,  uz_d, cfl_in, cfl_out, &dt);

		//dt=min2(0.005,dt);
//fixed time step size
		//dt=0.015;//0.0025//0.01;
	

		RK3_SSP(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);
		
		//single_time_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);


		//real time step ends here
		
		divergence_device(dimGrid_C, dimBlock_C, Nx, Ny, Mz, div_hat_d, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d,  kz_nabla_d, 1.0, Re);
		devergence_to_double(dimGrid, dimBlock, Nx, Ny, Nz, div_hat_d, div_pos_d);
 		
		Image_to_Domain(dimGrid, dimBlock, Nx, Ny, Nz, ux_d, ux_hat_d, uy_d, uy_hat_d, uz_d, uz_hat_d);



		//Helmholz_Fourier_Filter(dimGrid, dimBlock,  dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, Lx, Ly, Lz, ux_hat_d, uy_hat_d, uz_hat_d, ux_filt_hat_d, uy_filt_hat_d, uz_filt_hat_d, 1.0, ux_filt_d, uy_filt_d, uz_filt_d);

		CutOff_Fourier_Filter(dimGrid, dimBlock,  dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, Lx, Ly, Lz, ux_hat_d, uy_hat_d, uz_hat_d, ux_filt_hat_d, uy_filt_hat_d, uz_filt_hat_d, 0.1, ux_filt_d, uy_filt_d, uz_filt_d);


		velocity_to_abs_device<<<dimGrid, dimBlock>>>(Nx,  Ny, Nz, ux_d, uy_d, uz_d, u_abs_d);

 		if(DEBUG!=1)
 			write_drop_files_from_device(drop, (t+1), Nx, Ny, Nz, ux, uy, uz, u_abs, div_pos, dx, dy, dz, ux_d, uy_d, uz_d, u_abs_d, div_pos_d);
		



		//TODO: lame file operation =( Fix this in the future!

 		get_curl(dimGrid, dimBlock, Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, rot_x_d, rot_y_d, rot_z_d);

 		real energy=get_kinetic_energy(dimGrid, dimBlock, Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, energy_d, energy_out1_d, energy_out2_d);

 		energy=energy/(2.0*PI*2.0*PI*2.0*PI/alpha);

 		real dissipation=get_dissipation(dimGrid, dimBlock, Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, dissipation_d, energy_out1_d, energy_out2_d);

 		dissipation=dissipation/(2.0*PI*2.0*PI*2.0*PI/alpha);


		host_device_real_cpy(&ux_loc, &(ux_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
		host_device_real_cpy(&uy_loc, &(uy_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
 		host_device_real_cpy(&uz_loc, &(uz_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(div_pos_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);       
		
		host_device_real_cpy(&rot_x_loc, &(rot_x_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
		host_device_real_cpy(&rot_y_loc, &(rot_y_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
 		host_device_real_cpy(&rot_z_loc, &(rot_z_d[IN(Nx/2-4,Ny/2-3,Nz/2-2)]), 1, 1, 1);
		
		TotalTime+=dt;
		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e	%e ", TotalTime, ux_loc, uy_loc, uz_loc, rot_x_loc, rot_y_loc, rot_z_loc, energy);	//1 2 3 4 5 6 7 8

		host_device_real_cpy(&ux_loc, &(ux_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
		host_device_real_cpy(&uy_loc, &(uy_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
 		host_device_real_cpy(&uz_loc, &(uz_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(div_pos_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);    
       
        host_device_real_cpy(&rot_x_loc, &(rot_x_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
		host_device_real_cpy(&rot_y_loc, &(rot_y_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
 		host_device_real_cpy(&rot_z_loc, &(rot_z_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);   
		
		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e ", ux_loc, uy_loc, uz_loc, rot_x_loc, rot_y_loc, rot_z_loc, dissipation);	// 9 10 11 12 13 14 15

	//high or low wavenumber analysis
	//	get_high_wavenumbers(dimGrid,  dimBlock, dimGrid_C,  dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, ux_red_hat_d, uy_red_hat_d, uz_red_hat_d, u_temp_complex_d, ux_red_d, uy_red_d, uz_red_d, 1);

		host_device_real_cpy(&ux_loc, &(ux_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);
		host_device_real_cpy(&uy_loc, &(uy_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);
 		host_device_real_cpy(&uz_loc, &(uz_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(div_pos_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);    
       
        host_device_real_cpy(&rot_x_loc, &(rot_x_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);
		host_device_real_cpy(&rot_y_loc, &(rot_y_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1);
 		host_device_real_cpy(&rot_z_loc, &(rot_z_d[IN(Nx/8,Ny/8,Nz/8)]), 1, 1, 1); 

		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e ", ux_loc, uy_loc, uz_loc, rot_x_loc, rot_y_loc, rot_z_loc, (energy-dissipation/Re)); //16 17 18 19 20 21 22	



// Filtered velicties and curls
 		get_curl(dimGrid, dimBlock, Nx, Ny, Nz, dx, dy, dz, ux_filt_d, uy_filt_d, uz_filt_d, rot_x_d, rot_y_d, rot_z_d);

		host_device_real_cpy(&ux_loc, &(ux_filt_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
		host_device_real_cpy(&uy_loc, &(uy_filt_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
 		host_device_real_cpy(&uz_loc, &(uz_filt_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(div_pos_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);    
       
        host_device_real_cpy(&rot_x_loc, &(rot_x_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
		host_device_real_cpy(&rot_y_loc, &(rot_y_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1);
 		host_device_real_cpy(&rot_z_loc, &(rot_z_d[IN(Nx/3,Ny/3,Nz/3)]), 1, 1, 1); 

		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e ", ux_loc, uy_loc,  uz_loc, rot_x_loc, rot_y_loc, rot_z_loc, energy); //23 24 25 26 27 28 29
		
//Fourier hormonics from here on
		int local_Nz=Nz;
		Nz=Mz;
		
		all_Fourier2double(dimGrid_C, dimBlock_C, ux_hat_d, ux_Re_d,ux_Im_d, uy_hat_d, uy_Re_d, uy_Im_d, uz_hat_d, uz_Re_d, uz_Im_d, Nx, Ny, Mz);

		host_device_real_cpy(&ux_loc, &(ux_Re_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
		host_device_real_cpy(&rot_x_loc, &(ux_Im_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
 		host_device_real_cpy(&uy_loc, &(uy_Re_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
        host_device_real_cpy(&rot_y_loc, &(uy_Im_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
        host_device_real_cpy(&uz_loc, &(uz_Re_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
		host_device_real_cpy(&rot_z_loc, &(uz_Im_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(div_pos_d[IN(Nx-6,Ny-6,Nz-6)]), 1, 1, 1);

		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e ", ux_loc, rot_x_loc, uy_loc, rot_y_loc, uz_loc, rot_z_loc, dissipation); //30 31 32 33 34 35 36

		host_device_real_cpy(&ux_loc, &(ux_Im_d[IN(Nx/5,Ny/4-3,Nz/4-1)]), 1, 1, 1);
		host_device_real_cpy(&rot_x_loc, &(ux_Im_d[IN(Nx/5,Ny/5,Nz/5)]), 1, 1, 1);
 		host_device_real_cpy(&uy_loc, &(uy_Im_d[IN(Nx/5,Ny/4-3,Nz/4-1)]), 1, 1, 1);
        host_device_real_cpy(&rot_y_loc, &(uy_Im_d[IN(Nx/5,Ny/5,Nz/5)]), 1, 1, 1);
        host_device_real_cpy(&uz_loc, &(uz_Im_d[IN(Nx/5,Ny/4-3,Nz/4-1)]), 1, 1, 1);
		host_device_real_cpy(&rot_z_loc, &(uz_Im_d[IN(Nx/5,Ny/5,Nz/5)]), 1, 1, 1);
        host_device_real_cpy(&div_loc, &(ux_Im_d[IN(2,2,2)]), 1, 1, 1);

		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e ", ux_loc, rot_x_loc, uy_loc, rot_y_loc, uz_loc, rot_z_loc, div_loc); //37 38 39 40 41 42 43

		host_device_real_cpy(&ux_loc, &(ux_Im_d[IN(0,0,0)]), 1, 1, 1); //44
		host_device_real_cpy(&rot_x_loc, &(ux_Im_d[IN(1,0,0)]), 1, 1, 1); //45
 		host_device_real_cpy(&uy_loc, &(ux_Im_d[IN(1,1,0)]), 1, 1, 1); //46
        host_device_real_cpy(&rot_y_loc, &(ux_Im_d[IN(1,1,1)]), 1, 1, 1); //47
        host_device_real_cpy(&uz_loc, &(ux_Im_d[IN(Nx-1,0,0)]), 1, 1, 1); //48
		host_device_real_cpy(&rot_z_loc, &(ux_Im_d[IN(Nx-1,Ny-1,0)]), 1, 1, 1); //49
        host_device_real_cpy(&div_loc, &(ux_Im_d[IN(Nx-1,Ny-1,Nz-1)]), 1, 1, 1); //50

		fprintf( stream, "%e	%e	%e	%e	%e	%e	%e\n", ux_loc, rot_x_loc, uy_loc, rot_y_loc, uz_loc, rot_z_loc, div_loc); //44 45 46 47 48 49 50
		Nz=local_Nz;

		//Add random perturbation during execution
		if((t+1)%drop==0){
			//copy Fourier modes to device
			host_device_real_cpy(ux_Re, ux_Re_d, Nx, Ny, Mz);	
			host_device_real_cpy(ux_Im, ux_Im_d, Nx, Ny, Mz);	
			host_device_real_cpy(uy_Re, uy_Re_d, Nx, Ny, Mz);	
			host_device_real_cpy(uy_Im, uy_Im_d, Nx, Ny, Mz);	
			host_device_real_cpy(uz_Re, uz_Re_d, Nx, Ny, Mz);	
			host_device_real_cpy(uz_Im, uz_Im_d, Nx, Ny, Mz);		
			//add perturbations
			Fourier_Initial_perturbation(Perturbation, Nx, Ny, Nz, Mz, alpha, beta, nn, dx, dy, dz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);
			//copy Fourier modes to device
			device_host_real_cpy(ux_Re_d, ux_Re, Nx, Ny, Mz);	
			device_host_real_cpy(ux_Im_d, ux_Im, Nx, Ny, Mz);	
			device_host_real_cpy(uy_Re_d, uy_Re, Nx, Ny, Mz);	
			device_host_real_cpy(uy_Im_d, uy_Im, Nx, Ny, Mz);	
			device_host_real_cpy(uz_Re_d, uz_Re, Nx, Ny, Mz);	
			device_host_real_cpy(uz_Im_d, uz_Im, Nx, Ny, Mz);		

		}



		//lame file operation ends here.

		if(t%217==0)
			printf("run:---[%.03f\%]---dt=%.03e---U=%.03e---\r",(t+1)/(1.0*timesteps)*100.0,dt,ux_loc);		
		

		//Shapiro test case
		
		//real Shapiro_error=Shapiro_test_case(dimBlock, dimGrid,  one_over_n2, one_over_n3, dx, dy, dz, TotalTime, Re, ux_d, uy_d, uz_d, Nx, Ny, Nz, cfl_in, cfl_out, Sh_ret);
		//printf("%e %e\n",TotalTime, Shapiro_error);


	}
	gettimeofday(&end, NULL);
	fclose(stream);
	
	free(Sh_ret);

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}	
	real etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
	printf("\n\nWall time:%fsec\n",etime);	
//performing EIGS!
	cudaComplex *vx_hat_d, *vy_hat_d, *vz_hat_d, *vx_hat_d_1, *vy_hat_d_1, *vz_hat_d_1;

	device_allocate_all_complex(Nx, Ny, Mz, 6, &vx_hat_d, &vy_hat_d, &vz_hat_d, &vx_hat_d_1, &vy_hat_d_1, &vz_hat_d_1);

	struct_NS3D_RK3_Call *NS3D_Call = new struct_NS3D_RK3_Call[1];

	NS3D_Call->dimGrid=dimGrid;
	NS3D_Call->dimBlock=dimBlock;
	NS3D_Call->dimGrid_C=dimGrid_C;
	NS3D_Call->dimBlock_C=dimBlock_C;
	NS3D_Call->dx=dx;
	NS3D_Call->dy=dy; 
	NS3D_Call->dz=dz;
	NS3D_Call->dt=dt;
	NS3D_Call->Re=Re;
	NS3D_Call->Nx=Nx;
	NS3D_Call->Ny=Ny;
	NS3D_Call->Nz=Nz;
	NS3D_Call->Mz=Mz;
	NS3D_Call->ux_hat_d=ux_hat_d;
	NS3D_Call->uy_hat_d=uy_hat_d;
	NS3D_Call->uz_hat_d=uz_hat_d;
	NS3D_Call->vx_hat_d=vx_hat_d;
	NS3D_Call->vy_hat_d=vy_hat_d;
	NS3D_Call->vz_hat_d=vz_hat_d;
	NS3D_Call->ux_hat_d_1=ux_hat_d_1;
	NS3D_Call->uy_hat_d_1=uy_hat_d_1;
	NS3D_Call->uz_hat_d_1=uz_hat_d_1;
	NS3D_Call->vx_hat_d_1=vx_hat_d_1;
	NS3D_Call->vy_hat_d_1=vy_hat_d_1;
	NS3D_Call->vz_hat_d_1=vz_hat_d_1;
	NS3D_Call->ux_hat_d_2=ux_hat_d_2;
	NS3D_Call->uy_hat_d_2=uy_hat_d_2;
	NS3D_Call->uz_hat_d_2=uz_hat_d_2;
	NS3D_Call->ux_hat_d_3=ux_hat_d_3;
	NS3D_Call->uy_hat_d_3=uy_hat_d_3;
	NS3D_Call->uz_hat_d_3=uz_hat_d_3;
	NS3D_Call->fx_hat_d=fx_hat_d;
	NS3D_Call->fy_hat_d=fy_hat_d;
	NS3D_Call->fz_hat_d=fz_hat_d;
	NS3D_Call->Qx_hat_d=Qx_hat_d;
	NS3D_Call->Qy_hat_d=Qy_hat_d;
	NS3D_Call->Qz_hat_d=Qz_hat_d;
	NS3D_Call->div_hat_d=div_hat_d;
	NS3D_Call->kx_nabla_d=kx_nabla_d;
	NS3D_Call->ky_nabla_d=ky_nabla_d;
	NS3D_Call->kz_nabla_d=kz_nabla_d;
	NS3D_Call->din_diffusion_d=din_diffusion_d;
	NS3D_Call->din_poisson_d=din_poisson_d;
	NS3D_Call->AM_11_d=AM_11_d;
	NS3D_Call->AM_22_d=AM_22_d;
	NS3D_Call->AM_33_d=AM_33_d;
	NS3D_Call->AM_12_d=AM_12_d;
	NS3D_Call->AM_13_d=AM_13_d;
	NS3D_Call->AM_23_d=AM_23_d;
	//for periodic and quasiperiodic problems!
	//NS3D_Call->Timesteps_period=1;
	NS3D_Call->Timesteps_period=Timesteps_period;//3478;




	int N_Arnoldi=3*Nx*Ny*Mz;
	double *vec_f=new double[N_Arnoldi];
	double *vec_v=new double[N_Arnoldi];
	double *vec_f_d, *vec_v_d;

	device_allocate_all_real(N_Arnoldi-3, 1, 1, 2, &vec_f_d, &vec_v_d);


	for (int i = 0; i < N_Arnoldi; ++i)
	{
		vec_f[i]=rand_normal(0.0, 1.0);
		vec_v[i]=0.0;
		//vec_f[i]=2.0*rand()/RAND_MAX - 1;
	}
	vec_v[0]=1;
	normalize(N_Arnoldi, vec_f);
	
	device_host_real_cpy(vec_f_d, vec_f, N_Arnoldi-3, 1, 1);
	device_host_real_cpy(vec_v_d, vec_v, N_Arnoldi-3, 1, 1);

	real res_tol=1;
	//int k_A=6, m_A=3; //initialized at the start of main from parameters or from default k_A=6, m_A=3.
	int m=m_A*k_A;
	double complex *eigenvaluesA=new double complex[m];
	real *eigvs_real, *eigvs_imag;

	device_allocate_all_real(N_Arnoldi-3, k_A, 1, 2, &eigvs_real, &eigvs_imag);


	cublasHandle_t handle;		//init cublas
	cublasStatus_t ret;
	ret = cublasCreate(&handle);
	Arnoldi::checkError(ret, " cublasCreate(). ");

	//check invert Exponent structure init
	struct_NS3D_RK3_iExp_Call *NS3D_Call_iExp = new struct_NS3D_RK3_iExp_Call[1];

	NS3D_Call_iExp->dimGrid=dimGrid;
	NS3D_Call_iExp->dimBlock=dimBlock;
	NS3D_Call_iExp->dimGrid_C=dimGrid_C;
	NS3D_Call_iExp->dimBlock_C=dimBlock_C;
	NS3D_Call_iExp->dx=dx;
	NS3D_Call_iExp->dy=dy; 
	NS3D_Call_iExp->dz=dz;
	NS3D_Call_iExp->dt=dt;
	NS3D_Call_iExp->Re=Re;
	NS3D_Call_iExp->Nx=Nx;
	NS3D_Call_iExp->Ny=Ny;
	NS3D_Call_iExp->Nz=Nz;
	NS3D_Call_iExp->Mz=Mz;
	NS3D_Call_iExp->ux_hat_d=ux_hat_d;
	NS3D_Call_iExp->uy_hat_d=uy_hat_d;
	NS3D_Call_iExp->uz_hat_d=uz_hat_d;
	NS3D_Call_iExp->vx_hat_d=vx_hat_d;
	NS3D_Call_iExp->vy_hat_d=vy_hat_d;
	NS3D_Call_iExp->vz_hat_d=vz_hat_d;
	NS3D_Call_iExp->ux_hat_d_1=ux_hat_d_1;
	NS3D_Call_iExp->uy_hat_d_1=uy_hat_d_1;
	NS3D_Call_iExp->uz_hat_d_1=uz_hat_d_1;
	NS3D_Call_iExp->vx_hat_d_1=vx_hat_d_1;
	NS3D_Call_iExp->vy_hat_d_1=vy_hat_d_1;
	NS3D_Call_iExp->vz_hat_d_1=vz_hat_d_1;
	NS3D_Call_iExp->ux_hat_d_2=ux_hat_d_2;
	NS3D_Call_iExp->uy_hat_d_2=uy_hat_d_2;
	NS3D_Call_iExp->uz_hat_d_2=uz_hat_d_2;
	NS3D_Call_iExp->ux_hat_d_3=ux_hat_d_3;
	NS3D_Call_iExp->uy_hat_d_3=uy_hat_d_3;
	NS3D_Call_iExp->uz_hat_d_3=uz_hat_d_3;
	NS3D_Call_iExp->fx_hat_d=fx_hat_d;
	NS3D_Call_iExp->fy_hat_d=fy_hat_d;
	NS3D_Call_iExp->fz_hat_d=fz_hat_d;
	NS3D_Call_iExp->Qx_hat_d=Qx_hat_d;
	NS3D_Call_iExp->Qy_hat_d=Qy_hat_d;
	NS3D_Call_iExp->Qz_hat_d=Qz_hat_d;
	NS3D_Call_iExp->div_hat_d=div_hat_d;
	NS3D_Call_iExp->kx_nabla_d=kx_nabla_d;
	NS3D_Call_iExp->ky_nabla_d=ky_nabla_d;
	NS3D_Call_iExp->kz_nabla_d=kz_nabla_d;
	NS3D_Call_iExp->din_diffusion_d=din_diffusion_d;
	NS3D_Call_iExp->din_poisson_d=din_poisson_d;
	NS3D_Call_iExp->AM_11_d=AM_11_d;
	NS3D_Call_iExp->AM_22_d=AM_22_d;
	NS3D_Call_iExp->AM_33_d=AM_33_d;
	NS3D_Call_iExp->AM_12_d=AM_12_d;
	NS3D_Call_iExp->AM_13_d=AM_13_d;
	NS3D_Call_iExp->AM_23_d=AM_23_d;
	NS3D_Call_iExp->Timesteps_period=20;//3478;

	//init BICGStab(L) properties
	NS3D_Call_iExp->shift_real=1.0005;
	NS3D_Call_iExp->BiCG_L=7;
	NS3D_Call_iExp->BiCG_tol=1.0e-9;
	NS3D_Call_iExp->BiCG_Iter=N_Arnoldi-3;
	NS3D_Call_iExp->handle=handle;



	//call Arnoldi method with linearised function
	char what[]="LR";
	int IRA_iterations=3000;
	double IRA_tol=1.0e-9;




	printf("\nArnoldi starts\n");
	if(Timesteps_period==0){
		printf("\nSkipping IRA! \n");
	}
	else if(Timesteps_period==1){
		
		//res_tol=Implicit_restart_Arnoldi_GPU_data(handle, true, N_Arnoldi-3, (user_map_vector) NSCallMatrixVector_reduced, (struct_NS3D_RK3_Call *) NS3D_Call,  vec_f_d, what, k_A, m, eigenvaluesA, IRA_tol, IRA_iterations,eigvs_real,eigvs_imag); //_exponential
		
		res_tol=Implicit_restart_Arnoldi_GPU_data_Matrix_Exponent(handle, true, N_Arnoldi-3, (user_map_vector) Axb_exponent_invert, (struct_NS3D_RK3_iExp_Call *) NS3D_Call_iExp, (user_map_vector) NSCallMatrixVector_reduced, (struct_NS3D_RK3_Call *) NS3D_Call, vec_f_d, "LR", "LM", k_A, m, eigenvaluesA, IRA_tol, IRA_iterations, eigvs_real,eigvs_imag);
	}
	else{
		what[0]='L';
		what[1]='M';
		res_tol=Implicit_restart_Arnoldi_GPU_data(handle, true, N_Arnoldi-3, (user_map_vector) NSCallMatrixVector_exponential, (struct_NS3D_RK3_Call *) NS3D_Call,  vec_f_d, what, k_A, m, eigenvaluesA, IRA_tol, IRA_iterations,eigvs_real,eigvs_imag); //_exponential		

	}

	



	printf("\nArnoldi ends\n");
	cublasDestroy(handle);

	delete [] NS3D_Call_iExp;

	//printing out eigenvectors in physical domain
	
//	if(DEBUG!=1)   //XXX! Debug in debug! Recursion.
	for(int ll=0;ll<k_A;ll++){
		int N_point=Nx*Ny*Mz;
		//velocities_from_A_vector(N_point, N_point, &eigvs_real[ll*N_Arnoldi], vx_hat_d, vy_hat_d, vz_hat_d);
		velocities_from_A_vector_reduced(N_point, N_point, &eigvs_real[ll*(N_Arnoldi-3)], vx_hat_d, vy_hat_d, vz_hat_d);

		velocity_to_double(dimGrid, dimBlock, Nx, Ny, Nz, vx_hat_d, ux_d, vy_hat_d, uy_d, vz_hat_d, uz_d);
		velocity_to_abs_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d, uy_d, uz_d, u_abs_d);
		
		host_device_real_cpy(ux, ux_d, Nx, Ny, Nz);
		host_device_real_cpy(uy, uy_d, Nx, Ny, Nz);
		host_device_real_cpy(uz, uz_d, Nx, Ny, Nz );
		host_device_real_cpy(u_abs, u_abs_d, Nx, Ny, Nz );
		char f1_name[100];
		sprintf(f1_name, "Eigenvector_%i_abs.pos",ll);	
		write_out_file_pos(f1_name, Nx, Ny, Nz, dx, dy, dz, u_abs);
		sprintf(f1_name, "Eigenvector_%i.pos",ll);
		write_out_file_vec_pos_interp(f1_name, Nx, Ny, Nz, dx, dy, dz, ux, uy, uz);	

	}

	delete [] eigenvaluesA;
	device_deallocate_all_real(2,eigvs_real, eigvs_imag);

	host_device_real_cpy(vec_f, vec_f_d, N_Arnoldi-3, 1, 1); //!!! -3 !!!
	host_device_real_cpy(vec_v, vec_v_d, N_Arnoldi-3, 1, 1); //!!! -3 !!!

	print_vector("f0.dat", N_Arnoldi, vec_f);
	print_vector("v0.dat", N_Arnoldi, vec_v);


	device_deallocate_all_real(2, vec_f_d, vec_v_d );
	device_deallocate_all_complex(3, vx_hat_d, vy_hat_d, vz_hat_d);
	device_deallocate_all_complex(3, vx_hat_d_1, vy_hat_d_1, vz_hat_d_1);
	delete [] vec_f, vec_v;

	delete [] NS3D_Call;

/*
	if(Nx<33){
		printf("Calculating Jacobian matrix...\n");
		print_Jacobian(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);
	}
//*/
//performing EIGS ends!

	printf("DEVICE->HOST...\n");
	
	velocity_to_double(dimGrid, dimBlock, Nx, Ny, Nz, ux_hat_d, ux_d, uy_hat_d, uy_d, uz_hat_d, uz_d);
	velocity_to_abs_device<<<dimGrid, dimBlock>>>(Nx,  Ny, Nz, ux_d, uy_d, uz_d, u_abs_d);
	velocity_to_abs_device<<<dimGrid, dimBlock>>>(Nx,  Ny, Nz, ux_filt_d, uy_filt_d, uz_filt_d, u_abs_filt_d);
	host_device_real_cpy(ux, ux_d, Nx, Ny, Nz);
	host_device_real_cpy(uy, uy_d, Nx, Ny, Nz);
	host_device_real_cpy(uz, uz_d, Nx, Ny, Nz );
	host_device_real_cpy(u_abs, u_abs_d, Nx, Ny, Nz );
	host_device_real_cpy(u_abs_filt, u_abs_filt_d, Nx, Ny, Nz );
	host_device_real_cpy(div_pos, div_pos_d, Nx, Ny, Nz );
	host_device_real_cpy(mask_2_3, mask_2_3_d, Nx, Ny, Mz );


	all_Fourier2double(dimGrid_C, dimBlock_C, ux_hat_d, ux_Re_d,ux_Im_d, uy_hat_d, uy_Re_d, uy_Im_d, uz_hat_d, uz_Re_d, uz_Im_d, Nx, Ny, Mz);
//copy Fourier modes to device
	host_device_real_cpy(ux_Re, ux_Re_d, Nx, Ny, Mz);	
	host_device_real_cpy(ux_Im, ux_Im_d, Nx, Ny, Mz);	
	host_device_real_cpy(uy_Re, uy_Re_d, Nx, Ny, Mz);	
	host_device_real_cpy(uy_Im, uy_Im_d, Nx, Ny, Mz);	
	host_device_real_cpy(uz_Re, uz_Re_d, Nx, Ny, Mz);	
	host_device_real_cpy(uz_Im, uz_Im_d, Nx, Ny, Mz); 


	host_device_real_cpy(AM_11, AM_11_d, Nx, Ny, Mz );
	host_device_real_cpy(AM_22, AM_22_d, Nx, Ny, Mz );
	host_device_real_cpy(AM_33, AM_33_d, Nx, Ny, Mz );
	host_device_real_cpy(AM_12, AM_12_d, Nx, Ny, Mz );
	host_device_real_cpy(AM_13, AM_13_d, Nx, Ny, Mz );
	host_device_real_cpy(AM_23, AM_23_d, Nx, Ny, Mz );

	printf("done\n");
	
	//remove all CUDA device memory	
	printf("cleaning up DEVICE memory...\n");
	//clean dealiasing and WENO
	clean_dealiasing();
	clean_WENO();
	//clean cufft
	result = cufftDestroy(planR2C);
	if (result != CUFFT_SUCCESS) { printf ("*CUFFT DestrotPlanR2C failed\n"); return; }
	result = cufftDestroy(planC2R);
	if (result != CUFFT_SUCCESS) { printf ("*CUFFT DestrotPlanC2R failed\n"); return; }
	//clean all other arrays
	device_deallocate_all_real(15, din_poisson_d, din_diffusion_d, ux_d, uy_d, uz_d, div_pos_d, fx_d, fy_d, fz_d,kx_nabla_d,ky_nabla_d,kz_nabla_d,u_abs_d,cfl_out,cfl_in);	
	
	//for WENO advection
	device_deallocate_all_real(6, ux1_d, uy1_d, uz1_d, ux2_d, uy2_d, uz2_d);

	device_deallocate_all_complex(7,  ux_hat_d, uy_hat_d, uz_hat_d, div_hat_d, fx_hat_d, fy_hat_d, fz_hat_d);
	device_deallocate_all_complex(3, Qx_hat_d, Qy_hat_d, Qz_hat_d);

	device_deallocate_all_complex(9, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1, ux_hat_d_2, uy_hat_d_2, uz_hat_d_2, ux_hat_d_3, uy_hat_d_3, uz_hat_d_3);

//for high wavenumber analysis
	device_deallocate_all_complex(4, ux_red_hat_d, uy_red_hat_d, uz_red_hat_d, u_temp_complex_d);
	device_deallocate_all_real(3, ux_red_d, uy_red_d, uz_red_d);

//LES filtered velocities
	device_deallocate_all_complex(3, ux_filt_hat_d, uy_filt_hat_d, uz_filt_hat_d);
	device_deallocate_all_real(4, ux_filt_d, uy_filt_d, uz_filt_d, u_abs_filt_d);


//kinetic energy and dissipation

	device_deallocate_all_real(4, energy_d,energy_out1_d,energy_out2_d,dissipation_d);

//rotor components
	device_deallocate_all_real(3, rot_x_d, rot_y_d, rot_z_d);
	
//projection matrix
	device_deallocate_all_real(6, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);
	device_deallocate_all_real(6, ux_Re_d, ux_Im_d, uy_Re_d, uy_Im_d, uz_Re_d, uz_Im_d);
	device_deallocate_all_real(1, mask_2_3_d);

	printf("done\n");
	

	
	
	printf("Total energy:%e\n",TotalEnergy(Nx, Ny, Nz, ux, uy, uz, dx, dy, dz, alpha, beta));
	printf("Total Dissipation:%e\n",TotalDissipation(Nx, Ny, Nz, ux, uy, uz, dx, dy, dz,alpha, beta, Re));
	
	printf("calculating and wrighting energy spectra...\n");	
	calculate_energy_spectrum("Ek.dat", Nx, Ny, Mz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);


	printf("wrighting file...\n");
	if(DEBUG!=1){
		//control files
		write_control_file(Nx, Ny, Nz, ux, uy, uz);
		write_control_file_fourier(Nx, Ny, Mz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);
		write_line_specter(Nx, Ny, Mz, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);
		//visualization files

		write_res_files(ux, uy, uz, div_pos, u_abs, Nx, Ny, Nz, dx, dy, dz);
		
		write_out_file_pos("mask_2_3.pos", Nx, Ny, Mz, dx, dy, dz, mask_2_3, 1);

		write_out_file_pos("AM_11.pos", Nx, Ny, Mz, dx, dy, dz, AM_11, 1);
		write_out_file_pos("AM_22.pos", Nx, Ny, Mz, dx, dy, dz, AM_22, 1);
		write_out_file_pos("AM_33.pos", Nx, Ny, Mz, dx, dy, dz, AM_33, 1);
		write_out_file_pos("AM_12.pos", Nx, Ny, Mz, dx, dy, dz, AM_12, 1);
		write_out_file_pos("AM_13.pos", Nx, Ny, Mz, dx, dy, dz, AM_13, 1);
		write_out_file_pos("AM_23.pos", Nx, Ny, Mz, dx, dy, dz, AM_23, 1);

		write_out_file_pos("Uabs_filt.pos", Nx, Ny, Nz, dx, dy, dz, u_abs_filt, 2);

	//solution for Shapiro test case
	//	Initial_Shapiro(Nx, Ny, Nz, dx, dy, dz, ux, uy, uz, uxABC, uyABC, uzABC, Re, TotalTime);
	//	write_out_file_vec_pos_interp("p_outShapiroVec.pos", Nx, Ny, Nz, dx, dy, dz, uxABC, uyABC, uzABC);
	}
	



	printf("done\n");
	printf("cleaning up HOST memory...\n");	
	deallocate_real(16, din_poisson, din_diffusion, kx_laplace, kx_nabla, ky_laplace, ky_nabla, kz_laplace, kz_nabla, ux, uy, uz, div_pos, fx, fy, fz, u_abs);
//projection matrix
	deallocate_real(6, AM_11, AM_22, AM_33, AM_12, AM_13, AM_23);
	deallocate_real(6, ux_Re, ux_Im, uy_Re, uy_Im, uz_Re, uz_Im);
	free(mask_2_3);
	deallocate_real(4, ux_filt, uy_filt, uz_filt, u_abs_filt);

	free(uxABC); free(uyABC); free(uzABC);
	printf("done\n");	
	printf("=============all done=============\n");
	return 0;
}



