#include "RK_time_step.h"




__global__ void copy_arrays_device(int Nx, int Ny, int Nz,  cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/ Ny; 
yIndex = t1 - Ny * xIndex ;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){

		destination1[IN(j,k,l)].x=source1[IN(j,k,l)].x;
		destination1[IN(j,k,l)].y=source1[IN(j,k,l)].y;

		destination2[IN(j,k,l)].x=source2[IN(j,k,l)].x;
		destination2[IN(j,k,l)].y=source2[IN(j,k,l)].y;

		destination3[IN(j,k,l)].x=source3[IN(j,k,l)].x;
		destination3[IN(j,k,l)].y=source3[IN(j,k,l)].y;


	}

}


}




__global__ void scale_RHS_device(int Nx, int Ny, int Nz, int size, cudaComplex *source1, cudaComplex *source2, cudaComplex *source3){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/ Ny; 
yIndex = t1 - Ny * xIndex ;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){

		source1[IN(j,k,l)].x=source1[IN(j,k,l)].x/size;
		source1[IN(j,k,l)].y=source1[IN(j,k,l)].y/size;

		source2[IN(j,k,l)].x=source2[IN(j,k,l)].x/size;
		source2[IN(j,k,l)].y=source2[IN(j,k,l)].y/size;

		source3[IN(j,k,l)].x=source3[IN(j,k,l)].x/size;
		source3[IN(j,k,l)].y=source3[IN(j,k,l)].y/size;


	}

}


}




__global__ void single_RK_step_device(int Nx, int Ny, int Nz,  cudaComplex *source0_1, cudaComplex *source0_2, cudaComplex *source0_3, real wight_0, cudaComplex *source1_1, cudaComplex *source1_2, cudaComplex *source1_3, real wight_1, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/ Ny; 
yIndex = t1 - Ny * xIndex ;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){

		destination1[IN(j,k,l)].x=wight_0*source0_1[IN(j,k,l)].x+wight_1*source1_1[IN(j,k,l)].x;
		destination1[IN(j,k,l)].y=wight_0*source0_1[IN(j,k,l)].y+wight_1*source1_1[IN(j,k,l)].y;

		destination2[IN(j,k,l)].x=wight_0*source0_2[IN(j,k,l)].x+wight_1*source1_2[IN(j,k,l)].x;
		destination2[IN(j,k,l)].y=wight_0*source0_2[IN(j,k,l)].y+wight_1*source1_2[IN(j,k,l)].y;

		destination3[IN(j,k,l)].x=wight_0*source0_3[IN(j,k,l)].x+wight_1*source1_3[IN(j,k,l)].x;
		destination3[IN(j,k,l)].y=wight_0*source0_3[IN(j,k,l)].y+wight_1*source1_3[IN(j,k,l)].y;


	}

}


}


__global__ void single_RK_step_three_device(int Nx, int Ny, int Nz,  cudaComplex *source0_1, cudaComplex *source0_2, cudaComplex *source0_3, real wight_0, cudaComplex *source1_1, cudaComplex *source1_2, cudaComplex *source1_3, real wight_1, cudaComplex *source2_1, cudaComplex *source2_2, cudaComplex *source2_3, real wight_2, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/ Ny; 
yIndex = t1 - Ny * xIndex ;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){

		destination1[IN(j,k,l)].x=wight_0*source0_1[IN(j,k,l)].x+wight_1*source1_1[IN(j,k,l)].x+wight_2*source2_1[IN(j,k,l)].x;
		destination1[IN(j,k,l)].y=wight_0*source0_1[IN(j,k,l)].y+wight_1*source1_1[IN(j,k,l)].y+wight_2*source2_1[IN(j,k,l)].y;

		destination2[IN(j,k,l)].x=wight_0*source0_2[IN(j,k,l)].x+wight_1*source1_2[IN(j,k,l)].x+wight_2*source2_2[IN(j,k,l)].x;
		destination2[IN(j,k,l)].y=wight_0*source0_2[IN(j,k,l)].y+wight_1*source1_2[IN(j,k,l)].y+wight_2*source2_2[IN(j,k,l)].y;

		destination3[IN(j,k,l)].x=wight_0*source0_3[IN(j,k,l)].x+wight_1*source1_3[IN(j,k,l)].x+wight_2*source2_3[IN(j,k,l)].x;
		destination3[IN(j,k,l)].y=wight_0*source0_3[IN(j,k,l)].y+wight_1*source1_3[IN(j,k,l)].y+wight_2*source2_3[IN(j,k,l)].y;


	}

}


}

__global__ void advection_device(int Nx, int Ny, int Nz, real dt, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *ux_hat_d_back, cudaComplex *uy_hat_d_back, cudaComplex *uz_hat_d_back){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/ Ny; 
yIndex = t1 - Ny * xIndex ;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){

		ux_hat_d_back[IN(j,k,l)].x=ux_hat_d[IN(j,k,l)].x-dt*Qx_hat_d[IN(j,k,l)].x;
		ux_hat_d_back[IN(j,k,l)].y=ux_hat_d[IN(j,k,l)].y-dt*Qx_hat_d[IN(j,k,l)].y;

		uy_hat_d_back[IN(j,k,l)].x=uy_hat_d[IN(j,k,l)].x-dt*Qy_hat_d[IN(j,k,l)].x;
		uy_hat_d_back[IN(j,k,l)].y=uy_hat_d[IN(j,k,l)].y-dt*Qy_hat_d[IN(j,k,l)].y;

		uz_hat_d_back[IN(j,k,l)].x=uz_hat_d[IN(j,k,l)].x-dt*Qz_hat_d[IN(j,k,l)].x;
		uz_hat_d_back[IN(j,k,l)].y=uz_hat_d[IN(j,k,l)].y-dt*Qz_hat_d[IN(j,k,l)].y;

	}

}


}




void copy_arrays(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz,  cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

	copy_arrays_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, source1, source2, source3, destination1, destination2, destination3);

}

void scale_RHS(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int size, cudaComplex *source1, cudaComplex *source2, cudaComplex *source3){

	scale_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, size, source1, source2, source3);


}



void single_RK_step(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz,  cudaComplex *source0_1, cudaComplex *source0_2, cudaComplex *source0_3, real wight_0, cudaComplex *source1_1, cudaComplex *source1_2, cudaComplex *source1_3, real wight_1, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

	single_RK_step_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz,  source0_1, source0_2, source0_3, wight_0, source1_1, source1_2, source1_3, wight_1, destination1, destination2, destination3);

}

void single_RK_step_three(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz,  cudaComplex *source0_1, cudaComplex *source0_2, cudaComplex *source0_3, real wight_0, cudaComplex *source1_1, cudaComplex *source1_2, cudaComplex *source1_3, real wight_1, cudaComplex *source2_1, cudaComplex *source2_2, cudaComplex *source2_3, real wight_2, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){

	single_RK_step_three_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz,  source0_1, source0_2, source0_3, wight_0, source1_1, source1_2, source1_3, wight_1, source2_1, source2_2, source2_3, wight_2, destination1, destination2, destination3);

}

void advection_step(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real dt, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *ux_hat_d_back, cudaComplex *uy_hat_d_back, cudaComplex *uz_hat_d_back){


	advection_device<<<dimGrid_C, dimBlock_C>>>(Nx,  Ny, Nz, dt, ux_hat_d, uy_hat_d, uz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, ux_hat_d_back, uy_hat_d_back, uz_hat_d_back);




}





void return_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d){
	

		
	
		calculate_convolution_2p3(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		
		//WenoAdvection(dimBlock, dimGrid, dimBlock_C, dimGrid_C, Nx, Ny, Nz, Mz, 5, ux_hat_d, uy_hat_d, uz_hat_d, dx, dy, dz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

		RHS_advection_diffusion_projection(dimGrid_C, dimBlock_C, Nx, Ny, Mz, din_diffusion_d, ux_hat_d, uy_hat_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Re, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d, RHSx_hat_d, RHSy_hat_d, RHSz_hat_d);

		//scale down fourier components
		//scale_RHS(dimGrid_C, dimBlock_C, Nx, Ny, Mz, (Nx*Ny*Nz), RHSx_hat_d, RHSy_hat_d, RHSz_hat_d);

}


void single_time_step(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, cudaComplex *ux_hat_d_back, cudaComplex *uy_hat_d_back, cudaComplex *uz_hat_d_back, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

		


	
		calculate_convolution_2p3(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		
		//WenoAdvection(dimBlock, dimGrid, dimBlock_C, dimGrid_C, Nx, Ny, Nz, Mz, 5, ux_hat_d, uy_hat_d, uz_hat_d, dx, dy, dz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

		solve_advection_diffusion_projection(dimGrid_C, dimBlock_C, Nx, Ny, Mz, din_diffusion_d, ux_hat_d, uy_hat_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	

}

void single_time_step_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

		

		calculate_convolution_2p3_UV(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d, vy_hat_d, vz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		
		//WenoAdvection(dimBlock, dimGrid, dimBlock_C, dimGrid_C, Nx, Ny, Nz, Mz, 5, ux_hat_d, uy_hat_d, uz_hat_d, dx, dy, dz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

		solve_advection_diffusion_projection_UV(dimGrid_C, dimBlock_C, Nx, Ny, Mz, din_diffusion_d, vx_hat_d, vy_hat_d, vz_hat_d, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	

}



void single_time_step_UV_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

		
		calculate_convolution_2p3_UV(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d, vy_hat_d, vz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		//WenoAdvection(dimBlock, dimGrid, dimBlock_C, dimGrid_C, Nx, Ny, Nz, Mz, 5, ux_hat_d, uy_hat_d, uz_hat_d, dx, dy, dz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

		solve_advection_diffusion_projection_UV_RHS(dimGrid_C, dimBlock_C, Nx, Ny, Mz, din_diffusion_d, vx_hat_d, vy_hat_d, vz_hat_d, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);


}



void single_time_step_iUV_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

		
		calculate_convolution_2p3_UV(dimGrid, dimBlock, dimGrid_C, dimBlock_C, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d, vy_hat_d, vz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		//WenoAdvection(dimBlock, dimGrid, dimBlock_C, dimGrid_C, Nx, Ny, Nz, Mz, 5, ux_hat_d, uy_hat_d, uz_hat_d, dx, dy, dz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

		solve_advection_implicit_diffusion_projection_UV(dimGrid_C, dimBlock_C, Nx, Ny, Mz, din_diffusion_d, vx_hat_d, vy_hat_d, vz_hat_d, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);


}



void RK3_SSP(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){

	copy_arrays(dimGrid_C, dimBlock_C, Nx, Ny, Mz,  ux_hat_d, uy_hat_d, uz_hat_d, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1);


	single_time_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d,  kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, ux_hat_d_3, uy_hat_d_3, uz_hat_d_3, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	single_time_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d,  kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, ux_hat_d_3, uy_hat_d_3, uz_hat_d_3, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	single_RK_step(dimGrid_C, dimBlock_C, Nx, Ny, Mz,  ux_hat_d, uy_hat_d, uz_hat_d, 3.0/4.0, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1, 1.0/4.0, ux_hat_d_2, uy_hat_d_2, uz_hat_d_2);


	single_time_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d,  kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, ux_hat_d_3, uy_hat_d_3, uz_hat_d_3, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	single_RK_step(dimGrid_C, dimBlock_C, Nx, Ny, Mz,  ux_hat_d, uy_hat_d, uz_hat_d, 1.0/3.0, ux_hat_d_2, uy_hat_d_2, uz_hat_d_2, 2.0/3.0, ux_hat_d, uy_hat_d, uz_hat_d);
	

}


void RK3_SSP_UV_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){


	//RK3_SSP(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);

	single_time_step_UV_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d, vy_hat_d, vz_hat_d,  Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);
	

}



void RK3_SSP_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d,  cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1, cudaComplex *vx_hat_d_1, cudaComplex *vy_hat_d_1, cudaComplex *vz_hat_d_1, cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){


	copy_arrays(dimGrid_C, dimBlock_C, Nx, Ny, Mz,  vx_hat_d, vy_hat_d, vz_hat_d, vx_hat_d_1, vy_hat_d_1, vz_hat_d_1);

	single_time_step_UV(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d_1, vy_hat_d_1, vz_hat_d_1,  Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	RK3_SSP(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);

	single_time_step_UV(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d_1, vy_hat_d_1, vz_hat_d_1,  Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	single_RK_step(dimGrid_C, dimBlock_C, Nx, Ny, Mz,  vx_hat_d, vy_hat_d, vz_hat_d, 1.0/2.0, vx_hat_d_1, vy_hat_d_1, vz_hat_d_1, 1.0/2.0, vx_hat_d, vy_hat_d, vz_hat_d);

}