#include "advection_2_3.h"


__global__ void set_to_zero_device(int Nx, int Ny, int Nz, cudaComplex *a1, cudaComplex *a2, cudaComplex *a3){
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
		a1[IN(j,k,l)].x=0.0;
		a1[IN(j,k,l)].y=0.0;
		
		a2[IN(j,k,l)].x=0.0;
		a2[IN(j,k,l)].y=0.0;

		a3[IN(j,k,l)].x=0.0;
		a3[IN(j,k,l)].y=0.0;

	}
}
}



__global__ void set_to_zero_real(int Nx, int Ny, int Nz, cudaComplex *a1, cudaComplex *a2, cudaComplex *a3){
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
		a1[IN(j,k,l)].x=0.0;
		//a1[IN(j,k,l)].y=0.0;
		
		a2[IN(j,k,l)].x=0.0;
		//a2[IN(j,k,l)].y=0.0;

		a3[IN(j,k,l)].x=0.0;
		//a3[IN(j,k,l)].y=0.0;

		a1[IN(0,0,0)].x=0.0;
		a1[IN(0,0,0)].y=0.0;
		a2[IN(0,0,0)].x=0.0;
		a2[IN(0,0,0)].y=0.0;
		a3[IN(0,0,0)].x=0.0;
		a3[IN(0,0,0)].y=0.0;


	}
}
}



__global__ void set_to_zero_device(int Nx, int Ny, int Nz, real *a1, real *a2, real *a3){
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
		a1[IN(j,k,l)]=0.0;
		a2[IN(j,k,l)]=0.0;
		a3[IN(j,k,l)]=0.0;


	}
}
}



__global__ void set_zero_to_zero_device(int Nx, int Ny, int Nz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){
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
//	if((j<Nx)&&(k<Ny)&&(l<Nz)){
	
		Qx_hat_d[IN(0,0,0)].x=0.0;
		Qx_hat_d[IN(0,0,0)].y=0.0;
		Qy_hat_d[IN(0,0,0)].x=0.0;
		Qy_hat_d[IN(0,0,0)].y=0.0;
		Qz_hat_d[IN(0,0,0)].x=0.0;
		Qz_hat_d[IN(0,0,0)].y=0.0;


//	}
}
}





void set_to_zero(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *a1, cudaComplex *a2, cudaComplex *a3){

	set_to_zero_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, a1, a2, a3);
}


void set_to_zero(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real *a1, real *a2, real *a3){

	set_to_zero_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, a1, a2, a3);
}


__global__ void build_vels_device(int Nx, int Ny, int Nz, cudaComplex *source1, cudaComplex *source2, cudaComplex *source3, cudaComplex *destination1, cudaComplex *destination2, cudaComplex *destination3){
unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
unsigned int j_ex,k_ex,l_ex;

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



__global__ void build_ders_device(int Nx, int Ny, int Nz, cudaComplex *source_hat, cudaComplex *der_ext_x, cudaComplex *der_ext_y, cudaComplex *der_ext_z, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d){
unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
unsigned int j_ex,k_ex,l_ex;

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

		der_ext_x[IN(j,k,l)].x=-source_hat[IN(j,k,l)].y*kx_nabla_d[j];
		der_ext_x[IN(j,k,l)].y=source_hat[IN(j,k,l)].x*kx_nabla_d[j];
		
		der_ext_y[IN(j,k,l)].x=-source_hat[IN(j,k,l)].y*ky_nabla_d[k];
		der_ext_y[IN(j,k,l)].y=source_hat[IN(j,k,l)].x*ky_nabla_d[k];

		der_ext_z[IN(j,k,l)].x=-source_hat[IN(j,k,l)].y*kz_nabla_d[l];
		der_ext_z[IN(j,k,l)].y=source_hat[IN(j,k,l)].x*kz_nabla_d[l];

	}
}
}






__global__ void calc_vels_nabla_vels(int Nx, int Ny, int Nz, real *ux_d_ext, real *uy_d_ext, real *uz_d_ext, real *der_x_d_ext, real *der_y_d_ext, real *der_z_d_ext, real *Qx_d_ext){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
unsigned int j_ex,k_ex,l_ex;

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


		Qx_d_ext[IN(j,k,l)]=(ux_d_ext[IN(j,k,l)]*der_x_d_ext[IN(j,k,l)]+uy_d_ext[IN(j,k,l)]*der_y_d_ext[IN(j,k,l)]+uz_d_ext[IN(j,k,l)]*der_z_d_ext[IN(j,k,l)]);
		


	}
}
}


__global__ void calc_vels_nabla_vels_plus(int Nx, int Ny, int Nz, real *ux_d_ext, real *uy_d_ext, real *uz_d_ext, real *der_x_d_ext, real *der_y_d_ext, real *der_z_d_ext, real *Qx_d_ext){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
unsigned int j_ex,k_ex,l_ex;

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


		Qx_d_ext[IN(j,k,l)]+=(ux_d_ext[IN(j,k,l)]*der_x_d_ext[IN(j,k,l)]+uy_d_ext[IN(j,k,l)]*der_y_d_ext[IN(j,k,l)]+uz_d_ext[IN(j,k,l)]*der_z_d_ext[IN(j,k,l)]);
		


	}
}
}


__global__ void filter_1p3_wavenumbers_device(int Nx, int Ny, int Nz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *mask_2_3_d_ext){
unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
unsigned int j_ex,k_ex,l_ex;


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
int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){
		
		Qx_hat_d[IN(j,k,l)].x*=mask_2_3_d_ext[IN(j,k,l)];
		Qx_hat_d[IN(j,k,l)].y*=mask_2_3_d_ext[IN(j,k,l)];
		
		Qy_hat_d[IN(j,k,l)].x*=mask_2_3_d_ext[IN(j,k,l)];
		Qy_hat_d[IN(j,k,l)].y*=mask_2_3_d_ext[IN(j,k,l)];	
		
		Qz_hat_d[IN(j,k,l)].x*=mask_2_3_d_ext[IN(j,k,l)];
		Qz_hat_d[IN(j,k,l)].y*=mask_2_3_d_ext[IN(j,k,l)];
		
	}

}
}



void build_vels_and_ders(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d){
	
	build_vels_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz,  ux_hat_d, uy_hat_d, uz_hat_d, ux_hat_d_ext, uy_hat_d_ext, uz_hat_d_ext);

	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, ux_hat_d, ux_x_hat_d_ext, ux_y_hat_d_ext, ux_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);
	
	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, uy_hat_d, uy_x_hat_d_ext, uy_y_hat_d_ext, uy_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);
	
	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, uz_hat_d, uz_x_hat_d_ext, uz_y_hat_d_ext, uz_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);

}



void build_vels_and_ders_V(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d){
	
	build_vels_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz,  ux_hat_d, uy_hat_d, uz_hat_d, vx_hat_d_ext, vy_hat_d_ext, vz_hat_d_ext);

	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, ux_hat_d, vx_x_hat_d_ext, vx_y_hat_d_ext, vx_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);
	
	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, uy_hat_d, vy_x_hat_d_ext, vy_y_hat_d_ext, vy_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);

	build_ders_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, uz_hat_d, vz_x_hat_d_ext, vz_y_hat_d_ext, vz_z_hat_d_ext, kx_nabla_d, ky_nabla_d, kz_nabla_d);	

}



void filter_wavenumbers(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){
	
	// reduced hat

	filter_1p3_wavenumbers_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Nz, Qx_hat_d, Qy_hat_d, Qz_hat_d, mask_2_3_d_ext);


}



void convolute_2p3(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){
	//go to Domain

	iFFTN_Device(dimGrid, dimBlock, ux_hat_d_ext, ux_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_hat_d_ext, uy_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uz_hat_d_ext, uz_d_ext, Nx, Ny, Nz);


	iFFTN_Device(dimGrid, dimBlock, ux_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, ux_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, ux_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);	
	
	//in physical space
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qx_d_ext);

	
	iFFTN_Device(dimGrid, dimBlock, uy_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uy_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);

	//in physical space
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qy_d_ext);


	iFFTN_Device(dimGrid, dimBlock, uz_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uz_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uz_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);

	//in physical space
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qz_d_ext);


	//return to Image
	FFTN_Device(Qx_d_ext, Qx_hat_d);
	FFTN_Device(Qy_d_ext, Qy_hat_d);
	FFTN_Device(Qz_d_ext, Qz_hat_d);


}



void convolute_2p3_UV(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){
	//go to Domain

	iFFTN_Device(dimGrid, dimBlock, ux_hat_d_ext, ux_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_hat_d_ext, uy_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uz_hat_d_ext, uz_d_ext, Nx, Ny, Nz);

	iFFTN_Device(dimGrid, dimBlock, vx_hat_d_ext, vx_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, vy_hat_d_ext, vy_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, vz_hat_d_ext, vz_d_ext, Nx, Ny, Nz);



	iFFTN_Device(dimGrid, dimBlock, ux_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, ux_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, ux_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);	

	//(v,\nabla)u_x)
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, vx_d_ext, vy_d_ext, vz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qx_d_ext);

	iFFTN_Device(dimGrid, dimBlock, vx_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, vx_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, vx_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);
	
	//(u,\nabla)v_x)
	calc_vels_nabla_vels_plus<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qx_d_ext);







	
	iFFTN_Device(dimGrid, dimBlock, uy_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uy_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);	

	//(v,\nabla)u_y)
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, vx_d_ext, vy_d_ext, vz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qy_d_ext);

	iFFTN_Device(dimGrid, dimBlock, vy_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, vy_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, vy_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);
	
	//(u,\nabla)v_y)
	calc_vels_nabla_vels_plus<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qy_d_ext);





	
	iFFTN_Device(dimGrid, dimBlock, uz_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uz_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uz_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);	

	//(v,\nabla)u_z)
	calc_vels_nabla_vels<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, vx_d_ext, vy_d_ext, vz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qz_d_ext);

	iFFTN_Device(dimGrid, dimBlock, vz_x_hat_d_ext, der_x_d_ext, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, vz_y_hat_d_ext, der_y_d_ext, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, vz_z_hat_d_ext, der_z_d_ext, Nx, Ny, Nz);
	
	//(u,\nabla)v_z)
	calc_vels_nabla_vels_plus<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext, der_x_d_ext, der_y_d_ext, der_z_d_ext, Qz_d_ext);


	//return to Image
	FFTN_Device(Qx_d_ext, Qx_hat_d);
	FFTN_Device(Qy_d_ext, Qy_hat_d);
	FFTN_Device(Qz_d_ext, Qz_hat_d);


}




void calculate_convolution_2p3(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){


	build_vels_and_ders(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d);

	//reduce velocity and derivatives wavenumbers 
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_hat_d_ext, uy_hat_d_ext, uz_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_x_hat_d_ext, ux_y_hat_d_ext, ux_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, uy_x_hat_d_ext, uy_y_hat_d_ext, uy_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, uz_x_hat_d_ext, uz_y_hat_d_ext, uz_z_hat_d_ext);

	convolute_2p3(dimGrid, dimBlock, Nx, Ny, Nz, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		
	set_to_zero_real<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

	//set_zero_to_zero_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

}


void calculate_convolution_2p3_UV(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){


	build_vels_and_ders(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d);

	build_vels_and_ders_V(dimGrid_C, dimBlock_C, Nx, Ny, Mz, vx_hat_d, vy_hat_d, vz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d);

	//reduce velocity and derivatives wavenumbers 
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_hat_d_ext, uy_hat_d_ext, uz_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_x_hat_d_ext, ux_y_hat_d_ext, ux_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, uy_x_hat_d_ext, uy_y_hat_d_ext, uy_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, uz_x_hat_d_ext, uz_y_hat_d_ext, uz_z_hat_d_ext);

//	reduce velocity and derivatives wavenumbers for V
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, vx_hat_d_ext, vy_hat_d_ext, vz_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, vx_x_hat_d_ext, vx_y_hat_d_ext, vx_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, vy_x_hat_d_ext, vy_y_hat_d_ext, vy_z_hat_d_ext);
	filter_wavenumbers(dimGrid_C, dimBlock_C, Nx, Ny, Mz, vz_x_hat_d_ext, vz_y_hat_d_ext, vz_z_hat_d_ext);

	convolute_2p3_UV(dimGrid, dimBlock, Nx, Ny, Nz, Qx_hat_d, Qy_hat_d, Qz_hat_d);
		
	set_to_zero_real<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

	//set_zero_to_zero_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);

}



void init_dealiasing(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, real *mask_2_3_d){
	

	mask_2_3_d_ext=mask_2_3_d;

	device_allocate_all_complex(Nx, Ny, Mz, 3, &ux_hat_d_ext, &uy_hat_d_ext, &uz_hat_d_ext);

	device_allocate_all_complex(Nx, Ny, Mz, 3, &ux_x_hat_d_ext, &ux_y_hat_d_ext, &ux_z_hat_d_ext);
	device_allocate_all_complex(Nx, Ny, Mz, 3, &uy_x_hat_d_ext, &uy_y_hat_d_ext, &uy_z_hat_d_ext);
	device_allocate_all_complex(Nx, Ny, Mz, 3, &uz_x_hat_d_ext, &uz_y_hat_d_ext, &uz_z_hat_d_ext);

	device_allocate_all_real(Nx, Ny, Nz, 3, &ux_d_ext, &uy_d_ext, &uz_d_ext);

//	for B(u,v)
	device_allocate_all_complex(Nx, Ny, Mz, 3, &vx_hat_d_ext, &vy_hat_d_ext, &vz_hat_d_ext);

	device_allocate_all_complex(Nx, Ny, Mz, 3, &vx_x_hat_d_ext, &vx_y_hat_d_ext, &vx_z_hat_d_ext);
	device_allocate_all_complex(Nx, Ny, Mz, 3, &vy_x_hat_d_ext, &vy_y_hat_d_ext, &vy_z_hat_d_ext);
	device_allocate_all_complex(Nx, Ny, Mz, 3, &vz_x_hat_d_ext, &vz_y_hat_d_ext, &vz_z_hat_d_ext);

	device_allocate_all_real(Nx, Ny, Nz, 3, &vx_d_ext, &vy_d_ext, &vz_d_ext);


	device_allocate_all_real(Nx, Ny, Nz, 3, &der_x_d_ext, &der_y_d_ext, &der_z_d_ext);
	
	device_allocate_all_real(Nx, Ny, Nz, 3, &Qx_d_ext, &Qy_d_ext, &Qz_d_ext);

	set_to_zero(dimGrid_C, dimBlock_C, Nx, Ny, Mz, ux_hat_d_ext, uy_hat_d_ext, uz_hat_d_ext);
	set_to_zero(dimGrid, dimBlock, Nx, Ny, Nz, ux_d_ext, uy_d_ext, uz_d_ext);
	set_to_zero(dimGrid, dimBlock, Nx, Ny, Nz, der_x_d_ext, der_y_d_ext, der_z_d_ext);
 

}

void clean_dealiasing(){
	
	device_deallocate_all_complex(3, ux_hat_d_ext, uy_hat_d_ext, uz_hat_d_ext);
	device_deallocate_all_complex(3, ux_x_hat_d_ext, ux_y_hat_d_ext, ux_z_hat_d_ext);
	device_deallocate_all_complex(3, uy_x_hat_d_ext, uy_y_hat_d_ext, uy_z_hat_d_ext);
	device_deallocate_all_complex(3, uz_x_hat_d_ext, uz_y_hat_d_ext, uz_z_hat_d_ext);

	device_deallocate_all_real(3, ux_d_ext, uy_d_ext, uz_d_ext);

	device_deallocate_all_complex(3, vx_hat_d_ext, vy_hat_d_ext, vz_hat_d_ext);
	device_deallocate_all_complex(3, vx_x_hat_d_ext, vx_y_hat_d_ext, vx_z_hat_d_ext);
	device_deallocate_all_complex(3, vy_x_hat_d_ext, vy_y_hat_d_ext, vy_z_hat_d_ext);
	device_deallocate_all_complex(3, vz_x_hat_d_ext, vz_y_hat_d_ext, vz_z_hat_d_ext);

	device_deallocate_all_real(3, vx_d_ext, vy_d_ext, vz_d_ext);


	device_deallocate_all_real(3, der_x_d_ext, der_y_d_ext, der_z_d_ext);
	
	device_deallocate_all_real(3, Qx_d_ext, Qy_d_ext, Qz_d_ext);


}
