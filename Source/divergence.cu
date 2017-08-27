#include "divergence.h"





/*3D divergence in Fourier space----------------------------------*/
__global__ void solve_divergence(cudaComplex *div_hat_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, int Nx, int Ny, int Nz, real dt) {
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
		
		div_hat_d[IN(j,k,l)].x=(-ux_hat_d[IN(j,k,l)].y*kx_nabla_d[j]-uy_hat_d[IN(j,k,l)].y*ky_nabla_d[k]-uz_hat_d[IN(j,k,l)].y*kz_nabla_d[l])/dt;
		
		div_hat_d[IN(j,k,l)].y=(ux_hat_d[IN(j,k,l)].x*kx_nabla_d[j]+uy_hat_d[IN(j,k,l)].x*ky_nabla_d[k]+uz_hat_d[IN(j,k,l)].x*kz_nabla_d[l])/dt;
	}
}
}






void divergence_device(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, cudaComplex *div_hat_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* kx_nabla_d, real* ky_nabla_d, real* kz_nabla_d, real dt, real Re){
	
	solve_divergence<<<dimGrid_C, dimBlock_C>>>(div_hat_d, ux_hat_d, uy_hat_d, uz_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, Nx, Ny, Nz, dt);

}

void devergence_to_double(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *div_hat_d, real* div_d){

	iFFTN_Device(dimGrid, dimBlock, div_hat_d, div_d, Nx, Ny, Nz);

}



