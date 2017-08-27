#include "Shapiro_test.h"






__global__ void Shapiro_test_case_device(real dx, real dy, real dz, real current_time, real Re, real *ux, real *uy, real *uz, int Nx, int Ny, int Nz, real *cfl_in) {
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

		real y=(k)*dy;
		real x=(j)*dx;		
		real z=(l)*dz;
		real factor=sqrt(3.0);
		real err_ux=ux[IN(j,k,l)]-(-0.5*(factor*cos(x)*sin(y)*sin(z)+sin(x)*cos(y)*cos(z) )*exp(-(factor*factor)*current_time/Re));
		real err_uy=uy[IN(j,k,l)]-(0.5*(factor*sin(x)*cos(y)*sin(z)-cos(x)*sin(y)*cos(z) )*exp(-(factor*factor)*current_time/Re));
		real err_uz=uz[IN(j,k,l)]-(cos(x)*cos(y)*sin(z)*exp(-(factor*factor)*current_time/Re));

		cfl_in[IN(j,k,l)]=sqrt(err_ux*err_ux+err_uy*err_uy+err_uz*err_uz);

	}
}
}




real Shapiro_test_case(dim3 dimGrid, dim3 dimBlock, real dx, real dy, real dz, real current_time, real Re, real *ux, real *uy, real *uz, int Nx, int Ny, int Nz, real *cfl_in, real *cfl_out, real *ret){
	
	Shapiro_test_case_device<<<dimGrid, dimBlock>>>(dx, dy, dz, current_time, Re, ux, uy, uz, Nx, Ny,  Nz, cfl_in);

	compute_reduction(cfl_in, cfl_out, Nx*Ny*Nz);
	retrive_Shapiro_step(ret, cfl_out);
	return ret[1];

}


