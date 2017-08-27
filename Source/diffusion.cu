#include "diffusion.h"





/*3D Poisson in Fourier space----------------------------------*/
__global__ void solve_diffusion(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d) {
	
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

		real multiplicator=(0.5*din_diffusion[IN(j,k,l)]*dt/Re);

		ux_hat_d[IN(j,k,l)].x=0*dt*fx_hat_d[IN(j,k,l)].x+ux_hat_d[IN(j,k,l)].x*(1.0-multiplicator)/(1.0+multiplicator);
		ux_hat_d[IN(j,k,l)].y=0*dt*fx_hat_d[IN(j,k,l)].y+ux_hat_d[IN(j,k,l)].y*(1.0-multiplicator)/(1.0+multiplicator);

		uy_hat_d[IN(j,k,l)].x=0*dt*fy_hat_d[IN(j,k,l)].x+uy_hat_d[IN(j,k,l)].x*(1.0-multiplicator)/(1.0+multiplicator);
		uy_hat_d[IN(j,k,l)].y=0*dt*fy_hat_d[IN(j,k,l)].y+uy_hat_d[IN(j,k,l)].y*(1.0-multiplicator)/(1.0+multiplicator);

		uz_hat_d[IN(j,k,l)].x=0*dt*fz_hat_d[IN(j,k,l)].x+uz_hat_d[IN(j,k,l)].x*(1.0-multiplicator)/(1.0+multiplicator);
		uz_hat_d[IN(j,k,l)].y=0*dt*fz_hat_d[IN(j,k,l)].y+uz_hat_d[IN(j,k,l)].y*(1.0-multiplicator)/(1.0+multiplicator);	

/*
		real din_implicit=(din_diffusion[IN(j,k,l)]*dt/Re+1.0);


		ux_hat_d[IN(j,k,l)].x=(ux_hat_d[IN(j,k,l)].x+dt*fx_hat_d[IN(j,k,l)].x-0*dt*Qx_hat_d[IN(j,k,l)].x)/din_implicit;
		ux_hat_d[IN(j,k,l)].y=(ux_hat_d[IN(j,k,l)].y+dt*fx_hat_d[IN(j,k,l)].y-0*dt*Qx_hat_d[IN(j,k,l)].y)/din_implicit;
		
		uy_hat_d[IN(j,k,l)].x=(uy_hat_d[IN(j,k,l)].x+dt*fy_hat_d[IN(j,k,l)].x-0*dt*Qy_hat_d[IN(j,k,l)].x)/din_implicit;
		uy_hat_d[IN(j,k,l)].y=(uy_hat_d[IN(j,k,l)].y+dt*fy_hat_d[IN(j,k,l)].y-0*dt*Qy_hat_d[IN(j,k,l)].y)/din_implicit;

		uz_hat_d[IN(j,k,l)].x=(uz_hat_d[IN(j,k,l)].x+dt*fz_hat_d[IN(j,k,l)].x-0*dt*Qz_hat_d[IN(j,k,l)].x)/din_implicit;
		uz_hat_d[IN(j,k,l)].y=(uz_hat_d[IN(j,k,l)].y+dt*fz_hat_d[IN(j,k,l)].y-0*dt*Qz_hat_d[IN(j,k,l)].y)/din_implicit;
*/


	
	}
}
}




__global__ void solve_diffusion_explicit(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d) {
	
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

		real G_explicit=(1.0-din_diffusion[IN(j,k,l)]*dt/Re);

		ux_hat_d[IN(j,k,l)].x=G_explicit*ux_hat_d[IN(j,k,l)].x+dt*fx_hat_d[IN(j,k,l)].x;
		ux_hat_d[IN(j,k,l)].y=G_explicit*ux_hat_d[IN(j,k,l)].y+dt*fx_hat_d[IN(j,k,l)].y;
		
		uy_hat_d[IN(j,k,l)].x=G_explicit*uy_hat_d[IN(j,k,l)].x+dt*fy_hat_d[IN(j,k,l)].x;
		uy_hat_d[IN(j,k,l)].y=G_explicit*uy_hat_d[IN(j,k,l)].y+dt*fy_hat_d[IN(j,k,l)].y;

		uz_hat_d[IN(j,k,l)].x=G_explicit*uz_hat_d[IN(j,k,l)].x+dt*fz_hat_d[IN(j,k,l)].x;
		uz_hat_d[IN(j,k,l)].y=G_explicit*uz_hat_d[IN(j,k,l)].y+dt*fz_hat_d[IN(j,k,l)].y;
		
	}
}
}




__global__ void solve_advection_diffusion_projection_device(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d) {
	
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


//assemble projection. Here we ommit the diagonal unity members of AM projector so in the equation we add -dt*Q explicitly!!!!
		real AM_Q_x_Re=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_x_Im=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;	

		real AM_Q_y_Re=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_y_Im=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;

		real AM_Q_z_Re=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_z_Im=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;


//assamble diffusion matrix
		real G_explicit=(1.0-din_diffusion[IN(j,k,l)]*dt/Re);



		ux_hat_d[IN(j,k,l)].x=dt*AM_Q_x_Re+G_explicit*ux_hat_d[IN(j,k,l)].x+dt*fx_hat_d[IN(j,k,l)].x-dt*Qx_hat_d[IN(j,k,l)].x;
		ux_hat_d[IN(j,k,l)].y=dt*AM_Q_x_Im+G_explicit*ux_hat_d[IN(j,k,l)].y+dt*fx_hat_d[IN(j,k,l)].y-dt*Qx_hat_d[IN(j,k,l)].y;
		
		uy_hat_d[IN(j,k,l)].x=dt*AM_Q_y_Re+G_explicit*uy_hat_d[IN(j,k,l)].x+dt*fy_hat_d[IN(j,k,l)].x-dt*Qy_hat_d[IN(j,k,l)].x;
		uy_hat_d[IN(j,k,l)].y=dt*AM_Q_y_Im+G_explicit*uy_hat_d[IN(j,k,l)].y+dt*fy_hat_d[IN(j,k,l)].y-dt*Qy_hat_d[IN(j,k,l)].y;

		uz_hat_d[IN(j,k,l)].x=dt*AM_Q_z_Re+G_explicit*uz_hat_d[IN(j,k,l)].x+dt*fz_hat_d[IN(j,k,l)].x-dt*Qz_hat_d[IN(j,k,l)].x;
		uz_hat_d[IN(j,k,l)].y=dt*AM_Q_z_Im+G_explicit*uz_hat_d[IN(j,k,l)].y+dt*fz_hat_d[IN(j,k,l)].y-dt*Qz_hat_d[IN(j,k,l)].y;
		
	}
}
}

__global__ void solve_advection_diffusion_projection_device_UV_RHS(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d) {
	
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


//assemble projection. Here we ommit the diagonal unity members of AM projector so in the equation we add -dt*Q explicitly!!!!
		real AM_Q_x_Re=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_x_Im=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;	

		real AM_Q_y_Re=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_y_Im=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;

		real AM_Q_z_Re=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_z_Im=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;


//assamble diffusion matrix explicit
/*		
		real G_explicit=(-din_diffusion[IN(j,k,l)]/Re);



		ux_hat_d[IN(j,k,l)].x=AM_Q_x_Re+G_explicit*ux_hat_d[IN(j,k,l)].x-Qx_hat_d[IN(j,k,l)].x;
		ux_hat_d[IN(j,k,l)].y=AM_Q_x_Im+G_explicit*ux_hat_d[IN(j,k,l)].y-Qx_hat_d[IN(j,k,l)].y;
		
		uy_hat_d[IN(j,k,l)].x=AM_Q_y_Re+G_explicit*uy_hat_d[IN(j,k,l)].x-Qy_hat_d[IN(j,k,l)].x;
		uy_hat_d[IN(j,k,l)].y=AM_Q_y_Im+G_explicit*uy_hat_d[IN(j,k,l)].y-Qy_hat_d[IN(j,k,l)].y;

		uz_hat_d[IN(j,k,l)].x=AM_Q_z_Re+G_explicit*uz_hat_d[IN(j,k,l)].x-Qz_hat_d[IN(j,k,l)].x;
		uz_hat_d[IN(j,k,l)].y=AM_Q_z_Im+G_explicit*uz_hat_d[IN(j,k,l)].y-Qz_hat_d[IN(j,k,l)].y;
//*/
//assamble diffusion matrix implicit	?	
//*
		real G_implicit=0.5*(Re/din_diffusion[IN(j,k,l)]);
		real G_explicit=0.5*(-din_diffusion[IN(j,k,l)]/Re);


		ux_hat_d[IN(j,k,l)].x=(AM_Q_x_Re-Qx_hat_d[IN(j,k,l)].x+G_explicit*ux_hat_d[IN(j,k,l)].x)*G_implicit;
		ux_hat_d[IN(j,k,l)].y=(AM_Q_x_Im-Qx_hat_d[IN(j,k,l)].y+G_explicit*ux_hat_d[IN(j,k,l)].y)*G_implicit;
		
		uy_hat_d[IN(j,k,l)].x=(AM_Q_y_Re-Qy_hat_d[IN(j,k,l)].x+G_explicit*uy_hat_d[IN(j,k,l)].x)*G_implicit;
		uy_hat_d[IN(j,k,l)].y=(AM_Q_y_Im-Qy_hat_d[IN(j,k,l)].y+G_explicit*uy_hat_d[IN(j,k,l)].y)*G_implicit;

		uz_hat_d[IN(j,k,l)].x=(AM_Q_z_Re-Qz_hat_d[IN(j,k,l)].x+G_explicit*uz_hat_d[IN(j,k,l)].x)*G_implicit;
		uz_hat_d[IN(j,k,l)].y=(AM_Q_z_Im-Qz_hat_d[IN(j,k,l)].y+G_explicit*uz_hat_d[IN(j,k,l)].y)*G_implicit;
//*/
	}
}
}


__global__ void solve_advection_diffusion_projection_device_UV(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d) {
	
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


//assemble projection. Here we ommit the diagonal unity members of AM projector so in the equation we add -dt*Q explicitly!!!!
		real AM_Q_x_Re=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_x_Im=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;	

		real AM_Q_y_Re=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_y_Im=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;

		real AM_Q_z_Re=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_z_Im=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;


//assamble diffusion matrix
		real G_explicit=(1.0-din_diffusion[IN(j,k,l)]*dt/Re);



		ux_hat_d[IN(j,k,l)].x=dt*AM_Q_x_Re+G_explicit*ux_hat_d[IN(j,k,l)].x-dt*Qx_hat_d[IN(j,k,l)].x;
		ux_hat_d[IN(j,k,l)].y=dt*AM_Q_x_Im+G_explicit*ux_hat_d[IN(j,k,l)].y-dt*Qx_hat_d[IN(j,k,l)].y;
		
		uy_hat_d[IN(j,k,l)].x=dt*AM_Q_y_Re+G_explicit*uy_hat_d[IN(j,k,l)].x-dt*Qy_hat_d[IN(j,k,l)].x;
		uy_hat_d[IN(j,k,l)].y=dt*AM_Q_y_Im+G_explicit*uy_hat_d[IN(j,k,l)].y-dt*Qy_hat_d[IN(j,k,l)].y;

		uz_hat_d[IN(j,k,l)].x=dt*AM_Q_z_Re+G_explicit*uz_hat_d[IN(j,k,l)].x-dt*Qz_hat_d[IN(j,k,l)].x;
		uz_hat_d[IN(j,k,l)].y=dt*AM_Q_z_Im+G_explicit*uz_hat_d[IN(j,k,l)].y-dt*Qz_hat_d[IN(j,k,l)].y;
		
	}
}
}




__global__ void RHS_advection_diffusion_projection_device(cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real* din_diffusion, int Nx, int Ny, int Nz, real Re, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d) {
	
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


//assemble projection. Here we ommit the diagonal unity members of AM projector so in the equation we add -dt*Q explicitly!!!!
		real AM_Q_x_Re=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_x_Im=AM_11_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_12_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_13_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;	

		real AM_Q_y_Re=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_y_Im=AM_12_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_22_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;

		real AM_Q_z_Re=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].x+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].x+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].x;
		real AM_Q_z_Im=AM_13_d[IN(j,k,l)]*Qx_hat_d[IN(j,k,l)].y+AM_23_d[IN(j,k,l)]*Qy_hat_d[IN(j,k,l)].y+AM_33_d[IN(j,k,l)]*Qz_hat_d[IN(j,k,l)].y;


//assamble diffusion matrix. Diffusion is pre-assembled as positive definite.
		real G_explicit=(-din_diffusion[IN(j,k,l)]/Re);



		RHSx_hat_d[IN(j,k,l)].x=AM_Q_x_Re+G_explicit*ux_hat_d[IN(j,k,l)].x+fx_hat_d[IN(j,k,l)].x-Qx_hat_d[IN(j,k,l)].x;
		RHSx_hat_d[IN(j,k,l)].y=AM_Q_x_Im+G_explicit*ux_hat_d[IN(j,k,l)].y+fx_hat_d[IN(j,k,l)].y-Qx_hat_d[IN(j,k,l)].y;
		
		RHSy_hat_d[IN(j,k,l)].x=AM_Q_y_Re+G_explicit*uy_hat_d[IN(j,k,l)].x+fy_hat_d[IN(j,k,l)].x-Qy_hat_d[IN(j,k,l)].x;
		RHSy_hat_d[IN(j,k,l)].y=AM_Q_y_Im+G_explicit*uy_hat_d[IN(j,k,l)].y+fy_hat_d[IN(j,k,l)].y-Qy_hat_d[IN(j,k,l)].y;

		RHSz_hat_d[IN(j,k,l)].x=AM_Q_z_Re+G_explicit*uz_hat_d[IN(j,k,l)].x+fz_hat_d[IN(j,k,l)].x-Qz_hat_d[IN(j,k,l)].x;
		RHSz_hat_d[IN(j,k,l)].y=AM_Q_z_Im+G_explicit*uz_hat_d[IN(j,k,l)].y+fz_hat_d[IN(j,k,l)].y-Qz_hat_d[IN(j,k,l)].y;
		
	}
}
}

void solve_advection_diffusion_projection(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

	solve_advection_diffusion_projection_device<<<dimGrid_C, dimBlock_C>>>(ux_hat_d, uy_hat_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, din_diffusion_d, Nx, Ny, Nz, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);

	//printf(" dimGrid_C: %dX%dX%d, dimBlock_C: %dX%dX%d. \n", dimGrid_C.x, dimGrid_C.y, dimGrid_C.z, dimBlock_C.x, dimBlock_C.y, dimBlock_C.z);
	//printf("Mz=%i\n",Mz);

}



void solve_advection_diffusion_projection_UV_RHS(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	
	solve_advection_diffusion_projection_device_UV_RHS<<<dimGrid_C, dimBlock_C>>>(vx_hat_d, vy_hat_d, vz_hat_d, din_diffusion_d, Nx, Ny, Nz, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);



}



void solve_advection_diffusion_projection_UV(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *vx_hat_d, cudaComplex *vy_hat_d, cudaComplex *vz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){
	

	solve_advection_diffusion_projection_device_UV<<<dimGrid_C, dimBlock_C>>>(vx_hat_d, vy_hat_d, vz_hat_d, din_diffusion_d, Nx, Ny, Nz, Re, dt, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d);



}



void RHS_advection_diffusion_projection(dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d){
	

	RHS_advection_diffusion_projection_device<<<dimGrid_C, dimBlock_C>>>(ux_hat_d, uy_hat_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, din_diffusion_d, Nx, Ny, Nz, Re, Qx_hat_d, Qy_hat_d, Qz_hat_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d, RHSx_hat_d, RHSy_hat_d, RHSz_hat_d);

}



void diffusion_device(int Nx, int Ny, int Nz, real* din_diffusion_d, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, real Re, real dt,cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){



}

void velocity_to_double(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, real* ux_d, cudaComplex *uy_hat_d, real* uy_d, cudaComplex *uz_hat_d, real* uz_d){

	iFFTN_Device(dimGrid, dimBlock, ux_hat_d, ux_d, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uz_hat_d, uz_d, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_hat_d, uy_d, Nx, Ny, Nz);


}