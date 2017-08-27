#include "advection_WENO.h"


//weno interpolation
__device__ real weno9_interpolation(real f1, real f2,real f3, real f4, real f5, real f6, real f7, real f8, real f9){
	const real eps=1.0E-22;
	real res=0.0;

	real q0=1.0/5.0*f1-21.0/20.0*f2+137.0/60.0*f3-163.0/60.0*f4+137.0/60.0*f5;
	real q1=-1.0/20.0*f2+17.0/60.0*f3-43.0/60.0*f4+77.0/60.0*f5+1.0/5.0*f6;
	real q2=1.0/30.0*f3-13.0/60.0*f4+47.0/60.0*f5+9.0/20.0*f6-1.0/20.0*f7;
	real q3=-1.0/20.0*f4+9.0/20.0*f5+47.0/60.0*f6-13.0/60.0*f7+1.0/30.0*f8;
	real q4=1.0/5.0*f5+77.0/60.0*f6-43.0/60.0*f7+17.0/60.0*f8-1.0/20.0*f9;
	
	real betta0 = f1*(22658.0*f1-208501.0*f2 + 364863.0*f3-288007.0*f4 + 86329.0*f5)+f2*(482963.0*f2-1704396.0*f3 + 1358458.0*f4-411487.0*f5 )+f3*(1521393.0*f3-2462076.0*f4 + 758823.0*f5)+f4*(1020563.0*f4-649501.0*f5) + 107918.0*f5*f5;
	real betta1=f2*(6908.0*f2-60871.0*f3+ 99213.0*f4-70237.0*f5 + 18079.0*f6 )+f3*(138563.0*f3-464976.0*f4 + 337018.0*f5-88297.0*f6 )+f4*(406293.0*f4-611976.0*f5 + 165153.0*f6 )+f5*(242723.0*f5-140251.0*f6 ) + 22658.0*f6*f6;
	real betta2=f3*(6908.0*f3-51001.0*f4 + 67923.0*f5-38947.0*f6 + 8209.0*f7 )+f4*(104963.0*f4-299076.0*f5 + 179098.0*f6-38947.0*f7 )+f5*(231153.0*f5-299076.0*f6 + 67923.0*f7 )+f6*(104963.0*f6-51001.0*f7 ) + 6908.0*f7*f7;
	real betta3=f4*(22658.0*f4-140251.0*f5 + 165153.0*f6-88297.0*f7 + 18079.0*f8 )+f5*(242723.0*f5-611976.0*f6 + 337018.0*f7-70237.0*f8 )+f6*(406293.0*f6-464976.0*f7 + 99213.0*f8 )+f7*(138563.0*f7-60871.0*f8 ) + 6908.0*f8*f8;
	real betta4=f5*(107918.0*f5-649501.0*f6 + 758823.0*f7-411487.0*f8 + 86329.0*f9 )+f6*(1020563.0*f6-2462076.0*f7 + 1358458.0*f8-288007.0*f9 )+f7*(1521393.0*f7-1704396.0*f8 + 364863.0*f9 )+f8*(482963.0*f8-208501.0*f9 ) + 22658.0*f9*f9;
	

	real d0=1.0/126.0,d1=10.0/63.0,d2=10.0/21.0,d3=20.0/63.0,d4=5.0/126.0;



	real alpha0=d0/((eps+betta0)*(eps+betta0));
	real alpha1=d1/((eps+betta1)*(eps+betta1));
	real alpha2=d2/((eps+betta2)*(eps+betta2));
	real alpha3=d3/((eps+betta3)*(eps+betta3));
	real alpha4=d4/((eps+betta4)*(eps+betta4));
	
	real alpha_sum=alpha0+alpha1+alpha2+alpha3+alpha4;
	real w0=0.0,w1=0.0,w2=0.0,w3=0.0,w4=0.0;
	w0=alpha0/alpha_sum;
	w1=alpha1/alpha_sum;
	w2=alpha2/alpha_sum;
	w3=alpha3/alpha_sum;
	w4=alpha4/alpha_sum;
	res=w0*q0+w1*q1+w2*q2+w3*q3+w4*q4;
	
	return(res);

}



__device__ real weno5_interpolation(real f1, real f2,real f3, real f4, real f5){

	real eps=1.0E-22;
	real res;

	real q0=1.0/3.0*f1-7.0/6.0*f2+11.0/6.0*f3;
	real q1=-1.0/6.0*f2+5.0/6.0*f3+1.0/3.0*f4;
	real q2=1.0/3.0*f3+5.0/6.0*f4-1.0/6.0*f5;


	real betta0=13.0/12.0*(f1-2.0*f2+f3)*(f1-2.0*f2+f3)+0.25*(f1-4.0*f2+3.0*f3)*(f1-4.0*f2+3.0*f3);
	real betta1=13.0/12.0*(f2-2.0*f3+f4)*(f2-2.0*f3+f4)+0.25*(f2-f4)*(f2-f4);
	real betta2=13.0/12.0*(f3-2.0*f4+f5)*(f3-2.0*f4+f5)+0.25*(3.0*f3-4.0*f4+f5)*(3.0*f3-4.0*f4+f5);


	real d0=0.1,d1=0.6,d2=0.3; 
	real alpha0=d0/((eps+betta0)*(eps+betta0));
	real alpha1=d1/((eps+betta1)*(eps+betta1));
	real alpha2=d2/((eps+betta2)*(eps+betta2));
	real alpha_sum=alpha0+alpha1+alpha2;
	real w0=0.0,w1=0.0,w2=0.0;
	w0=alpha0/alpha_sum;
	w1=alpha1/alpha_sum;
	w2=alpha2/alpha_sum;
	res=w0*q0+w1*q1+w2*q2;
	return(res);


}
__device__ real weno3_interpolation(real f1, real f2, real f3){

	real eps=1.0E-22;
	real res=0.0;

	real q0=-0.5*f1+1.5*f2;
	real q1=0.5*f2+0.5*f3;
	


	real betta0=(f2-f1)*(f2-f1);
	real betta1=(f3-f2)*(f3-f2);


	real d0=1.0/3.0, d1=2.0/3.0; 
	real alpha0=d0/((eps+betta0)*(eps+betta0));
	real alpha1=d1/((eps+betta1)*(eps+betta1));
	real alpha_sum=alpha0+alpha1;
	real w0=0.0,w1=0.0;
	w0=alpha0/alpha_sum;
	w1=alpha1/alpha_sum;
	res=w0*q0+w1*q1;
	return(res);


}



__device__ real flux_x_weno_L(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	

	if(scheme==3){
		real f1=x[I3(j+1,k,l)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j-1,k,l)]; 
		//if(dir=='L') 
		res=weno3_interpolation(f3,f2,f1);
		//else res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j+2,k,l)]; 
		real f2=x[I3(j+1,k,l)]; 
		real f3=x[I3(j,k,l)];   
		real f4=x[I3(j-1,k,l)]; 
		real f5=x[I3(j-2,k,l)]; 
		//if(dir=='L') 
		res=weno5_interpolation(f5,f4,f3,f2,f1);
		//else res=weno5_interpolation(f1,f2,f3,f4,f5);

	}
	else{
		res=x[I3(j,k,l)];		
	
	}
	
	return (res); 

}

__device__ real flux_x_weno_R(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	

	if(scheme==3){
		real f1=x[I3(j+1,k,l)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j-1,k,l)]; 
		//if(dir=='L') 
		//res=weno3_interpolation(f3,f2,f1);
		res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j+2,k,l)]; 
		real f2=x[I3(j+1,k,l)]; 
		real f3=x[I3(j,k,l)];   
		real f4=x[I3(j-1,k,l)]; 
		real f5=x[I3(j-2,k,l)]; 
		//if(dir=='L') 
		//res=weno5_interpolation(f5,f4,f3,f2,f1);
		res=weno5_interpolation(f1,f2,f3,f4,f5);

	}
	else{
		res=x[I3(j,k,l)];		
	
	}
	
	return (res); 

}


__device__ real flux_y_weno_L(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	
	
	if(scheme==3){
		real f1=x[I3(j,k+1,l)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j,k-1,l)]; 
		//if(dir=='L') 
		res=weno3_interpolation(f3,f2,f1);
		//else res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j,k+2,l)]; 
		real f2=x[I3(j,k+1,l)]; 
		real f3=x[I3(j,k,l)];  
		real f4=x[I3(j,k-1,l)]; 
		real f5=x[I3(j,k-2,l)];
		//if(dir=='L') 
		res=weno5_interpolation(f5,f4,f3,f2,f1);
		//else res=weno5_interpolation(f1,f2,f3,f4,f5);

	}	
	else{
		res=x[I3(j,k,l)];		

	}
	
	return (res); 

}



__device__ real flux_y_weno_R(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	
	
	if(scheme==3){
		real f1=x[I3(j,k+1,l)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j,k-1,l)]; 
		//if(dir=='L') 
		//res=weno3_interpolation(f3,f2,f1);
		res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j,k+2,l)]; 
		real f2=x[I3(j,k+1,l)]; 
		real f3=x[I3(j,k,l)];  
		real f4=x[I3(j,k-1,l)]; 
		real f5=x[I3(j,k-2,l)];
		//if(dir=='L') 
		//res=weno5_interpolation(f5,f4,f3,f2,f1);
		res=weno5_interpolation(f1,f2,f3,f4,f5);

	}	
	else{
		res=x[I3(j,k,l)];		

	}
	
	return (res); 

}


__device__ real flux_z_weno_L(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	
	
	if(scheme==3){
		real f1=x[I3(j,k,l+1)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j,k,l-1)]; 
		//if(dir=='L') 
		res=weno3_interpolation(f3,f2,f1);
		//else res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j,k,l+2)]; 
		real f2=x[I3(j,k,l+1)]; 
		real f3=x[I3(j,k,l)];  
		real f4=x[I3(j,k,l-1)]; 
		real f5=x[I3(j,k,l-2)];
		//if(dir=='L') 
		res=weno5_interpolation(f5,f4,f3,f2,f1);
		//else res=weno5_interpolation(f1,f2,f3,f4,f5);

	}	
	else{
		res=x[I3(j,k,l)];		

	}
	
	return (res); 

}



__device__ real flux_z_weno_R(real* x,  int j, int k, int l, int scheme, int Nx, int Ny, int Nz){
real res=0.0;	
	
	if(scheme==3){
		real f1=x[I3(j,k,l+1)]; 
		real f2=x[I3(j,k,l)];   
		real f3=x[I3(j,k,l-1)]; 
		//if(dir=='L') 
		//res=weno3_interpolation(f3,f2,f1);
		res=weno3_interpolation(f1,f2,f3);	
	
	}
	else if(scheme==5){
		real f1=x[I3(j,k,l+2)]; 
		real f2=x[I3(j,k,l+1)]; 
		real f3=x[I3(j,k,l)];  
		real f4=x[I3(j,k,l-1)]; 
		real f5=x[I3(j,k,l-1)];
		//if(dir=='L') 
		//res=weno5_interpolation(f5,f4,f3,f2,f1);
		res=weno5_interpolation(f1,f2,f3,f4,f5);

	}	
	else{
		res=x[I3(j,k,l)];		

	}
	
	return (res); 

}


__device__ real flux_device( real dx, real dy, real dz, int j, int k, int l, real* ux,  real* uy, real* uz, real* x,  int scheme, int Nx, int Ny, int Nz){
		
		real res=0.0;
	
		real U_w=0.5*(ux[I3(j-1,k,l)]+ux[I3(j,k,l)]);
		real U_e=0.5*(ux[I3(j,k,l)]+ux[I3(j+1,k,l)]);
		
		real space_w=0.0;
		if(U_w>=0.0) 
			space_w=flux_x_weno_L(x, j-1, k, l, scheme,Nx,Ny,Nz);
		else 
			space_w=flux_x_weno_R(x, j, k, l, scheme,Nx,Ny,Nz);
		
		real space_e=0.0; 
		if(U_e>=0.0) 
			space_e=flux_x_weno_L(x, j, k, l,scheme,Nx,Ny,Nz);
		else
			space_e=flux_x_weno_R(x, j+1, k, l,scheme,Nx,Ny,Nz);		
				
		real U_n=0.5*(uy[I3(j,k-1,l)]+uy[I3(j,k,l)]);
		real U_s=0.5*(uy[I3(j,k,l)]+uy[I3(j,k+1,l)]);
			
		real space_n=0.0;
		if(U_n>=0.0)
			space_n=flux_y_weno_L(x,  j, k-1, l, scheme,Nx,Ny,Nz);
		else
			space_n=flux_y_weno_R(x,  j, k, l, scheme,Nx,Ny,Nz);
		
		real space_s=0.0;
		if(U_s>=0.0)
			space_s=flux_y_weno_L(x,  j, k, l, scheme,Nx,Ny,Nz);
		else			
			space_s=flux_y_weno_R(x, j, k+1, l, scheme,Nx,Ny,Nz);	
		

		real U_r=0.5*(uz[I3(j,k,l-1)]+uz[I3(j,k,l)]);
		real U_f=0.5*(uz[I3(j,k,l)]+uz[I3(j,k,l+1)]);	
		
		real space_r=0.0;
		if(U_r>=0.0)
			space_r=flux_z_weno_L(x,  j, k, l-1, scheme,Nx,Ny,Nz);
		else
			space_r=flux_z_weno_R(x,  j, k, l, scheme,Nx,Ny,Nz);
		
		real space_f=0.0;
		if(U_f>=0.0)
			space_f=flux_z_weno_L(x,  j, k, l, scheme,Nx,Ny,Nz);
		else			
			space_f=flux_z_weno_R(x, j, k, l+1, scheme,Nx,Ny,Nz);			

		res=ux[IN(j,k,l)]*(space_e-space_w)/dx+uy[IN(j,k,l)]*(space_s-space_n)/dy+uz[IN(j,k,l)]*(space_f-space_r)/dz;
		//res=(U_e*space_e-U_w*space_w)/dx+(U_s*space_s-U_n*space_n)/dy+(U_f*space_f-U_r*space_r)/dz;
		//res=(space_e*space_e-space_w*space_w)/dx+(space_s*space_s-space_n*space_n)/dy+(space_f*space_f-space_r*space_r)/dz;
	
		return(res);
		
}



__global__ void advection(int Nx, int Ny, int Nz, real* ux_d, real* uy_d, real* uz_d, real* Qx_d_ext, real* Qy_d_ext, real* Qz_d_ext, real dx, real dy, real dz, int scheme){

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
			
		//perform advection
		Qx_d_ext[I3(j,k,l)]=flux_device(dx, dy, dz, j, k, l, ux_d, uy_d, uz_d, ux_d, scheme, Nx, Ny, Nz);
		Qy_d_ext[I3(j,k,l)]=flux_device(dx, dy, dz, j, k, l, ux_d, uy_d, uz_d, uy_d, scheme, Nx, Ny, Nz);				
		Qz_d_ext[I3(j,k,l)]=flux_device(dx, dy, dz, j, k, l, ux_d, uy_d, uz_d, uz_d, scheme, Nx, Ny, Nz);

	}
}

}




void WenoAdvection(dim3 dimBlock, dim3 dimGrid, dim3 dimBlock_C, dim3 dimGrid_C, int Nx, int Ny, int Nz, int Mz, int scheme, cudaComplex* ux_hat_d, cudaComplex* uy_hat_d, cudaComplex* uz_hat_d, real dx, real dy, real dz, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d){

	build_vels_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, uy_hat_d, uz_hat_d, ux_hat_d_ext_W, uy_hat_d_ext_W, uz_hat_d_ext_W);

	iFFTN_Device(dimGrid, dimBlock, ux_hat_d_ext_W, ux_d_ext_W, Nx, Ny, Nz);
	iFFTN_Device(dimGrid, dimBlock, uy_hat_d_ext_W, uy_d_ext_W, Nx, Ny, Nz);	
	iFFTN_Device(dimGrid, dimBlock, uz_hat_d_ext_W, uz_d_ext_W, Nx, Ny, Nz);

	advection<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ux_d_ext_W, uy_d_ext_W, uz_d_ext_W, Qx_d_ext_W, Qy_d_ext_W, Qz_d_ext_W, dx, dy, dz, scheme);
	

	//return to Image
	FFTN_Device(Qx_d_ext_W, Qx_hat_d);
	FFTN_Device(Qy_d_ext_W, Qy_hat_d);
	FFTN_Device(Qz_d_ext_W, Qz_hat_d);
	
	//set_to_zero(dimGrid_C, dimBlock_C, Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);
	//set_to_zero_real<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Qx_hat_d, Qy_hat_d, Qz_hat_d);
}



void init_WENO(int Nx, int Ny, int Nz, int Mz){
	

	device_allocate_all_complex(Nx, Ny, Mz, 3, &ux_hat_d_ext_W, &uy_hat_d_ext_W, &uz_hat_d_ext_W);

	device_allocate_all_real(Nx, Ny, Nz, 3, &ux_d_ext_W, &uy_d_ext_W, &uz_d_ext_W);

	device_allocate_all_real(Nx, Ny, Nz, 3, &Qx_d_ext_W, &Qy_d_ext_W, &Qz_d_ext_W);




}

void clean_WENO(){
	
	device_deallocate_all_complex(3, ux_hat_d_ext_W, uy_hat_d_ext_W, uz_hat_d_ext_W);

	device_deallocate_all_real(3, ux_d_ext_W, uy_d_ext_W, uz_d_ext_W);

	device_deallocate_all_real(3, Qx_d_ext_W, Qy_d_ext_W, Qz_d_ext_W);


}