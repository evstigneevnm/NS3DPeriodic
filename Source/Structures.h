#ifndef __H_STRUCTURES_H__
#define __H_STRUCTURES_H__



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
		real tau;	//value for ''timestep'' in the Newton's method
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
		// imag exponential shift
		real rotate_angle;
		//for BiCGStab(L):
		int BiCG_L;
		real BiCG_tol;
		int BiCG_Iter;
		cublasHandle_t handle;
} struct_NS3D_RK3_iExp_Call;




#endif