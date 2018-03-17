#ifndef __MATH_SUPPORT_H__
#define __MATH_SUPPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include "Macros.h"
#include "min_max_reduction.h"



double rand_normal(double mean, double stddev);

void init_fft_plans(cufftHandle planR2C_l, cufftHandle planC2R_l);



static cufftHandle planR2C_local;
static cufftHandle planC2R_local;
//device part
__global__ void init_fources_fourier_device(int Nx, int Ny, int Nz, cudaComplex *f_hat_x, cudaComplex *f_hat_y, cudaComplex *f_hat_z);

__global__ void velocity_to_abs_device(int Nx, int Ny, int Nz, real* ux_d, real* uy_d, real*  uz_d, real*  u_abs_d);


__global__ void scale_double(real* f, int Nx, int Ny, int Nz);

void FFTN_Device(real *source, cudaComplex *destination);
void iFFTN_Device(cudaComplex *source, real *destination);
void iFFTN_Device(dim3 dimGrid, dim3 dimBlock, cudaComplex *source, real *destination, int Nx, int Ny, int Nz);




void Image_to_Domain(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real* ux_d, cudaComplex *ux_hat_d, real* uy_d, cudaComplex *uy_hat_d, real* uz_d, cudaComplex *uz_hat_d);
void Domain_to_Image(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, real* ux_d, cudaComplex *uy_hat_d, real* uy_d, cudaComplex *uz_hat_d, real* uz_d);


void all_Fourier2double(dim3 dimGrid_C, dim3 dimBlock_C, cudaComplex *ux_hat_d, real* ux_Re_d, real* ux_Im_d, cudaComplex *uy_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uz_hat_d, real* uz_Re_d, real* uz_Im_d, int Nx, int Ny, int Nz);
void all_double2Fourier(dim3 dimGrid_C, dim3 dimBlock_C, real* ux_Re_d, real* ux_Im_d, cudaComplex *ux_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uy_hat_d,  real* uz_Re_d, real* uz_Im_d, cudaComplex *uz_hat_d, int Nx, int Ny, int Nz);


void get_high_wavenumbers(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_red_hat_d, cudaComplex *uy_red_hat_d, cudaComplex *uz_red_hat_d, cudaComplex *u_temp_complex_d, real *ux_red_d, real *uy_red_d, real *uz_red_d, int delta);

void get_curl(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* rot_x_d, real* rot_y_d, real* rot_z_d);

//host part
void build_Laplace_Wavenumbers(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *kx_laplace, real *ky_laplace, real *kz_laplace);
void build_Laplace_and_Diffusion(int Nx, int Ny, int Nz, real *din_poisson, real *din_diffusion, real *kx_laplace, real *ky_laplace, real *kz_laplace);
void build_Nabla_Wavenumbers(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *kx_nabla, real *ky_nabla, real *kz_nabla);

real TotalEnergy(int Nx, int Ny, int Nz, real *ux, real *uy, real *uz, real dx, real dy, real dz, real alpha, real beta);
real TotalDissipation(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz,real dx, real dy, real dz,real alpha,real beta, real Re);

void build_projection_matrix_elements(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *AM_11, real *AM_22, real *AM_33, real *AM_12, real *AM_13, real *AM_23);

void build_mask_matrix(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *mask_2_3);


real get_kinetic_energy(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* energy, real* energy_out, real* energy_out1);

real get_dissipation(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* dissipation, real* energy_out, real* energy_out1);


void calculate_energy_spectrum(char* file_name, int Nx, int Ny, int Mz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im);


void Helmholz_Fourier_Filter(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz,  real Lx, real Ly, real Lz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real filter_eps, real *ux_filt_d, real *uy_filt_d, real *uz_filt_d);


void CutOff_Fourier_Filter(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz,  real Lx, real Ly, real Lz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real Radius_to_one, real *ux_filt_d, real *uy_filt_d, real *uz_filt_d);


#endif
