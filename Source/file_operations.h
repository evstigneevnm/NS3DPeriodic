#ifndef __FILE_OPERATIONS_H__
#define __FILE_OPERATIONS_H__

#include <stdio.h>
#include <stdlib.h>
#include "Macros.h"
#include "cuda_supp.h"

void read_control_file(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz);
void write_control_file(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz);

void read_control_fourier(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im);
void write_control_file_fourier(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im);

void write_file(char* file_name, real *array, int dir, int N1, int N2, int Nsec, real dx, real dy, real dz);
void write_res_files(real *ux, real *uy, real *uz, real *div_pos, real *u_abs, int Nx, int Ny, int Nz, real dx, real dy, real dz);
void write_drop_files(int drop, int t, int Nx, int Ny, int Nz,  real *ux, real* uy, real* uz, real *div_pos, real *u_abs, real dx, real dy, real dz);
void write_drop_files_from_device(int drop, int t, int Nx, int Ny, int Nz, real *ux, real* uy, real* uz, real* u_abs, real *div_pos, real dx, real dy, real dz, real *ux_d, real* uy_d, real* uz_d, real* u_abs_d, real *div_pos_d);
void write_out_file_vec_pos_interp(char f_name[], int Nx, int Ny, int Nz, real dx, real dy, real dz, real *ux, real *uy, real *uz, int what=2);
void write_out_file_pos(char f_name[], int Nx, int Ny, int Nz, real dx, real dy, real dz, real *U, int what=2);

void write_line_specter(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im);


#endif
