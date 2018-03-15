#include "return_map.h"


//==========================================FILE=OPERATIONS====================================================

void debug_plot_points_2D(char f_name[], int Nx, int Ny, double *vals_x, double *vals_y){
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < Nx; ++j){
        for (int k = 0; k < Ny; ++k){

            fprintf(stream, "%le %le\n", vals_x[I2(j,k)], vals_y[I2(j,k)]);

        }
        
    }
    fclose(stream);

}


void debug_plot_points_3D(char f_name[], int Nx, int Ny, int Nz, double *vals_x, double *vals_y, double *vals_z){
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < Nx; ++j){
        for (int k = 0; k < Ny; ++k){
            for (int l = 0; l < Nz; ++l){

                fprintf(stream, "%le %le %le\n", vals_x[I3(j,k,l)], vals_y[I3(j,k,l)], vals_z[I3(j,k,l)]);
            }
        }
        
    }
    fclose(stream);
}

void debug_plot_points(char f_name[], int size, double *vals_x, double *vals_y, double *vals_z){
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < size; ++j){
        fprintf(stream, "%le %le %le\n", vals_x[j], vals_y[j], vals_z[j]);
    
    }
    fclose(stream);
}

void debug_plot_vector(char f_name[], double x0, double y0, double z0, double dx1, double dy1, double dz1, double scale){
    FILE *stream;
    stream=fopen(f_name,"w");

    double vec_x=dx1;
    double vec_y=dy1;
    double vec_z=dz1;

    normalize_vector(&vec_x, &vec_y, &vec_z);


    fprintf(stream, "%le %le %le\n", x0, y0, z0);
    fprintf(stream, "%le %le %le\n", x0+scale*vec_x, y0+scale*vec_y, z0+scale*vec_z);

    fclose(stream);
}



void debug_plot_vectors(char f_name[], int size, double *xp, double *yp, double *zp, double *vals_x, double *vals_y, double *vals_z, double scale){
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < size; ++j){
        fprintf(stream, "%le %le %le %le %le %le\n", xp[j], yp[j], zp[j], scale*vals_x[j], scale*vals_y[j], scale*vals_z[j]);
    
    }
    fclose(stream);
}


//==========================================FILE=OPERATIONS====================================================


inline __device__ construct_physical_vector_device(int Nx, int Ny, int Nz, int j_fixed, int k_fixed, int l_fixed, real x1, real x2, real x3, real *ux_d, real *uy_d, real *uz_d)
{
    ux_d[IN(j_fixed,k_fixed,l_fixed)]=x1;
    uy_d[IN(j_fixed,k_fixed,l_fixed)]=x2;
    uz_d[IN(j_fixed,k_fixed,l_fixed)]=x3;
}


void return_physical_vector3(int Nx, int Ny, int Nz, real *ux, real *uy, real *uz, real *v3x, real *v3y, real *v3z, int j_fixed, int k_fixed, int l_fixed){

    v3x[0]=ux[IN(j_fixed,k_fixed,l_fixed)];
    v3y[0]=uy[IN(j_fixed,k_fixed,l_fixed)];
    v3z[0]=uz[IN(j_fixed,k_fixed,l_fixed)];


}



void construct_plane_rectangular(int local_Nx, int local_Ny, real *local_x, real *local_y, real *local_z, real eps){

    int Nx=local_Nx;
    int Ny=local_Ny;
    real dx=2.0*eps/Nx;
    real dy=2.0*eps/Ny;

    for (int j = 0; j < Nx; ++j){
        real x=j*dx-eps;
        for (int k = 0; k < Ny; ++k){
            real y=k*dy-eps;    

            local_x[I2(j,k)]=x;
            local_y[I2(j,k)]=y;
            local_z[I2(j,k)]=0.0;
        }
    
    }

}



void create_matrix3_direct(real *Matrix, real cos_alpha, real sin_alpha, real ux, real uy, real uz){
    int Nx=3;
    
    double a=1.0/(1.0+cos_alpha);

/*
a=1.0/(1.0+cos_a)
N=np.array( [ [1 - a*(u[1]**2 + u[2]**2), a*u[0]*u[1] - u[2], u[1] + a*u[0]*u[2] ],
            [u[2] + a*u[0]*u[1], 1. - a*(u[0]**2 + u[2]**2), a*u[1]*u[2] - u[0]],
            [a*u[0]*u[2] - u[1],         u[0] + a*u[1]*u[2], 1. - a*(u[0]**2 + u[1]**2)]] )
*/


    Matrix[I2(0,0)]=1.0-a*(uy*uy+uz*uz);
    Matrix[I2(0,1)]=a*ux*uy-uz; 
    Matrix[I2(0,2)]=uy+a*ux*uz;
    
    Matrix[I2(1,0)]=uz+a*ux*uy; 
    Matrix[I2(1,1)]=1.0-a*(ux*ux+uz*uz); 
    Matrix[I2(1,2)]=a*uy*uz-ux;
    
    Matrix[I2(2,0)]=a*ux*uz-uy; 
    Matrix[I2(2,1)]=ux+a*uy*uz; 
    Matrix[I2(2,2)]=1.0-a*(ux*ux+uy*uy);


}

void MatrixVector_3_3(real *Matrix, real in_v1, real in_v2, real in_v3, real *out_v1, real *out_v2, real *out_v3){
    int Nx=3;
    
    out_v1[0]=Matrix[I2(0,0)]*in_v1+Matrix[I2(0,1)]*in_v2+Matrix[I2(0,2)]*in_v3;
    out_v2[0]=Matrix[I2(1,0)]*in_v1+Matrix[I2(1,1)]*in_v2+Matrix[I2(1,2)]*in_v3;
    out_v3[0]=Matrix[I2(2,0)]*in_v1+Matrix[I2(2,1)]*in_v2+Matrix[I2(2,2)]*in_v3;

}

real vector3_norm(real ux, real uy, real uz){

    return sqrt(ux*ux+uy*uy+uz*uz);
}


void vector3_normalize(real *ux, real *uy, real *uz){

    real norm=vector3_norm(ux[0], uy[0], uz[0]);

    if(norm==0.0) 
        norm=1.0;

    ux[0]/=norm;
    uy[0]/=norm;
    uz[0]/=norm;

}




void vector3_cross_product(real nx, real ny, real nz, real mx, real my, real mz, real *ux, real *uy, real *uz){

    ux[0]=ny*mz-nz*my;
    uy[0]=-(nx*mz-nz*mx);
    uz[0]=nx*my-ny*mx;

    //normalize_vector(ux, uy, uz);

}

real vector3_dot_product(real nx, real ny, real nz, real mx, real my, real mz){

    return nx*mx+ny*my+nz*mz;

}





void rotate(real *Matrix, real *plane_x, real *plane_y, real *plane_z, real *p_x, real *p_y, real *p_z, int size){


    for (int j = 0; j < size; ++j){
        real phase_x, phase_y, phase_z;
        MatrixVector_3_3(Matrix, plane_x[j], plane_y[j], plane_z[j], &phase_x, &phase_y, &phase_z);
        p_x[j]=phase_x;
        p_y[j]=phase_y;
        p_z[j]=phase_z;
    }


}


void translate_plane(real x0, real y0, real z0, int size, real *p_x, real *p_y, real *p_z){

    for(int j=0;j<size;++j){
        p_x[j]+=x0;
        p_y[j]+=y0;
        p_z[j]+=z0;
    }

}



void rotate_plane(real rhs_x, real rhs_y, real rhs_z, real plane_nx, real plane_ny, real plane_nz, real *Matrix, int size, real *plane_x, real *plane_y, real *plane_z, real *p_x, real *p_y, real *p_z){

    real nx=rhs_x, ny=rhs_y, nz=rhs_z; //vector of the RHS
    real mx=plane_nx, my=plane_ny, mz=plane_nz;
    //mx,my,mz are vectors of the normal to the translated 2D plane

    vector3_normalize(&nx, &ny, &nz);
    vector3_normalize(&mx, &my, &mz);
    
    real ux, uy, uz; //rotating axis vector
    vector3_cross_product(mx, my, mz, nx, ny, nz, &ux, &uy, &uz);
    real cos_alpha=vector3_dot_product(nx, ny, nz, mx, my, mz);
    real sin_alpha=vector3_norm(ux, uy, uz);

    create_matrix3_direct(Matrix, cos_alpha, sin_alpha, ux, uy, uz);
    //print_Matrix_3_3(Matrix);
    rotate(Matrix, plane_x, plane_y, plane_z, p_x, p_y, p_z, size);

}



void return_vector3_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d_plane, cudaComplex *uy_hat_d_plane, cudaComplex *uz_hat_d_plane,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d, real *RHSx_d, real *RHSy_d, real *RHSz_d, int j_fixed, int k_fixed, int l_fixed, real *rhs_x, real *rhs_y, real *rhs_z){


    return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C,  dx,  dy,  dz,  Re,  Nx,  Ny,  Nz,  Mz, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_hat_d, RHSy_hat_d, RHSz_hat_d);

    iFFTN_Device(RHSx_hat_d, RHSx_d);
    iFFTN_Device(RHSy_hat_d, RHSy_d);
    iFFTN_Device(RHSz_hat_d, RHSz_d);

    host_device_real_cpy(RHSx, RHSx_d, Nx, Ny, Nz);
    host_device_real_cpy(RHSy, RHSy_d, Nx, Ny, Nz);
    host_device_real_cpy(RHSz, RHSz_d, Nx, Ny, Nz );

    return_physical_vector3(Nx, Ny, Nz, RHSx, RHSy, RHSz, rhs_x, rhs_y, rhs_z, j_fixed,  k_fixed, l_fixed);

}


void single_forward_step(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d_plane, cudaComplex *uy_hat_d_plane, cudaComplex *uz_hat_d_plane, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, real *ux_d_plane, real *uy_d_plane, real *uz_d_plane,  real *ux_plane, real *uy_plane, real *uz_plane, int j_fixed, int k_fixed, int l_fixed, real *point_x, real *point_y, real *point_z){


    RK3_SSP(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane,  ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d,  kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);

    velocity_to_double(dimGrid, dimBlock, Nx, Ny, Nz, ux_hat_d_plane, ux_d_plane, uy_hat_d_plane, uy_d_plane, uz_hat_d_plane, uz_d_plane);
    
    host_device_real_cpy(ux_plane, ux_d_plane, Nx, Ny, Nz);
    host_device_real_cpy(uy_plane, uy_d_plane, Nx, Ny, Nz);
    host_device_real_cpy(uz_plane, uz_d_plane, Nx, Ny, Nz);

    return_physical_vector3(Nx, Ny, Nz, ux_plane, uy_plane, uz_plane, point_x, point_y, point_z, j_fixed,  k_fixed, l_fixed);

}

//0 select a point in the plane 
//1 call single_forward_step.
//2 If the condition of the interseciton is met, then we find the intersection point and store the result, else, goto 1.
//3 take next point in the plane
//4 goto 1.

void execute_return_map()
{








}