#include "return_map.h"


//==========================================FILE=OPERATIONS====================================================

void debug_plot_points_2D(char f_name[], int Nx, int Ny, double *vals_x, double *vals_y)
{
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < Nx; ++j){
        for (int k = 0; k < Ny; ++k){

            fprintf(stream, "%le %le\n", vals_x[I2(j,k)], vals_y[I2(j,k)]);

        }
        
    }
    fclose(stream);

}


void debug_plot_points_3D(char f_name[], int Nx, int Ny, int Nz, double *vals_x, double *vals_y, double *vals_z)
{
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

void debug_plot_points(char f_name[], int size, double *vals_x, double *vals_y, double *vals_z)
{
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < size; ++j){
        fprintf(stream, "%le %le %le\n", vals_x[j], vals_y[j], vals_z[j]);
    
    }
    fclose(stream);
}

void debug_plot_vector(char f_name[], double x0, double y0, double z0, double dx1, double dy1, double dz1, double scale)
{
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



void debug_plot_vectors(char f_name[], int size, double *xp, double *yp, double *zp, double *vals_x, double *vals_y, double *vals_z, double scale)
{
    FILE *stream;
    stream=fopen(f_name,"w");

    for (int j = 0; j < size; ++j){
        fprintf(stream, "%le %le %le %le %le %le\n", xp[j], yp[j], zp[j], scale*vals_x[j], scale*vals_y[j], scale*vals_z[j]);
    
    }
    fclose(stream);
}


//==========================================FILE=OPERATIONS====================================================



__global__ void construct_physical_vector_device(int Nx, int Ny, int Nz, int j_fixed, int k_fixed, int l_fixed,  real x_point, real y_point, real z_point, real *ux_d, real *uy_d, real *uz_d)
{


    ux_d[IN(j_fixed, k_fixed, l_fixed)]=x_point;
    uy_d[IN(j_fixed, k_fixed, l_fixed)]=y_point;
    uz_d[IN(j_fixed, k_fixed, l_fixed)]=z_point;


}


void construct_physical_vector(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, int j_fixed, int k_fixed, int l_fixed,  real x_point, real y_point, real z_point, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real *ux_d_plane, real *uy_d_plane, real *uz_d_plane, cudaComplex *ux_hat_d_plane, cudaComplex *uy_hat_d_plane, cudaComplex *uz_hat_d_plane)
{

    Domain_to_Image(dimGrid, dimBlock,  Nx,  Ny,  Nz, ux_hat_d, ux_d_plane, uy_hat_d, uy_d_plane, uz_hat_d, uz_d_plane);

    construct_physical_vector_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, j_fixed, k_fixed, l_fixed,  x_point, y_point, z_point, ux_d_plane , uy_d_plane, uz_d_plane);

    Image_to_Domain(dimGrid, dimBlock, Nx, Ny, Nz, ux_d_plane, ux_hat_d_plane, uy_d_plane, uy_hat_d_plane, uz_d_plane, uz_hat_d_plane);


}



void return_physical_vector3(int Nx, int Ny, int Nz, real *ux, real *uy, real *uz, real *v3x, real *v3y, real *v3z, int j_fixed, int k_fixed, int l_fixed)
{

    v3x[0]=ux[IN(j_fixed,k_fixed,l_fixed)];
    v3y[0]=uy[IN(j_fixed,k_fixed,l_fixed)];
    v3z[0]=uz[IN(j_fixed,k_fixed,l_fixed)];


}




void construct_plane_rectangular(int local_Nx, int local_Ny, real *local_x, real *local_y, real *local_z, real eps)
{

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

real test_plane_location(real nx, real ny, real nz, real x0, real y0, real z0, real x, real y, real z){

    real vx=(x-x0);
    real vy=(y-y0);
    real vz=(z-z0);

    return(vector3_dot_product(nx, ny, nz, vx, vy, vz));
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



void return_vector3_RHS(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *RHSx_hat_d, cudaComplex *RHSy_hat_d, cudaComplex *RHSz_hat_d, real *RHSx_d, real *RHSy_d, real *RHSz_d, real *RHSx, real *RHSy, real *RHSz, int j_fixed, int k_fixed, int l_fixed, real *rhs_x, real *rhs_y, real *rhs_z){


    return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C,  dx,  dy,  dz,  Re,  Nx,  Ny,  Nz,  Mz, ux_hat_d, uy_hat_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_hat_d, RHSy_hat_d, RHSz_hat_d);

    iFFTN_Device(RHSx_hat_d, RHSx_d);
    iFFTN_Device(RHSy_hat_d, RHSy_d);
    iFFTN_Device(RHSz_hat_d, RHSz_d);

    host_device_real_cpy(RHSx, RHSx_d, Nx, Ny, Nz);
    host_device_real_cpy(RHSy, RHSy_d, Nx, Ny, Nz);
    host_device_real_cpy(RHSz, RHSz_d, Nx, Ny, Nz );

    return_physical_vector3(Nx, Ny, Nz, RHSx, RHSy, RHSz, rhs_x, rhs_y, rhs_z, j_fixed, k_fixed, l_fixed);

}

void return_vector3_solution(int j_fixed,  int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, real *ux_d, real *uy_d, real *uz_d, real *ux, real *uy, real *uz, real *point_x, real *point_y, real *point_z)
{

    velocity_to_double(dimGrid, dimBlock, Nx, Ny, Nz, ux_hat_d, ux_d, uy_hat_d, uy_d, uz_hat_d, uz_d);
    
    host_device_real_cpy(ux, ux_d, Nx, Ny, Nz);
    host_device_real_cpy(uy, uy_d, Nx, Ny, Nz);
    host_device_real_cpy(uz, uz_d, Nx, Ny, Nz);

    return_physical_vector3(Nx, Ny, Nz, ux, uy, uz, point_x, point_y, point_z, j_fixed,  k_fixed, l_fixed);

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

bool find_intersection(real x_0, real y_0, real z_0, real *x_next, real x_prev, real *y_next, real y_prev, real *z_next, real z_prev, real rhs_x, real rhs_y, real rhs_z, int j_fixed, int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d_plane, cudaComplex *uy_hat_d_plane, cudaComplex *uz_hat_d_plane, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, real *ux_d_plane, real *uy_d_plane, real *uz_d_plane,  real *ux_plane, real *uy_plane, real *uz_plane)
{

    const real rho=1.0e-4;
    bool return_flag=false;

    single_forward_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2, ux_hat_d_3, uy_hat_d_3, uz_hat_d_3, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d, ux_d_plane, uy_d_plane, uz_d_plane, ux_plane, uy_plane, uz_plane, j_fixed, k_fixed, l_fixed, x_next, y_next, z_next);

    real test_vec_x=x_next-x_prev, test_vec_y=y_next-y_prev, test_vec_z=z_next-z_prev;

    //crosses plane
    real sign_1=test_plane_location(rhs_x, rhs_y, rhs_z, x_0, y_0, z_0, x_prev, y_prev, z_prev);
    real sign_2=test_plane_location(rhs_x, rhs_y, rhs_z, x_0, y_0, z_0, x_next, y_next, z_next); 
    
    //in the same direction
    real sign_3=vector3_dot_product(rhs_x, rhs_y, rhs_z, test_vec_x, test_vec_y, test_vec_z); 
    
    real vec_x=xp-x0;
    real vec_y=yp-y0;
    real vec_z=zp-z0;
     //in the ball_rho
    real vec_norm=vector3_norm(vec_x, vec_y, vec_z);

    //if(vec_norm<1.0e-3){
    //    printf("\n %le", vec_norm);
    //}

    if((vec_norm<rho)&&(sign_1*sign_2<0.0)&&(sign_3>0.0)){
        real xn1=x_prev, yn1=y_prev, zn1=z_prev;
        real err=std::fabs(test_plane_location(rhs_x, rhs_y, rhs_z, x_0, y_0, z_0, x_next, y_next, z_next));
        real del=std::fabs(test_plane_location(rhs_x, rhs_y, rhs_z, x_0, y_0, z_0, x_prev, y_prev, z_prev));
        real dt1=dt;
        int iter=0;
        //printf("%i ", j);
        //real xn2=0.0, yn2=0.0, zn2=0.0;
        while((err>1.0e-8)&&(iter<2500)){                        
            iter++;
            //real err1=err;
            xn1=x_prev; yn1=y_prev; zn1=z_prev;
            dt1=dt1*(del)/(del+err);
            //RK_3_SSP(dh1, 0.0, &xn1, &yn1, &zn1, mu, eps);
            

            //ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane

            //construct_physical_vector(dimGrid, dimBlock, Nx, Ny, Nz, j_fixed, k_fixed, l_fixed,  xn1, yn1, zn1, ux_d_plane, uy_d_plane, uz_d_plane, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane);

            single_forward_step(dimGrid, dimBlock, dimGrid_C, dimBlock_C,  dx, dy, dz, /*!*/dt1 /*!*/, Re,  Nx,  Ny,  Nz,  Mz, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, ux_d_plane, uy_d_plane, uz_d_plane,  ux_plane, uy_plane, uz_plane, j_fixed, k_fixed, l_fixed, xn1, yn1, zn1);


            err=std::fabs(test_plane_location(rhs_x, rhs_y, rhs_z, x0, y0, z0, xn1, yn1, zn1));
            //if(err1<err)
            //    break;
            //xn2=xn1; yn2=yn1; zn2=zn1;
        }
        x_next=xn1; y_next=yn1; z_next=zn1;
        return_flag=true;
    }

    return return_flag;
}



void execute_return_map(int j_fixed, int k_fixed, int l_fixed, dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real dt, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_hat_d_1, cudaComplex *uy_hat_d_1, cudaComplex *uz_hat_d_1,  cudaComplex *ux_hat_d_2, cudaComplex *uy_hat_d_2, cudaComplex *uz_hat_d_2,  cudaComplex *ux_hat_d_3, cudaComplex *uy_hat_d_3, cudaComplex *uz_hat_d_3,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d)
{

    int Nrad=30, Nphi=30;
    int number_of_points=Nrad*Nphi;
    real x0, y0, z0;
    real rhs_x, rhs_y, rhs_z;

    real *x_loc, *y_loc, *z_loc, *p_x, *p_y, *p_z, *Matrix;
    real *v_x, *v_y, *v_z;
    real *vx_loc, *vy_loc;
    allocate_real(Nrad, Nphi, 1, 11, &x_loc, &y_loc, &z_loc, &p_x, &p_y, &p_z, &v_x, &v_y, &v_z, &vx_loc, &vy_loc);
    Matrix=allocate_d(3,3,1);
    
    real *ux, *uy, *uz;
    allocate_real(Nx, Ny, Nz, 3, &ux, &uy, &uz);

    cudaComplex *ux_hat_d_plane, *uy_hat_d_plane, *uz_hat_d_plane;
    real *ux_d_plane, *uy_d_plane, *uz_d_plane;


    device_allocate_all_complex(Nx, Ny, Mz, 3, &ux_hat_d_plane, &uy_hat_d_plane, &uz_hat_d_plane);
    device_allocate_all_real(Nx, Ny, Nz, 3, &ux_d_plane, &uy_d_plane, &uz_d_plane);


    //obtaining the RHS vector at a currect solution point
    return_vector3_RHS(dimGrid,  dimBlock,  dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny,  Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane, ux_d_plane, uy_d_plane, uz_d_plane, ux, uy, uz, j_fixed,  k_fixed, l_fixed, rhs_x, rhs_y, rhs_z);

    //get a point of x0,y0,z0 from the solution
    return_vector3_solution(j_fixed,  k_fixed, l_fixed, dimGrid, dimBlock, Nx, Ny, Nz, ux_hat_d, uy_hat_d, uz_hat_d, ux_d_plane, uy_d_plane, uz_d_plane, ux, uy, uz, x_0, y_0, z_0);


    real radius=0.001;
    int size=Nrad*Nphi;
 
    construct_plane_rectangular(Nrad, Nphi, x_loc, y_loc, z_loc, radius, 0.0, 0.0, 0.0);


    rotate_plane(rhs_x, rhs_y, rhs_z, 0.0, 0.0, 1.0, Matrix, size, x_loc, y_loc, z_loc, p_x, p_y, p_z);
    translate_plane(x0, y0, z0, size, p_x, p_y, p_z);


    real tvec_x=p_x[0]-p_x[Nphi*Nrad-3];
    real tvec_y=p_y[0]-p_y[Nphi*Nrad-3];
    real tvec_z=p_z[0]-p_z[Nphi*Nrad-3];
    
    real ivec_x=x_loc[0]-x_loc[Nphi*Nrad-3];
    real ivec_y=y_loc[0]-y_loc[Nphi*Nrad-3];
    real ivec_z=z_loc[0]-z_loc[Nphi*Nrad-3];

    printf("\n[%lf %lf %lf]->([%lf %lf %lf],[%lf %lf %lf]) plane test=%le\n", ivec_x, ivec_y, ivec_z, tvec_x, tvec_y, tvec_z, rhs_x, rhs_y, rhs_z, vector3_dot_product(rhs_x, rhs_y, rhs_z,tvec_x, tvec_y, tvec_z) );


    debug_plot_points("res_3D_0.dat", size, x_loc, y_loc, z_loc);
    debug_plot_points("res_3D.dat", size, p_x, p_y, p_z);
    double scale=0.75*radius;
    debug_plot_vector("normal.dat", x0, y0, z0, rhs_x, rhs_y, rhs_z, scale);
    

    bool stop_flag=false
    int steps=0;
    real x_next=0.0, y_next=0.0, z_next=0.0;
    for (int j = 0; j < number_of_points; ++j){
        x_prev=p_x[j];
        y_prev=p_y[j];
        z_prev=p_z[j];


        construct_physical_vector(dimGrid, dimBlock, Nx, Ny, Nz, j_fixed, k_fixed, l_fixed,  x_prev, y_prev, z_prev, /* original solution */ ux_hat_d, uy_hat_d, uz_hat_d,/* ends */ ux_d_plane, uy_d_plane, uz_d_plane, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane);

        while(!stop_flag){
            stop_flag = find_intersection(x_0, y_0, z_0, x_next, real x_prev, y_next, y_prev, z_next, z_prev, real rhs_x, real rhs_y, real rhs_z, j_fixed, k_fixed, l_fixed, dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny, Nz, Mz, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane, ux_hat_d_1, uy_hat_d_1, uz_hat_d_1,  ux_hat_d_2, uy_hat_d_2, uz_hat_d_2,  ux_hat_d_3, uy_hat_d_3, uz_hat_d_3,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d,  kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, ux_d_plane, uy_d_plane, uz_d_plane,  ux_plane, uy_plane, uz_plane); 

            x_prev=x_next;
            y_prev=y_next;
            z_prev=z_next;

            printf("[ %i ]   \r", steps);
            steps++;
            
        }
        p_x[j]=x_next;
        p_y[j]=y_next;
        p_z[j]=z_next;

    
    }
    


    debug_plot_points("res_3D_1.dat", size, p_x, p_y, p_z);

    
    double test_vec_x0=0.0,test_vec_y0=0.0,test_vec_z0=1.0;
    double test_vec_x,test_vec_y,test_vec_z;
    rotate_plane(rhs_x, rhs_y, rhs_z, 0, 0, 1.0, Matrix, 1, &test_vec_x0, &test_vec_y0, &test_vec_z0, &test_vec_x, &test_vec_y, &test_vec_z);
    
    //translate_plane(x0, y0, z0, 1, &test_vec_x, &test_vec_y, &test_vec_z);
    debug_plot_vector("normal_1.dat", x0, y0, z0, test_vec_x, test_vec_y, test_vec_z, scale);

    debug_plot_vectors("vectors.dat", size, p_x, p_y, p_z, v_x, v_y, v_z, scale);

    device_deallocate_all_complex(3, ux_hat_d_plane, uy_hat_d_plane, uz_hat_d_plane);
    device_deallocate_all_real(3, ux_d_plane, uy_d_plane, uz_d_plane);

    free(ux);
    free(uy);
    free(uz);

    free(x_loc);
    free(y_loc);
    free(p_x);
    free(p_y);
    free(p_z);
    free(Matrix);   
    free(vx_loc);
    free(vy_loc);
    free(v_x);
    free(v_y);
    free(v_z);  

}


