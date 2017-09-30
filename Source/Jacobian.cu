#include "Jacobian.h"


//write Jacobian matrix
void write_file_matrix(char *file_name, real *Matrix, int Nx, int Ny){
    FILE *stream;
    stream=fopen(file_name, "w" );
    double value;
    int NJacobian=Nx;
    for(int j=3;j<Nx;j++){
        for(int k=3;k<Ny;k++){      
            value=(double)Matrix[IJ(k,j)]; //!!! fix the index!
            fprintf(stream, "%.016le ", value); 
        }
        fprintf(stream, "\n");
    }
    
    fclose(stream);

}


__global__ void copy_complex_device(int Nx, int Ny, int Nz, cudaComplex *source_1, cudaComplex *source_2, cudaComplex *source_3, cudaComplex *destination_1, cudaComplex *destination_2, cudaComplex *destination_3){

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


        destination_1[IN(j,k,l)].x=source_1[IN(j,k,l)].x;
        destination_1[IN(j,k,l)].y=source_1[IN(j,k,l)].y;

        destination_2[IN(j,k,l)].x=source_2[IN(j,k,l)].x;
        destination_2[IN(j,k,l)].y=source_2[IN(j,k,l)].y;

        destination_3[IN(j,k,l)].x=source_3[IN(j,k,l)].x;
        destination_3[IN(j,k,l)].y=source_3[IN(j,k,l)].y;



        
    }

}

}



__global__ void increment_jkl_device(int Nx, int Ny, int Nz, cudaComplex *xN_hat_source, cudaComplex *xN_hat_destination, real eps_re, real eps_im, int coordinate_x, int coordinate_y, int coordinate_z){

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

        int flag=0;     
        
        if((coordinate_x==j)&&(coordinate_y==k)&&(coordinate_z==l))
            flag=1;
        else
            flag=0;

        xN_hat_destination[IN(j,k,l)].x=xN_hat_source[IN(j,k,l)].x+eps_re*flag;
        xN_hat_destination[IN(j,k,l)].y=xN_hat_source[IN(j,k,l)].y+eps_im*flag;

        
    }

}

}





__global__ void increment_jkl_on_unit_vector_device(int Nx, int Ny, int Nz, cudaComplex *xN_hat_destination, real eps_re, real eps_im, int coordinate_x, int coordinate_y, int coordinate_z){

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

        int flag=0;     
        
        if((coordinate_x==j)&&(coordinate_y==k)&&(coordinate_z==l))
            flag=1;
        else
            flag=0;

        xN_hat_destination[IN(j,k,l)].x=eps_re*flag;
        xN_hat_destination[IN(j,k,l)].y=eps_im*flag;

        
    }

}

}





__global__ void Diff_RHS_device(int Nx, int Ny, int Nz,  cudaComplex *RHSx_plus, cudaComplex *RHSx_minus, cudaComplex *Diff_RHSx, cudaComplex *RHSy_plus, cudaComplex *RHSy_minus, cudaComplex *Diff_RHSy, cudaComplex *RHSz_plus, cudaComplex *RHSz_minus, cudaComplex *Diff_RHSz, real eps){

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
        
        Diff_RHSx[IN(j,k,l)].x=(RHSx_plus[IN(j,k,l)].x-RHSx_minus[IN(j,k,l)].x)/(2.0*eps);
        Diff_RHSx[IN(j,k,l)].y=(RHSx_plus[IN(j,k,l)].y-RHSx_minus[IN(j,k,l)].y)/(2.0*eps);
        Diff_RHSy[IN(j,k,l)].x=(RHSy_plus[IN(j,k,l)].x-RHSy_minus[IN(j,k,l)].x)/(2.0*eps);
        Diff_RHSy[IN(j,k,l)].y=(RHSy_plus[IN(j,k,l)].y-RHSy_minus[IN(j,k,l)].y)/(2.0*eps);  
        Diff_RHSz[IN(j,k,l)].x=(RHSz_plus[IN(j,k,l)].x-RHSz_minus[IN(j,k,l)].x)/(2.0*eps);
        Diff_RHSz[IN(j,k,l)].y=(RHSz_plus[IN(j,k,l)].y-RHSz_minus[IN(j,k,l)].y)/(2.0*eps);

    }

}
}



__global__ void Jacobian_per_Raw_device(int Nx, int Ny, int Nz, cudaComplex *Diff_RHSx,  cudaComplex *Diff_RHSy, cudaComplex *Diff_RHSz, real *Jacobian_d, int Jacobian_Row_index){
    
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

        int submatrix_size=3; // 1*3 - for Re or Im only
        int index=IN(j,k,l);
        int NJacobian=Nx*Ny*Nz*submatrix_size; 
        
        int Jacobian_Row=Jacobian_Row_index;

        real factor=1.0, add_value=0.0;//Jacobian_Row_index;
        

        int j_Jacobian=submatrix_size*index+0;
    
        //Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSx[index].x*factor+add_value;
        
        //j_Jacobian=submatrix_size*index+1;
        Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSx[index].y*factor+add_value;
        
        //j_Jacobian=submatrix_size*index+2;
        //Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSy[index].x*factor+add_value;
        
        j_Jacobian=submatrix_size*index+1;
        Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSy[index].y*factor+add_value;
        
        //j_Jacobian=submatrix_size*index+4;
        //Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSz[index].x*factor+add_value;
        
        j_Jacobian=submatrix_size*index+2;
        Jacobian_d[IJ(Jacobian_Row,j_Jacobian)]=Diff_RHSz[index].y*factor+add_value;

        __syncthreads();
        
    }
}

}






void build_Jacobian(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *U_eps_d, cudaComplex *RHSx_plus, cudaComplex *RHSx_minus, cudaComplex *RHSy_plus, cudaComplex *RHSy_minus, cudaComplex *RHSz_plus, cudaComplex *RHSz_minus, cudaComplex *Diff_RHSx, cudaComplex *Diff_RHSy, cudaComplex *Diff_RHSz, real *Jacobian_d){

    real eps=1.0e-5;



    for(int j=0;j<Nx;j++){
    for(int k=0;k<Ny;k++){
    for(int l=0;l<Mz;l++){
        //int jkl_index=IM(j,k,l);
        int jkl_index=(j)*(Ny*Mz)+(k)*(Mz)+(l);
        
        if(jkl_index>=Nx*Ny*Mz)
            printf("ind: j=%d, k=%d, l=%d \n", j,k,l);

        if((jkl_index*6+5)>=Nx*Ny*Mz*6)
            printf("Jind: j=%d, k=%d, l=%d \n", j,k,l);

        //New variables: cudaComplex *U_eps_d, *RHSx_plus, *RHSx_minus, *RHSy_plus, *RHSy_minus, *RHSz_plus, *RHSz_minus, Diff_RHSx, Diff_RHSy, Diff_RHSz
        //  real *Jacobian_d             

        //ux, Re
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, U_eps_d, eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, U_eps_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, U_eps_d, -eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, U_eps_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);
        //Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        //index of this very instance used to be: t*8+0);
        //Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+0);
        //cudaDeviceSynchronize();

        //ux, Im
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, U_eps_d, 0.0, eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, U_eps_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, U_eps_d, 0.0, -eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, U_eps_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);
        Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+0);
        cudaDeviceSynchronize();

        //uy, Re
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uy_hat_d, U_eps_d, eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uy_hat_d, U_eps_d, -eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        //Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        //Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+1);
        //cudaDeviceSynchronize();

        //uy, Im
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uy_hat_d, U_eps_d, 0.0, eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uy_hat_d, U_eps_d, 0.0, -eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+1);
         cudaDeviceSynchronize();

        //uz, Re
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        //increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, -eps, 0.0, j, k, l);
        //return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        //Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        //Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*6+4);
        //cudaDeviceSynchronize();
        
        //uz, Im
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, 0.0, eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, 0.0, -eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+2);
        cudaDeviceSynchronize();
        
        //printf("[%.03f\%]  \r",100*(IM(j,k,l))/(1.0*Nx*Ny*Nz));   

    }
    }
        printf("[%.03f\%]  \n",100.0*(j+1)/(1.0*Nx));   
    }

    
}




void build_variational_Jacobian(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d, cudaComplex *U_eps_d, cudaComplex *RHSx_plus, cudaComplex *RHSx_minus, cudaComplex *RHSy_plus, cudaComplex *RHSy_minus, cudaComplex *RHSz_plus, cudaComplex *RHSz_minus, cudaComplex *Diff_RHSx, cudaComplex *Diff_RHSy, cudaComplex *Diff_RHSz, real *Jacobian_d){



    for(int j=0;j<Nx;j++){
    for(int k=0;k<Ny;k++){
    for(int l=0;l<Mz;l++){
        //int jkl_index=IM(j,k,l);
        int jkl_index=(j)*(Ny*Mz)+(k)*(Mz)+(l);
        
        if(jkl_index>=Nx*Ny*Mz)
            printf("ind: j=%d, k=%d, l=%d \n", j,k,l);

        if((jkl_index*6+5)>=Nx*Ny*Mz*6)
            printf("Jind: j=%d, k=%d, l=%d \n", j,k,l);

        //New variables: cudaComplex *U_eps_d, *RHSx_plus, *RHSx_minus, *RHSy_plus, *RHSy_minus, *RHSz_plus, *RHSz_minus, Diff_RHSx, Diff_RHSy, Diff_RHSz
        //  real *Jacobian_d             

        increment_jkl_on_unit_vector_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, U_zero_d, 0.0, 0.0, j, k, l);

        //ux, Im


        increment_jkl_on_unit_vector_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, U_eps_d, 0.0, eps, j, k, l);

        //!! these vectors are overwritten!
        RK3_SSP_UV_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, dt, Re, Nx, Ny,Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  !!vx_hat_d, !!vy_hat_d, !!vz_hat_d, ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d, uy_hat_d, uz_hat_d,  ux_hat_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d,  ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d);


        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+0);

        //uy, Im
        increment_jkl_on_unit_vector_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, U_eps_d, 0.0, eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uy_hat_d, U_eps_d, 0.0, -eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, U_eps_d, uz_hat_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+1);

        //uz, Im
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, 0.0, eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_plus, RHSy_plus, RHSz_plus);
        increment_jkl_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, uz_hat_d, U_eps_d, 0.0, -eps, j, k, l);
        return_RHS(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, U_eps_d, fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d,  AM_12_d, AM_13_d, AM_23_d, RHSx_minus, RHSy_minus, RHSz_minus);

        Diff_RHS_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, RHSx_plus, RHSx_minus, Diff_RHSx, RHSy_plus, RHSy_minus, Diff_RHSy, RHSz_plus, RHSz_minus, Diff_RHSz, eps);
        Jacobian_per_Raw_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d, jkl_index*3+2);
        

    }
    }
        printf("[%.03f\%]  \n",100.0*(j+1)/(1.0*Nx));   
    }

    
}



void print_Jacobian(dim3 dimGrid, dim3 dimBlock, dim3 dimGrid_C, dim3 dimBlock_C, real dx, real dy, real dz, real Re, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d,  cudaComplex *fx_hat_d, cudaComplex *fy_hat_d, cudaComplex *fz_hat_d, cudaComplex *Qx_hat_d, cudaComplex *Qy_hat_d, cudaComplex *Qz_hat_d, cudaComplex *div_hat_d, real* kx_nabla_d, real* ky_nabla_d, real *kz_nabla_d, real *din_diffusion_d, real *din_poisson_d, real *AM_11_d, real *AM_22_d, real *AM_33_d,  real *AM_12_d, real *AM_13_d, real *AM_23_d){

    real *Jacobian;
    real *Jacobian_d;
    int submatrix_size=3; // 1*3 - for Re or Im only
    int NJacobian=Nx*Ny*Mz*submatrix_size; 

    cudaComplex *U_eps_d, *RHSx_plus, *RHSx_minus, *RHSy_plus, *RHSy_minus, *RHSz_plus, *RHSz_minus, *Diff_RHSx, *Diff_RHSy, *Diff_RHSz;

    allocate_real(NJacobian, NJacobian, 1, 1, &Jacobian);
    device_allocate_all_real(NJacobian, NJacobian, 1, 1, &Jacobian_d);
    device_allocate_all_complex(Nx, Ny, Mz, 10, &U_eps_d, &RHSx_plus, &RHSx_minus, &RHSy_plus, &RHSy_minus, &RHSz_plus, &RHSz_minus, &Diff_RHSx, &Diff_RHSy, &Diff_RHSz);
    
    cudaDeviceSynchronize();    

//  copy_complex_device<<<dimGrid_C, dimBlock_C>>>(Nx,Ny,Mz,ux_hat_d,uy_hat_d,uz_hat_d,RHSx_plus,RHSy_plus,RHSz_plus);

//  copy_complex_device<<<dimGrid_C, dimBlock_C>>>(Nx,Ny,Mz,fx_hat_d,fy_hat_d,fz_hat_d,RHSx_minus,RHSy_minus,RHSz_minus);




    build_Jacobian(dimGrid, dimBlock, dimGrid_C, dimBlock_C, dx, dy, dz, Re, Nx, Ny, Nz, Mz, ux_hat_d, uy_hat_d, uz_hat_d,  fx_hat_d, fy_hat_d, fz_hat_d, Qx_hat_d, Qy_hat_d, Qz_hat_d, div_hat_d, kx_nabla_d, ky_nabla_d, kz_nabla_d, din_diffusion_d, din_poisson_d, AM_11_d, AM_22_d, AM_33_d, AM_12_d, AM_13_d, AM_23_d, U_eps_d, RHSx_plus, RHSx_minus, RHSy_plus, RHSy_minus, RHSz_plus, RHSz_minus, Diff_RHSx, Diff_RHSy, Diff_RHSz, Jacobian_d);

    host_device_real_cpy(Jacobian, Jacobian_d, NJacobian, NJacobian, 1);

    device_deallocate_all_complex(10, U_eps_d, RHSx_plus, RHSx_minus, RHSy_plus, RHSy_minus, RHSz_plus, RHSz_minus, Diff_RHSx, Diff_RHSy, Diff_RHSz);
    device_deallocate_all_real(1, Jacobian_d);

    write_file_matrix("Jacobian.dat", Jacobian, NJacobian, NJacobian);

    printf("Jacobian size is %i X %i\n", NJacobian, NJacobian);
    deallocate_real(1, Jacobian);
    
}



