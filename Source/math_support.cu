#include "math_support.h"


//random normal distribution
double rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}



//cuda math common functions



__global__ void init_fources_fourier_device(int Nx, int Ny, int Nz, cudaComplex *f_hat_x, cudaComplex *f_hat_y, cudaComplex *f_hat_z){
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
        int n=1;

        f_hat_x[IN(j,k,l)].x=0.0;
        f_hat_x[IN(j,k,l)].y=0.0;
        f_hat_y[IN(j,k,l)].x=0.0;
        f_hat_y[IN(j,k,l)].y=0.0;
        f_hat_z[IN(j,k,l)].x=0.0;
        f_hat_z[IN(j,k,l)].y=0.0;

//Re:
        //f_hat_x[IN(0,n,n)].x=(Nx)*(Ny)*(Nz)*0.25;
                //f_hat_x[IN(0,Ny-n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;
        //f_hat_x[IN(0,Ny-n,n)].x=-(Nx)*(Ny)*(Nz)*0.25;
                //f_hat_x[IN(0,n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;
//Im:
        f_hat_x[IN(0,n,n)].y=-(Nx)*(Ny)*(Nz)*0.25;
        //f_hat_x[IN(0,Ny-n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;
        f_hat_x[IN(0,Ny-n,n)].y=-(Nx)*(Ny)*(Nz)*0.25;
        //f_hat_x[IN(0,n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;


    //  f_hat_x[IN(0,n,n)].y=-(Nx)*(Ny)*(Nz)*0.25;
        //f_hat_x[IN(0,Ny-n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;
    //  f_hat_x[IN(0,Ny-n,n)].y=-(Nx)*(Ny)*(Nz)*0.25;
        //f_hat_x[IN(0,n,Nz-n)].y=(Nx)*(Ny)*(Nz)*0.25;

    }
}
}



__global__ void velocity_to_abs_device(int Nx, int Ny, int Nz, real* ux_d, real* uy_d, real*  uz_d, real*  u_abs_d){
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
        //uz_d[IN(j,k,l)]=-uz_d[IN(j,k,l)];

        u_abs_d[IN(j,k,l)]=ux_d[IN(j,k,l)]*ux_d[IN(j,k,l)]+uy_d[IN(j,k,l)]*uy_d[IN(j,k,l)]+uz_d[IN(j,k,l)]*uz_d[IN(j,k,l)];
        //divide by number of elements to recover value
    }
}
}






__global__ void scale_double(real* f, int Nx, int Ny, int Nz){
unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * block_size_y + threadIdx.y )*block_size_x + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/Nz; 
zIndex = index_in - Nz*t1 ;
xIndex =  t1/Ny; 
yIndex = t1 - Ny * xIndex ;


//remove arbitrary constant!    
//real constant=fc[0].x/(1.0*Nx*Ny*Nz);
//divide by number of elements to recover value

unsigned int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){
        
        f[IN(j,k,l)]*=1.0/(1.0*Nx*Ny*Nz);
        //divide by number of elements to recover value
    }
}
}




__global__ void all_Fourier2double_device(cudaComplex *ux_hat_d, real* ux_Re_d, real* ux_Im_d, cudaComplex *uy_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uz_hat_d, real* uz_Re_d, real* uz_Im_d, int Nx, int Ny, int Nz){
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
        

        ux_Re_d[IN(j,k,l)]=ux_hat_d[IN(j,k,l)].x;
        ux_Im_d[IN(j,k,l)]=ux_hat_d[IN(j,k,l)].y;


        uy_Re_d[IN(j,k,l)]=uy_hat_d[IN(j,k,l)].x;
        uy_Im_d[IN(j,k,l)]=uy_hat_d[IN(j,k,l)].y;


        uz_Re_d[IN(j,k,l)]=uz_hat_d[IN(j,k,l)].x;
        uz_Im_d[IN(j,k,l)]=uz_hat_d[IN(j,k,l)].y;


    }
}
}



__global__ void all_double2Fourier_device(real* ux_Re_d, real* ux_Im_d, cudaComplex *ux_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uy_hat_d,  real* uz_Re_d, real* uz_Im_d, cudaComplex *uz_hat_d, int Nx, int Ny, int Nz){
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
        
        ux_hat_d[IN(j,k,l)].x=ux_Re_d[IN(j,k,l)];
        ux_hat_d[IN(j,k,l)].y=ux_Im_d[IN(j,k,l)];


        uy_hat_d[IN(j,k,l)].x=uy_Re_d[IN(j,k,l)];
        uy_hat_d[IN(j,k,l)].y=uy_Im_d[IN(j,k,l)];


        uz_hat_d[IN(j,k,l)].x=uz_Re_d[IN(j,k,l)];
        uz_hat_d[IN(j,k,l)].y=uz_Im_d[IN(j,k,l)];


    }
}
}





void all_Fourier2double(dim3 dimGrid_C, dim3 dimBlock_C, cudaComplex *ux_hat_d, real* ux_Re_d, real* ux_Im_d, cudaComplex *uy_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uz_hat_d, real* uz_Re_d, real* uz_Im_d, int Nx, int Ny, int Nz){


    all_Fourier2double_device<<<dimGrid_C, dimBlock_C>>>(ux_hat_d, ux_Re_d, ux_Im_d, uy_hat_d, uy_Re_d, uy_Im_d, uz_hat_d, uz_Re_d, uz_Im_d, Nx, Ny, Nz);

}




void all_double2Fourier(dim3 dimGrid_C, dim3 dimBlock_C, real* ux_Re_d, real* ux_Im_d, cudaComplex *ux_hat_d, real* uy_Re_d, real* uy_Im_d, cudaComplex *uy_hat_d,  real* uz_Re_d, real* uz_Im_d, cudaComplex *uz_hat_d, int Nx, int Ny, int Nz){


    all_double2Fourier_device<<<dimGrid_C, dimBlock_C>>>(ux_Re_d, ux_Im_d, ux_hat_d, uy_Re_d, uy_Im_d, uy_hat_d,  uz_Re_d, uz_Im_d, uz_hat_d, Nx, Ny, Nz);

}


 static const char *_cudaGetErrorEnum(cufftResult error)
    {
        switch (error)
        {
            case CUFFT_SUCCESS:
                return "The cuFFT operation was successful";

            case CUFFT_INVALID_PLAN:
                return "cuFFT was passed an invalid plan handle";

            case CUFFT_ALLOC_FAILED:
                return "cuFFT failed to allocate GPU or CPU memory";

            case CUFFT_INVALID_TYPE:
                return "Invalid type";

            case CUFFT_INVALID_VALUE:
                return "User specified an invalid pointer or parameter";

            case CUFFT_INTERNAL_ERROR:
                return "Driver or internal cuFFT library error";

            case CUFFT_EXEC_FAILED:
                return "Failed to execute an FFT on the GPU";

            case CUFFT_SETUP_FAILED:
                return "The cuFFT library failed to initialize";

            case CUFFT_INVALID_SIZE:
                return "User specified an invalid transform size";

            case CUFFT_UNALIGNED_DATA:
                return "Data is unaligned";

            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "Missing parameters in call";
            
            case CUFFT_INVALID_DEVICE:
                return "Execution of a plan was on different GPU than plan creation";
            
            case CUFFT_PARSE_ERROR:
                return "Internal plan database error";

            case CUFFT_NO_WORKSPACE:
                return "No workspace has been provided prior to plan execution";

        }

        return "<unknown>";
    }



void FFTN_Device(real *source, cudaComplex *destination){
cufftResult result;


    result=cufftExecRtoC(planR2C_local, source, destination);
    if (result != CUFFT_SUCCESS) { 
        fprintf (stderr,"*CUFFT R->C* failed: %s. \n", _cudaGetErrorEnum(result)); 
        exit(1);
        return; 
        
    }
}


void iFFTN_Device(cudaComplex *source, real *destination){
cufftResult result;
    result=cufftExecCtoR(planC2R_local, source, destination);
    if (result != CUFFT_SUCCESS) { 
        fprintf (stderr,"*CUFFT C->R* failed: %s.\n", _cudaGetErrorEnum(result)); 
        exit(1);
        return; 
    }
}

void iFFTN_Device(dim3 dimGrid, dim3 dimBlock, cudaComplex *source, real *destination, int Nx, int Ny, int Nz){
cufftResult result;
    result=cufftExecCtoR(planC2R_local, source, destination);
    if (result != CUFFT_SUCCESS) { 
        fprintf (stderr,"*CUFFT C->R* failed: %s.\n", _cudaGetErrorEnum(result)); 
        return; 
    }

    scale_double<<<dimGrid, dimBlock>>>(destination, Nx, Ny, Nz);
}





void init_fft_plans(cufftHandle planR2C_l, cufftHandle planC2R_l){

    planR2C_local=planR2C_l;
    planC2R_local=planC2R_l;

}



//host math functions
void build_Laplace_Wavenumbers(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *kx_laplace, real *ky_laplace, real *kz_laplace){

    for(int j=0;j<Nx; j++){
        int m=j;
        if(j>=Nx/2)
            m=j-Nx;
        kx_laplace[j]=2.0*M_PI/Lx*m;
    }
    for(int k=0;k<Ny; k++){
        int n=k;
        if(k>=Ny/2)
            n=k-Ny;
        ky_laplace[k]=2.0*M_PI/Ly*n;
    }
    for(int l=0;l<Nz; l++){
        int q=l;
        //if(l>=Nz/2)       due to reality condition
        //  q=l-Nz;
        kz_laplace[l]=2.0*M_PI/Lz*q;
    }

}


void build_Laplace_and_Diffusion(int Nx, int Ny, int Nz, real *din_poisson, real *din_diffusion, real *kx_laplace, real *ky_laplace, real *kz_laplace){

    for(int l=0;l<Nz;l++){  
        for(int j=0;j<Nx;j++){
            for(int k=0;k<Ny;k++){
                din_poisson[IN(j,k,l)]=-(kx_laplace[j]*kx_laplace[j]+ky_laplace[k]*ky_laplace[k]+kz_laplace[l]*kz_laplace[l]);

                din_diffusion[IN(j,k,l)]=-din_poisson[IN(j,k,l)];


            }
        }
    }   

    din_poisson[IN(0,0,0)]=1.0;

}


void build_Nabla_Wavenumbers(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *kx_nabla, real *ky_nabla, real *kz_nabla){

    for(int j=0;j<Nx; j++){
        int m=j;
        if(j>=Nx/2)
            m=j-Nx;
        kx_nabla[j]=2.0*M_PI/Lx*m;
    }
    for(int k=0;k<Ny; k++){
        int n=k;
        if(k>=Ny/2)
            n=k-Ny;
        ky_nabla[k]=2.0*M_PI/Ly*n;
    }
    for(int l=0;l<Nz; l++){
        int q=l;
        //if(l>=Nz/2)  Due to reality condition
        //  q=l-Nz;
        kz_nabla[l]=2.0*M_PI/Lz*q;
    }

}



void build_projection_matrix_elements(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *AM_11, real *AM_22, real *AM_33, real *AM_12, real *AM_13, real *AM_23){

int q,m,n;

    for(int l=0;l<Nz;l++){  
    q=l;
    //if(l>=Nz/2)   Due to reality condition
    //  q=l-Nz;
        
        for(int j=0;j<Nx;j++){
        m=j;
        if(j>=Nx/2)
            m=j-Nx;
            
            for(int k=0;k<Ny;k++){
            n=k;
            if(k>=Ny/2)
                n=k-Ny; 

                real k1=2.0*M_PI/Lx;
                real k2=2.0*M_PI/Ly;
                real k3=2.0*M_PI/Lz;
                // m - x direction; n - y direction; q - z direction.
                real din=k1*k1*m*m+k2*k2*n*n+k3*k3*q*q;
                if((m*m+n*n+q*q)==0) din=1.0;

                //diagonal
                AM_11[IN(j,k,l)]=k1*k1*m*m/din;
                AM_22[IN(j,k,l)]=k2*k2*n*n/din;
                AM_33[IN(j,k,l)]=k3*k3*q*q/din;

                //of diagonal
                AM_12[IN(j,k,l)]=k1*k2*m*n/din;
                AM_13[IN(j,k,l)]=k1*k3*m*q/din;
                AM_23[IN(j,k,l)]=k2*k3*n*q/din;

                // AM_21, AM_31, AM_32 are symmetric!

            }
        }
    }   

    AM_11[IN(0,0,0)]=0.0;
    AM_22[IN(0,0,0)]=0.0;
    AM_33[IN(0,0,0)]=0.0;
    AM_12[IN(0,0,0)]=0.0;
    AM_13[IN(0,0,0)]=0.0;
    AM_23[IN(0,0,0)]=0.0;   
}




void build_mask_matrix(int Nx, int Ny, int Nz, real Lx, real Ly, real Lz, real *mask_2_3){

int q,m,n;

    for(int l=0;l<Nz;l++){  
    q=l;
    //if(l>=Nz/2)  Due to reality condition
    //  q=l-Nz;
        
        for(int j=0;j<Nx;j++){
        m=j;
        if(j>=Nx/2)
            m=j-Nx;
            
            for(int k=0;k<Ny;k++){
            n=k;
            if(k>=Ny/2)
                n=k-Ny; 

                real kx=m*2*PI/Lx;
                real ky=n*2*PI/Ly;
                real kz=q*2*PI/Lz;
                real kxMax=2*Nx*PI/Lx;
                real kyMax=2*Ny*PI/Ly;
                real kzMax=2*Nz*PI/Lz;

                real sphere2=sqrt(kx*kx/(kxMax*kxMax)+ky*ky/(kyMax*kyMax)+kz*kz/(kzMax*kzMax));

                //  2/3 limitation!
                if(sphere2<2.0/3.0){
                    mask_2_3[IN(j,k,l)]=1.0;
                }
                else{
                    mask_2_3[IN(j,k,l)]=0.0; //!
                }
                //exponental limitation!
                double alpha=36.0;
                int power=36;
                //mask_2_3[IN(j,k,l)]=sphere2;//
                //mask_2_3[IN(j,k,l)]=exp(-alpha*pow(sphere2,power));

            }
        }
    }   
    
    //remove 0,0,0 wave from calculation!
    mask_2_3[IN(0,0,0)]=0.0;

}



real TotalEnergy(int Nx, int Ny, int Nz, real *ux, real *uy, real *uz, real dx, real dy, real dz, real alpha, real beta){
real energy=0.0;
    for(int l=0;l<Nz;l++){  
        for(int j=0;j<Nx;j++){
            for(int k=0;k<Ny;k++){
                
                energy+=(ux[IN(j,k,l)]*ux[IN(j,k,l)]+uy[IN(j,k,l)]*uy[IN(j,k,l)]+uz[IN(j,k,l)]*uz[IN(j,k,l)])*dx*dy*dz;

            }
        }
    }
    return(alpha*beta*energy/(4.0*2.0*PI));
}


real TotalDissipation(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz,real dx, real dy, real dz,real alpha, real beta, real Re){
real diss_energy=0.0;
    for(int l=0;l<Nz;l++){  
        for(int j=0;j<Nx;j++){
            for(int k=0;k<Ny;k++){
            
                real ux_x=(ux[I3(j+1,k,l)]-ux[I3(j-1,k,l)])/dx;
                real ux_y=(ux[I3(j,k+1,l)]-ux[I3(j,k-1,l)])/dy; 
                real ux_z=(ux[I3(j,k,l+1)]-ux[I3(j,k,l-1)])/dz;

                real uy_x=(uy[I3(j+1,k,l)]-uy[I3(j-1,k,l)])/dx;
                real uy_y=(uy[I3(j,k+1,l)]-uy[I3(j,k-1,l)])/dy; 
                real uy_z=(uy[I3(j,k,l+1)]-uy[I3(j,k,l-1)])/dz;
                
                real uz_x=(uz[I3(j+1,k,l)]-uz[I3(j-1,k,l)])/dx;
                real uz_y=(uz[I3(j,k+1,l)]-uz[I3(j,k-1,l)])/dy; 
                real uz_z=(uz[I3(j,k,l+1)]-uz[I3(j,k,l-1)])/dz;

                diss_energy+=dx*dy*dz*(2.0*(ux_x*ux_x+uy_y*uy_y+uz_z*uz_z)+(uy_z+uz_y)*(uy_z+uz_y)+(ux_z+uz_x)*(ux_z+uz_x)+(ux_y+uy_x)*(ux_y+uy_x));

            }
        }
    }
    return(alpha*diss_energy/(8.0*PI*PI*PI));

}



__global__ void get_high_wavenumbers_device(int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_red_hat_d, cudaComplex *uy_red_hat_d, cudaComplex *uz_red_hat_d, int delta){
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

        ux_red_hat_d[IN(j,k,l)].x=0.0;
        ux_red_hat_d[IN(j,k,l)].y=0.0;
        
        uy_red_hat_d[IN(j,k,l)].x=0.0;
        uy_red_hat_d[IN(j,k,l)].y=0.0;      
        
        uz_red_hat_d[IN(j,k,l)].x=0.0;
        uz_red_hat_d[IN(j,k,l)].y=0.0;
    
        if( ((j>0.5*Nx-delta)&&(j<=0.5*Nx+delta))&&((k>0.5*Ny-delta)&&(k<=0.5*Ny+delta))&&((l>0.5*Nz-delta)&&(l<=0.5*Nz+delta)) ){
                ux_red_hat_d[IN(j,k,l)].x=ux_hat_d[IN(j,k,l)].x;
                ux_red_hat_d[IN(j,k,l)].y=ux_hat_d[IN(j,k,l)].y;
        
                uy_red_hat_d[IN(j,k,l)].x=uy_hat_d[IN(j,k,l)].x;
                uy_red_hat_d[IN(j,k,l)].y=uy_hat_d[IN(j,k,l)].y;        
        
                uz_red_hat_d[IN(j,k,l)].x=uz_hat_d[IN(j,k,l)].x;
                uz_red_hat_d[IN(j,k,l)].y=uz_hat_d[IN(j,k,l)].y;


            }
    

    }

}
}


__global__ void get_low_wavenumbers_device(int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_red_hat_d, cudaComplex *uy_red_hat_d, cudaComplex *uz_red_hat_d, int delta){
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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){

        ux_red_hat_d[IN(j,k,l)].x=0.0;
        ux_red_hat_d[IN(j,k,l)].y=0.0;
        
        uy_red_hat_d[IN(j,k,l)].x=0.0;
        uy_red_hat_d[IN(j,k,l)].y=0.0;      
        
        uz_red_hat_d[IN(j,k,l)].x=0.0;
        uz_red_hat_d[IN(j,k,l)].y=0.0;
    
        if( ((j<delta)||(j>=Nx-delta))&&((k<delta)||(k>=Ny-delta))&&((l<delta)||(l>=Nz-delta)) ){
                ux_red_hat_d[IN(j,k,l)].x=ux_hat_d[IN(j,k,l)].x;
                ux_red_hat_d[IN(j,k,l)].y=ux_hat_d[IN(j,k,l)].y;
        
                uy_red_hat_d[IN(j,k,l)].x=uy_hat_d[IN(j,k,l)].x;
                uy_red_hat_d[IN(j,k,l)].y=uy_hat_d[IN(j,k,l)].y;        
        
                uz_red_hat_d[IN(j,k,l)].x=uz_hat_d[IN(j,k,l)].x;
                uz_red_hat_d[IN(j,k,l)].y=uz_hat_d[IN(j,k,l)].y;


            }
    

    }

}
}




void get_high_wavenumbers(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_red_hat_d, cudaComplex *uy_red_hat_d, cudaComplex *uz_red_hat_d, cudaComplex *u_temp_complex_d, real *ux_red_d, real *uy_red_d, real *uz_red_d, int delta){


    get_low_wavenumbers_device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, uy_hat_d, uz_hat_d, ux_red_hat_d, uy_red_hat_d, uz_red_hat_d, delta);

    iFFTN_Device(dimGrid, dimBlock, ux_red_hat_d, ux_red_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uz_red_hat_d, uz_red_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uy_red_hat_d, uy_red_d, Nx, Ny, Nz);



}

__global__ void get_curl_device(int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* rot_x_d, real* rot_y_d, real* rot_z_d){
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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){

        rot_x_d[I3(j,k,l)]=0.5*(uz_d[I3(j,k+1,l)]-uz_d[I3(j,k-1,l)])/dy-0.5*(uy_d[I3(j,k,l+1)]-uy_d[I3(j,k,l-1)])/dz;

        rot_y_d[I3(j,k,l)]=0.5*(ux_d[I3(j,k,l+1)]-ux_d[I3(j,k,l-1)])/dz-0.5*(uz_d[I3(j+1,k,l)]-ux_d[I3(j-1,k,l)])/dx;

        rot_z_d[I3(j,k,l)]=0.5*(uy_d[I3(j+1,k,l)]-uy_d[I3(j-1,k,l)])/dx-0.5*(ux_d[I3(j,k+1,l)]-ux_d[I3(j,k-1,l)])/dy;

    }

}
}


void get_curl(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* rot_x_d, real* rot_y_d, real* rot_z_d){


    get_curl_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, rot_x_d, rot_y_d, rot_z_d);

}


__global__ void get_kinetic_energy_device(int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* energy){

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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){


        //energy[IN(j,k,l)]=0.5*(ux_d[IN(j,k,l)]*ux_d[IN(j,k,l)]+uy_d[IN(j,k,l)]*uy_d[IN(j,k,l)]+uz_d[IN(j,k,l)]*uz_d[IN(j,k,l)])*dx*dy*dz;
        //using energy of the flow, thransversal to the base one.
        energy[IN(j,k,l)]=0.5*(uy_d[IN(j,k,l)]*uy_d[IN(j,k,l)]+uz_d[IN(j,k,l)]*uz_d[IN(j,k,l)])*dx*dy;



    }
}

}



__global__ void get_dissipation_device(int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* dissipation){

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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){


                real ux_x=(ux_d[I3(j+1,k,l)]-ux_d[I3(j-1,k,l)])/dx;
                real ux_y=(ux_d[I3(j,k+1,l)]-ux_d[I3(j,k-1,l)])/dy; 
                real ux_z=(ux_d[I3(j,k,l+1)]-ux_d[I3(j,k,l-1)])/dz;

                real uy_x=(uy_d[I3(j+1,k,l)]-uy_d[I3(j-1,k,l)])/dx;
                real uy_y=(uy_d[I3(j,k+1,l)]-uy_d[I3(j,k-1,l)])/dy; 
                real uy_z=(uy_d[I3(j,k,l+1)]-uy_d[I3(j,k,l-1)])/dz;
                
                real uz_x=(uz_d[I3(j+1,k,l)]-uz_d[I3(j-1,k,l)])/dx;
                real uz_y=(uz_d[I3(j,k+1,l)]-uz_d[I3(j,k-1,l)])/dy; 
                real uz_z=(uz_d[I3(j,k,l+1)]-uz_d[I3(j,k,l-1)])/dz;

                dissipation[IN(j,k,l)]=dx*dy*dz*(2.0*(ux_x*ux_x+uy_y*uy_y+uz_z*uz_z)+(uy_z+uz_y)*(uy_z+uz_y)+(ux_z+uz_x)*(ux_z+uz_x)+(ux_y+uy_x)*(ux_y+uy_x));



    }
}

}


real get_kinetic_energy(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* energy, real* energy_out, real* energy_out1){

get_kinetic_energy_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, energy);

return reduction_sum(Nx*Ny*Nz, energy, energy_out, energy_out1);


}


real get_dissipation(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real dx, real dy, real dz, real* ux_d, real* uy_d, real* uz_d, real* dissipation, real* energy_out, real* energy_out1){

get_dissipation_device<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, dx, dy, dz, ux_d, uy_d, uz_d, dissipation);

return reduction_sum(Nx*Ny*Nz, dissipation, energy_out, energy_out1);


}


void calculate_energy_spectrum(char* file_name, int Nx, int Ny, int Mz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im){

int Nz=Mz;
real *Energy_spectrum;
    int size=(Nx/2+Ny/2+Nz);
    int norm=Nx*Ny*Nz;
    Energy_spectrum=(real*)malloc(sizeof(real)*size*3);
    int n=0;
    for(int z=0;z<size*3;z++)
        Energy_spectrum[z]=0.0;


    for(int j=0;j<Nx/2;j++)
    for(int k=0;k<Ny/2;k++)
    for(int l=0;l<Nz;l++){
        n=j+k+l;
        Energy_spectrum[n+size*0]=n;//sqrt(j*j+k*k+l*l);//n;
        // (a+ib)(a-ib)=a^2+b^2
        Energy_spectrum[n+size*1]+=ux_hat_Re[IN(j,k,l)]*ux_hat_Re[IN(j,k,l)]/norm/norm+ux_hat_Im[IN(j,k,l)]*ux_hat_Im[IN(j,k,l)]/norm/norm;
        Energy_spectrum[n+size*1]+=uy_hat_Re[IN(j,k,l)]*uy_hat_Re[IN(j,k,l)]/norm/norm+uy_hat_Im[IN(j,k,l)]*uy_hat_Im[IN(j,k,l)]/norm/norm;
        Energy_spectrum[n+size*1]+=uz_hat_Re[IN(j,k,l)]*uz_hat_Re[IN(j,k,l)]/norm/norm+uz_hat_Im[IN(j,k,l)]*uz_hat_Im[IN(j,k,l)]/norm/norm;


//      Energy_spectrum[n+size*2]+=ux_hat_Re[IN(j,k,l)]*uy_hat_Re[IN(j,k,l)]/norm/norm+ux_hat_Im[IN(j,k,l)]*uy_hat_Im[IN(j,k,l)]/norm/norm;
//      Energy_spectrum[n+size*2]+=uy_hat_Re[IN(j,k,l)]*uz_hat_Re[IN(j,k,l)]/norm/norm+uy_hat_Im[IN(j,k,l)]*uz_hat_Im[IN(j,k,l)]/norm/norm;
//      Energy_spectrum[n+size*2]+=uz_hat_Re[IN(j,k,l)]*ux_hat_Re[IN(j,k,l)]/norm/norm+uz_hat_Im[IN(j,k,l)]*ux_hat_Im[IN(j,k,l)]/norm/norm;
        Energy_spectrum[n+size*2]=0.5*pow((1.0*n),-5.0/3.0);

    }


    FILE *stream;
    stream=fopen(file_name, "w" );
    
    for(int n=0;n<size;n++)
        fprintf(stream, "%i %.16le %.16le\n",(int)Energy_spectrum[n+size*0],  0.5*Energy_spectrum[n+size*1],0.5*Energy_spectrum[n+size*2]); 
    
    fclose(stream);

    free(Energy_spectrum);

}





__global__ void Helmholz_Fourier_Filter_Device(int Nx, int Ny, int Nz,  real Lx, real Ly, real Lz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real filter_eps){
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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){

        int m=j;
        if(j>=Nx/2)
            m=j-Nx;
        int n=k;
        if(k>=Ny/2)
            n=k-Ny;
        int q=l;
    
        real filter=1.0/(1.0+filter_eps*filter_eps*(2.0*M_PI/Lx*m*2.0*M_PI/Lx*m+2.0*M_PI/Ly*n*2.0*M_PI/Ly*n+2.0*M_PI/Lz*q*2.0*M_PI/Lz*q));

        ux_filt_hat_d[IN(j,k,l)].x=filter*ux_hat_d[IN(j,k,l)].x;
        ux_filt_hat_d[IN(j,k,l)].y=filter*ux_hat_d[IN(j,k,l)].y;
        uy_filt_hat_d[IN(j,k,l)].x=filter*uy_hat_d[IN(j,k,l)].x;
        uy_filt_hat_d[IN(j,k,l)].y=filter*uy_hat_d[IN(j,k,l)].y;
        uz_filt_hat_d[IN(j,k,l)].x=filter*uz_hat_d[IN(j,k,l)].x;
        uz_filt_hat_d[IN(j,k,l)].y=filter*uz_hat_d[IN(j,k,l)].y;

    }

}
}




void Helmholz_Fourier_Filter(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz,  real Lx, real Ly, real Lz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real filter_eps, real *ux_filt_d, real *uy_filt_d, real *uz_filt_d){


    Helmholz_Fourier_Filter_Device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, Lx, Ly, Lz, ux_hat_d, uy_hat_d, uz_hat_d, ux_filt_hat_d, uy_filt_hat_d, uz_filt_hat_d, filter_eps);

    iFFTN_Device(dimGrid, dimBlock, ux_filt_hat_d, ux_filt_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uy_filt_hat_d, uy_filt_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uz_filt_hat_d, uz_filt_d, Nx, Ny, Nz);



}


__global__ void CutOff_Fourier_Filter_Device(int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real Radius_to_one){
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
int j=xIndex, k=yIndex, l=zIndex;
    if((j<Nx)&&(k<Ny)&&(l<Nz)){

        int m=j;
        if(j>=Nx/2)
            m=j-Nx;
        int n=k;
        if(k>=Ny/2)
            n=k-Ny;
        int q=l;
    
        real filter=0.0;
        if((1.0*m*m/(1.0*Nx*Nx)+1.0*n*n/(1.0*Ny*Ny)+1.0*q*q/(1.0*Nz*Nz))<=Radius_to_one*Radius_to_one){
            filter=1.0;

        }

        ux_filt_hat_d[IN(j,k,l)].x=filter*ux_hat_d[IN(j,k,l)].x;
        ux_filt_hat_d[IN(j,k,l)].y=filter*ux_hat_d[IN(j,k,l)].y;
        uy_filt_hat_d[IN(j,k,l)].x=filter*uy_hat_d[IN(j,k,l)].x;
        uy_filt_hat_d[IN(j,k,l)].y=filter*uy_hat_d[IN(j,k,l)].y;
        uz_filt_hat_d[IN(j,k,l)].x=filter*uz_hat_d[IN(j,k,l)].x;
        uz_filt_hat_d[IN(j,k,l)].y=filter*uz_hat_d[IN(j,k,l)].y;


        

    }

}
}


void CutOff_Fourier_Filter(dim3 dimGrid, dim3 dimBlock,  dim3 dimGrid_C, dim3 dimBlock_C, int Nx, int Ny, int Nz, int Mz,  real Lx, real Ly, real Lz, cudaComplex *ux_hat_d, cudaComplex *uy_hat_d, cudaComplex *uz_hat_d, cudaComplex *ux_filt_hat_d, cudaComplex *uy_filt_hat_d, cudaComplex *uz_filt_hat_d, real Radius_to_one, real *ux_filt_d, real *uy_filt_d, real *uz_filt_d){



    CutOff_Fourier_Filter_Device<<<dimGrid_C, dimBlock_C>>>(Nx, Ny, Mz, ux_hat_d, uy_hat_d, uz_hat_d, ux_filt_hat_d, uy_filt_hat_d, uz_filt_hat_d, Radius_to_one);

    iFFTN_Device(dimGrid, dimBlock, ux_filt_hat_d, ux_filt_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uy_filt_hat_d, uy_filt_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uz_filt_hat_d, uz_filt_d, Nx, Ny, Nz);



}


void Image_to_Domain(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, real* ux_d, cudaComplex *ux_hat_d, real* uy_d, cudaComplex *uy_hat_d, real* uz_d, cudaComplex *uz_hat_d)
{

    iFFTN_Device(dimGrid, dimBlock, ux_hat_d, ux_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uy_hat_d, uy_d, Nx, Ny, Nz);
    iFFTN_Device(dimGrid, dimBlock, uz_hat_d, uz_d, Nx, Ny, Nz);

}


void Domain_to_Image(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, cudaComplex *ux_hat_d, real* ux_d, cudaComplex *uy_hat_d, real* uy_d, cudaComplex *uz_hat_d, real* uz_d)
{

    FFTN_Device(ux_d, ux_hat_d);
    FFTN_Device(uy_d, uy_hat_d);
    FFTN_Device(uz_d, uz_hat_d);

}