#ifndef __MACROS_H__
#define __MACROS_H__

#include <cufft.h>	//for global FFTs

#define BLOCKSIZE 128
#define block_size_x 16
#define block_size_y BLOCKSIZE/block_size_x

#ifndef IN
	#define IN(i,j,k) (i)*(Ny*Nz)+(j)*Nz+(k)
//	#define IN(i,j,k) (i)+(j)*(Nx)+(k)*(Nx*Ny)
#endif
#ifndef IM
	#define IM(i,j,k) (i)*(Ny*Mz)+(j)*Mz+(k)
//	#define IM(i,j,k) (i)+(j)*(Nx)+(k)*(Nx*Ny)
#endif
#ifndef I3
	#define I3(i,j,k)  (Ny*Nz)*((i)>(Nx-1)?(i)-Nx:(i)<0?(Nx+(i)):(i))+((j)>(Ny-1)?(j)-Ny:(j)<0?(Ny+(j)):(j))*(Nz)+((k)>(Nz-1)?(k)-Nz:(k)<0?(Nz+(k)):(k))
//	#define I3(i,j,k)  ((i)>(Nx-1)?(i)-Nx:(i)<0?(Nx+(i)):(i))+((j)>(Ny-1)?(j)-Ny:(j)<0?(Ny+(j)):(j))*(Nx)+((k)>(Nz-1)?(k)-Nz:(k)<0?(Nz+(k)):(k))*(Nx*Ny)
#endif
#ifndef IE
	#define IE(i,j,k) (i)*(Ny_ext*Nz_ext)+(j)*Nz_ext+(k)
//	#define IE(i,j,k) (i)+(j)*(Nx_ext)+(k)*(Nx_ext*Ny_ext)
#endif

#ifndef I2
    #define I2(j,k) (j)+Nx*(k)
#endif



#define FS for ( j=0 ; j<Nx ; j++ ) { for ( k=0 ; k<Ny ; k++ ) { for ( l=0 ; l<Nz ; l++ ) { 
#define fFS for ( j=0 ; j<Nx ; j++ ) { for ( k=0 ; k<Ny ; k++ ) { for ( l=0 ; l<Mz ; l++ ) { 
#define FE }}}

//*
#define real double
#define cudaComplex cufftDoubleComplex
#define ComplexToComplex CUFFT_Z2Z
#define cufftExecXtoX  cufftExecZ2Z

#define ComplexToReal CUFFT_Z2D
#define RealToComplex CUFFT_D2Z
#define cufftExecCtoR cufftExecZ2D
#define cufftExecRtoC cufftExecD2Z




//global plans for FFT:R->C and iFFT:C->R




//static CUDA grid declaration
/*
static dim3 dimBlock;
static dim3 dimGrid;
static dim3 dimBlock_C;
static dim3 dimGrid_C;
//This thing doesn't work!!!!!
*/


//*/
/*
#define real float
#define cudaComplex cufftComplex
#define ComplexToComplex CUFFT_C2C
#define cufftExecXtoX  cufftExecC2C
//*/


#ifndef PI
#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128
#endif

#define MI(i,j) ((i)+(4)*(j))

#define min2(X,Y) ((X) < (Y) ? (X) : (Y)) 
#define max2(X,Y) ((X) < (Y) ? (Y) : (X)) 
#define max3(X,Y,Z) max2(max2(X,Y),max2(Y,Z))

#define Labs(X) ((X) < 0 ? -(X):(X))



#endif
