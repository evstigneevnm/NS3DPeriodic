#ifndef __ARNOLDI_MACROS_H__
#define __ARNOLDI_MACROS_H__


#define BLOCKSIZE 128
#define block_size_x 16
#define block_size_y BLOCKSIZE/block_size_x


const double Im_eig_tol=1.0E-14;

#ifndef I2
	#define I2(i,j,Rows) (i)+(j)*(Rows)
#endif


#ifndef I2t
	#define I2t(i,j,Cols) (i)*(Cols)+(j) //transonent matrix indexing
#endif


#define real_double
//#define real_float

#ifdef real_double
	#define real double
	#define cublasComplex cuDoubleComplex
#endif

#ifdef real_float
	#define real float
	#define cublasComplex cuComplex
#endif



#ifndef PI
#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128
#endif

#define MI(i,j) ((i)+(4)*(j))

#define min2(X,Y) ((X) < (Y) ? (X) : (Y)) 
#define max2(X,Y) ((X) < (Y) ? (Y) : (X)) 
#define max3(X,Y,Z) max2(max2(X,Y),max2(Y,Z))

#define Labs(X) ((X) < 0 ? -(X):(X))


#endif
