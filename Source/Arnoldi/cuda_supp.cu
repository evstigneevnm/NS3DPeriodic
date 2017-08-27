#include "cuda_supp.h"
//using namespace std;

namespace Arnoldi
{


bool InitCUDA(int GPU_number)
{

	
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no compartable device found.\n");
		return false;
	}
	
	int deviceNumber=0;
	int deviceNumberTemp=0;
	
	if(count>1){
		

			
		for(i = 0; i < count; i++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			printf( "#%i:	%s, pci-bus id:%i %i %i	\n", i, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
		}
		
		if(GPU_number==-1){
			printf("Device number for it to use>>>\n",i);
			scanf("%i", &deviceNumberTemp);
		}
		else{
			printf("Using device number %i\n",GPU_number);
			deviceNumberTemp=GPU_number;
		}
   		deviceNumber=deviceNumberTemp;
	
	}
	else{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceNumber);
		printf( "#%i:	%s, pci-bus id:%i %i %i	\n", deviceNumber, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
		printf( "		using it...\n");	
	}

	cudaSetDevice(deviceNumber);
	
	return true;
}


void device_host_real_cpy(real* device, real* host, int Nx, int Ny){
	int mem_size=sizeof(real)*Nx*Ny;
	cudaError_t cuerr=cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice);
   	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot copy real array from host to device because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 

}


void host_device_real_cpy(real* host, real* device, int Nx, int Ny){
	int mem_size=sizeof(real)*Nx*Ny;
	cudaError_t cuerr=cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
}





void check_for_nans(char message[], int Size, real *array){
	real Array_CPU[2]={0,0};
	
	cudaError_t cuerr=cudaMemcpy(Array_CPU, array, 2*sizeof(real), cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
	if(Array_CPU[0]!=Array_CPU[0]){
		std::cerr << "NANS!!!";
		std::cerr << message << "\n";
		exit(1);
	}

}





void checkError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("%s", msg);
		switch(status){
		    case CUBLAS_STATUS_NOT_INITIALIZED:
                printf(" the library was not initialized!\n");
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                printf(" the parameters m,n<0 or incx,incy=0!\n");
                break;           	
            case CUBLAS_STATUS_ALLOC_FAILED:
                printf(" the reduction buffer could not be allocated!\n");
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                printf(" the device does not support double-precision!\n");
                break;    
            case CUBLAS_STATUS_MAPPING_ERROR:
                printf(" An access to GPU memory space failed.!\n");
                break;                     
            case CUBLAS_STATUS_EXECUTION_FAILED:
                printf(" the function failed to launch on the GPU!\n");
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                printf(" An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure!\n");
                break;
			default:
                printf(" Unknown error!\n");
                break;				
			}


        exit(EXIT_FAILURE);
    }
}


void vectors_add_GPU(cublasHandle_t handle, int N, real alpha, real *x, real *y){
/*

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy)
cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
                           const double          *alpha,
                           const double          *x, int incx,
                           double                *y, int incy)

This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.


*/
 	cublasStatus_t ret; 

 	#ifdef real_float		
		ret=cublasSaxpy(handle, N, &alpha, x, 1, y, 1);
	#endif
	#ifdef real_double	
		ret=cublasDaxpy(handle, N, &alpha, x, 1, y, 1);
	#endif	
	checkError(ret, " vectors_add_GPU(). "); 


}

void normalize_vector_GPU(cublasHandle_t handle, int N, real *x){

/*
cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx)
cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx)

This function scales the vector x by the scalar α and overwrites it with the result.



*/

 	cublasStatus_t ret; 
 	real norm2=0.0;                   
	norm2=Arnoldi::vector_norm2_GPU(handle, N, x);
	//if(norm2>1E-15){
 		norm2=1.0/norm2;
	 	#ifdef real_float		
			ret=cublasSscal(handle, N,  &norm2, x, 1);
		#endif
		#ifdef real_double	
			ret=cublasDscal(handle, N,  &norm2, x, 1);
		#endif	
		checkError(ret, " normalize_vector_GPU(). "); 

	//}
	//else{
	//	printf("\nVector length is less than 1E-15!\n");
	//	exit(-1);
	//}


}

real vector_norm2_GPU(cublasHandle_t handle, int N, real *x){

/*
cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result)

*/
	cublasStatus_t ret;  
	                      
    real result;
 	#ifdef real_float		
		ret=cublasSnrm2(handle, N,  x, 1, &result);
	#endif
	#ifdef real_double	
		ret=cublasDnrm2(handle, N,  x, 1, &result);
	#endif	

	checkError(ret, " vector_norm2_GPU(). ");   

	return result;

}


void vector_copy_GPU(cublasHandle_t handle, int N, real *vec_source, real *vec_dest){


/*
cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy)
cublasStatus_t cublasDcopy(cublasHandle_t handle, int n,
                           const double          *x, int incx,
                           double                *y, int incy)

This function copies the vector x into the vector y


*/
	cublasStatus_t ret;
	
	#ifdef real_float		
		ret=cublasScopy(handle, N, vec_source, 1, vec_dest, 1);
	#endif
	#ifdef real_double	
		ret=cublasDcopy(handle, N, vec_source, 1, vec_dest, 1);
	#endif


	checkError(ret, " vector_copy_GPU(). ");
}



real vector_dot_product_GPU(cublasHandle_t handle, int N, real *vec1, real *vec2){

/*
cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result)
cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
                           const double          *x, int incx,
                           const double          *y, int incy,
                           double          *result)

*/

	cublasStatus_t ret;
	real result;

	#ifdef real_float		
		ret=cublasSdot(handle, N, vec1, 1, vec2, 1, &result);
	#endif
	#ifdef real_double	
		ret=cublasDdot(handle, N, vec1, 1, vec2, 1, &result);
	#endif	

	checkError(ret, " vector_dot_product_GPU(). ");

	return result;

}



void matrixMultVector_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, real beta, real *res){ //   res=α*A*x+β*res){

	cublasStatus_t ret;
/*
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy)


This function performs the matrix-vector multiplication

y = α op ( A ) x + β y

where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. Also, for matrix A

 A  if transa == CUBLAS_OP_N 
 A^T  if transa == CUBLAS_OP_T 
 A^H  if transa == CUBLAS_OP_H 
handle 	input	handle to the cuBLAS library context.

trans 	input 			operation op(A) that is non- or (conj.) transpose.
m 		input 			number of rows of matrix A.
n 		input 			number of columns of matrix A.
α	 host or device input <type> scalar used for multiplication.
A device  input         <type> array of dimension lda x n with lda >= max(1,m) if transa==CUBLAS_OP_N and lda x m with lda >= max(1,n) otherwise.

lda 	input 			leading dimension of two-dimensional array used to store matrix A.
x 	device input		<type> vector with n elements if transa==CUBLAS_OP_N and m elements otherwise.

incx	input 		stride between consecutive elements of x.

β host or device input 	<type> scalar used for multiplication, if beta==0 then y does not have to be a valid input.

y device in/out 		<type> vector with m elements if transa==CUBLAS_OP_N and n elements otherwise.

incy 	input stride between consecutive elements of .y

*/
	int LDA=RowA;

	#ifdef real_float		
		ret=cublasSgemv(handle, CUBLAS_OP_N, RowA, ColA, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif
	#ifdef real_double	
		ret=cublasDgemv(handle, CUBLAS_OP_N, RowA, ColA, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif	

	checkError(ret, " matrixMultVector_GPU(). ");

}





__global__ void set_matrix_colomn_kernel(int Row, int Col, real* matrix, real *vec, int col_number){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if((i<Row)&&(col_number<Col)){
		matrix[I2(i,col_number,Row)]=vec[i];
	}
	
}
__global__ void get_matrix_colomn_kernel(int Row, int Col, real* matrix, real *vec, int col_number){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if((i<Row)&&(col_number<Col)){
		vec[i]=matrix[I2(i,col_number,Row)];
	}
	
}

void set_matrix_colomn_GPU(int Row, int Col, real *mat, real *vec, int col_number){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(Row+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	set_matrix_colomn_kernel<<< blocks, threads>>>(Row, Col, mat, vec, col_number);

}

void get_matrix_colomn_GPU(int Row, int Col, real *mat, real *vec, int col_number){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(Row+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	get_matrix_colomn_kernel<<< blocks, threads>>>(Row, Col, mat, vec, col_number);

}

__global__ void set_vector_value_kernel(int N, real val, real *vec){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec[i]=val;
	}
	
}


__global__ void set_initial_Krylov_vector_value_kernel(int N, real *vec){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec[i]=0.0;
	}
	vec[0]=1.0;
	vec[N/4]=3.0;
	vec[N/2]=0.5;

}


void set_vector_value_GPU(int N, real val, real *vec){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	set_vector_value_kernel<<< blocks, threads>>>(N, val,vec);

}




void set_initial_Krylov_vector_value_GPU(int N, real *vec){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	set_initial_Krylov_vector_value_kernel<<< blocks, threads>>>(N, vec);


}



__global__ void set_vector_inverce_kernel(int N, real *vec){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec[i]=-vec[i];
	}
	
}


void set_vector_inverce_GPU(int N, real *vec){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	set_vector_inverce_kernel<<< blocks, threads>>>(N, vec);

}




void matrixMultVector_part_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, int part_Cols, real beta, real *res){ //   res=α*A*x+β*res){

	cublasStatus_t ret;
/*
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy)


This function performs the matrix-vector multiplication

y = α op ( A ) x + β y

where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. Also, for matrix A

 A  if transa == CUBLAS_OP_N 
 A^T  if transa == CUBLAS_OP_T 
 A^H  if transa == CUBLAS_OP_H 
handle 	input	handle to the cuBLAS library context.

trans 	input 			operation op(A) that is non- or (conj.) transpose.
m 		input 			number of rows of matrix A.
n 		input 			number of columns of matrix A.
α	 host or device input <type> scalar used for multiplication.
A device  input         <type> array of dimension lda x n with lda >= max(1,m) if transa==CUBLAS_OP_N and lda x m with lda >= max(1,n) otherwise.

lda 	input 			leading dimension of two-dimensional array used to store matrix A.
x 	device input		<type> vector with n elements if transa==CUBLAS_OP_N and m elements otherwise.

incx	input 		stride between consecutive elements of x.

β host or device input 	<type> scalar used for multiplication, if beta==0 then y does not have to be a valid input.

y device in/out 		<type> vector with m elements if transa==CUBLAS_OP_N and n elements otherwise.

incy 	input stride between consecutive elements of .y

*/
	int LDA=RowA;

	#ifdef real_float		
		ret=cublasSgemv(handle, CUBLAS_OP_N, RowA, part_Cols, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif
	#ifdef real_double	
		ret=cublasDgemv(handle, CUBLAS_OP_N, RowA, part_Cols, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif	

	checkError(ret, " matrixMultVector_part_GPU(). ");

}


void matrixDotVector_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, real beta, real *res){ //   res=α*A*x+β*res){

	cublasStatus_t ret;
/*
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy)


This function performs the matrix-vector multiplication

y = α op ( A ) x + β y

where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. Also, for matrix A

 A  if transa == CUBLAS_OP_N 
 A^T  if transa == CUBLAS_OP_T 
 A^H  if transa == CUBLAS_OP_H 
handle 	input	handle to the cuBLAS library context.

trans 	input 			operation op(A) that is non- or (conj.) transpose.
m 		input 			number of rows of matrix A.
n 		input 			number of columns of matrix A.
α	 host or device input <type> scalar used for multiplication.
A device  input         <type> array of dimension lda x n with lda >= max(1,m) if transa==CUBLAS_OP_N and lda x m with lda >= max(1,n) otherwise.

lda 	input 			leading dimension of two-dimensional array used to store matrix A.
x 	device input		<type> vector with n elements if transa==CUBLAS_OP_N and m elements otherwise.

incx	input 		stride between consecutive elements of x.

β host or device input 	<type> scalar used for multiplication, if beta==0 then y does not have to be a valid input.

y device in/out 		<type> vector with m elements if transa==CUBLAS_OP_N and n elements otherwise.

incy 	input stride between consecutive elements of .y

*/
	int LDA=RowA;

	#ifdef real_float		
		ret=cublasSgemv(handle, CUBLAS_OP_T, RowA, ColA, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif
	#ifdef real_double	
		ret=cublasDgemv(handle, CUBLAS_OP_T, RowA, ColA, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif	

	checkError(ret, " matrixDotVector_GPU(). ");

}


void matrixDotVector_part_GPU(cublasHandle_t handle, int RowA, real *A, int ColA, real alpha, real *x, int part_Cols, real beta, real *res){ //   res=α*A*x+β*res)

	cublasStatus_t ret;
/*
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy)


This function performs the matrix-vector multiplication

y = α op ( A ) x + β y

where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. Also, for matrix A

 A  if transa == CUBLAS_OP_N 
 A^T  if transa == CUBLAS_OP_T 
 A^H  if transa == CUBLAS_OP_H 
handle 	input	handle to the cuBLAS library context.

trans 	input 			operation op(A) that is non- or (conj.) transpose.
m 		input 			number of rows of matrix A.
n 		input 			number of columns of matrix A.
α	 host or device input <type> scalar used for multiplication.
A device  input         <type> array of dimension lda x n with lda >= max(1,m) if transa==CUBLAS_OP_N and lda x m with lda >= max(1,n) otherwise.

lda 	input 			leading dimension of two-dimensional array used to store matrix A.
x 	device input		<type> vector with n elements if transa==CUBLAS_OP_N and m elements otherwise.

incx	input 		stride between consecutive elements of x.

β host or device input 	<type> scalar used for multiplication, if beta==0 then y does not have to be a valid input.

y device in/out 		<type> vector with m elements if transa==CUBLAS_OP_N and n elements otherwise.

incy 	input stride between consecutive elements of .y

*/
	int LDA=RowA;

	#ifdef real_float		
		ret=cublasSgemv(handle, CUBLAS_OP_T, RowA, part_Cols, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif
	#ifdef real_double	
		ret=cublasDgemv(handle, CUBLAS_OP_T, RowA, part_Cols, &alpha, A, LDA, x, 1, &beta, res, 1);
	#endif	

	checkError(ret, " matrixDotVector_part_GPU(). ");

}

void matrixMultMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, real *A, real alpha, real *B, real beta, real *C){   
// C = α op ( A ) op ( B ) + β C


/*
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)


This function performs the matrix-matrix multiplication

C = α op ( A ) op ( B ) + β C
where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A

op ( A ) = A if  transa == CUBLAS_OP_N 
		   A^T if  transa == CUBLAS_OP_T 
		   A^H if  transa == CUBLAS_OP_C

and op ( B ) is defined similarly for matrix B

handle 	input	handle to the cuBLAS library context.

transa	input	operation op(A) that is non- or (conj.) transpose.

transb 	input 	operation op(B) that is non- or (conj.) transpose.

m 		input	number of rows of matrix op(A) and C.

n 		input	number of columns of matrix op(B) and C.

k 		input	number of columns of op(A) and rows of op(B).

alpha 	host or device input	<type> scalar used for multiplication.

A 	device input	<type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.

lda 	input leading dimension of two-dimensional array used to store the matrix A.

B 	device input	<type> array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.

ldb 	input 	leading dimension of two-dimensional array used to store matrix B.

beta	 host or device input	<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.

C 	device in/out 	<type> array of dimensions ldc x n with ldc>=max(1,m).

ldc 	input	leading dimension of a two-dimensional array used to store the matrix C.


*/
	int LDA=RowAC;
	int LDB=ColA;
	int LDC=RowAC;
	cublasStatus_t ret;

	#ifdef real_float		
		ret=cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                          &beta,
                          C, LDC);
	#endif
	#ifdef real_double	
		ret=cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                           &beta,
                          C, LDC);
	#endif

	checkError(ret, " matrixMultMatrix_GPU(). ");

}

void matrixTMultMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, real *A, real alpha, real *B, real beta, real *C){   
// C = α op ( A ) op ( B ) + β C


/*
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)


This function performs the matrix-matrix multiplication

C = α op ( A ) op ( B ) + β C
where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A

op ( A ) = A if  transa == CUBLAS_OP_N 
		   A^T if  transa == CUBLAS_OP_T 
		   A^H if  transa == CUBLAS_OP_C

and op ( B ) is defined similarly for matrix B

handle 	input	handle to the cuBLAS library context.

transa	input	operation op(A) that is non- or (conj.) transpose.

transb 	input 	operation op(B) that is non- or (conj.) transpose.

m 		input	number of rows of matrix op(A) and C.

n 		input	number of columns of matrix op(B) and C.

k 		input	number of columns of op(A) and rows of op(B).

alpha 	host or device input	<type> scalar used for multiplication.

A 	device input	<type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.

lda 	input leading dimension of two-dimensional array used to store the matrix A.

B 	device input	<type> array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.

ldb 	input 	leading dimension of two-dimensional array used to store matrix B.

beta	 host or device input	<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.

C 	device in/out 	<type> array of dimensions ldc x n with ldc>=max(1,m).

ldc 	input	leading dimension of a two-dimensional array used to store the matrix C.


*/
	int LDA=ColA;
	int LDB=ColA;
	int LDC=RowAC;
	cublasStatus_t ret;

	#ifdef real_float		
		ret=cublasSgemm(handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                           &beta,
                          C, LDC);
	#endif
	#ifdef real_double	
		ret=cublasDgemm(handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                           &beta,
                          C, LDC);
	#endif

	checkError(ret, " matrixTMultMatrix_GPU(). ");

}


void matrixMultComplexMatrix_GPU(cublasHandle_t handle, int RowAC, int ColBC, int ColA, cublasComplex *A, cublasComplex *B, cublasComplex *C){   // C = α op ( A ) op ( B ) + β C


/*
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)


This function performs the matrix-matrix multiplication

C = α op ( A ) op ( B ) + β C
where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A

op ( A ) = A if  transa == CUBLAS_OP_N 
		   A^T if  transa == CUBLAS_OP_T 
		   A^H if  transa == CUBLAS_OP_C

and op ( B ) is defined similarly for matrix B

handle 	input	handle to the cuBLAS library context.

transa	input	operation op(A) that is non- or (conj.) transpose.

transb 	input 	operation op(B) that is non- or (conj.) transpose.

m 		input	number of rows of matrix op(A) and C.

n 		input	number of columns of matrix op(B) and C.

k 		input	number of columns of op(A) and rows of op(B).

alpha 	host or device input	<type> scalar used for multiplication.

A 	device input	<type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.

lda 	input leading dimension of two-dimensional array used to store the matrix A.

B 	device input	<type> array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.

ldb 	input 	leading dimension of two-dimensional array used to store matrix B.

beta	 host or device input	<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.

C 	device in/out 	<type> array of dimensions ldc x n with ldc>=max(1,m).

ldc 	input	leading dimension of a two-dimensional array used to store the matrix C.


*/
	int LDA=RowAC;
	int LDB=ColA;
	int LDC=RowAC;
	cublasStatus_t ret;
	cublasComplex alpha;
	cublasComplex beta;


	#ifdef real_float		
		
		alpha=make_cuComplex(1.0, 0.0);
		beta=make_cuComplex(0.0, 0.0);
		ret=cublasCgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                           &beta,
                          C, LDC);
	#endif
	#ifdef real_double	

		alpha=make_cuDoubleComplex(1.0, 0.0);
		beta=make_cuDoubleComplex(0.0, 0.0);

		ret=cublasZgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           RowAC, ColBC, ColA,
                          &alpha, A, LDA,
                          B, LDB,
                           &beta,
                          C, LDC);
	#endif

	checkError(ret, " matrixMultComplexMatrix_GPU(). ");

}




//namespace!
}




