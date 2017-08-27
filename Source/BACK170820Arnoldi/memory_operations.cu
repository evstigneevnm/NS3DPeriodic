#include "memory_operations.h"


namespace Arnoldi
{


real* allocate_d(int Nx, int Ny, int Nz){
	int size=(Nx)*(Ny)*(Nz);
	real* array;
	array=(real*)malloc(sizeof(real)*size);
	if ( !array ){
		fprintf(stderr,"\n unable to allocate real memeory!\n");
		exit(-1);
	}
	else{
		for(int i=0;i<Nx;i++)
		for(int j=0;j<Ny;j++)
		for(int k=0;k<Nz;k++)
				array[I2(i,j,Nx)]=0.0;
	}
	
	return array;
}

int* allocate_i(int Nx, int Ny, int Nz){
	int size=(Nx)*(Ny)*(Nz);
	int* array;
	array=(int*)malloc(sizeof(int)*size);
	if ( !array ){
		fprintf(stderr,"\n unable to allocate int memeory!\n");
		exit(-1);
	}
	else{
		for(int i=0;i<Nx;i++)
		for(int j=0;j<Ny;j++)
		for(int k=0;k<Nz;k++)
				array[I2(i,j,Nx)]=0;
	}
	
	return array;
}

real average(int count, ...)
{
    va_list ap;
    int j;
    real tot = 0;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(j = 0; j < count; j++)
        tot += va_arg(ap, real); /* Increments ap to the next argument. */
    va_end(ap);
    return tot / count;
}


void allocate_real(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real** value= va_arg(ap, real**); /* Increments ap to the next argument. */
    	real* temp=allocate_d(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}

void allocate_int(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	int** value= va_arg(ap, int**); /* Increments ap to the next argument. */
    	int* temp=allocate_i(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}


void deallocate_real(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real* value= va_arg(ap, real*); /* Increments ap to the next argument. */
		free(value);
    }
    va_end(ap);

}
void deallocate_int(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	int* value= va_arg(ap, int*); /* Increments ap to the next argument. */
		free(value);
    }
    va_end(ap);

}

//for GPU:

int* device_allocate_int(int Nx, int Ny, int Nz){
	int* m_device;
	int mem_size=sizeof(int)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate int device array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}

real* device_allocate_real(int Nx, int Ny, int Nz){
	real* m_device;
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate real device array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}

cublasComplex* device_allocate_complex(int Nx, int Ny, int Nz){
	cublasComplex* m_device;
	int mem_size=sizeof(cublasComplex)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate device complex array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}


void to_device_from_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice);
   	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot copy real array from host to device because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 

}


void to_host_from_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
}



void to_device_from_host_int_cpy(int* device, int* host, int Nx, int Ny, int Nz){
	int mem_size=sizeof(int)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice);
   	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot copy int array from host to device because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 

}


void to_host_from_device_int_cpy(int* host, int* device, int Nx, int Ny, int Nz){
	int mem_size=sizeof(int)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real int from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
}



void device_allocate_all_real(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real** value=va_arg(ap, real**); /* Increments ap to the next argument. */
    	real* temp=device_allocate_real(Nx, Ny, Nz);
    	value[0]=temp;    	
    }
    va_end(ap);

}



void device_deallocate_all_real(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	real* value=va_arg(ap, real*); /* Increments ap to the next argument. */
		cudaFree(value);
    }
    va_end(ap);

}

}