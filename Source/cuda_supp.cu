#include "cuda_supp.h"


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





cudaComplex* device_allocate_complex(int Nx, int Ny, int Nz){
	cudaComplex* m_device;
	int mem_size=sizeof(cudaComplex)*Nx*Ny*Nz;
	
    cudaError_t cuerr=cudaMalloc((void**)&m_device, mem_size);
	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot allocate device array because: %s\n",
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
		fprintf(stderr, "Cannot allocate device array because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    }  

    return m_device;	

}


void device_host_real_cpy(real* device, real* host, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice);
   	if (cuerr != cudaSuccess)
    {
		fprintf(stderr, "Cannot copy real array from host to device because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 

}


void host_device_real_cpy(real* host, real* device, int Nx, int Ny, int Nz){
	int mem_size=sizeof(real)*Nx*Ny*Nz;
	cudaError_t cuerr=cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array from device to host because: %s\n",
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

void device_allocate_all_complex(int Nx, int Ny, int Nz, int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	cudaComplex** value=va_arg(ap, cudaComplex**); /* Increments ap to the next argument. */
    	cudaComplex* temp=device_allocate_complex(Nx, Ny, Nz);
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

void device_deallocate_all_complex(int count, ...){

    va_list ap;
	va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
	for(int j = 0; j < count; j++){
    	cudaComplex* value=va_arg(ap, cudaComplex*); /* Increments ap to the next argument. */
		cudaFree(value);
    }
    va_end(ap);

}




__global__ void calculate_CFL_device(real CFL, int Nx, int Ny, int Nz,  real dx, real dy, real dz, real* ux, real* uy, real* uz, real* cfl_in){
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
yIndex = t1 - Ny * xIndex;
unsigned int j=xIndex, k=yIndex, l=zIndex;
	if((j<Nx)&&(k<Ny)&&(l<Nz)){
		cfl_in[IN(j,k,l)]=CFL/((1.0e-6+abs(ux[IN(j,k,l)]))/dx+(1.0e-6+abs(uy[IN(j,k,l)]))/dy+(1.0e-6+abs(uz[IN(j,k,l)]))/dz);

	}
}
	

}


void retrive_time_step(real* ret, real* cfl_out){

	
	cudaError_t cuerr=cudaMemcpy(ret, cfl_out, 1*sizeof(real), cudaMemcpyDeviceToHost);
	//cudaError_t cuerr=cudaMemcpyToSymbol( ret,cfl_out,1*sizeof(real),0,cudaMemcpyDeviceToHost); 
	
	if (cuerr != cudaSuccess)
    	{	
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    	} 	
	

}


void retrive_Shapiro_step(real* ret, real* cfl_out){

	
	cudaError_t cuerr=cudaMemcpy(ret, cfl_out, 2*sizeof(real), cudaMemcpyDeviceToHost);
	//cudaError_t cuerr=cudaMemcpyToSymbol( ret,cfl_out,1*sizeof(real),0,cudaMemcpyDeviceToHost); 
	
	if (cuerr != cudaSuccess)
    	{	
		printf("Cannot copy real array from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    	} 	
	

}


void calc_dt(dim3 dimBlock, dim3 dimGrid, real CFL, int Nx, int Ny, int Nz,  real dx, real dy, real dz, real* ux, real* uy, real* uz, real* cfl_in, real *cfl_out, real *dt_pointer){


	calculate_CFL_device<<<dimGrid, dimBlock>>>(CFL, Nx, Ny, Nz,  dx, dy, dz, ux, uy, uz, cfl_in);
	
	compute_reduction(cfl_in, cfl_out, Nx*Ny*Nz);
	retrive_time_step(dt_pointer, cfl_out);

	
}



__global__ void check_nans_kernel(int N, real *vec){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	



	if(i<N){
	
		if(isnan(vec[i])){
			vec[0]=NAN;
		}

	}

}




int check_nans_kernel(char message[], int N, real *vec){
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	check_nans_kernel<<< blocks, threads>>>(N, vec);

	real Array_CPU[2]={0,0};
	
	cudaError_t cuerr=cudaMemcpy(Array_CPU, vec, 2*sizeof(real), cudaMemcpyDeviceToHost);
 	if (cuerr != cudaSuccess)
    {
		printf("Cannot copy real array while nan check from device to host because: %s\n",
		cudaGetErrorString(cuerr));
		exit(-1);
    } 
    if(isnan(Array_CPU[0])){
		std::cerr << "NANS!!!";
		std::cerr << message << "\n";
		return -1;
	}
	return 0;

}


__global__ void check_nans_kernel(int N, cudaComplex *vec, bool *result){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	



	if(i<N){
	
		if(isnan(vec[i].x)){
			result[0]=false;
		}

		if(isnan(vec[i].y)){
			result[0]=false;
		}

	}

}


int check_nans_kernel(char message[], int N, cudaComplex *vec){
	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
		
	bool  *d_result, h_result=true;
	cudaMalloc((void **)&d_result, sizeof (bool));
	cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
	check_nans_kernel<<< blocks, threads>>>(N, vec, d_result);
	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	if (h_result==false) {
		std::cerr << "NANS!!!";
		std::cerr << message << "\n";
		exit(-1);
	}
	return 0;
}

