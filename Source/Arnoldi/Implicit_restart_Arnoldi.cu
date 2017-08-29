#include "Implicit_restart_Arnoldi.h"



__global__ void real_to_cublasComplex_kernel(int N, real *vec_source_re, real *vec_source_im, cublasComplex *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest[i].x=vec_source_re[i];
		vec_dest[i].y=vec_source_im[i];
	}
	
}

__global__ void real_to_cublasComplex_kernel(int N, real *vec_source_re, cublasComplex *vec_dest){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest[i].x=vec_source_re[i];
		vec_dest[i].y=0.0;
	}
	
}

__global__ void cublasComplex_to_real_kernel(int N, cublasComplex *vec_source,  real *vec_dest_re,  real *vec_dest_im){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
		vec_dest_re[i]=vec_source[i].x;
		vec_dest_im[i]=vec_source[i].y;
	}
	
}

void real_complex_to_cublas_complex(int N, complex real* cpu_complex,  cublasComplex *gpu_complex){


	real *cpu_real, *cpu_imag, *gpu_real, *gpu_imag;
	Arnoldi::device_allocate_all_real(N,1, 1, 2,&gpu_real, &gpu_imag);
	Arnoldi::allocate_real(N, 1, 1, 2,&cpu_real, &cpu_imag);
	for(int j=0;j<N;j++){
		cpu_real[j]=creal(cpu_complex[j]);
		cpu_imag[j]=cimag(cpu_complex[j]);
	}
	Arnoldi::to_device_from_host_real_cpy(gpu_real, cpu_real, N, 1,1);
	Arnoldi::to_device_from_host_real_cpy(gpu_imag, cpu_imag, N, 1,1);

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	real_to_cublasComplex_kernel<<< blocks, threads>>>(N, gpu_real, gpu_imag, gpu_complex);


	Arnoldi::deallocate_real(2,cpu_real, cpu_imag);
	Arnoldi::device_deallocate_all_real(2, gpu_real, gpu_imag);
}


void real_device_to_cublas_complex(int N, real* gpu_real, cublasComplex *gpu_complex){


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	real_to_cublasComplex_kernel<<< blocks, threads>>>(N, gpu_real, gpu_complex);


}



void cublas_complex_to_complex_real(int N, cublasComplex *gpu_complex, complex real* cpu_complex){

	real *cpu_real, *cpu_imag, *gpu_real, *gpu_imag;
	Arnoldi::device_allocate_all_real(N,1, 1, 2,&gpu_real, &gpu_imag);
	Arnoldi::allocate_real(N, 1, 1, 2,&cpu_real, &cpu_imag);


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	cublasComplex_to_real_kernel<<< blocks, threads>>>(N, gpu_complex,  gpu_real,  gpu_imag);

	Arnoldi::to_host_from_device_real_cpy(cpu_real, gpu_real, N, 1,1);
	Arnoldi::to_host_from_device_real_cpy(cpu_imag, gpu_imag, N, 1,1);

	for(int j=0;j<N;j++){
		cpu_complex[j]=cpu_real[j]+I*cpu_imag[j];
	}



	Arnoldi::deallocate_real(2,cpu_real, cpu_imag);
	Arnoldi::device_deallocate_all_real(2, gpu_real, gpu_imag);

}




void cublas_complex_to_device_real(int N, cublasComplex *gpu_complex, real* gpu_real, real* gpu_imag){


	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	cublasComplex_to_real_kernel<<< blocks, threads>>>(N, gpu_complex, gpu_real, gpu_imag);

}



__global__ void permute_matrix_colums_kernel(int MatrixRaw, int coloms, int *sorted_list_d, cublasComplex *vec_source,  cublasComplex *vec_dest){


	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<MatrixRaw){
	
		for(int j=0;j<coloms;j++){
			int index=sorted_list_d[j];
			vec_dest[I2(i,j,MatrixRaw)]=vec_source[I2(i,index,MatrixRaw)];
		}

	}
	
}



void permute_matrix_colums(int MatrixRaw, int coloms, int *sorted_list_d, cublasComplex *vec_source,  cublasComplex *vec_dest){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(MatrixRaw+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	permute_matrix_colums_kernel<<< blocks, threads>>>(MatrixRaw, coloms, sorted_list_d, vec_source,  vec_dest);

}



__global__ void  RHS_of_eigenproblem_real_device_kernel(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=lambda_real*Vec_real[i]-lambda_imag*Vec_imag[i];
		

	}



}


__global__ void  RHS_of_eigenproblem_imag_device_kernel(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=lambda_imag*Vec_real[i]+lambda_real*Vec_imag[i];
		

	}



}


void RHS_of_eigenproblem_device_real(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	RHS_of_eigenproblem_real_device_kernel<<< blocks, threads>>>(N, lambda_real, Vec_real, lambda_imag, Vec_imag, Vec_res);

}

void RHS_of_eigenproblem_device_imag(int N, real lambda_real, real* Vec_real, real lambda_imag, real* Vec_imag, real *Vec_res){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);

	RHS_of_eigenproblem_imag_device_kernel<<< blocks, threads>>>(N, lambda_real, Vec_real, lambda_imag, Vec_imag, Vec_res);

}


__global__ void  Residual_eigenproblem_device_kernel(int N, real* Vl_r_d, real* Vr_r_d, real* Vec_res){

	int i = blockIdx.x * blockDim.x + threadIdx.x;	

	if(i<N){
	
		Vec_res[i]=Vl_r_d[i]-Vr_r_d[i];
		

	}

}

void Residual_eigenproblem_device(int N, real* Vl_r_d, real* Vr_r_d, real* Vre_d){

	dim3 threads(BLOCKSIZE);
	int blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
	dim3 blocks(blocks_x);
	
	Residual_eigenproblem_device_kernel<<< blocks, threads>>>(N, Vl_r_d, Vr_r_d, Vre_d);


}


void get_upper_matrix_part_host(int N_source, real *source_matrix, real *dest_matrix, int N_dist){

	for(int i=0;i<N_dist;i++)
		for(int j=0;j<N_dist;j++){
			dest_matrix[I2(i,j,N_dist)]=source_matrix[I2(i,j,N_source)];
		}


}



void permute_matrix_colums(int MatrixRaw, int coloms, int *sorted_list, real complex *vec_source,  real complex *vec_dest){

	for(int i=0;i<MatrixRaw;i++){
	
		for(int j=0;j<coloms;j++){
			int index=sorted_list[j];
			vec_dest[I2(i,j,MatrixRaw)]=vec_source[I2(i,index,MatrixRaw)];
		}

	}


}



//which: 
//		"LR" - largest real, "LM" - largest magnitude
//


real Implicit_restart_Arnoldi_GPU_data(cublasHandle_t handle, bool verbose, int N, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], int k, int m, complex real* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real_d, real *eigenvectors_imag_d, int BLASThreads){
	//wrapper without external routine like matrix Exponent


	real ritz_norm=1.0;
	ritz_norm=Implicit_restart_Arnoldi_GPU_data_Matrix_Exponent(handle, verbose, N,  Axb, user_struct, Axb, user_struct, vec_f_d, which, which, k, m, eigenvaluesA, tol, max_iter, eigenvectors_real_d, eigenvectors_imag_d, BLASThreads);

	return ritz_norm;
}



real Implicit_restart_Arnoldi_GPU_data_Matrix_Exponent(cublasHandle_t handle, bool verbose, int N,  user_map_vector Axb_exponent_invert, void *user_struct_exponent_invert, user_map_vector Axb, void *user_struct, real *vec_f_d, char which[2], char which_exponent[2], int k, int m, complex real* eigenvaluesA, real tol, int max_iter, real *eigenvectors_real_d, real *eigenvectors_imag_d, int BLASThreads){


	
	openblas_set_num_threads(BLASThreads); //sets number of threads to be used by OpenBLAS
	
	real *vec_c=new real[m];
	real *vec_h=new real[m];
	real *vec_q=new real[m];
	real *H=new real[m*m];
	real *R=new real[m*m];
	real *Q=new real[m*m];
	real *H1=new real[m*m];
	real *H2=new real[m*m];
	matrixZero(m, m, H);
	matrixZero(m, m, R);
	matrixZero(m, m, Q);
	matrixZero(m, m, H1);
	matrixZero(m, m, H2);
	real complex *eigenvectorsH=new real complex[m*m];
	real complex *eigenvaluesH=new real complex[m*m];
	real complex *eigenvectorsH_kk=new real complex[k*k];
	real complex *eigenvectorsH_kk_sorted=new real complex[k*k];
	real complex *eigenvaluesH_kk=new real complex[k*k];
	real *ritz_vector=new real[m];

	real *V_d, *V1_d, *Q_d; //matrixes on GPU
	real *vec_f1_d, *vec_v_d, *vec_w_d, *vec_c_d, *vec_h_d, *vec_q_d; //vectors on GPU
	//real *Vl_r_d, *Vl_i_d, *Vr_r_d, *Vr_i_d, *Vre_d, *Vim_d; //vectors on GPU for eigenvector residuals
	//real *eigenvectors_real_d, *eigenvectors_imag_d;	//Matrix Eigenvectors
	bool external_eigenvectors=true;
	if(eigenvectors_real_d==NULL){
		external_eigenvectors=false;
		Arnoldi::device_allocate_all_real(N,k, 1, 2, &eigenvectors_real_d, &eigenvectors_imag_d);
	}

	Arnoldi::device_allocate_all_real(N,m, 1, 2, &V_d, &V1_d);
	Arnoldi::device_allocate_all_real(N, 1,1, 3, &vec_f1_d, &vec_w_d, &vec_v_d);
	Arnoldi::device_allocate_all_real(m, 1,1, 3, &vec_c_d, &vec_h_d, &vec_q_d);
	Arnoldi::device_allocate_all_real(m,m, 1, 1, &Q_d);
	//Arnoldi::device_allocate_all_real(N, 1,1, 6, &Vl_r_d, &Vl_i_d, &Vr_r_d, &Vr_i_d, &Vre_d, &Vim_d);
	//sets initial guesses for Krylov vectors
	Arnoldi::set_initial_Krylov_vector_value_GPU(N, vec_f1_d);
	Arnoldi::set_initial_Krylov_vector_value_GPU(N, vec_v_d);
	Arnoldi::set_initial_Krylov_vector_value_GPU(N, vec_w_d);

	// Allocate memroy for eigenvectors!
	cublasComplex *eigenvectorsH_d, *eigenvectorsA_d, *eigenvectorsA_unsorted_d;

	eigenvectorsH_d=Arnoldi::device_allocate_complex(k, k, 1);
	eigenvectorsA_d=Arnoldi::device_allocate_complex(N, k, 1);
	eigenvectorsA_unsorted_d=Arnoldi::device_allocate_complex(N, k, 1);


//	cublasHandle_t handle;		//init cublas
//	cublasStatus_t ret;
//	ret = cublasCreate(&handle);
//	Arnoldi::checkError(ret, " cublasCreate(). ");


	int k0=1;
	int iterations=0;
	real ritz_norm=1.0;
	timer_start();
	while(((iterations++)<max_iter)&&(ritz_norm>tol)){
	
		Arnoldi_driver(handle, N, Axb_exponent_invert, user_struct_exponent_invert, V_d, H, vec_f_d, k0-1, m, vec_v_d, vec_w_d, vec_c_d, vec_h_d, vec_h);	//Build orthogonal Krylov subspace
		

		select_shifts(m, H, which_exponent, eigenvectorsH, eigenvaluesH, ritz_vector); //select basisi shift depending on 'which'

		QR_shifts(k, m, Q, H, eigenvaluesH, &k0); //Do QR shifts of basis. Returns active eigenvalue indexes and Q-matrix for basis shift
		
		real vec_f_norm=Arnoldi::vector_norm2_GPU(handle, N, vec_f_d); 					
		for(int i=0;i<k0;i++){
			ritz_vector[i]=ritz_vector[i]*vec_f_norm;
		}
		get_matrix_colomn(m, m, Q, vec_q, k0);	
		real hl=H[I2(k0,k0-1,m)];
		real ql=Q[I2(m-1,k0-1,m)];
       	//f = V*vec_q*hl + f*ql;
		Arnoldi::to_device_from_host_real_cpy(vec_q_d, vec_q, m, 1,1); //vec_q -> vec_q_d
		Arnoldi::matrixMultVector_GPU(handle, N, V_d, m, hl, vec_q_d, ql, vec_f_d);
		//matrixMultVector(N, V, m, hl, vec_q, ql, vec_f1, vec_f);	//GG
		
		//fix this shit!!! V
		//we must apply Q only as matrix mXk0 on a submatrix  V NXm!!!
		for(int i=0;i<m;i++){
			for(int j=k0;j<m;j++){
				Q[I2(i,j,m)]=1.0*delta(i,j);
			}
		}
		//Copy matrixQtoGPUmemory!
		//here!
		Arnoldi::to_device_from_host_real_cpy(Q_d, Q, m, m, 1); //Q -> Q_d
		Arnoldi::matrixMultMatrix_GPU(handle, N, m, m, V_d, 1.0, Q_d, 0.0, V1_d);	//OK
		
		//matrix_copy(N, m, V1, V);									//GG
		Arnoldi::vector_copy_GPU(handle, N*m, V1_d, V_d);

		ritz_norm=vector_normC(k0,ritz_vector);
		if(verbose){
			printf("it=%i, ritz norms=", iterations);
			for(int ll=0;ll<k0;ll++){
				printf("%0.3le ",(double)ritz_vector[ll]);
			}
			printf("\n");
		}

		else{
		//	if(iterations%50==0)
		//		printf("it=%i, ritz norm_C=%.05e \n", iterations, ritz_norm);
		}
	
	}
	timer_stop();
	timer_print();

	if(verbose)
		printf("\ncomputing original map eigenvectors and eigenvalues...\n");

	//test Schur!
	real *Q_Schur=new real[k*k];
	real *H_Schur=new real[k*k];
	//get_upper_matrix_part_host(m, H, H_Schur, k);
	for(int i=0;i<k;i++){
		for(int j=0;j<k;j++){
			H_Schur[I2(i,j,k)]=H[I2(i,j,m)];
		}
	}
	print_matrix("H_pre.dat", k, k, H_Schur);
	//check pre-Galerkin eigenvalues of H matrix
	real complex *HC1=new real complex[k*k];
	for(int i=0;i<k;i++){
		for(int j=0;j<k;j++){
			HC1[I2(i,j,k)]=H_Schur[I2(i,j,k)]+0.0*I;
			//HC[I2(i,j,k)]=H[I2(i,j,m)]+0.0*I;
		}
	}
	MatrixComplexEigensystem(eigenvectorsH_kk, eigenvaluesH_kk, HC1, k);
	delete [] HC1;
	printf("\n Eigenvalues of H matrix before Galerkin projection:\n");
  	for(int i=0;i<k;i++){ 
  		real ritz_val=ritz_vector[i];
  		printf("\n (%.08le, %.08le), ritz: %.04le",  (double) creal(eigenvaluesH_kk[i]), (double) cimag(eigenvaluesH_kk[i]), (double)ritz_val );
  	}

	//check ends

	Schur_Hessinberg_matrix(H_Schur, k, Q_Schur); //returns Q as orthogonal matrix whose columns are the Schur vectors and the input matrix is overwritten as an upper quasi-triangular matrix (the Schur form of input matrix)


	print_matrix("H_Schur.dat", k, k, H_Schur);
	print_matrix("Q_Schur.dat", k, k, Q_Schur);
	//compute eigenvectors
   	//[Q,R] = schur(H(1:ko,1:ko));
   	//V = V(:,1:ko)*Q; <--- eigenvectors
	//R= V'*(A*V);
	//eigens=eig(R); <--- eigenvalues
	//residual: resid = norm(A*V - V*R);
	real *Q_Schur_d;
	real *Vcolres_d, *VRres_d;

	//real *V1_temp=new real[N*k];

	Arnoldi::device_allocate_all_real(k, k, 1, 1, &Q_Schur_d);
	Arnoldi::device_allocate_all_real(N, k, 1, 2, &Vcolres_d, &VRres_d);

	Arnoldi::to_device_from_host_real_cpy(Q_Schur_d, Q_Schur, k, k,1);
	Arnoldi::matrixMultMatrix_GPU(handle, N, k, k, V_d, 1.0, Q_Schur_d, 0.0, V1_d);	//Vectors are in V1_d!!!
	
	//Arnoldi::to_host_from_device_real_cpy(V1_temp, V1_d, N, k, 1);
	//print_matrix("V1_d.dat", N, k, V1_temp);	
	

	//form Vcolres_d=A*V1_d
	for(int i=0;i<k;i++){
		Axb(user_struct, &V1_d[i*N], &Vcolres_d[i*N]);
		Arnoldi::check_for_nans("IRA: Schur basis projeciton out", N, &Vcolres_d[i*N]);
	}
	Arnoldi::matrixTMultMatrix_GPU(handle, k, k, N, V1_d, 1.0, Vcolres_d, 0.0, Q_Schur_d);	//Vectors are in V1_d!!! Q_Schur_d := R in matlab
//	Arnoldi::to_host_from_device_real_cpy(V1_temp, Vcolres_d, N, k, 1);
//	print_matrix("Vcol_d.dat", N, k, V1_temp);

	//delete [] V1_temp;	

	//check residual!
	real *residualAV=new real[k];
	for(int i=0;i<k;i++){
		Axb(user_struct, &V1_d[i*N], &Vcolres_d[i*N]);
		Arnoldi::check_for_nans("IRA: Schur basis projeciton out in residual", N, &Vcolres_d[i*N]);
	}	
	Arnoldi::matrixMultMatrix_GPU(handle, N, k, k, V1_d, 1.0, Q_Schur_d, 0.0, VRres_d);
	for(int i=0;i<k;i++){
		Arnoldi::vectors_add_GPU(handle, N, -1.0, &Vcolres_d[i*N], &VRres_d[i*N]);
		residualAV[i]=Arnoldi::vector_norm2_GPU(handle, N, &VRres_d[i*N]);
	}
	//done

	Arnoldi::to_host_from_device_real_cpy(H_Schur, Q_Schur_d, k, k, 1);
	//print_matrix("RRR.dat", k, k, H_Schur);
	Arnoldi::device_deallocate_all_real(3, Q_Schur_d,Vcolres_d, VRres_d);

	//170820 stopped here!!!

	real complex *HC=new real complex[k*k];
	for(int i=0;i<k;i++){
		for(int j=0;j<k;j++){
			HC[I2(i,j,k)]=H_Schur[I2(i,j,k)]+0.0*I;
			//HC[I2(i,j,k)]=H[I2(i,j,m)]+0.0*I;
		}
	}
	MatrixComplexEigensystem(eigenvectorsH_kk, eigenvaluesH_kk, HC, k);

	delete [] HC;
	delete [] Q_Schur, H_Schur;

	int *sorted_list=new int[k];
	int *sorted_list_d=Arnoldi::device_allocate_int(k, 1, 1);
	get_sorted_index(k, which, eigenvaluesH_kk, sorted_list);

	//sort eigenvectors of Shur form of Hessinberg matrix
	permute_matrix_colums(k, k, sorted_list, eigenvectorsH_kk,  eigenvectorsH_kk_sorted);
	// Now store EigenvectorsH to GPU as cublasComplex.
	real_complex_to_cublas_complex(k*k, eigenvectorsH_kk_sorted,  eigenvectorsH_d);

	real_device_to_cublas_complex(N*k, V_d, eigenvectorsA_unsorted_d);
	Arnoldi::matrixMultComplexMatrix_GPU(handle, N, k, k, eigenvectorsA_unsorted_d, eigenvectorsH_d, eigenvectorsA_d); //here eigenvectorsA_d contain sorted eigenvectors of original problem

	cudaFree(sorted_list_d);
	delete [] sorted_list;
	cudaFree(eigenvectorsH_d);
	cudaFree(eigenvectorsA_unsorted_d);

	if(verbose)
		printf("\ndone\n");



	printf("\nNumber of correct eigenvalues=%i Eigenvalues: \n", k);
  	for(int i=0;i<k;i++){ 
  		real ritz_val=ritz_vector[i];
  		printf("\n (%.08le, %.08le), residual: %.04le",  (double) creal(eigenvaluesH_kk[i]), (double) cimag(eigenvaluesH_kk[i]), (double)residualAV[i] );
  	}
	printf("\n");
	delete [] residualAV;

	//get Real and Imag parts of eigenvectors
	cublas_complex_to_device_real(N*k, eigenvectorsA_d, eigenvectors_real_d, eigenvectors_imag_d);

	bool do_plot=true;
	if((verbose)&&(do_plot)){
		printf("plotting output matrixes and vectors...\n");
		real *vec_f_local=new real[N];
		real *V_local=new real[N*m];
		real *V1_local=new real[N*m];
		Arnoldi::to_host_from_device_real_cpy(vec_f_local, vec_f_d, N, 1, 1); //vec_f_d -> vec_f
		Arnoldi::to_host_from_device_real_cpy(V_local, V_d, N, m, 1); //vec_V_d -> vec_V
		Arnoldi::to_host_from_device_real_cpy(V1_local, V1_d, N, m, 1);
		real complex *eigenvectorsA=new real complex[N*k];

		cublas_complex_to_complex_real(N*k, eigenvectorsA_d, eigenvectorsA);

		real *V_real_local=new real[N*k];
		real *V_imag_local=new real[N*k];
		Arnoldi::to_host_from_device_real_cpy(V_real_local, eigenvectors_real_d, N, k, 1);
		Arnoldi::to_host_from_device_real_cpy(V_imag_local, eigenvectors_imag_d, N, k, 1);

		print_matrix("EigVecA.dat", N, k, eigenvectorsA);
		print_matrix("V1.dat", N, k, V1_local);
		print_matrix("V_real.dat", N, k, V_real_local);//eigenvectors_real_d
		print_matrix("V_imag.dat", N, k, V_imag_local);//eigenvectors_imag_d
		print_matrix("V.dat", N, k, V_local);
		print_matrix("H.dat", m, m, H);
		print_matrix("H1.dat", m, m, H1);
		print_matrix("H2.dat", m, m, H2);
		print_matrix("R.dat", m, m, R);
		print_matrix("Q.dat", m, m, Q);	
		print_matrix("EigVecH.dat", k, k, eigenvectorsH_kk_sorted);
		print_vector("EigH.dat", k, eigenvaluesH_kk);
		print_vector("f.dat", N, vec_f_local);	

		delete [] eigenvectorsA, vec_f_local, V_local, V1_local;
		delete [] V_real_local,V_imag_local;
		printf("done\n");
	}
	cudaFree(eigenvectorsA_d);
	if(!external_eigenvectors){
		cudaFree(eigenvectors_real_d);
		cudaFree(eigenvectors_imag_d);
	}

	Arnoldi::device_deallocate_all_real(9, V_d, V1_d, vec_f1_d, vec_w_d, vec_v_d, vec_c_d, vec_h_d, vec_q_d, Q_d);

	//Arnoldi::device_deallocate_all_real(6, Vl_r_d, Vl_i_d, Vr_r_d, Vr_i_d, Vre_d, Vim_d);

	//free cublas
	//cublasDestroy(handle);
	
	delete [] vec_c, vec_h, vec_q;
	delete [] H, R, Q, H1, H2;
	delete [] eigenvectorsH, eigenvaluesH, eigenvectorsH_kk, eigenvectorsH_kk_sorted, eigenvaluesH_kk, ritz_vector;


	return ritz_norm;


}