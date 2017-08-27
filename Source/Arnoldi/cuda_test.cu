#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <complex.h>

#include "Macros.h"
#include "cuda_supp.h"
#include "memory_operations.h"
#include "file_operations.h"
#include "timer.h"
#include "Implicit_restart_Arnoldi.h"
#include "Matrix_Vector_emulator.h"

real rand_normal(real mean, real stddev)
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
            double result = n1*((double)stddev) + (double)mean;
            n2_cached = 1;
            return (real)result;
        }
    }
    else
    {
        n2_cached = 0;
        double result = n2*((double)stddev) + (double)mean;
        return (real)result;
    }
}


void set_matrix(int Row, int Col, real *mat, real val){
	for (int i = 0; i<Row; ++i)
	{
		for(int j=0;j<Col;j++)
		{

			mat[I2(i,j,Row)]=rand_normal(0.0, val);

		}
	}

}



int main (int argc, char const* argv[])
{
	

	srand ( time(NULL) ); //seed pseudo_random
		
	int N=2300000;
	int k=6;
	int m=k*5;
	
	real *V, *V1, *A, *H, *R, *Q, *H1, *H2; //matrixes on CPU
	real *vec_f1, *vec_v, *vec_w, *vec_c, *vec_h, *vec_f, *vec_q; //vectors on CPU

	real *V_d, *V1_d, *A_d, *H_d, *R_d, *Q_d, *H1_d, *H2_d; //matrixes on GPU
	real *vec_f1_d, *vec_v_d, *vec_w_d, *vec_c_d, *vec_h_d, *vec_f_d, *vec_q_d; //vectors on GPU

	real complex *eigenvaluesA=new real complex[k];

//AA	Arnoldi::allocate_real(N,N,1, 1,&A);
	Arnoldi::allocate_real(N,m,1, 2, &V, &V1);
	Arnoldi::allocate_real(N,1,1, 4, &vec_f1, &vec_f, &vec_w, &vec_v);
	Arnoldi::allocate_real(m,1,1, 3, &vec_c, &vec_h, &vec_q);
	Arnoldi::allocate_real(m,m,1, 5, &H, &H1, &H2, &R, &Q);
	//set initial matrixes and vectors
//AA	if(read_matrix("A0.dat", N, N, A)==-1){
//AA		set_matrix(N, N, A, 1.0);
//AA		print_matrix("A0.dat", N, N, A);
//AA	}
//XX	if(read_matrix("V0.dat", N, m, V)==-1){
//XX		set_matrix(N, m, V, 1.0);
//XX		print_matrix("V0.dat", N, m, V);
//XX	}	
	if(read_matrix("H.dat", m, m, H)==-1){
		set_matrix(m, m, H, 1.0);
		print_matrix("H.dat", m, m, H);
	}

//	if(read_vector("f0.dat",  N,  vec_f)==-1){
		for (int i = 0; i < N; ++i)
		{
			vec_f[i]=rand_normal(0.0, 1.0);
		}
//		print_vector("f0.dat", N, vec_f);
//	}
	if(read_vector("c0.dat",  m,  vec_c)==-1){
		for (int i = 0; i < m; ++i)
		{
			vec_c[i]=rand_normal(0.0, 1.0);
		}
		print_vector("c0.dat", m, vec_c);
	}
	printf("\nCPU Preperations done.\n");
	//preparations	
	if(!Arnoldi::InitCUDA(0)) {
		return 0;
	}
	cudaDeviceReset();		
//	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	
//AA	Arnoldi::device_allocate_all_real(N,N, 1, 1,&A_d);
//AA	Arnoldi::to_device_from_host_real_cpy(A_d, A, N, N,1);

	Arnoldi::device_allocate_all_real(N,1,1, 1, &vec_f_d);
	Arnoldi::to_device_from_host_real_cpy(vec_f_d, vec_f, N, 1,1);

	

	real res_tol=1.0;
	cublasHandle_t handle;		//init cublas
	cublasStatus_t ret;
	ret = cublasCreate(&handle);
	Arnoldi::checkError(ret, " cublasCreate(). ");

	Ax_struct *SC=new Ax_struct[1];
	SC->N=N;


	//definition of auxiliary vectors to be used in matrix Exponent estimation
	real *vec_step0_d, *vec_step1_d, *vec_step2_d;
	Arnoldi::device_allocate_all_real(N,1,1, 3, &vec_step0_d, &vec_step1_d, &vec_step2_d);


	Ax_struct_exponential *SC_exp=new Ax_struct_exponential[1];
	SC_exp->N=N;
	SC_exp->tau=1.0/100.0;
	SC_exp->T=100;
	SC_exp->shift_real=1.0+1.0e-3;

	SC_exp->BiCG_L=4;
	SC_exp->BiCG_tol=1.0e-9;
	SC_exp->BiCG_Iter=N;
	SC_exp->handle=handle;

	SC_exp->vec_step0=vec_step0_d;
	SC_exp->vec_step1=vec_step1_d;
	SC_exp->vec_step2=vec_step2_d;

//	res_tol=Implicit_restart_Arnoldi_GPU_data(handle, true, N, (user_map_vector) user_Ax_function_1, (Ax_struct_1 *) SC_1,  vec_f_d, "LR", k, m, eigenvaluesA, 1.0e-12, 2000);
	
//	res_tol=Implicit_restart_Arnoldi_GPU_data(handle, true, N, (user_map_vector) user_Ax_function, (Ax_struct *) SC,  vec_f_d, "LR", k, m, eigenvaluesA, 1.0e-12, 1000);

	res_tol=Implicit_restart_Arnoldi_GPU_data_Matrix_Exponent(handle, true, N, (user_map_vector) Axb_exponent_invert, (Ax_struct_exponential *) SC_exp, (user_map_vector) user_Ax_function, (Ax_struct *) SC, vec_f_d, "LR", "LM", k, m, eigenvaluesA, 1.0e-8, 1000);

	delete [] SC;
	delete [] SC_exp;
	Arnoldi::device_deallocate_all_real(3, vec_step0_d, vec_step1_d, vec_step2_d);
	

	if(res_tol>0.0)
		printf("\n convergence in norm_C=%.05e\n", res_tol);
	else
		printf("\n first desired eigenvalues are exact!\n");


	
	//clean up

//XX	to_host_from_device_real_cpy(A, A_d, N, N,1);
	Arnoldi::to_host_from_device_real_cpy(vec_f, vec_f_d, N, 1, 1);


//AA	Arnoldi::device_deallocate_all_real(1, A_d);
	Arnoldi::device_deallocate_all_real(1, vec_f_d);

/*
	print_matrix("A.dat", N, N, A);
	print_matrix("V.dat", N, m, V);
	print_matrix("V1.dat", N, m, V1);
	print_matrix("H.dat", m, m, H);
	print_vector("f.dat", N, vec_f);
	print_vector("w.dat", N, vec_w);
	print_vector("h.dat", m, vec_h);
*/
//AA	Arnoldi::deallocate_real(1, A);
	Arnoldi::deallocate_real(2, V, V1);
	Arnoldi::deallocate_real(4, vec_f1, vec_f, vec_w, vec_v);
	Arnoldi::deallocate_real(3, vec_c, vec_h, vec_q);
	Arnoldi::deallocate_real(5, H, H1, H2, R, Q);
    delete [] eigenvaluesA;
	cublasDestroy(handle);
    cudaDeviceReset();
	printf("\ndone.\n");
	return 0;
}

