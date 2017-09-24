#include "Newton.h"







int Newton(cublasHandle_t cublasHandle, user_map_vector Jacobi_Axb, void *user_struct_Jacobi,  user_map_vector RHS_Axb, void *user_struct_RHS, int N, real *x, real *tol, int *iter, real tol_linsolve, int iter_linsolve, bool verbose, unsigned int skip){

	real *tolBCG=new real[1];
	int *iterBCG=new int[1];


	tolBCG[0]=tol_linsolve;
	iterBCG[0]=iter_linsolve;
	int L=6;

//	GPU arrays:
  	real *delta_d, *bn_d, *v_d;
  	Arnoldi::device_allocate_all_real(N, 1 ,1, 3, &delta_d, &bn_d, &v_d);
  	Arnoldi::set_initial_Krylov_vector_value_GPU(N, delta_d);
  	//Arnoldi::set_initial_Krylov_vector_value_GPU(N, x);

  	real tolNewton=1.0;
  	int flag=1;
  	int ll=0;
  	if(verbose)
  		printf("\nNewton: Linsolve parameters: tol=%le, max iterations=%i\n", (double)tolBCG[0], iterBCG[0]);

  	RHS_Axb(user_struct_RHS, x, bn_d);
  	real norm_RHS=Arnoldi::vector_norm2_GPU(cublasHandle, N, bn_d);
  	if(norm_RHS<tol[0]){
  		flag=0;
  		printf("\nNewton: Steady state solution is provided by the initial guess with tolerance=%le\n", (double)norm_RHS);
  	}
  	else{

	  	for(ll=0;ll<iter[0];ll++){

			tolBCG[0]=tol_linsolve;
			iterBCG[0]=iter_linsolve;

	  		RHS_Axb(user_struct_RHS, x, bn_d); //all to get b=F(u);
			Arnoldi::set_vector_inverce_GPU(N, bn_d);

			int flag_linsolve=BiCGStabL(cublasHandle, L, N, Jacobi_Axb, user_struct_Jacobi, delta_d, bn_d, tolBCG, iterBCG, true, 100); // call to get delta=inv(J) b;
			switch (flag_linsolve){
				case 0: printf("%i|", iterBCG[0]); 
						fflush(stdout);
						break;
				case 1: printf("\nNewton LinSolve: not converged with: %le, and %i iterations\n", tolBCG[0], iterBCG[0]);
						exit(-1);
						break;
				case -1: printf("\nNewton LinSolve: rho is 0 with: %le, and %i iterations\n", tolBCG[0], iterBCG[0]);
						exit(-1);
						break;
				case -2: printf("\nNewton LinSolve: omega is with: %le, and %i iterations\n", tolBCG[0], iterBCG[0]);
						exit(-1);
						break;
				case -3: printf("\nNewton LinSolve: NANs with: %le, and %i iterations\n", tolBCG[0], iterBCG[0]);
						exit(-1);
						break;
			}

			real alpha=0.8;	//must be adjusted!
			Arnoldi::vectors_add_GPU(cublasHandle, N, alpha, delta_d, x);

			tolNewton=Arnoldi::vector_norm2_GPU(cublasHandle, N, delta_d);
			real norm_solution=Arnoldi::vector_norm2_GPU(cublasHandle, N, x);
			norm_RHS=Arnoldi::vector_norm2_GPU(cublasHandle, N, bn_d);
			if(norm_solution<1.0e-15){
				printf("\nNewton Warning: solution 2norm=%le!", (double)norm_solution);
				norm_solution=1.0;	
			} 


			if((verbose)&&(ll%skip==0)){
				printf("Newton: Iteration %i, solution norm is %le, RHS norm is %le, Newton tolerance is %le\n", ll, (double)norm_solution, (double)norm_RHS, (double)tolNewton);
				fflush(stdout);
			}


			if((norm_RHS<tol[0])&&(ll>2)){
				tol[0]=tolNewton;		
				flag=0;
				break;
			}


		}
	}
	iter[0]=ll;

	Arnoldi::device_deallocate_all_real(3, delta_d, bn_d, v_d);

	return flag;
}
