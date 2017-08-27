//#include "BiCGStabL.h"
#include "Matrix_Vector_emulator.h"

void get_residualL(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *source, real *RHS, real* r){
	

    Axb(user_struct, source, r);
    Arnoldi::vectors_add_GPU(handle, N, -1.0, RHS, r);
    Arnoldi::set_vector_inverce_GPU(N, r);
    
}
    
real get_errorL(cublasHandle_t handle,int N, real *r, real RHSnorm){
	return Arnoldi::vector_norm2_GPU(handle, N, r)/RHSnorm;
}


int BiCGStabL(cublasHandle_t handle, int L, int N, user_map_vector Axb, void *user_struct, real *x, real* RHS, real *tol, int *Iter, bool verbose, unsigned int skip){
	

	int iter = 0; //number of iterations
	int flag = 1; //termiantion flag
	real error = 1.0; //residual
	real tollerance = tol[0];
  	int maxIter = Iter[0];
//  unsigned int skip=round(maxIter/50.0);

//  CUBLAS init
//    cublasHandle_t handle; 
//    cublasStatus_t ret;
//    ret = cublasCreate(&handle);
//    Arnoldi::checkError(ret, " cublasCreate(). ");

//	CPU arrays:
    real *tau, *sigma, *gamma, *gamma_p, *gamma_pp;

    tau=new real[L*L];
    for(int j=0;j<L*L;j++)
    	tau[j]=0.0;
    sigma=new real[(L+1)];
    gamma=new real[(L+1)];
    gamma_p=new real[(L+1)];
    gamma_pp=new real[(L+1)];

//	GPU arrays:
  	real *r, *u, *rtilde, *v_help;

  	Arnoldi::device_allocate_all_real(N, (L+1),1, 2, &r, &u);
  	Arnoldi::device_allocate_all_real(N, 1,1, 2, &rtilde, &v_help);


	
    for(int j=0;j<=L;j++){
        Arnoldi::set_vector_value_GPU(N, 0.0, &r[j*N]);
        Arnoldi::set_vector_value_GPU(N, 0.0, &u[j*N]);
    }

  	real bnrm = Arnoldi::vector_norm2_GPU(handle, N, RHS);
	if(bnrm < 1.0e-15){
		if(verbose)
            printf( "||b||_2 <1e-15! assuming ||b||_2=1.0\n");
		bnrm = 1.0;
	}
    real vn = Arnoldi::vector_norm2_GPU(handle, N, x);
    if(verbose)
        printf("\n||b||_2=%le, ||x0||_2=%le\n",bnrm, vn);

	get_residualL(handle, N, Axb, user_struct, x, RHS, &r[0]);
	error = get_errorL(handle, N, &r[0], bnrm);
	if(error < tollerance){
		flag = 0;		
	}
    if(isnan(error)){
        printf("\nBiCGStab(L): Nans in user defined function!\n");
        flag = -3;
    }
	Arnoldi::vector_copy_GPU(handle, N, &r[0], rtilde); //vec_w -> vec_f
	Arnoldi::normalize_vector_GPU(handle, N, rtilde); 
	real rho = 1.0, alpha = 0.0, omega = 1.0;	
	if(flag == 1)
	for(iter=0; iter<maxIter; iter++){
		
		rho = -omega * rho;
		for (int j = 0; j < L; ++j){			//j=0,...L-1
			if ( rho == 0.0 ){ 					//check rho break
	        	flag=-1;
	        	rho=1; 
	     	}
    		real rho1 = Arnoldi::vector_dot_product_GPU(handle, N, &r[j*N], rtilde); //rho1=(r[j],rtilde)
    		real beta = alpha * rho1 / rho;	
    		rho = rho1;	
    		for (int i = 0; i <= j; ++i){
    			//vector_add(N, 1.0, &r[i*N], -beta, &u[i*N], &u[i*N]); //u[i]=r[i]-beta*u[i]
    			Arnoldi::vector_copy_GPU(handle, N, &r[i*N], v_help); //vec_w -> vec_f
    			Arnoldi::vectors_add_GPU(handle, N, -beta, &u[i*N], v_help);
    			Arnoldi::vector_copy_GPU(handle, N, v_help, &u[i*N]);
    		}
    		Axb(user_struct,&u[j*N], &u[(j+1)*N]); //u[j+1]=A*u[j]
           // real vn0 = Arnoldi::vector_norm2_GPU(handle, N, &u[(j+1)*N]);
           // printf("\n|u[(j+1)*N]|=%le\n",vn0);
    		
    		alpha = rho / Arnoldi::vector_dot_product_GPU(handle, N,  &u[(j+1)*N], rtilde); //alpha=rho/(u[j+1],rtilde)
			for (int i = 0; i <= j; ++i){
				Arnoldi::vectors_add_GPU(handle, N, -alpha, &u[(i+1)*N], &r[i*N]); //r[i]=r[i]-alpha.*u[i+1]
			}
			Axb(user_struct, &r[j*N], &r[(j+1)*N]); //r[j+1]=A*r[j] Krylov subspace
           // real vn1 = Arnoldi::vector_norm2_GPU(handle, N, &r[(j+1)*N]);
           // printf("\n|r[(j+1)*N]|=%le\n",vn1);

			Arnoldi::vectors_add_GPU(handle, N, alpha, &u[0], x); //x=x+alpha.*u[0]
	   	}
	   	for (int j = 1; j <= L; ++j){
	   		for (int i = 1; i < j; ++i){
	   			int ij = (j-1)*L + (i-1);			//calculation of index ij=(j-1)*L+(i-1) for matrix tau
	   			tau[ij] = Arnoldi::vector_dot_product_GPU(handle, N, &r[j*N], &r[i*N]) / sigma[i];
	   			Arnoldi::vectors_add_GPU(handle, N, -tau[ij], &r[i*N], &r[j*N]); //r[j]=r[j]-tau[i,j].*r[i]
	   		}
	   		sigma[j] = Arnoldi::vector_dot_product_GPU(handle, N, &r[j*N],&r[j*N]);					//sigma[j]=(r[j],r[j]);
    		gamma_p[j] = Arnoldi::vector_dot_product_GPU(handle, N, &r[0], &r[j*N]) / sigma[j];		//gamma_p[j]=(r[0],r[j])/sigma[j];
	   	}
	   	gamma[L] = gamma_p[L];						//gamma[L]=gamma_p[L];
    	omega = gamma[L];							//gamma=gamma[L];
    	for (int j = L-1; j >= 1; --j){
    		gamma[j] = gamma_p[j];					//gamma[j]=gamma_p[j]
    		for (int i = j+1; i <= L; ++i){
    			gamma[j] -= tau[(i-1)*L + (j-1)] * gamma[i]; //gamma[j]=gamma[j]-tau[j,i].*gamma[i];
    		}
    	}
    	for (int j = 1; j < L; ++j){
    		gamma_pp[j] = gamma[j+1];						//gamma_pp[j]=gamma[j+1]
    		for (int i = j+1; i < L; ++i){
    			gamma_pp[j] += tau[(i-1)*L + (j-1)] * gamma[i+1]; //gamma_pp[j]=gamma_pp[j]+tau[j,i]*gamma[i+1]
    		}
    	}
    	Arnoldi::vectors_add_GPU(handle, N, gamma[1], &r[0], x); 			//x=x+gamma[1].*r[0];
    	Arnoldi::vectors_add_GPU(handle, N, -gamma_p[L], &r[L*N], &r[0]);	//r[0]=r[0]-gamma_p[L].*r[L]
    	Arnoldi::vectors_add_GPU(handle, N, -gamma[L], &u[L*N], &u[0]);	//u[0]=u[0]-gamma[L].*u[L];
		
    	for (int j = 1; j < L; ++j) { 					//j=1,..L-1
    		Arnoldi::vectors_add_GPU(handle, N, gamma_pp[j], &r[j*N], x); //x=x+gamma_pp[j].*r[j]
    		Arnoldi::vectors_add_GPU(handle, N, -gamma_p[j], &r[j*N], &r[0]);	//r[0]=r[0]-gamma_p[j].*r[j]
    		Arnoldi::vectors_add_GPU(handle, N, -gamma[j], &u[j*N], &u[0]);	//u[0]=u[0]-gamma[j].*u[j]
    	}
    	//check convergence
        
    	error = get_errorL(handle, N, &r[0], bnrm);
    	if ( error < tollerance ){
        	flag = 0;
        	break;
		}
        if(error!=error){
            printf("\nBiCGStab(L): Nans in user defined function in iterations!\n");
            flag = -3;
            break;
        }
    	if((verbose)&&(iter%skip==0)){
      		printf("%i %le\n", iter, error);
    	}
    	if(flag<0)	//exit if nans or rho==0;
    		break;

	}

	//return residual and iteration count
  	Iter[0] = iter;
  	tol[0] = error;


    //clean up
	Arnoldi::device_deallocate_all_real(4, v_help, rtilde, u, r);

  	delete[] tau;
  	delete[] sigma;
  	delete[] gamma_pp;
  	delete[] gamma_p;
  	delete[] gamma;

    //delete CUBLAS
  
  //  cublasDestroy(handle);
  
  	//return flag
  	return flag;
}