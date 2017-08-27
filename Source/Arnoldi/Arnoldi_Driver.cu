#include "Arnoldi_Driver.h"


/*
   Input:  A - an n by n matrix
           V - an n by k orthogonal matrix
           H - a k  by k upper Hessenberg matrix
           f - an nonzero n vector
 
           with   AV = VH + fe_k' (if k > 1)

           k - a positive integer (k << n assumed)

           m - a positive integer (k < m << n assumed)
          

   Output: V - an n by m orthogonal matrix
           H - an m by m upper Hessenberg matrix
           f - an n vector

 
           with   AV = VH + fe_m'

           Leading k columns of V agree with input V (!)
*/



void Arnoldi_driver(cublasHandle_t handle, int N, user_map_vector Axb, void *user_struct, real *V_d, real *H, real *vec_f_d, int k, int m, real *vec_v_d, real *vec_w_d, real *vec_c_d, real *vec_h_d, real *vec_h){
	
	real tolerance=1.0e-12;
	#ifdef real_float
		tolerance=1.0e-8;
	#endif
	
	if(k==0){


		real beta = Arnoldi::vector_norm2_GPU(handle, N, vec_f_d);
		if(beta<1.0e-15){
			printf( "Norm of the input vector ||f||<1e-15! Quit!\n");
        	exit(-1); 
		}
		Arnoldi::vector_copy_GPU(handle, N, vec_f_d, vec_v_d);
		Arnoldi::normalize_vector_GPU(handle, N, vec_v_d);
//		Call matrix_vector value on GPU
//		Using user definded funciton via structure	
        real vvpp = Arnoldi::vector_norm2_GPU(handle, N, vec_v_d);
    	Arnoldi::check_for_nans("\nFirst vector to Axb\n", N, vec_v_d);
        Axb(user_struct, vec_v_d, vec_w_d);
        Arnoldi::check_for_nans("\nFirst call of Axb\n", N, vec_w_d);
//    	if(use_matrix==1)
//        call_vector_map_GPU(handle, N, A_d, vec_v_d, vec_w_d);   
//    	else
//     	  call_vector_map_GPU(N, vec_v_d, vec_w_d);  //change this for abstract function call!!!
//

		real alpha=Arnoldi::vector_dot_product_GPU(handle, N, vec_v_d, vec_w_d);   //GG
		Arnoldi::vector_copy_GPU(handle, N, vec_w_d, vec_f_d); //vec_w -> vec_f
    	Arnoldi::vectors_add_GPU(handle, N, -alpha, vec_v_d, vec_f_d); //f = w - v*alpha; % orthogonalization once.  //GG
		real c=1.0;
    	int it=0;
       	while(c>tolerance){
        	it++;
       		c=Arnoldi::vector_dot_product_GPU(handle, N, vec_v_d, vec_f_d);    //(vec_v,vec_f)
       		Arnoldi::vectors_add_GPU(handle, N, -c, vec_v_d, vec_f_d);   //vec_f=vec_f-c*vec_v
        	alpha+=c;  
        	if(it>10){
        		printf("\nArnoldi orthogonalization failed at k==0: %.05e\n", c);
        		break;
        	}
            Arnoldi::check_for_nans("\nFirst reorthogonalization\n", N, vec_f_d);
       	}
       	H[I2(0,0,m)]=alpha;  //set H in HOST!
       	Arnoldi::set_matrix_colomn_GPU(N, m, V_d, vec_v_d, 0);  //V(:,0)=vec_v
	}
	for(int j=k+1;j<m;j++){

		real beta = Arnoldi::vector_norm2_GPU( handle, N, vec_f_d);   
		Arnoldi::vector_copy_GPU(handle, N, vec_f_d, vec_v_d);          
		Arnoldi::normalize_vector_GPU(handle, N, vec_v_d);                  
 		H[I2(j,j-1,m)] = beta;  //set H in HOST!
 		Arnoldi::set_matrix_colomn_GPU(N, m, V_d, vec_v_d, j);   //V(:,j)=vec_v
//	Call matrix_vector value on GPU
//    	if(use_matrix==1)
//    		call_vector_map_GPU(handle, N, A_d, vec_v_d, vec_w_d);   
//    	else
//    		call_vector_map_GPU(N, vec_v_d, vec_w_d);

        Arnoldi::check_for_nans("\nOther vector to Axb\n", N, vec_v_d);
 		Axb(user_struct, vec_v_d, vec_w_d);
        Arnoldi::check_for_nans("\nOther call of Axb\n", N, vec_w_d);
	    Arnoldi::set_vector_value_GPU(m, 0.0, vec_h_d); //set all to 0
	    Arnoldi::set_vector_value_GPU(m, 0.0, vec_c_d);
	    
	   
	    Arnoldi::matrixDotVector_part_GPU(handle, N, V_d, m, 1.0, vec_w_d, j+1, 0.0, vec_h_d);  //from 0 to j STRICT!
	    Arnoldi::vector_copy_GPU(handle, N, vec_w_d, vec_f_d); //vec_w -> vec_f
    	Arnoldi::matrixMultVector_part_GPU(handle, N, V_d, m, -1.0, vec_h_d, j+1, 1.0, vec_f_d);

		//matrixMultVector_part(N, V, m, 0, j, -1.0, vec_h, 1.0, vec_w, vec_f); //from 0 to j strict!  

		real c=1.0;
    	int it=0;
       	while(c>tolerance){
        	it++;
         
            Arnoldi::matrixDotVector_part_GPU(handle, N, V_d, m, 1.0, vec_f_d, j+1, 0.0, vec_c_d);     
        	Arnoldi::matrixMultVector_part_GPU(handle, N, V_d, m, -1.0, vec_c_d, j+1, 1.0, vec_f_d);
        	Arnoldi::vectors_add_GPU(handle, m, 1.0, vec_c_d, vec_h_d);
       		c=Arnoldi::vector_norm2_GPU(handle, m, vec_c_d);                                
          	if(it>10){
            	printf("\nArnoldi orthogonalization failed: %.05e at %i\n", c, j);
            	break;
       		}
            Arnoldi::check_for_nans("\nOther reorthogonalization\n", N, vec_f_d);
        }
		Arnoldi::to_host_from_device_real_cpy(vec_h, vec_h_d, m, 1,1); //vec_h_d -> vec_h
 		set_matrix_colomn(m, m, H, vec_h, j);   //set H on HOST
	}


}
