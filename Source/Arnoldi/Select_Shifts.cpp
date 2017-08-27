#include "Select_Shifts.h"
#include "Products.h"
#include "LAPACK_routines.h"




void check_nans_H(int N, real *H){
	for(int i=0;i<N;i++)
		if(H[i]!=H[i]){
			printf("\nNANs in H-matrix detected!!!\n");
		}


}



int struct_cmp_by_value(const void *a, const void *b) 
{ 
	struct sort_struct *ia = (struct sort_struct *)a;
	struct sort_struct *ib = (struct sort_struct *)b;
	real val_a=ia->value;
	real val_b=ib->value;
	if(val_a>val_b)
		return 1;
	else if (val_a<val_b)
		return -1;
	else
		return 0;
 
}



void get_sorted_index(int m, char which[2],  real complex *eigenvaluesH, int *sorted_list){

	sort_struct *eigs_struct=new sort_struct[m];
	real complex *eigs_local=new real complex[m];


	for(int i=0;i<m;i++){
		real value=0.0;
		if((which[0]=='L')&&(which[1]=='R'))
			value=-creal(eigenvaluesH[i]);
		else if((which[0]=='L')&&(which[1]=='M'))
			value=-cabs(eigenvaluesH[i]);

		eigs_struct[i].index=i;
		eigs_struct[i].value=value;
		eigs_local[i]=eigenvaluesH[i];
	}
	size_t structs_len = m;
	qsort(eigs_struct, structs_len, sizeof(struct sort_struct), struct_cmp_by_value);
	
	for(int i=0;i<m;i++){
		int j=eigs_struct[i].index;
		sorted_list[i]=j;
		eigenvaluesH[i]=eigs_local[j];
	}


	delete [] eigs_struct;
	delete [] eigs_local;
}




void filter_small_imags(int N, real complex *eigenvaluesH){

	for(int i=0;i<N;i++){
		real eig_imag=cimag(eigenvaluesH[i]);
		real eig_real=creal(eigenvaluesH[i]);

		if(fabsf(eig_imag)<Im_eig_tol)
			eig_imag=0.0;

		real complex Ctemp=eig_real+eig_imag*I;
		eigenvaluesH[i]=Ctemp;
	}
}



void  select_shifts(int m, real *H, char which[2], real complex *eigenvectorsH, real complex *eigenvaluesH, real *ritz_vector){

	real complex *HC=new real complex[m*m];
	real complex *eigs_local=new real complex[m];
	//real complex *eigvs_local=new real complex[m*m];
	
	check_nans_H(m*m, H);

	real_to_complex_matrix(m, m, H, HC);

	MatrixComplexEigensystem(eigenvectorsH, eigs_local, HC, m);
	
	//filter_small_imags(m, eigs_local);

	sort_struct *eigs_struct=new sort_struct[m];
	for(int i=0;i<m;i++){
		real value=0.0;
		if((which[0]=='L')&&(which[1]=='R'))
			value=-creal(eigs_local[i]);
		else if((which[0]=='L')&&(which[1]=='M'))
			value=-cabs(eigs_local[i]);

		eigs_struct[i].index=i;
		eigs_struct[i].value=value;

	}

	size_t structs_len = m;
	qsort(eigs_struct, structs_len, sizeof(struct sort_struct), struct_cmp_by_value);

	for(int i=0;i<m;i++){
		int j=eigs_struct[i].index;
		eigenvaluesH[i]=eigs_local[j];
		//for(int k=0;k<m;k++){
		//	eigenvectorsH[I2(k,i,m)]=eigvs_local[I2(k,j,m)];
		//}


		ritz_vector[i]=cabs(eigenvectorsH[I2(m-1,j,m)]);
	}




	delete [] eigs_struct;
	delete [] HC;
	delete [] eigs_local;
	//delete [] eigvs_local;
}