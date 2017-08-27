#include "QR_Shifts.h"





void QR_step(real *V, real *H, real eig_Re, real eig_Im, int m, real *H1, real *H2, real *Q){

	if(abs(eig_Im)>Im_eig_tol){
		matrixAddToDiagonal(m, m, H, -eig_Re, H1); //ok
		matrixPower2(H1, m, m, H2); //ok
		real eig_Im2=(eig_Im*eig_Im);
		matrixAddToDiagonal(m, m, H2, eig_Im2, H1); //ok
		//result in H1
	}
	else{
		matrixAddToDiagonal(m, m, H, -eig_Re, H1);
		//result in H1
	}


	QR_square_matrix_no_R(H1, m, Q); //ok

	transpose_matrix(m, Q, H1);  //H1 now contains Q^T //ok

	matrixmul(H1, m, m, H, m, m, H2); //H2=Q^T H; //ok
	matrixmul(H2, m, m, Q, m, m, H); //H = H2 Q;

	for(int j=0;j<m-1;j++){
		for(int i=m-1;i>j+1;i--){
			H[I2(i,j,m)]=0.0;	//remove noise below first subdiagonal
		}
	}


	matrixmul(V, m, m, Q, m, m, H1); //H1=V Q;
	matrix_copy(m, m, H1, V);


}




void QR_shifts(int k, int m, real *Q, real *H, complex real *eigenvaluesH, int *ko){
	//Q input is an eye matrix that we make here

real *H1=new real[m*m];
real *H2=new real[m*m];
real *Q_loc=new real[m*m];  //temp matrixes

int j=m;
	Ident(m, Q);
	Ident(m, Q_loc);
	while(j>k){
		
		real eig_Re=creal(eigenvaluesH[j-1]);
		real eig_Im=cimag(eigenvaluesH[j-1]);

		QR_step(Q, H, eig_Re, eig_Im, m, H1, H2, Q_loc);
		
		if(fabsf(eig_Im)>Im_eig_tol){
			j=j-2;
		}else{
			j=j-1;
		}
	}
	ko[0]=j; //set number of vectors to be made by another iteration of Arnoli procedure


delete [] H1, H2, Q_loc;
}

