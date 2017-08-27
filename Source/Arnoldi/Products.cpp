#include "Products.h"


real vector_dot_product(int N, real *vec1, real *vec2){

	real dot=0.0;
	for(int j=0;j<N;j++)
		dot+=vec1[j]*vec2[j];

	return(dot);
}


real vector_norm2(int N, real *vec){

	real norm=vector_dot_product(N, vec, vec);
	norm=sqrt(norm);
	return norm;

}

real vector_normC(int N, real *vec){

	real norm=0.0;
	for(int j=0;j<N;j++)
		norm=max2(norm,fabsf(vec[j]));

	return norm;

}


int normalize(int N, real *vec){

	real norm=vector_norm2(N, vec);
	if(norm>1E-15){
		for(int j=0;j<N;j++)
			vec[j]/=norm;
		return 0;
	}
	else
		return -1;

}

void transpose_matrix(int N, real *matrix, real *matrixT){


int i,j;
for(i=0;i<N;i++)
  for(j=0;j<N;j++) 
  	matrixT[I2t(i,j,N)] = matrix[I2(i,j,N)];


}
	
void transpose_matrix(int N, real complex *matrix, real complex *matrixT)
{

int i,j;
for(i=0;i<N;i++)
	for(j=0;j<N;j++)
		matrixT[I2t(i,j,N)] = conj(matrix[I2(i,j,N)]);

}


int matrixmul(real *A, int RowA, int ColA, real *B, int RowB, int ColB, real *C){

if(ColA=RowB){
	for (int i=0; i<RowA; i++) { // Do each row of A with each column of B
		for (int j=0; j<ColB; j++) {
//			real dummy=0;
			C[i+j*RowB]=0;
			for (int k=0; k<ColA; k++) {
				C[i+j*RowA]+=A[i+k*RowA]*B[k+j*RowB];
			} // End for k
		} // End for j
	} // End for i

	return 0;
}
else
	return -1;
}



int matrixPower2(real *A, int RowA, int ColA, real *C){
	
	int res=matrixmul(A, RowA, ColA, A, RowA, ColA, C);

	return res;
}

void Ident(int N, real *A){

	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			A[I2(i,j,N)]=0;
		}
		A[I2(i,i,N)]=1;
	}

}

void matrixAdd(int RowA, int ColA, real *A, real factor, real *B, real *C){
	int N=ColA;
	for (int i=0; i<RowA; i++) { 
		for (int j=0; j<ColA; j++) {
			C[I2(i,j,RowA)]=A[I2(i,j,RowA)]+factor*B[I2(i,j,RowA)];
		}
	}

}

void matrixMultVector(int RowA, real *A, int ColA, real val_y, real *y, real val_z, real *z, real *res){ //   res=val_y*A*y+val_z*z

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=0;j<ColA;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)]*val_y;			
		}
		res[i]+=val_z*z[i];
	}

}

void matrixMultVector(int RowA, real *A, int ColA, real *y, real *res){

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=0;j<ColA;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)];			
		}
	}

}



void matrixMultVector(int RowA, real complex *A, int ColA, real complex *y, real complex *res){

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=0;j<ColA;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)];			
		}
	}

}


void matrixDotVector(int RowA,  real *A, int ColA, real *y, real *res){

	int N=ColA;
	for(int j=0;j<ColA;j++){
		res[j]=0.0;
		for(int i=0;i<RowA;i++){
			res[j]+=y[i]*A[I2(i,j,RowA)];			
		}
	}

}


void matrixMultVector_part(int RowA, real  *A, int ColA, int from_Col, int to_Col, real  *y, real *res){

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=from_Col;j<=to_Col;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)];			
		}
	}

}

void matrixMultVector_part(int RowA, real *A, int ColA, int from_Col, int to_Col,  real val_y, real *y, real val_z, real *z, real *res){ //   res=val_y*A*y+val_z*z

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=from_Col;j<=to_Col;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)]*val_y;			
		}
		res[i]+=val_z*z[i];
	}

}



void matrixMultVector_part(int RowA, real complex *A, int ColA, int from_Col, int to_Col,  real val_y, real complex *y, real complex *res){ //   res=val_y*A*y+val_z*z

	int N=ColA;
	for(int i=0;i<RowA;i++){
		res[i]=0.0;
		for(int j=from_Col;j<=to_Col;j++){
			res[i]+=y[j]*A[I2(i,j,RowA)]*val_y;			
		}
	}

}



void matrixDotVector_part(int RowA,  real *A, int ColA, int from_Col, int to_Col, real *y, real *res){

	int N=ColA;
	for(int j=from_Col;j<=to_Col;j++){
		res[j]=0.0;
		for(int i=0;i<RowA;i++){
			res[j]+=y[i]*A[I2(i,j,RowA)];			
		}
	}

}




void vector_set_val(int N, real *vec, real val){

	for(int i=0;i<N;i++){
		vec[i]=val;
	}

}

void vector_copy(int N, real *vec_source, real *vec_dest){

	for(int i=0;i<N;i++){
		vec_dest[i]=vec_source[i];
	}

}

void matrix_copy(int RowA, int ColA, real *A_source, real *A_dest){

	int N=ColA;
	for(int j=0;j<ColA;j++){
		for(int i=0;i<RowA;i++){
			A_dest[I2(i,j,RowA)]=A_source[I2(i,j,RowA)];
		}
	}
}

void matrix_copy(int RowA, int ColA, real complex *A_source, real complex *A_dest){

	int N=ColA;
	for(int j=0;j<ColA;j++){
		for(int i=0;i<RowA;i++){
			A_dest[I2(i,j,RowA)]=A_source[I2(i,j,RowA)];
		}
	}
}


void vector_add(int N, real v1, real *vec1, real v2, real *vec2, real *vec_dest){

	for(int i=0;i<N;i++){
		vec_dest[i]=v1*vec1[i]+v2*vec2[i];
	}

}


void set_matrix_colomn(int Row, int Col, real *mat, real *vec, int col_number){

	int N=Col;
	for(int i=0;i<Row;i++){
		mat[I2(i,col_number,Row)]=vec[i];
	}


}

void get_matrix_colomn(int Row, int Col, real *mat, real *vec, int col_number){

	int N=Col;
	for(int i=0;i<Row;i++){
		vec[i]=mat[I2(i,col_number,Row)];
	}


}


void real_to_complex_matrix(int Row, int Col, real *input_matrix, real complex *output_matrix){


	int N=Col;
	for(int j=0;j<Col;j++){
		for(int i=0;i<Row;i++){
			real real_part=input_matrix[I2(i,j,Row)];
			output_matrix[I2(i,j,Row)]=real_part+0.*I;
		}
	}

}


real delta(int i, int j){
	if(i==j) 
		return 1.0;
	else 
		return 0.0;
}


void matrixAddToDiagonal(int RowA, int ColA, real *A, real factor, real *B){
	int N=ColA;
	for (int i=0; i<RowA; i++) { 
		for (int j=0; j<ColA; j++) {
			B[I2(i,j,RowA)]=A[I2(i,j,RowA)]+factor*delta(i,j);
		}
	}

}


void matrixZero(int RowA, int ColA, real *A){
	int N=ColA;
	for (int i=0; i<RowA; i++) { 
		for (int j=0; j<ColA; j++) {
			A[I2(i,j,RowA)]=0.0;
		}
	}

}


