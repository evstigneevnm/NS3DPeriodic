#include "file_operations.h"



void print_vector(const char *f_name, int N, real *vec){
		FILE *stream;
		stream=fopen(f_name, "w" );

		for (int i = 0; i < N; ++i)
		 {
		 	fprintf(stream, "%.16le\n",(double)vec[i]);
		 } 

		fclose(stream);


}


void print_vector(const char *f_name, int N, real complex *vec){
		FILE *stream;
		stream=fopen(f_name, "w" );

		for (int i = 0; i < N; ++i)
		 {
		 	fprintf(stream, "%.16le+%.16lei\n",(double complex)vec[i]);
		 } 

		fclose(stream);


}

void print_matrix(const char *f_name, int Row, int Col, real *matrix){
	int N=Col;
	FILE *stream;
	stream=fopen(f_name, "w" );
	for (int i = 0; i<Row; i++)
	{
		for(int j=0;j<Col;j++)
		{
	 	
		 	fprintf(stream, "%.16le ",(double) matrix[I2(i,j,Row)]);
	 	}
		fprintf(stream, "\n");
	} 
	
	fclose(stream);


}

void print_matrix(const char *f_name, int Row, int Col, real complex *matrix){
	int N=Col;
	FILE *stream;
	stream=fopen(f_name, "w" );
	for (int i = 0; i<Row; i++)
	{
		for(int j=0;j<Col;j++)
		{
	 		//if(cimag(matrix[I2(i,j)])<0.0)
			// 	fprintf(stream, "%.16le%.16leI ",matrix[I2(i,j)]);
			//else
			 	fprintf(stream, "%.16le+%.16lei ",(double complex)matrix[I2(i,j,Row)]);				
	 	}
		fprintf(stream, "\n");
	} 
	
	fclose(stream);


}


int read_matrix(const char *f_name,  int Row, int Col,  real *matrix){

	FILE *stream;
	stream=fopen(f_name, "r" );
	if (stream == NULL)
  	{
  		return -1;
  	}
  	else{
		for (int i = 0; i<Row; i++)
		{
			for(int j=0;j<Col;j++)
			{
				 double val=0;	
				 fscanf(stream, "%le",&val);				
				 matrix[I2(i,j,Row)]=(real)val;
		 	}
			
		} 
	
		fclose(stream);
		return 0;
	}

}

int read_vector(const char *f_name,  int N,  real *vec){

	FILE *stream;
	stream=fopen(f_name, "r" );
	if (stream == NULL)
  	{
  		return -1;
  	}
  	else{
		for (int i = 0; i<N; i++)
		{
			double val=0;	
			fscanf(stream, "%le",&val);				
			vec[i]=(real)val;			
		} 
	
		fclose(stream);
		return 0;
	}

}