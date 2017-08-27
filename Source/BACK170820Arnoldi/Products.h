#ifndef __ARNOLDI_H_PRODUCTS_H__
#define __ARNOLDI_H_PRODUCTS_H__

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include "Macros.h"


void matrixDotVector(int RowA,  real *A, int ColA, real *y, real *res);

void set_matrix_colomn(int Col, int Row, real *mat, real *vec, int col_number);

void get_matrix_colomn(int Row, int Col, real *mat, real *vec, int col_number);

void vector_copy(int N, real *vec_source, real *vec_dest);

void vector_add(int N, real v1, real *vec1, real v2, real *vec2, real *vec_dest);

void vector_set_val(int N, real *vec, real val);

int normalize(int N, real *vec);

real vector_dot_product(int N, real *vec1, real *vec2);

real vector_norm2(int N, real *vec);

real vector_normC(int N, real *vec);

void transpose_matrix(int N, real *matrix, real *matrixT);
	
void transpose_matrix(int N, real complex *matrix, real complex *matrixT);

int matrixmul(real *A, int RowA, int ColA, real *B, int RowB, int ColB, real *C);

int matrixPower2(real *A, int RowA, int ColA, real *C);

void Ident(int N, real *A);

void matrixAdd(int RowA, int ColA, real *A, real factor, real *B, real *C);

real delta(int i, int j);

void matrixMultVector(int RowA, real *A, int ColA, real *y, real *res);

void matrixMultVector(int RowA, real *A, int ColA, real val_y, real *y, real val_z, real *z, real *res);

void matrixMultVector(int RowA, real complex *A, int ColA, real complex *y, real complex *res);


void matrixMultVector_part(int RowA, real *A, int ColA, int from_Col, int to_Col, real *y, real *res);

void matrixDotVector_part(int RowA,  real *A, int ColA, int from_Col, int to_Col, real *y, real *res);

void matrixMultVector_part(int RowA, real *A, int ColA, int from_Col, int to_Col,  real val_y, real *y, real val_z, real *z, real *res);

void matrixMultVector_part(int RowA, real complex *A, int ColA, int from_Col, int to_Col,  real val_y, real complex *y, real complex *res);


void real_to_complex_matrix(int Row, int Col, real *input_matrix, real complex *output_matrix);

void matrix_copy(int RowA, int ColA, real *A_source, real *A_dest);

void matrix_copy(int RowA, int ColA, real complex *A_source, real complex *A_dest);

void matrixAddToDiagonal(int RowA, int ColA, real *A, real factor, real *B);

void matrixZero(int RowA, int ColA, real *A);

#endif