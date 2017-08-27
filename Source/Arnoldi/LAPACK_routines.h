#ifndef __ARNOLDI_LAPACK_routines_H__
#define __ARNOLDI_LAPACK_routines_H__

#include "Macros.h"
#include "Products.h"
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
//
#include "file_operations.h"

extern "C" void zgeev_(char *jobvl, char *jobvr, int *n, complex real *a,
              int *lda, complex real *w, complex real *vl,
              int *ldvl, complex real *vr, int *ldvr,
              complex real *work, int *lwork, complex real *rwork, int *info);

extern "C" void cgeev_(char *jobvl, char *jobvr, int *n, complex real *a,
              int *lda, complex real *w, complex real *vl,
              int *ldvl, complex real *vr, int *ldvr,
              complex real *work, int *lwork, complex real *rwork, int *info);

extern "C"	void dgeqrf_(int* M, int* N, 
                    real* A, int* LDA, real* TAU, 
                    real* WORK, int* LWORK, int* INFO );
extern "C"  void sgeqrf_(int* M, int* N, 
                    real* A, int* LDA, real* TAU, 
                    real* WORK, int* LWORK, int* INFO );

extern "C"	void dormqr_(char*  SIDE, char* TRANS, 
                    int* M, int* N, int* K, 
                    real* A, int* LDA, real* TAU, 
                    real* C, int* LDC,
                    real* WORK, int* LWORK, int* INFO );
extern "C"  void sormqr_(char*  SIDE, char* TRANS, 
                    int* M, int* N, int* K, 
                    real* A, int* LDA, real* TAU, 
                    real* C, int* LDC,
                    real* WORK, int* LWORK, int* INFO );

extern "C" void dhseqr_(char*  JOB,char*  COMPZ, 
					int* N, int* ILO, 
					int* IHI,	real *H,
					int* LDH, 
					real *WR, real *WI,
					real *Z, int* LDZ,
					real* WORK,
					int* LWORK,int *INFO);
extern "C" void shseqr_(char*  JOB,char*  COMPZ, 
          int* N, int* ILO, 
          int* IHI, real *H,
          int* LDH, 
          real *WR, real *WI,
          real *Z, int* LDZ,
          real* WORK,
          int* LWORK,int *INFO);


void MatrixComplexEigensystem( real complex *eigenvectorsVR, real complex *eigenvaluesW, real complex *A, int N);


void QR_square_matrix(real *A, int N, real *Q, real *R);

void QR_square_matrix_no_R(real *A, int N, real *Q);

void Schur_Hessinberg_matrix(real *H, int Nl, real *Q);


#endif