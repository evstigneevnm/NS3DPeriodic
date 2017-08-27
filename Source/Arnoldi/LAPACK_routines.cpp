#include "LAPACK_routines.h"




void MatrixComplexEigensystem( real complex *eigenvectorsVR, real complex *eigenvaluesW, real complex *A, int N)
{

int i;
  
real complex *AT = (real complex*) malloc( N*N*sizeof(real complex) );

//transpose_matrix(N, A, AT);
matrix_copy(N, N, A, AT);

char JOBVL ='N';   // Compute Right eigenvectors

char JOBVR ='V';   // Do not compute Left eigenvectors

real complex VL[1];

int LDVL = 1; 
int LDVR = N;

int LWORK = 4*N; 

real complex *WORK =  (real complex*)malloc( LWORK*sizeof(real complex));

real complex *RWORK = (real complex*)malloc( 2*N*sizeof(real complex));

 
int INFO;

#ifdef real_double
	zgeev_( &JOBVL, &JOBVR, &N, AT ,  &N , eigenvaluesW ,
	   	VL, &LDVL,
	   	eigenvectorsVR, &LDVR, 
	   	WORK, 
	   	&LWORK, RWORK, &INFO );
#endif
#ifdef real_float	
	cgeev_( &JOBVL, &JOBVR, &N, AT ,  &N , eigenvaluesW ,
		VL, &LDVL,
	   	eigenvectorsVR, &LDVR, 
	   	WORK, 
	   	&LWORK, RWORK, &INFO );
#endif	


//transpose_matrix(N, eigenvectorsVR, AT);


//for(i=0;i<N*N;i++) 
//	eigenvectorsVR[i]=AT[i];

 
	free(WORK);
	free(RWORK);
	free(AT);
}


void QR_square_matrix(real *A, int N, real *Q, real *R){


/*
SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )

*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M-by-N matrix A.
*          On exit, the elements on and above the diagonal of the array
*          contain the min(M,N)-by-N upper trapezoidal matrix R (R is
*          upper triangular if m >= n); the elements below the diagonal,
*          with the array TAU, represent the orthogonal matrix Q as a
*          product of min(m,n) elementary reflectors (see Further
*          Details).
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
*          The scalar factors of the elementary reflectors (see Further
*          Details).
*
*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*
*  LWORK   (input) INTEGER
*          The dimension of the array WORK.  LWORK >= max(1,N).
*          For optimum performance LWORK >= N*NB, where NB is
*          the optimal blocksize.
*
*          If LWORK = -1, then a workspace query is assumed; the routine
*          only calculates the optimal size of the WORK array, returns
*          this value as the first entry of the WORK array, and no error
*          message related to LWORK is issued by XERBLA.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*  Further Details
*  ===============
*
*  The matrix Q is represented as a product of elementary reflectors
*
*     Q = H(1) H(2) . . . H(k), where k = min(m,n).
*
*  Each H(i) has the form
*
*     H(i) = I - tau * v * v'
*
*  where tau is a real scalar, and v is a real vector with
*  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
*  and tau in TAU(i).
*
*  =====================================================================
*/

/*
SUBROUTINE DORMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
     $                   WORK, LWORK, INFO )

*  Purpose
*  =======
*
*  DORMQR overwrites the general real M-by-N matrix C with
*
*                  SIDE = 'L'     SIDE = 'R'
*  TRANS = 'N':      Q * C          C * Q
*  TRANS = 'T':      Q**T * C       C * Q**T
*
*  where Q is a real orthogonal matrix defined as the product of k
*  elementary reflectors
*
*        Q = H(1) H(2) . . . H(k)
*
*  as returned by DGEQRF. Q is of order M if SIDE = 'L' and of order N
*  if SIDE = 'R'.
*
*  Arguments
*  =========
*
*  SIDE    (input) CHARACTER*1
*          = 'L': apply Q or Q**T from the Left;
*          = 'R': apply Q or Q**T from the Right.
*
*  TRANS   (input) CHARACTER*1
*          = 'N':  No transpose, apply Q;
*          = 'T':  Transpose, apply Q**T.
*
*  M       (input) INTEGER
*          The number of rows of the matrix C. M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix C. N >= 0.
*
*  K       (input) INTEGER
*          The number of elementary reflectors whose product defines
*          the matrix Q.
*          If SIDE = 'L', M >= K >= 0;
*          if SIDE = 'R', N >= K >= 0.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,K)
*          The i-th column must contain the vector which defines the
*          elementary reflector H(i), for i = 1,2,...,k, as returned by
*          DGEQRF in the first k columns of its array argument A.
*          A is modified by the routine but restored on exit.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.
*          If SIDE = 'L', LDA >= max(1,M);
*          if SIDE = 'R', LDA >= max(1,N).
*
*  TAU     (input) DOUBLE PRECISION array, dimension (K)
*          TAU(i) must contain the scalar factor of the elementary
*          reflector H(i), as returned by DGEQRF.
*
*  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
*          On entry, the M-by-N matrix C.
*          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
*
*  LDC     (input) INTEGER
*          The leading dimension of the array C. LDC >= max(1,M).
*
*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*
*  LWORK   (input) INTEGER
*          The dimension of the array WORK.
*          If SIDE = 'L', LWORK >= max(1,N);
*          if SIDE = 'R', LWORK >= max(1,M).
*          For optimum performance LWORK >= N*NB if SIDE = 'L', and
*          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
*          blocksize.
*
*          If LWORK = -1, then a workspace query is assumed; the routine
*          only calculates the optimal size of the WORK array, returns
*          this value as the first entry of the WORK array, and no error
*          message related to LWORK is issued by XERBLA.
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================

*/

	real *AT=new real[N*N];
	real *TAU=new real[N];
	//transpose_matrix(N, A, AT);
	matrix_copy(N, N, A, AT);
	
	int M=N,LDA=N;
	int INFO=0;
	int LWORK = 4*N; 
	real *WORK=new real[LWORK];

	#ifdef real_double		
	//DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
		dgeqrf_(&M, &N, AT, &LDA, TAU, WORK, &LWORK, &INFO);
		if(INFO!=0){
			printf("DGEQRF: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif
	#ifdef real_float	
		//FGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
		sgeqrf_(&M, &N, AT, &LDA, TAU, WORK, &LWORK, &INFO);
		if(INFO!=0){
			printf("SGEQRF: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif	




	//transpose_matrix(N, AT, R);
	matrix_copy(N, N, AT, R);
	
	for(int j=0;j<N-1;j++){
		for(int i=N-1;i>j;i--){
			R[I2(i,j,N)]=0.0;	//remove (v)-s below diagonal
		}
	}
	

	char SIDE='L';
	char TRANS='N'; //transposed output
	int K=N, LDC=N;
	Ident(N, Q);
	#ifdef real_double	
	dormqr_(&SIDE, &TRANS, &M, &N, &K, 
                    AT, &LDA, TAU, 
                    Q, &LDC,
                    WORK, &LWORK, &INFO );
	if(INFO!=0){
		printf("DORMQR: Argument %i has an illegal value. Aborting.\n", INFO);
		exit(-1);
	}

	#endif	
	#ifdef real_float		
	sormqr_(&SIDE, &TRANS, &M, &N, &K, 
                    AT, &LDA, TAU, 
                    Q, &LDC,
                    WORK, &LWORK, &INFO );
	if(INFO!=0){
		printf("SORMQR: Argument %i has an illegal value. Aborting.\n", INFO);
		exit(-1);
	}
	#endif




	//we have A=QR in Q and in R =)

	delete [] AT, WORK, TAU;
}

void QR_square_matrix_no_R(real *A, int N, real *Q){


	real *AT=new real[N*N];
	real *TAU=new real[N];
	//transpose_matrix(N, A, AT);
	matrix_copy(N, N, A, AT);
	int M=N,LDA=N;
	int INFO=0;
	int LWORK = 4*N; 
	real *WORK=new real[LWORK];
	#ifdef real_double	
		//DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
		dgeqrf_(&M, &N, AT, &LDA, TAU, WORK, &LWORK, &INFO);
		if(INFO!=0){
			printf("DGEQRF: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif	
	#ifdef real_float		
		//FGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
		sgeqrf_(&M, &N, AT, &LDA, TAU, WORK, &LWORK, &INFO);
		if(INFO!=0){
			printf("SGEQRF: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif





	char SIDE='L';
	char TRANS='N'; //transposed output
	int K=N, LDC=N;
	Ident(N, Q);
	#ifdef real_double	
		dormqr_(&SIDE, &TRANS, &M, &N, &K, 
	                    AT, &LDA, TAU, 
	                    Q, &LDC,
	                    WORK, &LWORK, &INFO );
		if(INFO!=0){
			printf("DORMQR: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif	
	#ifdef real_float		
		sormqr_(&SIDE, &TRANS, &M, &N, &K, 
	                    AT, &LDA, TAU, 
	                    Q, &LDC,
	                    WORK, &LWORK, &INFO );
		if(INFO!=0){
			printf("SORMQR: Argument %i has an illegal value. Aborting.\n", INFO);
			exit(-1);
		}
	#endif




	//we have A=QR in Q and no R =)

	delete [] AT, WORK, TAU;
}



void Schur_Hessinberg_matrix(real *H, int N, real *Q){

/*
F08PEF (DHSEQR) computes all the eigenvalues and, optionally, the Schur factorization of a real Hessenberg matrix or a real general matrix which has been reduced to Hessenberg form.
F08PEF (DHSEQR) computes all the eigenvalues and, optionally, the Schur factorization of a real upper Hessenberg matrix H: H=Z T Z^T,
where T is an upper quasi-triangular matrix (the Schur form of H), and Z is the orthogonal matrix whose columns are the Schur vectors z_i


extern "C" void dhseqr_(char*  JOB,char*  COMPZ, 
					int* N, int* ILO, 
					int* IHI,	real *H,
					int* LDH, 
					real *WR, real *WI,
					real *Z, int* LDZ,
					real* WORK,
					int* LWORK,int *INFO);


1:	JOB='E'
	Eigenvalues only are required.
	JOB='S'
	The Schur form T is required.
	
2:     COMPZ – CHARACTER*1Input
	On entry: indicates whether the
	Schur vectors are to be computed.

	COMPZ='N'
	No Schur vectors are computed (and the array Z is not referenced).
	COMPZ='I' The Schur vectors of H are computed (and the array Z is initialized by the routine).
	COMPZ='V'The Schur vectors of A are computed (and the array Z must contain the matrix Q on entry). 
	Constraint:  COMPZ='N', 'V' or 'I'.
3:  N – INTEGERInput
	On entry: n, the order of the matrix H.

4: ILO – INTEGER
5: IHI – INTEGER
	IHI – INTEGERInput 
	On entry: if the matrix A has been balanced by
	F08NHF (DGEBAL), then ILO and IHI must contain the values returned by that routine. Otherwise, ILO must be set to 1 and IHI to N.Constraint:
  	ILO≥1 and min⁡ILO,N≤IHI≤N.

6:  H(LDH,*) – real precision array
	Note: the second dimension of the array H must be at least max⁡1,N.
	On entry: the n by n upper Hessenberg matrix H, as returned by 	F08NEF (DGEHRD).
	On exit: if JOB=E, the array contains no useful information. 
	If JOB=S, H is overwritten by the upper	quasi-triangular matrix T from the Schur decomposition (the Schur form) unless INFO>0.	

7:  LDH – INTEGER
	On entry: the first dimension of the array H as declared in the (sub)program from which F08PEF (DHSEQR) is called.Constraint: LDH≥max⁡1,N.
8:  WR(*) – real precision arrayOutput
9:  WI(*) – real precision arrayOutput
	Note: the dimension of the array WR and WI must be at least max⁡1,N.

	On exit: the real and imaginary parts, respectively, of the computed eigenvalues, unless INFO>0 (in which case see Section 6). Complex conjugate pairs of eigenvalues appear consecutively with the eigenvalue having positive imaginary part first. The eigenvalues are stored in the same order as on the diagonal of the Schur form T (if computed); see Section 8 for details.

10:  Z(LDZ,*) – real precision arrayInput/Output
	Note: the second dimension of the array Z must be at least max⁡1,N if COMPZ = 'V' or 'I' and at least 1 if COMPZ=N. 
	On entry: if COMPZ=V, Z must contain the orthogonal matrix Q from the reduction to Hessenberg form.
	If COMPZ=I, Z need not be set.
	On exit: if COMPZ = 'V' or 'I', Z contains the orthogonal matrix of the required Schur vectors, unless INFO>0.
	If COMPZ=N, Z is not referenced.

11:  LDZ – INTEGERInput
	On entry: the first dimension of the array Z as declared in the (sub)program from which F08PEF (DHSEQR) is called.Constraints:
   if COMPZ = 'V' or 'I', LDZ≥max⁡1,N; 
   if COMPZ=N, LDZ≥1. 

12:  WORK(*) – real precision arrayWorkspace
	Note: the dimension of the array WORK must be at least max⁡1,LWORK.
	On exit: if INFO=0, WORK1 contains the minimum value of LWORK required for optimal performance.

13: LWORK – INTEGERInput
	On entry: the dimension of the array WORK as declared in the (sub)program from which F08PEF (DHSEQR) is called, unless LWORK=-1, in which case a workspace query is assumed and the routine only calculates the minimum dimension of WORK.Constraint:
  	LWORK≥max⁡1,N or LWORK=-1.


14: INFO – INTEGEROutputOn exit: INFO=0 unless the routine detects an error (see Section 6).

Results:
	If all the computed eigenvalues are real, T is upper triangular, and the diagonal elements of T are the eigenvalues; WRi=tii, for i=1,2,…,n and WIi=0.0.
	
	If some of the computed eigenvalues form complex conjugate pairs, then T has 2 by 2 diagonal blocks.  Each diagonal block has the form

	(  t_{i,i}     t_{i,i+1}  )     (α    β)
	(                         )  =  (      )
	(  t_{i+1,i}  t_{i+1,i+1} )     (γ    α)
 


	where βγ<0.  The corresponding eigenvalues are α±sqrt(βγ); WRi=WRi+1=α; WIi=+sqrt(|βγ|); WIi+1=-WIi.
	
	test:

	real *H=new real(4*4);
	int m=4;
	H[I2m(0,0)]=0.350000000000000; H[I2m(0,1)]=-0.116000000000000;  H[I2m(0,2)]=-0.388600000000000; H[I2m(0,3)]=-0.294200000000000;
	H[I2m(1,0)]=-0.514000000000000; H[I2m(1,1)]=0.122500000000000;  H[I2m(1,2)]=0.100400000000000; H[I2m(1,3)]=0.112600000000000;
	H[I2m(2,0)]=0; H[I2m(2,1)]=0.644300000000000;  H[I2m(2,2)]=-0.135700000000000; H[I2m(2,3)]=-0.0977000000000000;
	H[I2m(3,0)]=0; H[I2m(3,1)]=0;  H[I2m(3,2)]=0.426200000000000; H[I2m(3,3)]=0.163200000000000;

	H=Z T Z':

	Z=		
	-0.655090395498507	-0.345013851172607	-0.103603585070971	0.664144798507900
	0.597210340109255	-0.170562202236235	0.524576452614526	0.582295363365769
	0.384529263999376	-0.714337682936540	-0.578893336510767	-0.0821061800864399
	0.257553156688664	0.584461848544650	-0.615604050885172	0.461630124243687

	T=
	0.799520573289046	0.00605342023308397	-0.114442607963615	-0.0335132570006511
	0					-0.0994330557135366	-0.648335317381854	-0.202608709663576
	0					0.247741704276945	-0.0994330557135366	-0.347395407590160
	0					0					0					-0.100654461861972

	eigs:
	   7.9952e-01 
	  -9.9433e-02 + 4.0077e-01i
	  -9.9433e-02 - 4.0077e-01i
	  -1.0065e-01 

*/

/*	
	//DEBUG!!!!

	real *Ht=new real[4*4];
	int m=4;
	Ht[I2(0,0,m)]=0.350000000000000; Ht[I2(0,1,m)]=-0.116000000000000;  Ht[I2(0,2,m)]=-0.388600000000000; Ht[I2(0,3,m)]=-0.294200000000000;
	Ht[I2(1,0,m)]=-0.514000000000000; Ht[I2(1,1,m)]=0.122500000000000;  Ht[I2(1,2,m)]=0.100400000000000; Ht[I2(1,3,m)]=0.112600000000000;
	Ht[I2(2,0,m)]=0; Ht[I2(2,1,m)]=0.644300000000000;  Ht[I2(2,2,m)]=-0.135700000000000; Ht[I2(2,3,m)]=-0.0977000000000000;
	Ht[I2(3,0,m)]=0; Ht[I2(3,1,m)]=0;  Ht[I2(3,2,m)]=0.426200000000000; Ht[I2(3,3,m)]=0.163200000000000;
	

	int N=4;//debug!
	print_matrix("Ht.dat", N, N, Ht);
	real *Z=new real [N*N];
*/

	print_matrix("Ht.dat", N, N, H);
	char JOB='S';
	char COMPZ='I';
	int ILO=1;
	int IHI=N;
	int LDH=N;
	real *WR=new real [N];
	real *WI=new real [N];
	real *HT=new real [N*N];

	//transpose_matrix(N, Ht, HT);
	int LDZ=N;
	int INFO=0;
	int LWORK = 4*N; 
	real *WORK=new real[LWORK];

	//DEBUG: change /*Ht*/ ... in funcction call

	dhseqr_(&JOB, &COMPZ, &N, &ILO, 
					&IHI, H	/*Ht*/,
					&LDH, 
					WR, WI,
					/*Z*/ Q, &LDZ,
					WORK,
					&LWORK, &INFO);


	if(INFO!=0){
		printf("DHSEQR: Argument %i has an illegal value. Aborting.\n", INFO);
		exit(-1);
	}
/*
	//DEBUG!!!
	//transpose_matrix(N, HT, Ht);
	print_matrix("T.dat", N, N, Ht);
	//transpose_matrix(N, Z, HT);
	print_matrix("Z.dat", N, N, Z);
	
	delete [] Ht, HT ,Z;
*/
	
	delete [] WORK, WR, WI; 
	

}