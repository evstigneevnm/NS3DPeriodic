#ifndef __ARNOLDI_file_operations_H__
#define __ARNOLDI_file_operations_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include "Macros.h"

using namespace std;
void print_vector(const char *f_name, int N, creal *vec);
void print_vector(const char *f_name, int N, complex<creal> *vec);
void print_matrix(const char *f_name, int Row, int Col, creal *matrix);
void print_matrix(const char *f_name, int Row, int Col, complex<creal> *matrix);
int read_matrix(const char *f_name,  int Row, int Col,  creal *matrix);
int read_vector(const char *f_name,  int N,  creal *vec);



#endif