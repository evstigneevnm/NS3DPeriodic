#ifndef __ARNOLDI_file_operations_H__
#define __ARNOLDI_file_operations_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "Macros.h"


void print_vector(const char *f_name, int N, real *vec);
void print_vector(const char *f_name, int N, real complex *vec);
void print_matrix(const char *f_name, int Row, int Col, real *matrix);
void print_matrix(const char *f_name, int Row, int Col, real complex *matrix);
int read_matrix(const char *f_name,  int Row, int Col,  real *matrix);
int read_vector(const char *f_name,  int N,  real *vec);



#endif