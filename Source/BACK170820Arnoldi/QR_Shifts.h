#ifndef __ARNOLDI_QR_Shifts_H__
#define __ARNOLDI_QR_Shifts_H__

#include "Macros.h"
#include "Products.h"
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include "LAPACK_routines.h"



void QR_shifts(int k, int m, real *Q, real *H, complex real *eigenvaluesH, int *ko);


#endif