#ifndef __ARNOLDI_H_Select_Shifts_H__
#define __ARNOLDI_H_Select_Shifts_H__

#include "Macros.h"
#include "Products.h"
	

struct sort_struct{ 
   	int index;
   	real value;
};


void get_sorted_index(int m, char which[2],  real complex *eigenvaluesH, int *sorted_list);

void  select_shifts(int m,real *H, char which[2], real complex *eigenvectorsH, real complex *eigenvaluesH, real *ritz_vector);


#endif
