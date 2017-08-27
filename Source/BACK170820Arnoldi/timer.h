#ifndef __ARNOLDI_timer_H__
#define __ARNOLDI_timer_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> //for timer
#include "Macros.h"

static struct timeval start_w, end_w;
void timer_start(void);
void timer_stop(void);
void timer_print(void);


#endif
