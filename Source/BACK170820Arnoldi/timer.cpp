#include "timer.h"





void timer_start(void){
	gettimeofday(&start_w, NULL);
}

void timer_stop(void){
	gettimeofday(&end_w, NULL);
}

double timer_read_seconds(void){

	double etime=((end_w.tv_sec-start_w.tv_sec)*1000000u+(end_w.tv_usec-start_w.tv_usec))/1.0E6;
	return etime;
}

void timer_print(void){
	double seconds_total=timer_read_seconds();
	int days=(int)seconds_total/60.0/60.0/24.0;
	int hours=(int)seconds_total/60.0/60.0-days*24.0;
	int minutes=(int)seconds_total/60.0-hours*60-days*24*60;
	double seconds=seconds_total-minutes*60.0-hours*60.0*60.0-days*24.0*60.0*60.0;
	printf("\nWall time (%f sec):",seconds_total);
	if((days==0)&&(hours==0)&&(minutes==0)){
		printf("%.03f sec.\n",seconds);
	}
	else if((days==0)&&(hours==0)){
		printf("%i min. %.03f sec.\n",minutes, seconds);
	}
	else if(days==0){
		printf("%i h. %i min. %.03f sec.\n",hours, minutes, seconds);
	}		
	else{
		printf("%i days %i h. %i min. %.03f sec.\n",days, hours, minutes, seconds);	
	}
}
