#include "file_operations.h"


void read_control_file(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz){
	double ux_l, uy_l, uz_l;
	FILE *stream;
	stream=fopen("control_file.dat", "r" );
	for(int j=0;j<Nx;j++){
		for(int k=0;k<Ny;k++){
			for(int l=0;l<Nz;l++){
			
				fscanf( stream, "%lf %lf %lf",  &ux_l, &uy_l, &uz_l);	

				ux[IN(j,k,l)]=ux_l;

				uy[IN(j,k,l)]=uy_l;

				uz[IN(j,k,l)]=uz_l;


			}
		}
	}
	fclose(stream);
}


void write_control_file(int Nx, int Ny, int Nz, real* ux, real* uy, real* uz){
	double ux_l, uy_l, uz_l;
	FILE *stream;
	stream=fopen("control_file.dat", "w" );
	for(int j=0;j<Nx;j++){
		for(int k=0;k<Ny;k++){
			for(int l=0;l<Nz;l++){
				ux_l=ux[IN(j,k,l)];
				uy_l=uy[IN(j,k,l)];
				uz_l=uz[IN(j,k,l)];

				fprintf( stream, "%.16le %.16le %.16le\n", ux_l, uy_l, uz_l );	
			}
		}
	}
	fclose(stream);
}



int get_control_fourier_size(char *file_name, int Nx, int Ny, int Nz, int *Nx_file, int *Ny_file, int *Mz_file){
	double ux_l_Re, uy_l_Re, uz_l_Re;
	double ux_l_Im, uy_l_Im, uz_l_Im;
	
	FILE *stream;
	stream=fopen(file_name, "r+" );
	double temp=0;
	int aa=0;

	while(1){		

			fscanf(stream, "%lf %lf %lf %lf %lf %lf",  &ux_l_Re, &ux_l_Im, &uy_l_Re, &uy_l_Im, &uz_l_Re, &uz_l_Im);

		if(feof(stream))
			break; // hit end-of-file while getting the array
		aa++;
    }

	fclose(stream);
	
	int ratio_xy=Nx/Ny;
	int ratio_zy=(2*Nz-1)/Ny;
	int Ny_guess=2;
	int Nx_guess=ratio_xy*Ny_guess;
	int Nz_guess=ratio_zy*Ny_guess;
	int Mz_guess=0;

	if(Nx*Ny*Nz==aa){
		printf("\ninput size=%i, read size=%i\n",Nx*Ny*Nz,aa);
		return -1;
	}

	//while((Nx_guess<=Nx)&&(Ny_guess<=Ny)&&(Mz_guess<=Nz)){
	while(true){
		Ny_guess+=2;
		Nx_guess=ratio_xy*Ny_guess;
		Nz_guess=ratio_zy*Ny_guess;	
		Mz_guess=Nz_guess/2+1;	
		printf(" Nx=%i, Ny=%i, Mz=%i\n",Nx_guess,Ny_guess,Mz_guess);


		if((Nx_guess*Ny_guess*Mz_guess)==aa){
			printf("\nread size: Nx=%i, Ny=%i, Mz=%i\n",Nx_guess,Ny_guess,Mz_guess);
			break;
		}



	}

	Nx_file[0]=Nx_guess;
	Ny_file[0]=Ny_guess;
	Mz_file[0]=Mz_guess;
	return 1;

}




void read_control_fourier(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im){
	double ux_l_Re, uy_l_Re, uz_l_Re;
	double ux_l_Im, uy_l_Im, uz_l_Im;
	

	//XXX put here: read for smaller file to bigger fourier space with the same wave vector direction
	int *Nx_file=(int*)malloc(1*sizeof(int));
	int *Ny_file=(int*)malloc(1*sizeof(int));
	int *Mz_file=(int*)malloc(1*sizeof(int));
	int Nx_read=Nx;
	int Ny_read=Ny;
	int Nz_read=Nz;
	int ret=get_control_fourier_size("control_file_Fourier.dat", Nx, Ny, Nz, Nx_file, Ny_file, Mz_file);
	if(ret==1){

		Nx_read=Nx_file[0]; Ny_read=Ny_file[0]; Nz_read=Mz_file[0];

	}


	FILE *stream;
	stream=fopen("control_file_Fourier.dat", "r" );
	
	if(ret!=1){
	for(int j=0;j<Nx;j++){
		for(int k=0;k<Ny;k++){
			for(int l=0;l<Nz;l++){
			
				fscanf( stream, "%lf %lf %lf %lf %lf %lf",  &ux_l_Re, &ux_l_Im, &uy_l_Re, &uy_l_Im, &uz_l_Re, &uz_l_Im);	

				ux_hat_Re[IN(j,k,l)]=ux_l_Re;
				ux_hat_Im[IN(j,k,l)]=ux_l_Im;

				uy_hat_Re[IN(j,k,l)]=uy_l_Re;
				uy_hat_Im[IN(j,k,l)]=uy_l_Im;

				uz_hat_Re[IN(j,k,l)]=uz_l_Re;
				uz_hat_Im[IN(j,k,l)]=uz_l_Im;


			}
		}
	}
	}
	else{	
	double scale=Nx*Ny*(2*Nz-1)/(Nx_read*Ny_read*(2*Nz_read-1));
	for(int j=0;j<Nx_read;j++){
		for(int k=0;k<Ny_read;k++){
			for(int l=0;l<Nz_read;l++){
			
				fscanf( stream, "%lf %lf %lf %lf %lf %lf",  &ux_l_Re, &ux_l_Im, &uy_l_Re, &uy_l_Im, &uz_l_Re, &uz_l_Im);	

				int jj=j;
				if(j>=Nx_read/2)
					jj=j+(Nx-Nx_read);
				int kk=k;
				if(k>=Ny_read/2)
					kk=k+(Ny-Ny_read);


				ux_hat_Re[IN(jj,kk,l)]=scale*ux_l_Re;
				ux_hat_Im[IN(jj,kk,l)]=scale*ux_l_Im;

				uy_hat_Re[IN(jj,kk,l)]=scale*uy_l_Re;
				uy_hat_Im[IN(jj,kk,l)]=scale*uy_l_Im;

				uz_hat_Re[IN(jj,kk,l)]=scale*uz_l_Re;
				uz_hat_Im[IN(jj,kk,l)]=scale*uz_l_Im;


			}
		}
	}
	}

	fclose(stream);
	free(Mz_file); free(Nx_file); free(Ny_file);
}



void write_control_file_fourier(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im){
	double ux_l_Re, uy_l_Re, uz_l_Re;
	double ux_l_Im, uy_l_Im, uz_l_Im;


	double maxRe=0.0;

	

	FILE *stream;
	stream=fopen("control_file_Fourier.dat", "w" );
	for(int j=0;j<Nx;j++){
		for(int k=0;k<Ny;k++){
			for(int l=0;l<Nz;l++){
				ux_l_Re=ux_hat_Re[IN(j,k,l)];
				ux_l_Im=ux_hat_Im[IN(j,k,l)];
				uy_l_Re=uy_hat_Re[IN(j,k,l)];
				uy_l_Im=uy_hat_Im[IN(j,k,l)];
				uz_l_Re=uz_hat_Re[IN(j,k,l)];
				uz_l_Im=uz_hat_Im[IN(j,k,l)];

				double maxRe_temp=max2(Labs(ux_l_Re),Labs(uy_l_Re));
				maxRe_temp=max2(Labs(uz_l_Re),maxRe_temp);
				maxRe=max2(maxRe,maxRe_temp);

				fprintf( stream, "%.16le %.16le %.16le %.16le %.16le %.16le\n", ux_l_Re, ux_l_Im, uy_l_Re, uy_l_Im, uz_l_Re, uz_l_Im);		
			}
		}
	}
	fclose(stream);
	
	printf("\n      Maximum Real:%.16le\n",maxRe);

}




void write_file(char* file_name, real *array, int dir, int Nx, int Ny, int Nz, real dx, real dy, real dz){
// dir=0 - XY; dir=1 - XZ; dir = 2 - YZ
	FILE *stream;
	stream=fopen(file_name, "w" );
	int index=0;
	real x,y;
	for(int k=0;k<Nx;k++){
		for(int j=0;j<Ny;j++){		
			if(dir==0){
				x=dx*j;	
				y=dy*k;
				index=IN(j,k,Nz);
			}
			else if(dir==1){
				x=dx*j;	
				y=dz*k;
				index=IN(j,Nz,k);

			}
			else if(dir==2){
				x=dy*j;	
				y=dz*k;
				index=IN(Nz,j,k);

			}
			fprintf(stream, "%f	%f	%.016e\n", x, y, array[index]);	
		}

	}
	
	fclose(stream);

}




void write_out_file_vec_pos_interp(char f_name[], int Nx, int Ny, int Nz, real dx, real dy, real dz, real *ux, real *uy, real *uz, int what){
		real Xmin=0.0, Ymin=0.0, Zmin=0.0;

		FILE *stream;


		stream=fopen(f_name, "w" );


		fprintf( stream, "View");
		fprintf( stream, " '");
		fprintf( stream, f_name);
		fprintf( stream, "' {\n");
		fprintf( stream, "TIME{0};\n");
	

		for(int j=0;j<Nx;j++)
		for(int k=0;k<Ny;k++)
		for(int l=0;l<Nz;l++){
			real par_x=0.0,par_y=0.0,par_z=0.0;
			real par_x_mmm=0.0;
			real par_x_pmm=0.0;
			real par_x_ppm=0.0;
			real par_x_ppp=0.0;
			real par_x_mpp=0.0;
			real par_x_mmp=0.0;
			real par_x_pmp=0.0;
			real par_x_mpm=0.0;
			real par_y_mmm=0.0;
			real par_y_pmm=0.0;
			real par_y_ppm=0.0;
			real par_y_ppp=0.0;
			real par_y_mpp=0.0;
			real par_y_mmp=0.0;
			real par_y_pmp=0.0;
			real par_y_mpm=0.0;
			real par_z_mmm=0.0;
			real par_z_pmm=0.0;
			real par_z_ppm=0.0;
			real par_z_ppp=0.0;
			real par_z_mpp=0.0;
			real par_z_mmp=0.0;
			real par_z_pmp=0.0;
			real par_z_mpm=0.0;

			
			par_x=ux[I3(j,k,l)];
			par_y=uy[I3(j,k,l)];
			par_z=uz[I3(j,k,l)];
			par_x_mmm=0.125f*(ux[I3(j,k,l)]+ux[I3(j-1,k,l)]+ux[I3(j,k-1,l)]+ux[I3(j,k,l-1)]+ux[I3(j-1,k-1,l)]+ux[I3(j,k-1,l-1)]+ux[I3(j-1,k,l-1)]+ux[I3(j-1,k-1,l-1)]);
			par_x_pmm=0.125f*(ux[I3(j,k,l)]+ux[I3(j+1,k,l)]+ux[I3(j,k-1,l)]+ux[I3(j,k,l-1)]+ux[I3(j+1,k-1,l)]+ux[I3(j,k-1,l-1)]+ux[I3(j+1,k,l-1)]+ux[I3(j+1,k-1,l-1)]);
			par_x_ppm=0.125f*(ux[I3(j,k,l)]+ux[I3(j+1,k,l)]+ux[I3(j,k+1,l)]+ux[I3(j,k,l-1)]+ux[I3(j+1,k+1,l)]+ux[I3(j,k+1,l-1)]+ux[I3(j+1,k,l-1)]+ux[I3(j+1,k+1,l-1)]);
			par_x_ppp=0.125f*(ux[I3(j,k,l)]+ux[I3(j+1,k,l)]+ux[I3(j,k+1,l)]+ux[I3(j,k,l+1)]+ux[I3(j+1,k+1,l)]+ux[I3(j,k+1,l+1)]+ux[I3(j+1,k,l+1)]+ux[I3(j+1,k+1,l+1)]);
			par_x_mpp=0.125f*(ux[I3(j,k,l)]+ux[I3(j-1,k,l)]+ux[I3(j,k+1,l)]+ux[I3(j,k,l+1)]+ux[I3(j-1,k+1,l)]+ux[I3(j,k+1,l+1)]+ux[I3(j-1,k,l+1)]+ux[I3(j-1,k+1,l+1)]);
			par_x_mmp=0.125f*(ux[I3(j,k,l)]+ux[I3(j-1,k,l)]+ux[I3(j,k-1,l)]+ux[I3(j,k,l+1)]+ux[I3(j-1,k-1,l)]+ux[I3(j,k-1,l+1)]+ux[I3(j-1,k,l+1)]+ux[I3(j-1,k-1,l+1)]);
			par_x_pmp=0.125f*(ux[I3(j,k,l)]+ux[I3(j+1,k,l)]+ux[I3(j,k-1,l)]+ux[I3(j,k,l+1)]+ux[I3(j+1,k-1,l)]+ux[I3(j,k-1,l+1)]+ux[I3(j+1,k,l+1)]+ux[I3(j+1,k-1,l+1)]);
			par_x_mpm=0.125f*(ux[I3(j,k,l)]+ux[I3(j-1,k,l)]+ux[I3(j,k+1,l)]+ux[I3(j,k,l-1)]+ux[I3(j-1,k+1,l)]+ux[I3(j,k+1,l-1)]+ux[I3(j-1,k,l-1)]+ux[I3(j-1,k+1,l-1)]);
			
			par_y_mmm=0.125f*(uy[I3(j,k,l)]+uy[I3(j-1,k,l)]+uy[I3(j,k-1,l)]+uy[I3(j,k,l-1)]+uy[I3(j-1,k-1,l)]+uy[I3(j,k-1,l-1)]+uy[I3(j-1,k,l-1)]+uy[I3(j-1,k-1,l-1)]);
			par_y_pmm=0.125f*(uy[I3(j,k,l)]+uy[I3(j+1,k,l)]+uy[I3(j,k-1,l)]+uy[I3(j,k,l-1)]+uy[I3(j+1,k-1,l)]+uy[I3(j,k-1,l-1)]+uy[I3(j+1,k,l-1)]+uy[I3(j+1,k-1,l-1)]);
			par_y_ppm=0.125f*(uy[I3(j,k,l)]+uy[I3(j+1,k,l)]+uy[I3(j,k+1,l)]+uy[I3(j,k,l-1)]+uy[I3(j+1,k+1,l)]+uy[I3(j,k+1,l-1)]+uy[I3(j+1,k,l-1)]+uy[I3(j+1,k+1,l-1)]);
			par_y_ppp=0.125f*(uy[I3(j,k,l)]+uy[I3(j+1,k,l)]+uy[I3(j,k+1,l)]+uy[I3(j,k,l+1)]+uy[I3(j+1,k+1,l)]+uy[I3(j,k+1,l+1)]+uy[I3(j+1,k,l+1)]+uy[I3(j+1,k+1,l+1)]);
			par_y_mpp=0.125f*(uy[I3(j,k,l)]+uy[I3(j-1,k,l)]+uy[I3(j,k+1,l)]+uy[I3(j,k,l+1)]+uy[I3(j-1,k+1,l)]+uy[I3(j,k+1,l+1)]+uy[I3(j-1,k,l+1)]+uy[I3(j-1,k+1,l+1)]);
			par_y_mmp=0.125f*(uy[I3(j,k,l)]+uy[I3(j-1,k,l)]+uy[I3(j,k-1,l)]+uy[I3(j,k,l+1)]+uy[I3(j-1,k-1,l)]+uy[I3(j,k-1,l+1)]+uy[I3(j-1,k,l+1)]+uy[I3(j-1,k-1,l+1)]);
			par_y_pmp=0.125f*(uy[I3(j,k,l)]+uy[I3(j+1,k,l)]+uy[I3(j,k-1,l)]+uy[I3(j,k,l+1)]+uy[I3(j+1,k-1,l)]+uy[I3(j,k-1,l+1)]+uy[I3(j+1,k,l+1)]+uy[I3(j+1,k-1,l+1)]);
			par_y_mpm=0.125f*(uy[I3(j,k,l)]+uy[I3(j-1,k,l)]+uy[I3(j,k+1,l)]+uy[I3(j,k,l-1)]+uy[I3(j-1,k+1,l)]+uy[I3(j,k+1,l-1)]+uy[I3(j-1,k,l-1)]+uy[I3(j-1,k+1,l-1)]);
			
					
			par_z_mmm=0.125f*(uz[I3(j,k,l)]+uz[I3(j-1,k,l)]+uz[I3(j,k-1,l)]+uz[I3(j,k,l-1)]+uz[I3(j-1,k-1,l)]+uz[I3(j,k-1,l-1)]+uz[I3(j-1,k,l-1)]+uz[I3(j-1,k-1,l-1)]);
			par_z_pmm=0.125f*(uz[I3(j,k,l)]+uz[I3(j+1,k,l)]+uz[I3(j,k-1,l)]+uz[I3(j,k,l-1)]+uz[I3(j+1,k-1,l)]+uz[I3(j,k-1,l-1)]+uz[I3(j+1,k,l-1)]+uz[I3(j+1,k-1,l-1)]);
			par_z_ppm=0.125f*(uz[I3(j,k,l)]+uz[I3(j+1,k,l)]+uz[I3(j,k+1,l)]+uz[I3(j,k,l-1)]+uz[I3(j+1,k+1,l)]+uz[I3(j,k+1,l-1)]+uz[I3(j+1,k,l-1)]+uz[I3(j+1,k+1,l-1)]);
			par_z_ppp=0.125f*(uz[I3(j,k,l)]+uz[I3(j+1,k,l)]+uz[I3(j,k+1,l)]+uz[I3(j,k,l+1)]+uz[I3(j+1,k+1,l)]+uz[I3(j,k+1,l+1)]+uz[I3(j+1,k,l+1)]+uz[I3(j+1,k+1,l+1)]);
			par_z_mpp=0.125f*(uz[I3(j,k,l)]+uz[I3(j-1,k,l)]+uz[I3(j,k+1,l)]+uz[I3(j,k,l+1)]+uz[I3(j-1,k+1,l)]+uz[I3(j,k+1,l+1)]+uz[I3(j-1,k,l+1)]+uz[I3(j-1,k+1,l+1)]);
			par_z_mmp=0.125f*(uz[I3(j,k,l)]+uz[I3(j-1,k,l)]+uz[I3(j,k-1,l)]+uz[I3(j,k,l+1)]+uz[I3(j-1,k-1,l)]+uz[I3(j,k-1,l+1)]+uz[I3(j-1,k,l+1)]+uz[I3(j-1,k-1,l+1)]);
			par_z_pmp=0.125f*(uz[I3(j,k,l)]+uz[I3(j+1,k,l)]+uz[I3(j,k-1,l)]+uz[I3(j,k,l+1)]+uz[I3(j+1,k-1,l)]+uz[I3(j,k-1,l+1)]+uz[I3(j+1,k,l+1)]+uz[I3(j+1,k-1,l+1)]);
			par_z_mpm=0.125f*(uz[I3(j,k,l)]+uz[I3(j-1,k,l)]+uz[I3(j,k+1,l)]+uz[I3(j,k,l-1)]+uz[I3(j-1,k+1,l)]+uz[I3(j,k+1,l-1)]+uz[I3(j-1,k,l-1)]+uz[I3(j-1,k+1,l-1)]);

			fprintf( stream, "VH(%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)",
					Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
					Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz,
					Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz, 
					Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz,
					Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz);

			if(what==2){
				fprintf( stream,"{");
				fprintf(stream, "%e,%e,%e,",par_x_mmm,par_y_mmm,par_z_mmm);
				fprintf(stream, "%e,%e,%e,",par_x_pmm,par_y_pmm,par_z_pmm);
				fprintf(stream, "%e,%e,%e,",par_x_ppm,par_y_ppm,par_z_ppm);
				fprintf(stream, "%e,%e,%e,",par_x_ppp,par_y_ppp,par_z_ppp);
				fprintf(stream, "%e,%e,%e,",par_x_mpp,par_y_mpp,par_z_mpp);
				fprintf(stream, "%e,%e,%e,",par_x_mmp,par_y_mmp,par_z_mmp);
				fprintf(stream, "%e,%e,%e,",par_x_pmp,par_y_pmp,par_z_pmp);
				fprintf(stream, "%e,%e,%e",par_x_mpm,par_y_mpm,par_z_mpm);
				fprintf(stream, "};\n");
			}
			else if(what==1){
				fprintf( stream,"{");
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
				fprintf(stream, "%e,%e,%e",par_x,par_y,par_z);
				fprintf(stream, "};\n");
			}

		}	
		
		

		fprintf( stream, "};");

	fclose(stream);
	



}



void write_out_file_pos(char f_name[], int Nx, int Ny, int Nz, real dx, real dy, real dz, real *U, int what){
		real Xmin=0.0, Ymin=0.0, Zmin=0.0;

		FILE *stream;


		stream=fopen(f_name, "w" );


		fprintf( stream, "View");
		fprintf( stream, " '");
		fprintf( stream, f_name);
		fprintf( stream, "' {\n");
		fprintf( stream, "TIME{0};\n");
	

		for(int j=0;j<Nx;j++)
		for(int k=0;k<Ny;k++)
		for(int l=0;l<Nz;l++){
			real par=0.0;
			real par_mmm=0.0;
			real par_pmm=0.0;
			real par_ppm=0.0;
			real par_ppp=0.0;
			real par_mpp=0.0;
			real par_mmp=0.0;
			real par_pmp=0.0;
			real par_mpm=0.0;

			par=U[I3(j,k,l)];
			if(what==2){					
				par_mmm=0.125f*(U[I3(j,k,l)]+U[I3(j-1,k,l)]+U[I3(j,k-1,l)]+U[I3(j,k,l-1)]+U[I3(j-1,k-1,l)]+U[I3(j,k-1,l-1)]+U[I3(j-1,k,l-1)]+U[I3(j-1,k-1,l-1)]);
				par_pmm=0.125f*(U[I3(j,k,l)]+U[I3(j+1,k,l)]+U[I3(j,k-1,l)]+U[I3(j,k,l-1)]+U[I3(j+1,k-1,l)]+U[I3(j,k-1,l-1)]+U[I3(j+1,k,l-1)]+U[I3(j+1,k-1,l-1)]);
				par_ppm=0.125f*(U[I3(j,k,l)]+U[I3(j+1,k,l)]+U[I3(j,k+1,l)]+U[I3(j,k,l-1)]+U[I3(j+1,k+1,l)]+U[I3(j,k+1,l-1)]+U[I3(j+1,k,l-1)]+U[I3(j+1,k+1,l-1)]);
				par_ppp=0.125f*(U[I3(j,k,l)]+U[I3(j+1,k,l)]+U[I3(j,k+1,l)]+U[I3(j,k,l+1)]+U[I3(j+1,k+1,l)]+U[I3(j,k+1,l+1)]+U[I3(j+1,k,l+1)]+U[I3(j+1,k+1,l+1)]);
				par_mpp=0.125f*(U[I3(j,k,l)]+U[I3(j-1,k,l)]+U[I3(j,k+1,l)]+U[I3(j,k,l+1)]+U[I3(j-1,k+1,l)]+U[I3(j,k+1,l+1)]+U[I3(j-1,k,l+1)]+U[I3(j-1,k+1,l+1)]);
				par_mmp=0.125f*(U[I3(j,k,l)]+U[I3(j-1,k,l)]+U[I3(j,k-1,l)]+U[I3(j,k,l+1)]+U[I3(j-1,k-1,l)]+U[I3(j,k-1,l+1)]+U[I3(j-1,k,l+1)]+U[I3(j-1,k-1,l+1)]);
				par_pmp=0.125f*(U[I3(j,k,l)]+U[I3(j+1,k,l)]+U[I3(j,k-1,l)]+U[I3(j,k,l+1)]+U[I3(j+1,k-1,l)]+U[I3(j,k-1,l+1)]+U[I3(j+1,k,l+1)]+U[I3(j+1,k-1,l+1)]);
				par_mpm=0.125f*(U[I3(j,k,l)]+U[I3(j-1,k,l)]+U[I3(j,k+1,l)]+U[I3(j,k,l-1)]+U[I3(j-1,k+1,l)]+U[I3(j,k+1,l-1)]+U[I3(j-1,k,l-1)]+U[I3(j-1,k+1,l-1)]);
					}
					
				

					fprintf( stream, "SH(%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)",
					Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
					Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
					Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
					Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz,
					Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz);

					if(what==2){
						fprintf( stream,"{");
						fprintf(stream, "%e,",par_mmm);
						fprintf(stream, "%e,",par_pmm);
						fprintf(stream, "%e,",par_ppm);
						fprintf(stream, "%e,",par_mpm);
						fprintf(stream, "%e,",par_mmp);
						fprintf(stream, "%e,",par_pmp);
						fprintf(stream, "%e,",par_ppp);
						fprintf(stream, "%e",par_mpp);
						fprintf(stream, "};\n");
					}
					else if(what==1){
						fprintf( stream,"{");
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e,",par);
						fprintf(stream, "%e",par);
						fprintf(stream, "};\n");
					}

		}	

			

		fprintf( stream, "};");

		fclose(stream);
	



}





void write_res_files(real *ux, real *uy, real *uz, real *div_pos, real *u_abs, int Nx, int Ny, int Nz, real dx, real dy, real dz){
	
	
	printf("printing files: ");
	write_out_file_vec_pos_interp("p_outVec.pos", Nx, Ny, Nz, dx, dy, dz, ux, uy, uz);	
	printf(".");	
	write_out_file_pos("p_outDiv_pos.pos", Nx, Ny, Nz, dx, dy, dz, div_pos);	
	printf(".");	
	write_out_file_pos("p_outUabs.pos", Nx, Ny, Nz, dx, dy, dz, u_abs);
	printf(".\n");	

}


void write_drop_files(int drop, int t, int Nx, int Ny, int Nz,  real *ux, real* uy, real* uz, real *div_pos, real *u_abs, real dx, real dy, real dz){

	if(t%drop==0){
		//drop down intermediate results
		//	'r'	- rot interpolation
		//	's' - psi interpolation
		char f1_name[100];
		sprintf(f1_name, "p%08d_outVec.pos",(t/drop));
		write_out_file_vec_pos_interp(f1_name, Nx, Ny, Nz, dx, dy, dz, ux, uy, uz);	
		printf(".");
		sprintf(f1_name, "p%08d_outDiv.pos",(t/drop));	
		write_out_file_pos(f1_name, Nx, Ny, Nz, dx, dy, dz, div_pos);
		printf(".");
		sprintf(f1_name, "p%08d_outUabs.pos",(t/drop));	
		write_out_file_pos(f1_name, Nx, Ny, Nz, dx, dy, dz, u_abs);
		printf(".");	
		printf("\n");	
	}
}




void write_drop_files_from_device(int drop, int t, int Nx, int Ny, int Nz, real *ux, real* uy, real* uz, real* u_abs, real *div_pos, real dx, real dy, real dz, real *ux_d, real* uy_d, real* uz_d, real* u_abs_d, real *div_pos_d){



	if(t%drop==0){
		host_device_real_cpy(ux, ux_d, Nx, Ny, Nz);
		host_device_real_cpy(uy, uy_d, Nx, Ny, Nz);	
		host_device_real_cpy(uz, uz_d, Nx, Ny, Nz);
		host_device_real_cpy(u_abs, u_abs_d, Nx, Ny, Nz);
		host_device_real_cpy(div_pos, div_pos_d, Nx, Ny, Nz);	
		write_drop_files(drop, t, Nx, Ny, Ny,  ux, uy, uz, div_pos, u_abs, dx, dy, dz);
	}
}






void write_line_specter(int Nx, int Ny, int Nz, real* ux_hat_Re, real* ux_hat_Im, real* uy_hat_Re, real* uy_hat_Im, real* uz_hat_Re, real* uz_hat_Im){
	double ux_l_Re, uy_l_Re, uz_l_Re;
	double ux_l_Im, uy_l_Im, uz_l_Im;

	
	double Kx=1.0*Nx*Nx;
	double Ky=1.0*Ny*Ny;
	double Kz=1.0*Nz*Nz;

	double sum=1.0/Kx+1.0/Ky+1.0/Kz;

	double J_inedex=sqrt(1.0/sum);
	int J=floor(J_inedex)+1;

	FILE *stream;
	stream=fopen("Specter.dat", "w" );
	for(int j=0;j<Nx;j++){
		for(int k=0;k<Ny;k++){
			for(int l=0;l<Nz;l++){
				if((j==k)&&(j==l)&&(j<=J)&&(j>0)){
					ux_l_Re=ux_hat_Re[IN(j,k,l)];
					ux_l_Im=ux_hat_Im[IN(j,k,l)];
					uy_l_Re=uy_hat_Re[IN(j,k,l)];
					uy_l_Im=uy_hat_Im[IN(j,k,l)];
					uz_l_Re=uz_hat_Re[IN(j,k,l)];
					uz_l_Im=uz_hat_Im[IN(j,k,l)];

					double amplitude_x=sqrt(ux_l_Re*ux_l_Re+ux_l_Im*ux_l_Im);
					double amplitude_y=sqrt(uy_l_Re*uy_l_Re+uy_l_Im*uy_l_Im);
					double amplitude_z=sqrt(uz_l_Re*uz_l_Re+uz_l_Im*uz_l_Im);
					double amplitude=sqrt(amplitude_z*amplitude_z+amplitude_y*amplitude_y+amplitude_x*amplitude_x);
				
					fprintf( stream, "%.16le %.16le %.16le %.16le %.16le\n", 1.0*j, amplitude_x, amplitude_y, amplitude_z, amplitude);
				}		
			}
		}
	}
	fclose(stream);
	
}


