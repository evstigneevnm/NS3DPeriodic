Arnoldi_LAPACKs:
	2>result_make.txt
	g++ -m64 -g Source/Arnoldi/LAPACK_routines.cpp -o Obj/Arnoldi/Arnoldi_LAPACK_routines.o -c 2>>result_make.txt

Arnoldi_Products:
	2>result_make.txt
	g++ -m64 -g Source/Arnoldi/Products.cpp -o Obj/Arnoldi/Arnoldi_Products.o -c 2>>result_make.txt


Arnoldi_Shifts:
	2>result_make.txt
	g++ -m64 -g Source/Arnoldi/Select_Shifts.cpp -o Obj/Arnoldi/Arnoldi_Select_Shifts.o -c 2>>result_make.txt


Arnoldi_QRshifts:
	2>result_make.txt
	g++ -m64  -g Source/Arnoldi/QR_Shifts.cpp -o Obj/Arnoldi/Arnoldi_QR_Shifts.o -c 2>>result_make.txt


Arnoldi_timer:
	2>result_make.txt
	g++ -m64 -g Source/Arnoldi/timer.cpp -o Obj/Arnoldi/Arnoldi_timer.o -c 2>>result_make.txt

Arnoldi_Matrix_Vector_emulator:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/Matrix_Vector_emulator.cu -o Obj/Arnoldi/Arnoldi_Matrix_Vector_emulator.o -c 2>>result_make.txt

Arnoldi_Arnoldi_Driver:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/Arnoldi_Driver.cu -o Obj/Arnoldi/Arnoldi_Arnoldi_Driver.o -c 2>>result_make.txt

Arnoldi_file_operations:
	2>result_make.txt
	g++ -g Source/Arnoldi/file_operations.cpp -o Obj/Arnoldi/Arnoldi_file_operations.o -c 2>>result_make.txt


Arnoldi_memory_operations:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/memory_operations.cu -o Obj/Arnoldi/Arnoldi_memory_operations.o -c 2>>result_make.txt

Arnoldi_cuda_supp:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/cuda_supp.cu -o Obj/Arnoldi/Arnoldi_cuda_supp.o -c 2>>result_make.txt

Arnoldi_Implicit_restart_Arnoldi:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/Implicit_restart_Arnoldi.cu -o Obj/Arnoldi/Arnoldi_Implicit_restart_Arnoldi.o  -c 2>>result_make.txt


Arnoldi_deb:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/cuda_test.cu -o Arnoldi_test Obj/Arnoldi/Arnoldi_LAPACK_routines.o Obj/Arnoldi/Arnoldi_Products.o Obj/Arnoldi/Arnoldi_Select_Shifts.o Obj/Arnoldi/Arnoldi_QR_Shifts.o Obj/Arnoldi/Arnoldi_timer.o Obj/Arnoldi/Arnoldi_Matrix_Vector_emulator.o Obj/Arnoldi/Arnoldi_Arnoldi_Driver.o Obj/Arnoldi/Arnoldi_file_operations.o Source/Arnoldi/memory_operations.cu Obj/Arnoldi/Arnoldi_cuda_supp.o Obj/Arnoldi/Arnoldi_Implicit_restart_Arnoldi.o -L/usr/local/cuda-9.1/lib64 -I/usr/local/cuda-9.1/include -llapack -lblas -lgfortran -lm -lcublas  2>>result_make.txt


BiCGstabL:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/BiCGStabL.cu -o Obj/Arnoldi/BiCGStabL.o -c 2>>result_make.txt

Newton:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Arnoldi/Newton.cu -o Obj/Arnoldi/Newton.o -c 2>>result_make.txt


Arnoldi_all:
	make Arnoldi_LAPACKs Arnoldi_Products Arnoldi_Shifts Arnoldi_QRshifts Arnoldi_timer Arnoldi_Matrix_Vector_emulator Arnoldi_Arnoldi_Driver Arnoldi_file_operations Arnoldi_memory_operations Arnoldi_cuda_supp Arnoldi_Implicit_restart_Arnoldi BiCGstabL Newton


Jacobian:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Jacobian.cu -o Obj/Jacobian.o -c 2>>result_make.txt

Shapiro:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/Shapiro_test.cu -o Obj/Shapiro_test.o -c 2>>result_make.txt
file:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g  Source/file_operations.cu -o Obj/file_operations.o -c 2>>result_make.txt

supp:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/cuda_supp.cu -o Obj/cuda_supp.o -c 2>>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/file_operations.cu -o Obj/file_operations.o -c 2>>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/memory_operations.cu -o Obj/memory_operations.o -c 2>>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/min_max_reduction.cu -o Obj/min_max_reduction.o -c 2>>result_make.txt
	
adv_2p3:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/advection_2_3.cu -o Obj/advection_2_3.o -c 2>>result_make.txt


adv_WENO:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/advection_WENO.cu -o Obj/advection_WENO.o -c 2>>result_make.txt

math:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/diffusion.cu -o Obj/diffusion.o -c 2>>result_make.txt   
	/usr/local/cuda-9.1/bin/nvcc -g Source/math_support.cu -o Obj/math_support.o -c 2>>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/divergence.cu -o Obj/divergence.o -c 2>>result_make.txt

rkstep:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/RK_time_step.cu -o Obj/RK_time_step.o -c 2>>result_make.txt


return_map:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/return_map.cu -o Obj/return_map.o -c 2>>result_make.txt 

NS_deb:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/NS3D_periodic.cu -o NS3D_periodic -lcufft -lm Obj/cuda_supp.o Obj/file_operations.o  Obj/math_support.o Obj/memory_operations.o Obj/divergence.o Obj/diffusion.o Obj/RK_time_step.o Obj/min_max_reduction.o Obj/advection_2_3.o Obj/advection_WENO.o Obj/Shapiro_test.o Obj/Jacobian.o 2>>result_make.txt


NS_all:
	make Jacobian adv_2p3 adv_WENO supp math rkstep Shapiro return_map


clear:
	cd Obj/Arnoldi; \
	rm *.o
	cd Obj; \
	rm *.o  
	rm *.pos *.dat NS3D_periodic result_make.txt

deb_all:
	2>result_make.txt
	/usr/local/cuda-9.1/bin/nvcc -g Source/NS3D_periodic.cu -o NS3D_periodic Obj/cuda_supp.o Obj/file_operations.o  Obj/math_support.o Obj/memory_operations.o Obj/divergence.o Obj/diffusion.o Obj/RK_time_step.o Obj/min_max_reduction.o Obj/advection_2_3.o Obj/advection_WENO.o Obj/Shapiro_test.o Obj/Jacobian.o  Obj/Arnoldi/Arnoldi_LAPACK_routines.o Obj/Arnoldi/Arnoldi_Products.o Obj/Arnoldi/Arnoldi_Select_Shifts.o Obj/Arnoldi/Arnoldi_QR_Shifts.o Obj/Arnoldi/Arnoldi_timer.o Obj/Arnoldi/Arnoldi_Matrix_Vector_emulator.o Obj/Arnoldi/Arnoldi_Arnoldi_Driver.o Obj/Arnoldi/Arnoldi_file_operations.o Source/Arnoldi/memory_operations.cu Obj/Arnoldi/Arnoldi_cuda_supp.o Obj/Arnoldi/Arnoldi_Implicit_restart_Arnoldi.o  Obj/Arnoldi/BiCGStabL.o Obj/Arnoldi/Newton.o  -L/opt/OpenBLAS/lib Obj/return_map.o -L/usr/local/cuda-9.1/lib64 -I/usr/local/cuda-9.1/include -llapack -lblas -lopenblas -lgfortran -lm -lcublas -lcufft 2>>result_make.txt 
