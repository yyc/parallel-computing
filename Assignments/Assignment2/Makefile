training_build:
	#export OMPI_CC=gcc-7
	mkdir -p bin out
	touch hostfile
	mpicc -O3 -o bin/training training_mpi.c -fopenmp;

training: training_build:
		mpirun -hostfile hostfile -n 3 bin/blur_omp
