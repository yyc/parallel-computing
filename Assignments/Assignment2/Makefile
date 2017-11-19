make_dir:
	mkdir -p bin out
	touch hostfile

training_build: make_dir
	export OMPI_CC=gcc-7
	# gcc-7 -O3 -std=c11 training_mpi.c -o bin/training -fopenmp
	mpicc -O3 -std=c11 -o bin/training training_mpi.c;

training: training_build
		mpirun -hostfile hostfile -np 12 bin/training > training.lab.o; \
		tail training.lab.o;