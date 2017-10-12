buildall: seq_build thread_build

seq_build:
	gcc -O3 -o bin/blur_seq blur_seq.c -lm;
thread_build:
	gcc -O3 -o bin/blur_threads blur_threads.c -lm;
omp_build:
	#export OMPI_CC=gcc-7
	#gcc -O3 blur_omp.c -fopenmp -o bin/blur_omp -lm
	mpicc -O3 -o bin/blur_omp blur_omp.c -lm;


threads: thread_build
	for number in 1 2 3 4 5 ; do \
		time bin/blur_threads in/image$$number.bmp 3 9 out/image_threads$$number.bmp; \
		echo $$(diff out/image_ref$$number.bmp out/image_threads$$number.bmp); \
	done

omp: omp_build
	touch hostfile
	for number in 1 2 3 4 5 ; do \
		time mpirun -hostfile hostfile -np 1 bin/blur_omp in/image$$number.bmp 3 9 out/image_omp$$number.bmp; \
		echo $$(diff out/image_ref$$number.bmp out/image_omp$$number.bmp); \
	done


seq: seq_build
	for number in 1 2 3 4 5 ; do \
		time bin/blur_seq in/image$$number.bmp 3 9 out/image_ref$$number.bmp; \
	done
