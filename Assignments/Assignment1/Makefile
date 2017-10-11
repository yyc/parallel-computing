buildall: seq_build thread_build

seq_build:
	gcc -O3 -o bin/blur_seq -lm blur_seq.c;
thread_build:
	gcc -O3 -o bin/blur_threads -lm blur_threads.c;
omp_build:
	export OMPI_CC=gcc-7
	gcc-7 -O3 -lm blur_omp.c -fopenmp -o bin/blur_omp
	# mpicc -O3 -o bin/blur_omp -lm blur_omp.c;


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
