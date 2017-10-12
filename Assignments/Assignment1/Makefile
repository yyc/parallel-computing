buildall: seq_build thread_build omp_build

seq_build:
	mkdir -p bin out
	gcc -O3 -o bin/blur_seq blur_seq.c -lm;
thread_build:
	mkdir -p bin out
	gcc -O3 -o bin/blur_threads blur_threads.c -lm -pthread;
thread_seq_build:
	mkdir -p bin out
	gcc -O3 -o bin/blur_thread_seq blur_thread_seq.c -lm;
omp_build:
	#export OMPI_CC=gcc-7
	#gcc -O3 blur_omp.c -fopenmp -o bin/blur_omp -lm
	mkdir -p bin out
	mpicc -O3 -o bin/blur_omp blur_omp.c -lm -fopenmp;


threads: thread_build
	for number in 1 2 3 4 5 ; do \
		 /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/blur_threads in/image$$number.bmp 3 9 out/image_threads$$number.bmp; \
		echo $$(diff out/image_ref$$number.bmp out/image_threads$$number.bmp); \
	done
thread_seq: thread_seq_build
	for number in 1 2 3 4 5 ; do \
		/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w"  bin/blur_thread_seq in/image$$number.bmp 3 9 out/image_threadsseq$$number.bmp; \
		echo $$(diff out/image_ref$$number.bmp out/image_threadsseq$$number.bmp); \
	done


omp: omp_build
	touch hostfile
	for number in 1 2 3 4 5 ; do \
		/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w"  mpirun -hostfile hostfile -np 1 bin/blur_omp in/image$$number.bmp 3 9 out/image_omp$$number.bmp; \
		echo $$(diff out/image_ref$$number.bmp out/image_omp$$number.bmp); \
	done

seq: seq_build
	for number in 1 2 3 4 5 ; do \
		/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w"  bin/blur_seq in/image$$number.bmp 3 9 out/image_ref$$number.bmp; \
	done

2a: thread_seq_build thread_build  omp_build
	for cores in 0 1 3 5 7 ; do \
		echo "====USING $$cores CORES ==="; \
		echo "====SEQ===="; \
		numactl --physcpubind=0-$$cores /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/blur_thread_seq in/image6.bmp 3 9 out/image_seq6.bmp; \
		echo "====THREAD===="; \
		numactl --physcpubind=0-$$cores /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/blur_threads in/image6.bmp 3 9 out/image_threads6.bmp; \
		echo "====OMP===="; \
		numactl --physcpubind=0-$$cores /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" mpirun -hostfile hostfile -np 1 bin/blur_omp in/image6.bmp 3 9 out/image_omp6.bmp; \
	done
