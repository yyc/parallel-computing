buildall: seq_build thread_build omp_build

seq_build:
	mkdir -p bin out
	gcc -O3 -o bin/blur_seq blur_seq.c -lm;
cuda_build:
	mkdir -p bin out
	nvcc -o bin/mm-cuda -arch=sm_32 -lm -ccbin clang-3.8 mm-cuda.cu
optimized_build:
	mkdir -p bin out
	nvcc -o bin/mm-optimized -arch=sm_32 -lm -ccbin clang-3.8 mm-optimized.cu
sm_build:
	mkdir -p bin out
	nvcc -o bin/mm-sm -arch=sm_32 -lm -ccbin clang-3.8 mm-sm.cu
banks_build:
	mkdir -p bin out
	nvcc -o bin/mm-banks -arch=sm_32 -lm -ccbin clang-3.8 mm-banks.cu


cuda: cuda_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-cuda 1000;
optimized: optimized_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-optimized 1000;
sm: sm_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-sm 1000;
banks: banks_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-banks 1000;


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
