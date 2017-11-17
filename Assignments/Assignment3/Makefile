## Run testall to compile and test all 3 programs

buildall: seq_build thread_build omp_build

cuda_build:
	mkdir -p bin out
	nvcc -o bin/mm-cuda -arch=sm_32 -lm mm-cuda.cu
optimized_build:
	mkdir -p bin out
	nvcc -o bin/mm-optimized -arch=sm_32 mm-optimized.cu
sm_build:
	mkdir -p bin out
	nvcc -o bin/mm-sm -arch=sm_32 -lm mm-sm.cu
banks_build:
	mkdir -p bin out
	nvcc -o bin/mm-banks-sm -arch=sm_32 mm-banks-sm.cu


cuda: cuda_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-cuda 1000;
optimized: optimized_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-optimized 1000;
sm: sm_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-sm 1000;
banks: banks_build
	/usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-banks-sm 1000;

test_optimized: optimized_build
	for number in 256 512 1024; do \
		bin/mm-optimized $$number; \
	done

test_sm: sm_build
	for number in 256 512 1024; do \
		bin/mm-sm $$number; \
	done

test_banks_sm: banks_build
	for number in 256 512 1024; do \
		bin/mm-banks-sm $$number; \
	done


testall: test_optimized test_sm test_banks_sm