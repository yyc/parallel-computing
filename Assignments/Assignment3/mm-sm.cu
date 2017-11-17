/**
 *
 * Matrix Multiplication - CUDA for GPUs
 *
 * CS3210
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

int size, paddedSize;
#define BLOCKSIZE 64

typedef struct
{
	float ** element;
} matrix;


long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/**
 * Allocates memory for a matrix of size SIZE
 * The memory is allocated row-major order, i.e.
 *  elements from the same row are allocated at contiguous
 *  memory addresses.
 **/
void allocate_matrix(matrix* m)
{
	int i;
	cudaError_t rc;

	// allocate array for all the rows
	rc = cudaMallocManaged((void**)&(m->element), sizeof(float*) * paddedSize);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}

	// allocate an array for each row of the matrix
	for (i = 0; i < paddedSize; i++)
	{
		rc = cudaMallocManaged((void**)&(m->element[i]), sizeof(float) * paddedSize);
		if (rc != cudaSuccess)
		{
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
			exit(1);
		}
	}
}

/**
 * Free the memory allocated for a matrix.
 **/
void free_matrix(matrix* m) {
	int i;
	for (i = 0; i < paddedSize; i++)
		cudaFree(m->element[i]);
	cudaFree(m->element);
}

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			// m.element[i][j] = rand() % 10;
			m.element[i][j] = 1;
		}
}

/**
 * Initializes the elements of the matrix with
 * element 0.
 **/
void init_matrix_zero(matrix m)
{
	int i, j;

	for (i = 0; i < paddedSize; i++)
		for (j = 0; j < paddedSize; j++)
		{
			m.element[i][j] = 0.0;
		}
}

__global__ void transpose_kernel(matrix src, matrix dest, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= size || j >= size)		return;

	dest.element[i][j] = src.element[j][i];
}

// SHARED MEMORY KERNEL HERE
__global__ void sm_kernel(matrix a, matrix b, matrix result, int size)
{
	// Initialize shared memory
	__shared__ float aMat[BLOCKSIZE][BLOCKSIZE];
	__shared__ float bMat[BLOCKSIZE][BLOCKSIZE];
	// Calculate the index in the resulting matrix
	int i = (blockIdx.x) * blockDim.x + threadIdx.x;
	int j = (blockIdx.y) * blockDim.y + threadIdx.y;
	int k, m, numBlocks;
	float sum = 0.0f;

// Require M blocks to finish
	numBlocks = ((size % BLOCKSIZE) == 0) ? (size / BLOCKSIZE) : (size / BLOCKSIZE + 1);

	// For each block in turn
	for(k = 0; k < numBlocks; k++){
		// each thread copy one element to the buffer
		aMat[threadIdx.x][threadIdx.y] = a.element[i][k * BLOCKSIZE + threadIdx.y];
		bMat[threadIdx.y][threadIdx.x] = b.element[j][k * BLOCKSIZE + threadIdx.x];
		__syncthreads();

		// Do a partial sum of all available elements
		for(m = 0; m < BLOCKSIZE; m++)
			sum += aMat[threadIdx.x][m] * bMat[threadIdx.y][m];
		__syncthreads();
	}

	// When done, the sum is complete and we write it back to global
	result.element[i][j] = sum;
}

/**
 * Each kernel computes the result element (i,j).
 */
__global__ void mm_kernel(matrix a, matrix b, matrix result, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k;
	float sum = 0.0f;

	if (i >= size || j >= size)
		return;

	for(k = 0; k < size; k++)
		sum += a.element[i][k] * b.element[j][k];
	result.element[i][j] = sum;
}

void print_matrix(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
	{
		printf("row %4d: ", i);
		for (j = 0; j < size; j++)
			printf("%6.2f  ", m.element[i][j]);
		printf("\n");
	}
}



void work()
{
	matrix a, b, bt, result1, result2;
	long long before, after;
	float time1, time2;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&bt);
	allocate_matrix(&result1);
	allocate_matrix(&result2);

	// Initialize matrix elements
	init_matrix_zero(a);
	init_matrix_zero(b);
	init_matrix(a);
	init_matrix(b);


	// Perform CUDA matrix  multiplication
	dim3 transblock(32, 32);
	dim = (size % 32 == 0) ? size / 32 : size / 32 + 1;
	dim3 transgrid(dim, dim);

	before = wall_clock_time();
	init_matrix_zero(bt);
	transpose_kernel<<<transgrid, transblock>>>(b, bt, size);
	cudaDeviceSynchronize();
	mm_kernel<<<transblock, transgrid>>>(a, bt, result1, size);
	cudaDeviceSynchronize();
	after = wall_clock_time();
	time1 = ((float)(after - before))/1000000000;
	fprintf(stderr, "Optimized MM on GPU took %1.2f seconds\n", time1);
	// was there any error?
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
            printf("Last CUDA error %s\n", cudaGetErrorString(rc));


	dim3 block(BLOCKSIZE, BLOCKSIZE);			// a block of 32 x 32 CUDA threads
	// dim = (size % BLOCKSIZE == 0) ? size / BLOCKSIZE : size / BLOCKSIZE + 1;
	dim = paddedSize / BLOCKSIZE;
	dim3 grid(dim , dim );	// a grid of CUDA thread blocks
	before = wall_clock_time();
	init_matrix_zero(bt);
	transpose_kernel<<<transgrid, transblock>>>(b, bt, size);
	cudaDeviceSynchronize();

	// fprintf(stderr,"Starting SM with blocksize %dx%d and grid %dx%d", BLOCKSIZE, BLOCKSIZE, dim, dim);
	sm_kernel<<<grid, block>>>(a, bt, result2, size);
	cudaDeviceSynchronize();

	after = wall_clock_time();
	time2 = ((float)(after - before))/1000000000;
	fprintf(stderr, "SM MM on GPU took %1.2f seconds\n", time2);
	// was there any error?
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
            printf("Last CUDA error %s\n", cudaGetErrorString(rc));



	// Compare the results
	correct = 1;
	for (i = 0; correct && i < size; i++)
		for (j = 0; j < size; j++)
			if (result1.element[i][j] != result2.element[i][j]) {
				correct = 0;
				break;
			}

	if (correct) {
		//printf("The result matrices are identical!\n");
		printf("Speedup: %1.4f\n", time1/time2);
	}
	else {
		printf("Difference in result matrices at element (%d, %d)!\n", i, j);
		// print_matrix(result1);
		// print_matrix(result2);
	}
	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&result1);
	free_matrix(&result2);
}


int main(int argc, char ** argv)
{
	srand(0);

	// printf("Usage: %s <size>\n", argv[0]);

	if (argc >= 2)
		size = atoi(argv[1]);
	else
		size = 1024;

	paddedSize = (size % BLOCKSIZE == 0) ? size : (1 + size / BLOCKSIZE) * BLOCKSIZE;
	fprintf(stderr,"Optimized/SM multiplication of size %d\n", size);

	// Multiply the matrices
	work();

	return 0;
}
