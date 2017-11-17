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

int size;
#define BLOCKSIZE 32

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
	rc = cudaMallocManaged((void**)&(m->element), sizeof(float*) * size);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}

	// allocate an array for each row of the matrix
	for (i = 0; i < size; i++)
	{
		rc = cudaMallocManaged((void**)&(m->element[i]), sizeof(float) * size);
		if (rc != cudaSuccess)
		{
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
			exit(1);
		}
	}
}

void allocate_3d(matrix* m) {
	int i, j;
	cudaError_t rc;

	// allocate array for all the rows
	rc = cudaMallocManaged((void**)&(m->element), sizeof(float*) * size);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}

	// allocate an array for each row of the matrix
	for (i = 0; i < size; i++)
	{
		rc = cudaMallocManaged((void**)&(m->element[i]), sizeof(float*) * size);
		if (rc != cudaSuccess)
		{
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
			exit(1);
		}
		for(j = 0; j < size; j++) {
			rc = cudaMallocManaged((void**)&(m->element[i][j]), sizeof(float*) * size);
		}
	}
}

/**
 * Free the memory allocated for a matrix.
 **/
void free_matrix(matrix* m) {
	int i;
	for (i = 0; i < size; i++)
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
			m.element[i][j] = rand() % 10;
		}
}

/**
 * Initializes the elements of the matrix with
 * element 0.
 **/
void init_matrix_zero(matrix m)
{
	int i, j;

	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = 0.0;
		}
}

void matrix_transpose(matrix src, matrix dest) {
	int i,j;
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			dest.element[i][j] = src.element[j][i];
		}
	}
}

__global__ void transpose_kernel(matrix src, matrix dest, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= size || j >= size)		return;

	dest.element[i][j] = src.element[j][i];
}

__global__ void multiply_kernel(matrix src, matrix dest, int size) {

}

/**
 * Multiplies matrix @a with matrix @b storing
 * the result in matrix @result
 *
 * The multiplication algorithm is the O(n^3)
 * algorithm
 */
void mm(matrix a, matrix b, matrix result)
{
	int i, j, k;

	// Do the multiplication
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			for(k = 0; k < size; k++)
				result.element[i][j] += a.element[i][k] * b.element[k][j];
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
	float seqtime, gputime;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&bt);
	allocate_matrix(&result1);
	allocate_matrix(&result2);

	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);

	// Perform sequential matrix multiplication
	before = wall_clock_time();
	mm(a, b, result1);
	after = wall_clock_time();
	seqtime = ((float)(after - before))/1000000000;
        fprintf(stderr, "Matrix multiplication on CPU took %1.2f seconds\n", seqtime);

	// Perform CUDA matrix  multiplication
	dim3 transblock(BLOCKSIZE, BLOCKSIZE);
	dim = (size % BLOCKSIZE == 0) ? size /BLOCKSIZE : size / BLOCKSIZE + 1;
	dim3 transgrid(dim, dim);

	before = wall_clock_time();

	init_matrix_zero(bt);
  transpose_kernel<<<transgrid, transblock>>>(b, bt, size);

	mm_kernel<<<transgrid, transblock>>>(a, bt, result2, size);

	cudaDeviceSynchronize();
	after = wall_clock_time();
	gputime = ((float)(after - before))/1000000000;
	fprintf(stderr, "Matrix multiplication on GPU took %1.2f seconds\n", gputime);

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
		printf("The result matrices are identical!\n");
		printf("Speedup: %1.4f\n", seqtime/gputime);
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

//	printf("Usage: %s <size>\n", argv[0]);

	if (argc >= 2)
		size = atoi(argv[1]);
	else
		size = 1024;

	fprintf(stderr,"Sequential/Optimized Matrix Multiplication of size %d\n", size);

	// Multiply the matrices
	work();

	return 0;
}
