/* Miriam Assraf */

#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

__global__ void calcScoresKernel(double* dev_results, double* dev_scores, int num_rows, int num_cols);
__host__ void checkErrors(cudaError_t err, const char* error_msg);

int calcScoreWithCuda(double* similarities, double* scores, int num_rows, int num_cols)
{
	cudaError_t err = cudaSuccess;
	double* dev_similarities;
	double* dev_scores;

	// Allocate memory on GPU 
	err = cudaMalloc((void**)&dev_similarities, num_rows * num_cols * sizeof(double));
	checkErrors(err, "Failed to allocate similarities to device memory - %s\n");

	err = cudaMalloc((void**)&dev_scores, num_rows * sizeof(double));
	checkErrors(err, "Failed to allocate scores to device memory - %s\n");
	// Initialize device scores with zeros
	err = cudaMemset(dev_scores, 0, num_rows * sizeof(double));
	checkErrors(err, "Failed to initiate device scores - %s\n");

	// Copy similarities from host to the GPU memory
	err = cudaMemcpy(dev_similarities, similarities, num_rows * num_cols * sizeof(double), cudaMemcpyHostToDevice);
	checkErrors(err, "Failed to copy similarities from host to device - %s\n");

	// Launch the Kernel
	int threadsPerBlock = (int)ceil(sqrt(num_rows));
	if (threadsPerBlock % 32 != 0)
	{
		threadsPerBlock = threadsPerBlock + 32 - threadsPerBlock % 32;	// Make sure block size is multiple of 32
	}
	int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;	// We get total number of threads as number of rows (or a bit more so it is a multiple of 32)
	calcScoresKernel << <blocksPerGrid, threadsPerBlock >> > (dev_similarities, dev_scores, num_rows, num_cols);
	err = cudaGetLastError();
	checkErrors(err, "Failed to launch comparison kernel -  %s\n");
	
	// Copy the  scores from GPU to the host memory
	err = cudaMemcpy(scores, dev_scores, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
	checkErrors(err, "Failed to copy calculated scores from device to host -%s\n");
	
	// Free allocated memory on GPU
	err = cudaFree(dev_similarities);
	checkErrors(err, "Failed to free device similarities - %s\n");

	err = cudaFree(dev_scores);
	checkErrors(err, "Failed to free device scores - %s\n");

	return 0;
}

__global__ void calcScoresKernel(double* dev_similarities, double* dev_scores, int num_rows, int num_cols)
{
	// Each thread calculates score for one row of similarities (one mutant similarity)
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < num_cols; i++) {
		dev_scores[row] += dev_similarities[num_cols * row + i];
	}
}

__host__ void checkErrors(cudaError_t err, const char* error_msg)
{
	// If didn't return cudaSuccess print error
	if (err != cudaSuccess) {
		fprintf(stderr, error_msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
