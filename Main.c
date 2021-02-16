/* Miriam Assraf */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include "Functions.h"
#include "Sequential.h"

#define MASTER 0
#define NUM_WEIGHTS 4

char** readFile(char* NS1, double* weights, int* num_seqs);
void writeToFile(Result* results, int num_seqs);
MPI_Datatype newType();

int main(int argc, char* argv[]) {
	int my_rank, num_procs;
	clock_t begin_parallel = clock(); // Get time for execution time calculation

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Datatype MPI_RESULT = newType();
	MPI_Status status;

	// Get values from input file
	double* weights = (double*)malloc(NUM_WEIGHTS * sizeof(double));
	if (weights == NULL) {
		error_handler("Failed to allocate memory for weights\n");
	}
	char* NS1 = (char*)malloc(3000 * sizeof(char));
	if (NS1 == NULL) {
		error_handler("Failed to allocate memory for NS1\n");
	}
	int num_seqs = 0;
	char** NS2 = readFile(NS1, weights, &num_seqs);

	// Initialize parallel result array for all sequences results
	Result* parallel_results = (Result*)malloc(num_seqs * sizeof(Result));
	if (parallel_results == NULL) {
		error_handler("Failed to allocate memory for parallel results\n");
	}
	// For each SN2
	for (int seq = 0; seq < num_seqs; seq++) {
		Result final_result = getBestScore(NS2[seq], NS1, weights, my_rank, num_procs);
		//Result final_result = getBestScore2(NS2[seq], NS1, weights, my_rank, num_procs);
		if (my_rank != MASTER) {
			// Replicas send final result - highest result for all mutants checked
			MPI_Send(&final_result, 1, MPI_RESULT, MASTER, 0, MPI_COMM_WORLD);
		}

		if (my_rank == MASTER) {
			// Master receive from all replicas their highest result
			for (int i = 1; i < num_procs; i++) {
				Result result;
				MPI_Recv(&result, 1, MPI_RESULT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				// Update to final result if higher score
				final_result = compareResults(result, final_result);
			}
			// Master gets final result for all calculations to current SN2 to parallel results
			parallel_results[seq] = final_result;
		}
		// All processes work on each SN2, so need to wait before continuing to next SN2
		MPI_Barrier(MPI_COMM_WORLD);
	}
	// After getting all SN2 results
	if (my_rank == MASTER) {
		writeToFile(parallel_results, num_seqs);  // Master prints results to output file

		clock_t end_parallel = clock();
		double time_parallel = (double)(end_parallel - begin_parallel) / CLOCKS_PER_SEC;

		testAndCompareTime(NS2, NS1, weights, num_seqs, parallel_results);
		// Print parallel calculation execution time		
		printf("Total time for parallel calculation: %.2f minutes\n", time_parallel / 60);
	}

	free(NS1);
	free(NS2);
	free(weights);
	free(parallel_results);

	MPI_Finalize();

	return 0;
}

char** readFile(char* NS1, double* weights, int* num_seqs) {
	FILE* inputFile = fopen("input.txt", "r");

	if (inputFile == NULL) {
		error_handler("Failed to open file\n");
	}
	fscanf(inputFile, "%lf %lf %lf %lf", &weights[0], &weights[1], &weights[2],
		&weights[3]);
	fscanf(inputFile, "%s", NS1);
	fscanf(inputFile, "%d", num_seqs);

	char** NS2 = (char**)malloc(sizeof(char*) * *num_seqs);
	if (NS2 == NULL) {
		error_handler("Failed to allocate memory for NS2\n");
	}
	for (int i = 0; i < *num_seqs; i++) {
		NS2[i] = (char*)malloc(sizeof(char) * 2000);
		if (NS2[i] == NULL) {
			error_handler("Failed to allocate memory for NS2[i]\n");
		}
		fscanf(inputFile, "%s", NS2[i]);
	}

	fclose(inputFile);
	return NS2;
}

void writeToFile(Result* results, int num_seqs)
{
	FILE* outputFile;
	outputFile = fopen("output.txt", "w");

	if (outputFile == NULL) {
		error_handler("Failed to open file\n");
	}

	for (int seq = 0; seq < num_seqs; seq++) {
		fprintf(outputFile, "Offset n = %d\tMS(%d)\n", results[seq].offset, results[seq].mutant);
	}
	fclose(outputFile);
}

MPI_Datatype newType() {
	// Create new mpi data type for Result to transfer between processes
	MPI_Datatype MPI_RESULT;
	int lengths[3] = { 1, 1, 1 }; // none is array - each contains 1 value
	// Where each Result variable starts 
	// First one is score -  starts at 0
	// Second is offset - starts after score which is double
	// Last is mutant - starts after offset and score which are int and double
	const MPI_Aint displacements[3] = { 0, sizeof(double), sizeof(double) + sizeof(int) };
	// Result variables types	
	MPI_Datatype types[3] = { MPI_DOUBLE, MPI_INT, MPI_INT };
	MPI_Type_create_struct(3, lengths, displacements, types, &MPI_RESULT);
	MPI_Type_commit(&MPI_RESULT);

	return MPI_RESULT;
}
