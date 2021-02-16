/* Miriam Assraf */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "Functions.h"

#define STAR 0
#define COLON 1
#define POINT 2
#define SPACE 3

const char* consrvatives[] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK",
		"FYW", "HY", "MILF" };
const char* semi_consrvatives[] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK",
		"NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };

int isConservative(char ns1, char ns2) {
	// Check if both chars appear at same sequence of conservatives
	for (int j = 0; j < sizeof(consrvatives) / sizeof(consrvatives[0]); j++) {
		if (strchr(consrvatives[j], ns1) != NULL
			&& strchr(consrvatives[j], ns2) != NULL) {
			return TRUE;
		}
	}

	return FALSE;
}

int isSemiConservative(char ns1, char ns2) {
	// Check if both chars appear at same sequence of semi-conservatives
	for (int j = 0; j < sizeof(semi_consrvatives) / sizeof(semi_consrvatives[0]); j++) {
		if (strchr(semi_consrvatives[j], ns1) != NULL
			&& strchr(semi_consrvatives[j], ns2) != NULL) {
			return TRUE;
		}
	}

	return FALSE;
}

double charsComparison(char ns1, char ns2, double* weights) {
	double value;

	// Compare chars and return weight
	if (ns1 == ns2) {  // Star
		value = weights[STAR];
	}
	else if (isConservative(ns1, ns2) == TRUE) {  // Colon
		value = -weights[COLON];
	}
	else if (isSemiConservative(ns1, ns2) == TRUE) {  // Point
		value = -weights[POINT];
	}
	else {	// Space
		value = -weights[SPACE];
	}
	return value;
}

double* getMutantSimilarity(char* ns2, char* ns1, double* weights, int k, int n)
{
	// Compare mutant with give k with matching char of NS1 and build weights array
	// For each index compare mutant and NS1 char 
	// and enter to similarity array compatible weight 
	int i = 0;
	double* similarity = (double*)malloc(sizeof(double) * (strlen(ns2) + 1));
	if (similarity == NULL) {
		error_handler("Failed to allocate memory for similarity\n");
	}

	while (i < k) {
		similarity[i] = charsComparison(ns1[n + i], ns2[i], weights);
		i++;
	}

	similarity[i] = charsComparison(ns1[n + i], '-', weights);
	i++;

	while (i < strlen(ns2) + 1) {
		similarity[i] = charsComparison(ns1[n + i], ns2[i - 1], weights);
		i++;
	}

	return similarity;
}

Result getBestScore(char* NS2, char* NS1, double* weights, int my_rank, int num_procs)
{
	int max_n = strlen(NS1) - (strlen(NS2) + 1); // If same length than max_n will be 0
	Result final_result = initializeResult(-DBL_MAX, 0, 1);	// Initialize final_result with minimum double value, first offset and k

	int start, end;
	assignPart(&start, &end, max_n, my_rank, num_procs);

	// Each process calculates equal amount of offsets possible
	for (int n = start; n < end; n++) {
		// Allocate memory to save all mutants similarities of current offset
		double* similarities = (double*)malloc(strlen(NS2) * (strlen(NS2) + 1) * sizeof(double));
		if (similarities == NULL) {
			error_handler("Failed to allocate memory for similarities\n");
		}
		// Get all similarities for all mutants of current offset with OpenMP
		getMutantsSimilaritiesWithOpenMP(similarities, NS2, NS1, weights, n);
		// Get best result of all similarities with CUDA - best mutant for current offset
		Result best_result = getBestMutantScoreWithCUDA(similarities, strlen(NS2), strlen(NS2) + 1, n, start);
		// Get final result comparing offsets results
		// After getting each offset best result - update final result if higher score
		final_result = compareResults(best_result, final_result);
	}
	// Return best result - higher score of all offsets mutants
	return final_result;
}

void getMutantsSimilaritiesWithOpenMP(double* similarities, char* NS2, char* NS1, double* weights, int n)
{
	// Set number of threads to number of mutants for current offset
	int num_threads = strlen(NS2);

#pragma omp parallel num_threads(num_threads)
	{
		// Each thread calculates one mutant weights array, based on mutant comparison with NS1
		int tid = omp_get_thread_num();
		// Allocate memory for each mutant similarity
		double* similarity = getMutantSimilarity(NS2, NS1, weights, tid+1, n);
		// Enter similarity for current mutant k to similarities array
		memcpy(similarities + tid * (strlen(NS2) + 1), similarity, (strlen(NS2) + 1) * sizeof(double));
		free(similarity);
	}
}

Result getBestMutantScoreWithCUDA(double* similarities, int num_mutants, int mutant_sz, int n, int first_offset)
{
	// Allocate memory for scores array - each index for a score of different mutant
	double* scores = (double*)malloc(num_mutants * sizeof(double));
	if (scores == NULL) {
		error_handler("Failed to allocate memory for scores\n");
	}
	// Calculate scores of all mutants with CUDA
	calcScoreWithCuda(similarities, scores, num_mutants, mutant_sz);
	
	Result best_result = initializeResult(-DBL_MAX, first_offset, 1);	// Initialize best score for all mutants with minimum double value, first offset this process is checking and first k

	// Get best result by comparing all mutants results
	for (int k = 0; k < num_mutants; k++) {
		Result current_result = initializeResult(0, n, k + 1);
		current_result.score = scores[k];
		// For each mutant score - update best result if higher score
		best_result = compareResults(current_result, best_result);
	}
	free(scores);
	free(similarities);

	// Return best result of all mutants for current offset
	return best_result;
}

Result initializeResult(double score, int n, int k)
{
	// Initialize Result values
	Result result;
	result.score = score;
	result.offset = n;
	result.mutant = k;

	return result;
}

Result compareResults(Result res1, Result res2) {
	// Compare both results scores and return result with higher score
	if (res1.score > res2.score) {
		return res1;
	}
	return res2;
}

void assignPart(int* start, int* end, int max_val, int my_rank, int num_procs)
{
	// Divide work equally
	int part = max_val / num_procs;
	*start = my_rank * part;
	// If can't divide equally - assign the rest to last process
	if (max_val % num_procs != 0 && my_rank == num_procs - 1) {
		part += max_val % num_procs;
	}
	*end = *start + part;
}

void error_handler(const char* error_msg)
{
	fprintf(stderr, "%s\n", error_msg);
	MPI_Abort(MPI_COMM_WORLD, __LINE__);
}
