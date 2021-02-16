/* Miriam Assraf */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include "Functions.h"

Result* sequential(char* NS1, char** NS2, double* weights, int num_seqs)
{
	// Calculate best results sequentially and return all results
	Result* results = (Result*)malloc(num_seqs * sizeof(Result));
	if (results == NULL) {
		error_handler("Failed to allocate memory for results\n");
	}
	// For each sequence
	for (int seq = 0; seq < num_seqs; seq++)
	{
		int max_offset = strlen(NS1) - (strlen(NS2[seq]) + 1);
		int max_k = strlen(NS2[seq]) + 1;
		
		Result final_result = initializeResult(-DBL_MAX, 0, 1);
		// For each offset n
		for (int n = 0; n < max_offset; n++)
		{
			Result best_result = initializeResult(-DBL_MAX, n, 1);
			// For each mutant k
			for (int k = 1; k < max_k; k++)
			{
				// Get similarity array
				double* mutant_results = getMutantSimilarity(NS2[seq], NS1, weights, k, n);
				Result current_result = initializeResult(0, n, k);

				// Calculate current result by similarity
				for (int i = 0; i <= strlen(NS2[seq]); i++)
				{
					current_result.score += mutant_results[i];
				}
				// Update best result to current result if higher score
				best_result = compareResults(current_result, best_result);
				free(mutant_results);
			}
			// Update final result to best offset result if higher score
			final_result = compareResults(best_result, final_result);
		}
		// Enter final result for current SN2 to results array
		results[seq] = final_result;
	}

	return results;
}

void testResults(Result* parallel_res, Result* sequential_res, int num_seqs)
{
	// Check for parallel results if matches sequential results
	int passed = TRUE;

	for (int i = 0; i < num_seqs; i++)
	{
		if (parallel_res[i].score != sequential_res[i].score ||
			parallel_res[i].offset != sequential_res[i].offset ||
			parallel_res[i].mutant != sequential_res[i].mutant)
		{
			passed = FALSE;
		}
	}

	if (passed == TRUE) {
		printf("Test passed successfully!\n");
	}
	else {
		fprintf(stderr, "Test failed!\n");
	}
}

void testAndCompareTime(char** NS2, char* NS1, double* weights, int num_seqs, Result* parallel_results)
{
	// Get sequential results and calculate execution time
	clock_t begin_sequential = clock();
	Result* sequential_results = sequential(NS1, NS2, weights, num_seqs);
	clock_t end_sequential = clock();
	double time_sequential = (double)(end_sequential - begin_sequential) / CLOCKS_PER_SEC;

	// Check if parallel results matches sequential results
	testResults(parallel_results, sequential_results, num_seqs);
	// Print execution time of sequential calculation
	printf("Total time for sequential calculation: %.2f minutes\n", time_sequential / 60);
	free(sequential_results);
}
