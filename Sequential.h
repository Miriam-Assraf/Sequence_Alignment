#pragma once
#include "Functions.h"

Result* sequential(char* NS1, char** NS2, double* weights, int num_seqs);
void testResults(Result* parallel_res, Result* sequential_res, int num_seqs);
void testAndCompareTime(char** NS2, char* NS1, double* weights, int num_seqs, Result* parallel_results);
