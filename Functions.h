#pragma once
#define TRUE 1
#define FALSE 0

typedef struct Result {
	double score;
	int offset, mutant;
} Result;

int isConservative(char ns1, char ns2);
int isSemiConservative(char ns1, char ns2);
double* getMutantSimilarity(char* ns2, char* ns1, double* weights, int k, int n);
double charsComparison(char ns1, char ns2, double* weights);
Result getBestScore(char* NS2, char* NS1, double* weights, int my_rank, int num_procs);
void getMutantsSimilaritiesWithOpenMP(double* similarities, char* NS2, char* NS1, double* weights, int n);
Result getBestMutantScoreWithCUDA(double* similarities, int num_mutants, int mutant_sz, int n, int first_offset);
int calcScoreWithCuda(double* similarities, double* scores, int num_rows, int num_cols);
Result initializeResult(double score, int n, int k);
Result compareResults(Result res1, Result res2);
void assignPart(int* start, int* end, int max_val, int my_rank, int num_procs);
void error_handler(const char* error_msg);
