# Sequence_Alignment
A naïve approach for performing pairwise sequence alignment in order to maximize the similarity between two sequences. </br>
Defining offset n as the number of gaps inserted before the second sequence, and MS(k) as a gap inserted after the k-th letter of the </br>
second sequence, the purpose is to find the optimal n and k, leading to the best alignment score. </br>
Parallel implementation using MPI, OpenMP and CUDA. </br>

Input file contains different weights for different conservation groups, the first sequense SN1, the number of sequences SN2 to align and the sequences SN2. </br>
Produces output file including the best alignment for each sequence. </br>
