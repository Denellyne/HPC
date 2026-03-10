#ifndef MATRIX
#define MATRIX

double get_gflops_matrix_naive(int N, int R);
double get_gflops_matrix_vectorized(int N, int R);
double get_gflops_matrix_vectorized_transposed(int N, int R);
double get_gflops_matrix_vectorized_transposed_blocking(int N, int R);

#endif // !MATRIX
