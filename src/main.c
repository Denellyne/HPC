#include "tests/matrix.h"
#include "tests/vector_triad.h"
#include <stdio.h>
#include <stdlib.h>
#define N 32 * 8192 * 4
#define NSTACK 32 * 1024
#define MATRIX_N 64
#define R 256

int main(int argc, char *argv[]) {

  // printf("Calculating GFLOPS Stack:%f\n", get_GFLOPS_Stack(NSTACK, R));
  // printf("Calculating GFLOPS Stack Vectorized:%f\n",
  //        get_GFLOPS_Stack_Vec(NSTACK, R));
  // printf("Calculating GFLOPS Stack Intrinsics Vectorized:%f\n",
  //        get_GFLOPS_Stack_Vec_SIMD(NSTACK, R));
  // printf("Calculating GFLOPS:%f\n", get_GFLOPS(NSTACK, R));
  // printf("Calculating GFLOPS Vectorized:%f\n", get_GFLOPS_Vec(NSTACK, R));
  printf("Calculating GFLOPS for Matrix Multiply Naive:%f\n",
         get_gflops_matrix_naive(MATRIX_N, R));
  printf("Calculating GFLOPS for Matrix Multiply:%f\n",
         get_gflops_matrix_vectorized(MATRIX_N, R));
  printf("Calculating GFLOPS for Matrix Multiply transposed:%f\n",
         get_gflops_matrix_vectorized_transposed(MATRIX_N, R));
  printf("Calculating GFLOPS for Matrix Multiply transposed blocking:%f\n",
         get_gflops_matrix_vectorized_transposed_blocking(MATRIX_N, R));

  return EXIT_SUCCESS;
}
