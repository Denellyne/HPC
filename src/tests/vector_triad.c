#include "vector_triad.h"
#include "../lib/utils.h"
#include <immintrin.h>
#include <stdlib.h>
void dummy(double *__restrict A, double *__restrict B, double *__restrict D) {}

double get_GFLOPS_Stack_Vec(int arr_size, int repetitions) {
  double A[arr_size], B[arr_size], D[arr_size];
  double start, end;
  for (int i = 0; i < arr_size; i++) {
    A[i] = 0.0;
    B[i] = 1.0;
    D[i] = 3.0;
  }
  get_walltime(&start);
  for (unsigned i = 0; i < repetitions; i++) {
    for (unsigned j = 0; j + 3 < arr_size; j += 4) {
      A[j] = B[j] + 2.0 * D[j];
      A[j + 1] = B[j + 1] + 2.0 * D[j + 1];
      A[j + 2] = B[j + 2] + 2.0 * D[j + 2];
      A[j + 3] = B[j + 3] + 2.0 * D[j + 3];
    }
    for (unsigned j = arr_size - arr_size % 4; j < arr_size; j++)
      A[j] = B[j] + 2.0 * D[j];

    if (A[2] < 0.0)
      dummy(A, B, D);
  }

  get_walltime(&end);

  return repetitions * arr_size * 2.0 / ((end - start) * 1.0e9);
}
double get_GFLOPS_Stack(int arr_size, int repetitions) {
  double A[arr_size], B[arr_size], D[arr_size];
  double start, end;
  for (int i = 0; i < arr_size; i++) {
    A[i] = 0.0;
    B[i] = 1.0;
    D[i] = 3.0;
  }
  get_walltime(&start);
  for (unsigned i = 0; i < repetitions; i++) {
    for (unsigned j = 0; j < arr_size; j++) {
      A[j] = B[j] + 2.0 * D[j];
    }
    if (A[2] < 0.0)
      dummy(A, B, D);
  }

  get_walltime(&end);

  return repetitions * arr_size * 2.0 / ((end - start) * 1.0e9);
}

double get_GFLOPS(int arr_size, int repetitions) {
  double *__restrict A, *__restrict B, *__restrict D;
  A = (double *)alloc(sizeof(double), arr_size);
  B = (double *)alloc(sizeof(double), arr_size);
  D = (double *)alloc(sizeof(double), arr_size);
  double start, end;
  for (int i = 0; i < arr_size; i++) {
    A[i] = 0.0;
    B[i] = 1.0;
    D[i] = 3.0;
  }
  get_walltime(&start);
  for (unsigned i = 0; i < repetitions; i++) {
    for (unsigned j = 0; j < arr_size; j++) {
      A[j] = B[j] + 2.0 * D[j];
    }
    if (A[2] < 0.0)
      dummy(A, B, D);
  }

  get_walltime(&end);
  free(A);
  free(B);
  free(D);

  return repetitions * arr_size * 2.0 / ((end - start) * 1.0e9);
}

double get_GFLOPS_Vec(int arr_size, int repetitions) {
  double *__restrict A, *__restrict B, *__restrict D;
  A = (double *)alloc(sizeof(double), arr_size);
  B = (double *)alloc(sizeof(double), arr_size);
  D = (double *)alloc(sizeof(double), arr_size);
  double start, end;
  for (int i = 0; i < arr_size; i++) {
    A[i] = 0.0;
    B[i] = 1.0;
    D[i] = 3.0;
  }
  __m256d scalar = _mm256_set1_pd(2.0);
  get_walltime(&start);
  for (unsigned r = 0; r < repetitions; r++) {
    for (unsigned j = 0; j + 15 < arr_size; j += 16) {
      __m256d b0 = _mm256_loadu_pd(&B[j]);
      __m256d d0 = _mm256_loadu_pd(&D[j]);
      _mm256_storeu_pd(&A[j], _mm256_add_pd(b0, _mm256_mul_pd(d0, scalar)));

      __m256d b1 = _mm256_loadu_pd(&B[j + 4]);
      __m256d d1 = _mm256_loadu_pd(&D[j + 4]);
      _mm256_storeu_pd(&A[j + 4], _mm256_add_pd(b1, _mm256_mul_pd(d1, scalar)));

      __m256d b2 = _mm256_loadu_pd(&B[j + 8]);
      __m256d d2 = _mm256_loadu_pd(&D[j + 8]);
      _mm256_storeu_pd(&A[j + 8], _mm256_add_pd(b2, _mm256_mul_pd(d2, scalar)));

      __m256d b3 = _mm256_loadu_pd(&B[j + 12]);
      __m256d d3 = _mm256_loadu_pd(&D[j + 12]);
      _mm256_storeu_pd(&A[j + 12],
                       _mm256_add_pd(b3, _mm256_mul_pd(d3, scalar)));
    }

    for (unsigned j = arr_size - arr_size % 16; j < arr_size; j++)
      A[j] = B[j] + 2.0 * D[j];

    if (A[2] < 0.0)
      dummy(A, B, D);
  }

  get_walltime(&end);

  free(A);
  free(B);
  free(D);
  return repetitions * arr_size * 2.0 / ((end - start) * 1.0e9);
}
double get_GFLOPS_Stack_Vec_SIMD(int arr_size, int repetitions) {
  double A[arr_size], B[arr_size], D[arr_size];
  double start, end;
  for (int i = 0; i < arr_size; i++) {
    A[i] = 0.0;
    B[i] = 1.0;
    D[i] = 3.0;
  }
  __m256d scalar = _mm256_set1_pd(2.0);
  get_walltime(&start);
  for (unsigned r = 0; r < repetitions; r++) {
    for (unsigned j = 0; j + 15 < arr_size; j += 16) {
      __m256d b0 = _mm256_loadu_pd(&B[j]);
      __m256d d0 = _mm256_loadu_pd(&D[j]);
      _mm256_storeu_pd(&A[j], _mm256_add_pd(b0, _mm256_mul_pd(d0, scalar)));

      __m256d b1 = _mm256_loadu_pd(&B[j + 4]);
      __m256d d1 = _mm256_loadu_pd(&D[j + 4]);
      _mm256_storeu_pd(&A[j + 4], _mm256_add_pd(b1, _mm256_mul_pd(d1, scalar)));

      __m256d b2 = _mm256_loadu_pd(&B[j + 8]);
      __m256d d2 = _mm256_loadu_pd(&D[j + 8]);
      _mm256_storeu_pd(&A[j + 8], _mm256_add_pd(b2, _mm256_mul_pd(d2, scalar)));

      __m256d b3 = _mm256_loadu_pd(&B[j + 12]);
      __m256d d3 = _mm256_loadu_pd(&D[j + 12]);
      _mm256_storeu_pd(&A[j + 12],
                       _mm256_add_pd(b3, _mm256_mul_pd(d3, scalar)));
    }

    for (unsigned j = arr_size - arr_size % 16; j < arr_size; j++)
      A[j] = B[j] + 2.0 * D[j];

    if (A[2] < 0.0)
      dummy(A, B, D);
  }

  get_walltime(&end);

  return repetitions * arr_size * 2.0 / ((end - start) * 1.0e9);
}
