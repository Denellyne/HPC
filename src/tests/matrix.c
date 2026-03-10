#include "matrix.h"
#include "../lib/utils.h"
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>

double get_gflops_matrix_naive(int N, int R) {
  float A[N][N], B[N][N], C[N][N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() % 128;
      B[i][j] = rand() % 128;
      C[i][j] = 0;
    }
  }
  double start, end;

  get_walltime(&start);
  for (int r = 0; r < R; r++) {

    for (int i = 0; i < N; i++) {
      for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
          C[i][k] += A[i][j] * B[j][i];
        }
      }
    }
  }
  get_walltime(&end);
  return R * 2.0 * N * N * N / ((end - start) * 1.0e9);
}

double get_gflops_matrix_vectorized(int N, int R) {
  float A[N][N], B[N][N], C[N][N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() % 128;
      B[i][j] = rand() % 128;
      C[i][j] = 0;
    }
  }
  double start, end;

  get_walltime(&start);
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < N; i++) {
      for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j += 4) {
          C[i][k] += A[i][j] * B[j][i];
          C[i][k] += A[i][j + 1] * B[j + 1][i];
          C[i][k] += A[i][j + 2] * B[j + 2][i];
          C[i][k] += A[i][j + 3] * B[j + 3][i];
        }
      }
    }
  }
  get_walltime(&end);
  return R * 2.0 * N * N * N / ((end - start) * 1.0e9);
}

double get_gflops_matrix_vectorized_transposed(int N, int R) {
  float A[N][N], D[N][N], C[N][N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() % 128;
      D[i][j] = rand() % 128;
      C[i][j] = 0;
    }
  }
  double start, end;

  get_walltime(&start);
  float B[N][N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      B[i][j] = D[j][i];
    }
  }

  for (int r = 0; r < R; r++) {
    for (int i = 0; i < N; i++) {
      for (int k = 0; k < N; k++) {

        __m256 sum = _mm256_setzero_ps();

        for (int j = 0; j < N; j += 8) {

          __m256 a = _mm256_loadu_ps(&A[i][j]);
          __m256 b = _mm256_loadu_ps(&B[k][j]);

          __m256 prod = _mm256_mul_ps(a, b);
          sum = _mm256_add_ps(sum, prod);
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, sum);

        for (int t = 0; t < 8; t++)
          C[i][k] += tmp[t];
      }
    }
  }
  get_walltime(&end);
  return R * 2.0 * N * N * N / ((end - start) * 1.0e9);
}

double get_gflops_matrix_vectorized_transposed_blocking(int N, int R) {
  float A[N][N], D[N][N], C[N][N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() % 128;
      D[i][j] = rand() % 128;
      C[i][j] = 0;
    }
  }
  double start, end;

  get_walltime(&start);
  float B[N][N];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      B[i][j] = D[j][i];

  int blockSize = 64;
  for (int r = 0; r < R; r++)
    for (int ii = 0; ii < N; ii += blockSize)
      for (int jj = 0; jj < N; jj += blockSize)
        for (int kk = 0; kk < N; kk += blockSize)

          for (int i = ii; i < ii + blockSize; i++)
            for (int k = kk; k < kk + blockSize; k++) {

              __m256 sum = _mm256_setzero_ps();

              for (int j = jj; j < jj + blockSize; j += 8) {

                __m256 a = _mm256_loadu_ps(&A[i][j]);
                __m256 b = _mm256_loadu_ps(&B[k][j]);

                __m256 prod = _mm256_mul_ps(a, b);
                sum = _mm256_add_ps(sum, prod);
              }

              float tmp[8];
              _mm256_storeu_ps(tmp, sum);

              for (int t = 0; t < 8; t++)
                C[i][k] += tmp[t];
            }

  get_walltime(&end);
  return R * 2.0 * N * N * N / ((end - start) * 1.0e9);
}
