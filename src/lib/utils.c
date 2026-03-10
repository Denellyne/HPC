#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
void get_walltime_(double *wcTime) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec / 1000000.0);
}
void get_walltime(double *wcTime) { get_walltime_(wcTime); }
void **alloc_array(unsigned size, unsigned N) {
  void **arr = aligned_alloc(32, size * N);
  if (!arr) {
    printf("Unable to allocate memory for array: %d", size * N);
    abort();
  }

  for (int i = 0; i < N; i++)
    arr[i] = alloc(size, 1);

  return arr;
}
void delete_array(void **arr, unsigned N) {
  for (int i = 0; i < N; i++) {
    free(arr[i]);
  }
  free(arr);
}

void *alloc(unsigned size, unsigned N) {

  void *arr = malloc(size * N);
  if (!arr) {
    printf("Unable to allocate memory of size: %d", size * N);
    abort();
  }
  return arr;
}
