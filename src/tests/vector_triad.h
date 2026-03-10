#ifndef VECTOR_TRIAD
#define VECTOR_TRIAD
double get_GFLOPS_Stack(int arr_size, int repetitions);
double get_GFLOPS(int arr_size, int repetitions);
double get_GFLOPS_Vec(int arr_size, int repetitions);
double get_GFLOPS_Stack_Vec(int arr_size, int repetitions);
double get_GFLOPS_Stack_Vec_SIMD(int arr_size, int repetitions);

#endif // !VECTOR_TRIAD
