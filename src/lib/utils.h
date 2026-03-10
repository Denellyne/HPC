#ifndef UTILS
#define UTILS
#include <sys/time.h>

void get_walltime(double *wcTime);

void **alloc_array(unsigned size, unsigned N);
void delete_array(void **arr, unsigned N);
void *alloc(unsigned size, unsigned N);

#endif // !UTILS
