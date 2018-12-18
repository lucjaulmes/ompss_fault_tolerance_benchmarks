#pragma omp target device(cuda) ndrange(1, bs, 512) copy_out(a[j;bs], b[j;bs], c[j;bs]) label(init_kernel)
#pragma omp task
//#pragma omp task out(a[j;bs], b[j;bs], c[j;bs]) label(init_kernel)
__global__ void init_kernel (double *a, double *b, double *c, int bs, int j);

#pragma omp target device(cuda) ndrange(1, bs, 512) copy_in (a[j;bs]) copy_out (c[j;bs]) label(copy_kernel)
#pragma omp task in
//#pragma omp task in (a[j;bs]) out (c[j;bs]) label(copy_kernel)
__global__ void copy_kernel (double *a, double *c, int bs, int j);

#pragma omp target device(cuda) ndrange(1, bs, 512) copy_in (c[j;bs]) copy_out (b[j;bs]) label(scale_kernel)
#pragma omp task
//#pragma omp task in (c[j;bs]) out (b[j;bs]) label(scale_kernel)
__global__ void scale_kernel (double *b, double *c, double scalar, int bs, int j);

#pragma omp target device(cuda) ndrange(1, bs, 512) copy_in (a[j;bs], b[j;bs]) copy_out (c[j;bs]) label(add_kernel)
#pragma omp task
//#pragma omp task in (a[j;bs], b[j;bs]) out (c[j;bs]) label(add_kernel)
__global__ void add_kernel (double *a, double *b, double *c, int bs, int j);

#pragma omp target device(cuda) ndrange(1, bs, 512) copy_in (b[j;bs], c[j;bs]) copy_out (a[j;bs]) label(triad_kernel)
#pragma omp task
//#pragma omp task in (b[j;bs], c[j;bs]) out (a[j;bs]) label(triad_kernel)
__global__ void triad_kernel (double *a, double *b, double *c, double scalar, int bs, int j);
