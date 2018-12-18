extern "C" {
__global__ void init_kernel (double *a, double *b, double *c, int bs, int j)
{
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + j;
  if ( i >= j+bs ) return;
  a[i] = 1.0;
  b[i] = 2.0;
  c[i] = 0.0;
  a[i] = 2.0E0 * a[i];
}

__global__ void copy_kernel (double *a, double *c, int bs, int j)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + j;
  if ( i >= j+bs ) return;
  c[i] = a[i];
}

__global__ void scale_kernel (double *b, double *c, double scalar, int bs, int j)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + j;
  if ( i >= j+bs ) return;
  b[i] = scalar * c[i];
}

__global__ void add_kernel (double *a, double *b, double *c, int bs, int j)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + j;
  if ( i >= j+bs ) return;
  c[i] = a[i] + b[i];
}
__global__ void triad_kernel (double *a, double *b, double *c, double scalar, int bs, int j)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + j;
  if ( i >= j+bs ) return;
  a[i] = b[i] + scalar * c[i];
}
}
