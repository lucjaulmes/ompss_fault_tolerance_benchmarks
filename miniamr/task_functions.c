#include "block.h"
#include "proto_task.h"
#include <stdlib.h>
#include <string.h>

void init_task(size_t xdim, size_t ydim, size_t zdim,
               double array[xdim][ydim][zdim],
               long int rand_seed)
{
  struct drand48_data buff;
  srand48_r(rand_seed, &buff);

  for (size_t ib = 1; ib <= xdim - 2; ib++) {
    for (size_t jb = 1; jb <= ydim - 2; jb++) {
      for (size_t kb = 1; kb <= zdim - 2; kb++) {
        drand48_r(&buff, &array[ib][jb][kb]);
      }
    }
  }
}


void split_block_task(size_t xdim, size_t ydim, size_t zdim,
                      double array_from[xdim][ydim][zdim],
                      double array_to[xdim][ydim][zdim],
                      int i1, int j1, int k1)
{
  int i, j, k, i2, j2, k2;

  for (i2 = i = 1; i <= x_block_half; i++, i2+=2) {
    for (j2 = j = 1; j <= y_block_half; j++, j2+=2) {
      for (k2 = k = 1; k <= z_block_half; k++, k2+=2) {
        array_to[i2  ][j2  ][k2  ] =
          array_to[i2+1][j2  ][k2  ] =
          array_to[i2  ][j2+1][k2  ] =
          array_to[i2+1][j2+1][k2  ] =
          array_to[i2  ][j2  ][k2+1] =
          array_to[i2+1][j2  ][k2+1] =
          array_to[i2  ][j2+1][k2+1] =
          array_to[i2+1][j2+1][k2+1] =
          array_from[i+i1][j+j1][k+k1]/8.0;
      }
    }
  }
}


void consolidate_block_task(size_t xdim, size_t ydim, size_t zdim,
                            double array_from[xdim][ydim][zdim], double array_to[xdim][ydim][zdim],
                            int i1, int j1, int k1)
{

  int i, j, k, i2, j2, k2;

  for (i2 = i = 1; i <= x_block_half; i++, i2+=2) {
    for (j2 = j = 1; j <= y_block_half; j++, j2+=2) {
      for (k2 = k = 1; k <= z_block_half; k++, k2+=2) {
        array_to[i+i1][j+j1][k+k1] =
          array_from[i2  ][j2  ][k2  ] +
          array_from[i2+1][j2  ][k2  ] +
          array_from[i2  ][j2+1][k2  ] +
          array_from[i2+1][j2+1][k2  ] +
          array_from[i2  ][j2  ][k2+1] +
          array_from[i2+1][j2  ][k2+1] +
          array_from[i2  ][j2+1][k2+1] +
          array_from[i2+1][j2+1][k2+1];
      }
    }
  }
}


void stencil_task7(size_t xdim, size_t ydim, size_t zdim,
                   double array[xdim][ydim][zdim])
{
  double work[xdim][ydim][zdim];
  memcpy(work, array, sizeof(work));

  for (size_t i = 1; i <= xdim-2; i++)
    for (size_t j = 1; j <= ydim-2; j++)
      for (size_t k = 1; k <= zdim-2; k++)
        array[i][j][k] = (work[i-1][j  ][k  ] +
                          work[i  ][j-1][k  ] +
                          work[i  ][j  ][k-1] +
                          work[i  ][j  ][k  ] +
                          work[i  ][j  ][k+1] +
                          work[i  ][j+1][k  ] +
                          work[i+1][j  ][k  ])/7.0;
}


void stencil_task27(size_t xdim, size_t ydim, size_t zdim,
                    double array[xdim][ydim][zdim])
{
  double work[xdim][ydim][zdim];
  memcpy(work, array, sizeof(work));
  double sb, sm, sf;

  for (size_t i = 1; i <= xdim-2; i++)
    for (size_t j = 1; j <= ydim-2; j++)
      for (size_t k = 1; k <= zdim-2; k++) {
        sb = work[i-1][j-1][k-1] +
          work[i-1][j-1][k  ] +
          work[i-1][j-1][k+1] +
          work[i-1][j  ][k-1] +
          work[i-1][j  ][k  ] +
          work[i-1][j  ][k+1] +
          work[i-1][j+1][k-1] +
          work[i-1][j+1][k  ] +
          work[i-1][j+1][k+1];
        sm = work[i  ][j-1][k-1] +
          work[i  ][j-1][k  ] +
          work[i  ][j-1][k+1] +
          work[i  ][j  ][k-1] +
          work[i  ][j  ][k  ] +
          work[i  ][j  ][k+1] +
          work[i  ][j+1][k-1] +
          work[i  ][j+1][k  ] +
          work[i  ][j+1][k+1];
        sf = work[i+1][j-1][k-1] +
          work[i+1][j-1][k  ] +
          work[i+1][j-1][k+1] +
          work[i+1][j  ][k-1] +
          work[i+1][j  ][k  ] +
          work[i+1][j  ][k+1] +
          work[i+1][j+1][k-1] +
          work[i+1][j+1][k  ] +
          work[i+1][j+1][k+1];
        array[i][j][k] = (sb + sm + sf)/27.0;
      }
}


double check_sum_task(size_t xdim, size_t ydim, size_t zdim,
                      double array[xdim][ydim][zdim])
{
  double ret = 0.0;
  for (size_t i = 1; i <= xdim-2; i++)
    for (size_t j = 1; j <= ydim-2; j++)
      for (size_t k = 1; k <= zdim-2; k++)
        ret += array[i][j][k];

  return ret;
}
