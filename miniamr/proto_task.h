#ifndef __PROTO_TASK_H
#define __PROTO_TASK_H

void init_task(size_t xdim, size_t ydim, size_t zdim, double array[xdim][ydim][zdim], long int rand_seed);

void split_block_task(size_t xdim, size_t ydim, size_t zdim,
                      double array_from[xdim][ydim][zdim], double array_to[xdim][ydim][zdim],
                      int i1, int j1, int k1);

void consolidate_block_task(size_t xdim, size_t ydim, size_t zdim,
                            double array_from[xdim][ydim][zdim], double array_to[xdim][ydim][zdim],
                            int i1, int j1, int k1);

void stencil_task7(size_t xdim, size_t ydim, size_t zdim, double array[xdim][ydim][zdim]);
void stencil_task27(size_t xdim, size_t ydim, size_t zdim, double array[xdim][ydim][zdim]);
double check_sum_task(size_t xdim, size_t ydim, size_t zdim, double array[xdim][ydim][zdim]);

#endif
