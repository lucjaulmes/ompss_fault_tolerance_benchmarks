/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <err.h>

#include <catchroi.h>

typedef double elem_t;      // float
#define gemm cblas_dgemm    // cblas_sgemm
#define abs fabs            // fabsl

static inline
void gemm_kernel(const size_t bsize, elem_t *A, elem_t *B, elem_t *C)
{
    #pragma omp task in([bsize*bsize]A, [bsize*bsize]B) commutative([bsize*bsize]C) label(gemm)
    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bsize, bsize, bsize, 1, A, bsize, B, bsize, 1, C, bsize);
}

int verify(size_t nblocks, size_t bsize, int niter, elem_t **C)
{
    unsigned int ii, jj;
    int res = 0;

    for (ii = 0; ii < nblocks; ++ii)
        for (jj = 0; jj < nblocks; ++jj)
        {
            elem_t *Cblk = C[ii * nblocks + jj];
            #pragma omp task in(C[bsize * bsize]) reduction(+:res)
            {
                unsigned int i, j;
                for (i = 0; i < bsize; ++i)
                    for (j = 0; j < bsize; ++j)
                    {
                        if (ii * bsize + i < 2)
                            continue;

                        elem_t i1 = 1. / (elem_t)(ii * bsize + i + 1);
                        elem_t shb = niter * (pow(i1, (elem_t)(jj * bsize + j + 1)) - 1) / (i1 - 1);
                        elem_t diff = abs((Cblk[i * bsize + j] - shb) / shb);

                        if (!(diff <= 1e-4))
                        {
                            res++;
                            fprintf(stderr, "error at %d, %d, %d, %d\n", ii, jj, i, j);
                        }
                   }
            }
        }
    #pragma omp taskwait

    return res;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s size block_size\n", argv[0]);
        exit(-1);
    }

    const unsigned long bsize = atoi(argv[2]);
    const unsigned long nblocks = atoi(argv[1]) / bsize;
    const unsigned long n = nblocks * bsize;
    unsigned int ii, jj, kk;

    printf("Allocating 3 matrices of size %lu x %lu as %lu x %lu blocks of %lu x %lu elements  =  %lu bytes\n", n, n, nblocks, nblocks, bsize, bsize, n * n * sizeof(elem_t));

    elem_t **A = malloc(nblocks * nblocks * sizeof(elem_t*));
    elem_t **B = malloc(nblocks * nblocks * sizeof(elem_t*));
    elem_t **C = malloc(nblocks * nblocks * sizeof(elem_t*));

    if (A == NULL || B == NULL || C == NULL)
        err(1, NULL);

    for (ii = 0; ii < nblocks * nblocks; ++ii)
    {
        A[ii] = CATCHROI_INSTRUMENT(aligned_alloc)(4096, bsize * bsize * sizeof(elem_t));
        B[ii] = CATCHROI_INSTRUMENT(aligned_alloc)(4096, bsize * bsize * sizeof(elem_t));
        C[ii] = CATCHROI_INSTRUMENT(aligned_alloc)(4096, bsize * bsize * sizeof(elem_t));

        if (A[ii] == NULL || B[ii] == NULL || C[ii] == NULL)
            err(1, NULL);
    }

    printf("Starting initialization\n");

    for (ii = 0; ii < nblocks; ++ii)
        for (jj = 0; jj < nblocks; ++jj)
        {
            #pragma omp task label(init) // out([bsize * bsize](A[ii * nblocks + jj]), [bsize * bsize](B[ii * nblocks + jj]))
            // do not use out() outside of ROI for refresh-skipping reasons
            {
                unsigned int i, j;
                for (i = 0; i < bsize; ++i)
                    for (j = 0; j < bsize; ++j)
                    {
                        A[ii * nblocks + jj][i * bsize + j] = pow(1. / (elem_t)(ii * bsize + i + 1), (elem_t)(jj * bsize + j));
                        B[ii * nblocks + jj][i * bsize + j] = (ii * bsize + i) <= (jj * bsize + j);
                        C[ii * nblocks + jj][i * bsize + j] = 0;
                    }
            }
        }
    #pragma omp taskwait

    printf("Starting execution\n");

    struct timeval start, stop;
    gettimeofday(&start, NULL);
    start_roi();

    for (ii = 0; ii < nblocks; ++ii)
        for (jj = 0; jj < nblocks; ++jj)
            for (kk = 0; kk < nblocks; ++kk)
                gemm_kernel(bsize, A[ii * nblocks + kk], B[kk * nblocks + jj], C[ii * nblocks + jj]);

    #pragma omp taskwait

    stop_roi(-1);
    gettimeofday(&stop, NULL);

    double time_seconds = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1e6;
    double gflops = 2.0e-9 * n * n * n / time_seconds;


#ifdef VALIDATE
    printf("Starting verification\n");
    if (verify(nblocks, bsize, 1, C) == 0)
        printf("Verification ok.\n");
    else
        printf("Verification failed.\n");
#endif

    printf("============= GEMM RESULTS =============\n" );
    printf("  Execution time (sec): %f\n", time_seconds);
    printf("  Performance (GFLOPS): %f\n", gflops);
    printf("  Memory alocated (GB): %.2lf\n", (double)(3 * n * n * sizeof(elem_t)) / (1024. * 1024. * 1024.));
    printf("========================================\n" );

    return 0;
}

