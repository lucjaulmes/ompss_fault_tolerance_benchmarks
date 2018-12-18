/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    Stencil

PURPOSE: This program tests the efficiency with which a space-invariant,
         linear, symmetric filter (stencil) can be applied to a square
         grid or image.

USAGE:   The program takes as input the linear dimension of the grid,
		 and the number of iterations on the grid

               <progname> <iterations> <grid size>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than OpenMP or standard C functions, the following
         functions are used in this program:

         wtime()

HISTORY: - Written by Rob Van der Wijngaart, November 2006.
         - RvdW: Removed unrolling pragmas for clarity;
           added constant to array "in" at end of each iteration to force
           refreshing of neighbor data in parallel versions; August 2013

*******************************************************************/

#include <catchroi.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <err.h>


#define RESTRICT_KEYWORD 1
#define DOUBLE 1
#define STAR 1
#define RADIUS 2

#ifndef MIN
	#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
	#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#if DOUBLE
	#define DTYPE   double
	#define EPSILON 1.e-8
	#define COEFX   1.0
	#define COEFY   1.0
	#define FSTR    "%lf"
	#define ABS		fabs
#else
	#define DTYPE   float
	#define EPSILON 0.0001f
	#define COEFX   1.0f
	#define COEFY   1.0f
	#define FSTR    "%f"
	#define ABS		fabsf
#endif

/* define shorthand for indexing a multi-dimensional array                       */
#define IN(i,j)       in [((i) / tile_size) + ((j) / tile_size) * n_tiles][((i) % tile_size) + ((j) % tile_size) * tile_size]
#define OUT(i,j)      out[((i) / tile_size) + ((j) / tile_size) * n_tiles][((i) % tile_size) + ((j) % tile_size) * tile_size]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]

#define DO_PRAGMA(x) _Pragma (#x)
#define OMP_TASK(dependencies) DO_PRAGMA(omp task dependencies)


#if RESTRICT_KEYWORD
	#define RESTRICT restrict
#else
	#define RESTRICT
#endif


#include <sys/time.h>
#define USEC_TO_SEC 1.0e-6

double wtime()
{
	struct timeval time_data;
	gettimeofday(&time_data, NULL);
	return (double)time_data.tv_sec + (double)time_data.tv_usec * USEC_TO_SEC;
}


int main(int argc, char **argv)
{
	int    i, j, ii, jj, bi, bj, iter;
	double stencil_time;

	printf("Parallel Research Kernels version %s\n", "2.17 + OmpSs");
	printf("OmpSs stencil execution on 2D grid\n");

	/*******************************************************************************
	** process and test input parameters
	********************************************************************************/

	if (argc != 4)
		errx(EXIT_FAILURE, "Usage: %s <# iterations> <array dimension> <tile size>", argv[0]);

	const int iterations = atoi(argv[1]); // number of times to run the algorithm
	const long n         = atol(argv[2]); // linear grid dimension
	const long tile_size = atol(argv[3]);
	const long n_tiles   = n / tile_size;
	const long total_length = n * n;

	if (iterations < 1)
		errx(EXIT_FAILURE, "ERROR: iterations must be >= 1 : %d ", iterations);

	if (n < 1)
		errx(EXIT_FAILURE, "ERROR: grid dimension must be positive: %ld", n);

	if (RADIUS < 1)
		errx(EXIT_FAILURE, "ERROR: Stencil radius %d should be positive", RADIUS);

	if (2 * RADIUS + 1 > n)
		errx(EXIT_FAILURE, "ERROR: Stencil radius %d exceeds grid size %ld", RADIUS, n);


	DTYPE **RESTRICT in  = aligned_alloc(sysconf(_SC_PAGESIZE), n_tiles * n_tiles * sizeof(DTYPE));	/* input grid values                    */
	DTYPE **RESTRICT out = aligned_alloc(sysconf(_SC_PAGESIZE), n_tiles * n_tiles * sizeof(DTYPE));	/* output grid values                   */
	for (ii = 1; ii < n_tiles - 1; ii++)
		for (jj = 1; jj < n_tiles - 1; jj++)
		{
			in[ii * n_tiles + jj]  = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
			out[ii * n_tiles + jj] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
		}

	for (ii = 1; ii < n_tiles - 1; ii++)
	{
		in[ii * n_tiles + 0]             = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * (RADIUS + 1) * sizeof(DTYPE));
		in[ii * n_tiles + n_tiles - 1]   = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * (RADIUS + 1) * sizeof(DTYPE));
		in[ii]                           = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * (RADIUS + 1) * sizeof(DTYPE));
		in[ii + (n_tiles - 1) * n_tiles] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * (RADIUS + 1) * sizeof(DTYPE));
	}

	#if STAR
	in[0]                       = NULL;
	in[n_tiles - 1]             = NULL;
	in[n_tiles * (n_tiles - 1)] = NULL;
	in[n_tiles *  n_tiles - 1]  = NULL;
	#else
	in[0]                       = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), (RADIUS + 1) * (RADIUS + 1) * sizeof(DTYPE));
	in[n_tiles - 1]             = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), (RADIUS + 1) * (RADIUS + 1) * sizeof(DTYPE));
	in[n_tiles * (n_tiles - 1)] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), (RADIUS + 1) * (RADIUS + 1) * sizeof(DTYPE));
	in[n_tiles *  n_tiles - 1]  = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), (RADIUS + 1) * (RADIUS + 1) * sizeof(DTYPE));
	#endif
	DTYPE weight[2 * RADIUS + 1][2 * RADIUS + 1];										/* weights of points in the stencil     */

	if (!in || !out)
		err(EXIT_FAILURE, "ERROR: could not allocate space for input or output array: %ld", total_length * sizeof(DTYPE));

	/* fill the stencil weights to reflect a discrete divergence operator         */
	for (jj = -RADIUS; jj <= RADIUS; jj++)
		for (ii = -RADIUS; ii <= RADIUS; ii++)
			WEIGHT(ii, jj) = (DTYPE) 0.0;

	#if STAR
	int stencil_size = 4 * RADIUS + 1;
	for (ii = 1; ii <= RADIUS; ii++)
	{
		WEIGHT(0, ii) = WEIGHT(ii, 0) = (DTYPE)(1.0 / (2.0 * ii * RADIUS));
		WEIGHT(0, -ii) = WEIGHT(-ii, 0) = -(DTYPE)(1.0 / (2.0 * ii * RADIUS));
	}
	#else
	int stencil_size = (2 * RADIUS + 1) * (2 * RADIUS + 1);
	for (jj = 1; jj <= RADIUS; jj++)
	{
		for (ii = -jj + 1; ii < jj; ii++)
		{
			WEIGHT(ii, jj)  = (DTYPE)(1.0 / (4.0 * jj * (2.0 * jj - 1) * RADIUS));
			WEIGHT(ii, -jj) = -(DTYPE)(1.0 / (4.0 * jj * (2.0 * jj - 1) * RADIUS));
			WEIGHT(jj, ii)  = (DTYPE)(1.0 / (4.0 * jj * (2.0 * jj - 1) * RADIUS));
			WEIGHT(-jj, ii) = -(DTYPE)(1.0 / (4.0 * jj * (2.0 * jj - 1) * RADIUS));
		}
		WEIGHT(jj, jj)    = (DTYPE)(1.0 / (4.0 * jj * RADIUS));
		WEIGHT(-jj, -jj)  = -(DTYPE)(1.0 / (4.0 * jj * RADIUS));
	}
	#endif

	printf("Grid size            = %ld\n", n);
	printf("Radius of stencil    = %d\n", RADIUS);
	printf("Number of iterations = %d\n", iterations);
	#if STAR
	printf("Type of stencil      = star\n");
	#else
	printf("Type of stencil      = compact\n");
	#endif
	#if DOUBLE
	printf("Data type            = double precision\n");
	#else
	printf("Data type            = single precision\n");
	#endif
	#if RESTRICT_KEYWORD
	printf("No aliasing          = on\n");
	#else
	printf("No aliasing          = off\n");
	#endif
	printf("Compact representation of stencil loop body\n");
	printf("Parallel regions     = tasked (ompss tasks)\n");


	/* intialize the input and output arrays                                     */
	for (j = 0; j < n_tiles; j++)
		for (i = 0; i < n_tiles; i++)
		{
			const int bi_size = (i == 0 || i == (n_tiles - 1)) ? RADIUS + 1 : tile_size;
			const int bj_size = (j == 0 || j == (n_tiles - 1)) ? RADIUS + 1 : tile_size;
			const int ii_shift = (i == 0) ? tile_size - RADIUS - 1 : 0;
			const int jj_shift = (j == 0) ? tile_size - RADIUS - 1 : 0;
			DTYPE *in_b = in[i + j * n_tiles], add = (COEFX * i + COEFY * j) * tile_size;

			if (in_b == NULL) continue;

			#pragma omp task out([bj_size]in_b) private(ii, jj) firstprivate(i, j, add)
			for (jj = 0; jj < bj_size; jj++)
				for (ii = 0; ii < bi_size; ii++)
					in_b[jj * bi_size + ii] = add + COEFX * (ii + ii_shift) + COEFY * (jj + jj_shift);
		}

	for (j = 1; j < n_tiles - 1; j++)
		for (i = 1; i < n_tiles - 1; i++)
		{
			DTYPE *out_b = out[i + j * n_tiles];
			#pragma omp task out([tile_size * tile_size]out_b) private(ii, jj) firstprivate(i, j)
			for (jj = (j == 0 ? RADIUS : 0); jj < (j + 1 < n_tiles ? tile_size : tile_size - RADIUS); jj++)
				for (ii = (i == 0 ? RADIUS : 0); ii < (i + 1 < n_tiles ? tile_size : tile_size - RADIUS); ii++)
					out_b[ii + jj * tile_size] = 0.0;
		}

	#pragma omp taskwait

	stencil_time = wtime();
	start_roi();

	for (iter = 0; iter <= iterations; iter++)
	{
		for (bj = 1; bj < n_tiles - 1; bj ++)
			for (bi = 1; bi < n_tiles - 1; bi ++)
			{
				const int up_h  = (bj == 1)           ? (RADIUS + 1) : tile_size,
						down_h  = (bj == n_tiles - 2) ? (RADIUS + 1) : tile_size,
						left_w  = (bi == 1)           ? (RADIUS + 1) : tile_size,
						right_w = (bi == n_tiles - 2) ? (RADIUS + 1) : tile_size;

				/* from..to for all blocks that are neighbours */
				DTYPE (*o)[tile_size]       = (DTYPE (*)[tile_size])out[bi + n_tiles * bj];
				DTYPE (*in_b)[tile_size]    = (DTYPE (*)[tile_size])in[bi + n_tiles * bj];
				DTYPE (*in_up)[tile_size]   = (DTYPE (*)[tile_size])in[bi + n_tiles * (bj - 1)];
				DTYPE (*in_down)[tile_size] = (DTYPE (*)[tile_size])in[bi + n_tiles * (bj + 1)];
				DTYPE (*in_left) [left_w]   = (DTYPE (*)[left_w   ])in[(bi - 1) + n_tiles * bj];
				DTYPE (*in_right)[right_w]  = (DTYPE (*)[right_w  ])in[(bi + 1) + n_tiles * bj];

				OMP_TASK(inout([tile_size]o) in([tile_size]in_b, [up_h]in_up, [down_h]in_down, [tile_size]in_left, [tile_size]in_right) \
						firstprivate(bi, bj, tile_size, n, up_h, down_h) private(i, j, ii, jj))
				{
					for (j = 0; j < RADIUS; j++)
						for (i = 0; i < tile_size; i++)
						{
							/* i + ii < 0 => ii < -i */
							for (ii = -RADIUS; ii < -i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_left[j][i + ii + tile_size];

							for (; ii <= RADIUS && ii < tile_size - i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							for (; ii <= RADIUS; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_right[j][i + ii - tile_size];

							/* j + jj < 0 */
							for (jj = -RADIUS; jj < -j; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_up[up_h + j + jj][i];

							for (jj = -j; jj < 0; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];

							for (jj = 1; jj <= RADIUS; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];
						}

					for (j = tile_size - RADIUS; j < tile_size; j++)
						for (i = 0; i < tile_size; i++)
						{
							for (ii = -RADIUS; ii < -i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_left[j][i + ii + tile_size];

							for (; ii <= RADIUS && ii < tile_size - i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							for (; ii <= RADIUS; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_right[j][i + ii - tile_size];

							for (jj = 1; jj < tile_size - j; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];

							/* j + jj >= tile_size */
							for (jj = tile_size - j ; jj <= RADIUS; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_down[j + jj - tile_size][i];

							for (jj = -RADIUS; jj < 0; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];
						}

					for (j = RADIUS; j < tile_size - RADIUS; j++)
						for (i = 0; i < RADIUS; i++)
						{
							for (jj = -RADIUS; jj <= RADIUS; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];

							/* i + ii < 0 */
							for (ii = -RADIUS; ii < -i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_left[j][tile_size + i + ii];

							for (ii = -i; ii < 0; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							for (ii = 1; ii <= RADIUS; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];
						}

					for (j = RADIUS; j < tile_size - RADIUS; j++)
						for (i = tile_size - RADIUS; i < tile_size; i++)
						{
							for (jj = -RADIUS; jj <= RADIUS; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];

							for (ii = -RADIUS; ii < 0; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							for (ii = 1; ii < tile_size - i; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							/* i + ii >= tile_size */
							for (ii = tile_size - i; ii <= RADIUS; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_right[j][i + ii - tile_size];
						}

					/* center */
					for (j = RADIUS; j < tile_size - RADIUS; j++)
						for (i = RADIUS; i < tile_size - RADIUS; i++)
						{
							for (jj = -RADIUS; jj <= RADIUS; jj++)
								o[j][i] += weight[RADIUS][jj + RADIUS] * in_b[j + jj][i];

							for (ii = -RADIUS; ii < 0; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];

							for (ii = 1; ii <= RADIUS; ii++)
								o[j][i] += weight[ii + RADIUS][RADIUS] * in_b[j][i + ii];
						}
				}
			}

		/* add constant to solution to force refresh of neighbor data, if any       */
		for (bj = 0; bj < n_tiles; bj++)
			for (bi = 0; bi < n_tiles; bi++)
			{
				const int j_size = (bj == 0 || bj == n_tiles - 1) ? RADIUS + 1 : tile_size;
				const int i_size = (bi == 0 || bi == n_tiles - 1) ? RADIUS + 1 : tile_size;
				DTYPE (*in_bij)[i_size] = (DTYPE(*)[i_size])in[bi + bj * n_tiles];

				if (in_bij == NULL) continue;

				OMP_TASK(inout([j_size]in_bij) firstprivate(tile_size) private(i, j))
				for (j = 0; j < j_size; j++)
					for (i = 0; i < i_size; i++)
						in_bij[j][i] += 1.0;
			}
	}

	#pragma omp taskwait
	stop_roi(iter);
	stencil_time = wtime() - stencil_time;


	DTYPE norm = 0.0;

	for (j = 1; j < n_tiles - 1; j++)
		for (i = 1; i < n_tiles - 1; i++)
		{
			DTYPE (*block)[tile_size] = (DTYPE (*)[tile_size])out[i + j * n_tiles];
			#pragma omp task reduction(+:norm) out([tile_size]block) private(ii, jj) firstprivate(i, j)
			for (jj = 0; jj < tile_size; jj++)
				for (ii = 0; ii < tile_size; ii++)
					norm += ABS(block[jj][ii]);
		}

	#pragma omp taskwait

	DTYPE f_active_points = (n - 2 * tile_size) * (n - 2 * tile_size);
	norm /= f_active_points;

	/*******************************************************************************
	** Analyze and output results.
	********************************************************************************/

	for (ii = 0; ii < n_tiles * n_tiles; ii++)
	{
		free(in[ii]);
		free(out[ii]);
	}
	free(out);
	free(in);

	/* verify correctness                                                            */
	DTYPE reference_norm = (DTYPE)(iterations + 1) * (COEFX + COEFY);
	if (ABS(norm - reference_norm) > EPSILON)
		printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n", norm, reference_norm);
	else
	{
		printf("Solution validates\n");
		#if VERBOSE
		printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n", reference_norm, norm);
		#endif
	}

	DTYPE flops = (DTYPE)(2 * stencil_size + 1) * f_active_points;
	DTYPE avgtime = stencil_time / iterations;
	printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n", 1.0E-06 * flops/avgtime, avgtime);

	return EXIT_SUCCESS;
}
