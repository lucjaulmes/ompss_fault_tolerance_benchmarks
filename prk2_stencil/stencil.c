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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <err.h>


#define DOUBLE 1
#define STAR 1
#define RADIUS 2

#if DOUBLE
	#define DTYPE   double
	#define EPSILON 1.e-8
	#define COEFY   1.0
	#define COEFX   1.0
	#define FSTR    "%lf"
	#define ABS		fabs
#else
	#define DTYPE   float
	#define EPSILON 0.0001f
	#define COEFY   1.0f
	#define COEFX   1.0f
	#define FSTR    "%f"
	#define ABS		fabsf
#endif


#define DO_PRAGMA(x) _Pragma (#x)
#define OMP_TASK(dependencies) DO_PRAGMA(omp task dependencies)


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


	DTYPE *(*in )[n_tiles] = calloc(n_tiles * n_tiles, sizeof(DTYPE*));
	DTYPE *(*out)[n_tiles] = calloc(n_tiles * n_tiles, sizeof(DTYPE*));
	for (int bi = 1; bi < n_tiles - 1; bi++)
		for (int bj = 1; bj < n_tiles - 1; bj++)
		{
			in[bi][bj]  = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
			out[bi][bj] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
		}

	for (int b = 1; b < n_tiles - 1; b++)
	{
		in[b][0]           = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * RADIUS * sizeof(DTYPE));
		in[b][n_tiles - 1] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * RADIUS * sizeof(DTYPE));
		in[0][b]           = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * RADIUS * sizeof(DTYPE));
		in[n_tiles - 1][b] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * RADIUS * sizeof(DTYPE));
	}

	#if STAR
	in[0][0]                     = NULL;
	in[0][n_tiles - 1]           = NULL;
	in[n_tiles - 1][0]           = NULL;
	in[n_tiles - 1][n_tiles - 1] = NULL;
	#else
	in[0][0]                     = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
	in[0][n_tiles - 1]           = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
	in[n_tiles - 1][0]           = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
	in[n_tiles - 1][n_tiles - 1] = CATCHROI_INSTRUMENT(aligned_alloc)(sysconf(_SC_PAGESIZE), tile_size * tile_size * sizeof(DTYPE));
	#endif

	if (!in || !out)
		err(EXIT_FAILURE, "ERROR: could not allocate space for input or output array: %ld", total_length * sizeof(DTYPE));

	DTYPE weight_arr[2 * RADIUS + 1][2 * RADIUS + 1] = {{0.}};
	// get a pointer to the middle element so we can access with values from -RADIUS to +RADIUS
	DTYPE (*weight)[2 * RADIUS + 1] = (DTYPE (*)[2 * RADIUS + 1])(weight_arr[RADIUS] + RADIUS);

	#if STAR
	int stencil_size = 4 * RADIUS + 1;
	for (int jj = 1; jj <= RADIUS; jj++)
	{
		weight[0][ jj] = weight[ jj][0] =  1. / (2 * jj * RADIUS);
		weight[0][-jj] = weight[-jj][0] = -1. / (2 * jj * RADIUS);
	}
	#else
	int stencil_size = (2 * RADIUS + 1) * (2 * RADIUS + 1);
	for (int ii = 1; ii <= RADIUS; ii++)
	{
		for (int jj = -ii + 1; jj < ii; jj++)
		{
			weight[ jj][ ii] =  1. / (4 * ii * (2 * ii - 1) * RADIUS);
			weight[ jj][-ii] = -1. / (4 * ii * (2 * ii - 1) * RADIUS);
			weight[ ii][ jj] =  1. / (4 * ii * (2 * ii - 1) * RADIUS);
			weight[-ii][ jj] = -1. / (4 * ii * (2 * ii - 1) * RADIUS);
		}
		weight[ ii][ ii] =  1. / (4 * ii * RADIUS);
		weight[-ii][-ii] = -1. / (4 * ii * RADIUS);
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
	printf("Compact representation of stencil loop body\n");
	printf("Parallel regions     = tasked (ompss tasks)\n");


	/* intialize the input and output arrays                                     */
	for (int bi = 0; bi < n_tiles; bi++)
		for (int bj = 0; bj < n_tiles; bj++)
		{
			const int tile_height = (bi == 0 || bi == n_tiles - 1) ? RADIUS : tile_size;
			const int tile_width  = (bj == 0 || bj == n_tiles - 1) ? RADIUS : tile_size;

			DTYPE (*block)[tile_width] = (DTYPE (*)[tile_width])in[bi][bj];
			if (!block) continue;

			DTYPE add = (COEFX * bi + COEFY * bj) * tile_size;
			if (bi == 0) add += (tile_size - tile_height) * COEFX;
			if (bj == 0) add += (tile_size - tile_width ) * COEFY;

			#pragma omp task out([tile_size * tile_size]block) firstprivate(add)
			for (int ii = 0; ii < tile_height; ii++)
				for (int jj = 0; jj < tile_width; jj++)
					block[ii][jj] = add + COEFX * ii + COEFY * jj;
		}

	for (int bi = 1; bi < n_tiles - 1; bi++)
		for (int bj = 1; bj < n_tiles - 1; bj++)
		{
			#pragma omp task out([tile_size * tile_size](out[bj][bi])) firstprivate(bj, bi)
			memset(out[bj][bi], 0, tile_size * tile_size * sizeof(DTYPE));
		}

	#pragma omp taskwait

	double stencil_time = wtime();
	int iter;
	start_roi();

	for (iter = 0; iter <= iterations; iter++)
	{
		for (int bi = 1; bi < n_tiles - 1; bi ++)
			for (int bj = 1; bj < n_tiles - 1; bj ++)
			{
				const int height_up   = bi == 1           ? RADIUS : tile_size;
				const int height_down = bi == n_tiles - 2 ? RADIUS : tile_size;
				const int width_left  = bj == 1           ? RADIUS : tile_size;
				const int width_right = bj == n_tiles - 2 ? RADIUS : tile_size;

				/* from..to for all blocks that are neighbours */
				DTYPE (*o		)[tile_size] = (DTYPE (*)[tile_size])out[bi][bj];
				DTYPE (*in_b	)[tile_size] = (DTYPE (*)[tile_size])in[bi][bj];
				DTYPE (*in_up   )[tile_size] = (DTYPE (*)[tile_size])in[bi - 1][bj];
				DTYPE (*in_down )[tile_size] = (DTYPE (*)[tile_size])in[bi + 1][bj];
				DTYPE (*in_left )[width_left ] = (DTYPE (*)[width_left ])in[bi][bj - 1];
				DTYPE (*in_right)[width_right] = (DTYPE (*)[width_right])in[bi][bj + 1];

				OMP_TASK(inout([tile_size]o) in([tile_size]in_b, [height_up]in_up, \
							[height_down]in_down, [tile_size]in_left, [tile_size]in_right))
				{
					for (int ii, i = 0; i < RADIUS; i++)
						for (int jj, j = 0; j < tile_size; j++)
						{
							/* j + jj < 0 => jj < -j */
							for (jj = -RADIUS; jj < -j; jj++)
								o[i][j] += weight[jj + 0][0] * in_left[i][j + jj + width_left];

							for (; jj <= RADIUS && jj < tile_size - j; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							for (; jj <= RADIUS; jj++)
								o[i][j] += weight[jj + 0][0] * in_right[i][j + jj - tile_size];

							/* i + ii < 0 */
							for (ii = -RADIUS; ii < -i; ii++)
								o[i][j] += weight[0][ii + 0] * in_up[i + ii + height_up][j];

							for (ii = -i; ii < 0; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];

							for (ii = 1; ii <= RADIUS; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];
						}

					for (int ii, i = tile_size - RADIUS; i < tile_size; i++)
						for (int jj, j = 0; j < tile_size; j++)
						{
							for (jj = -RADIUS; jj < -j; jj++)
								o[i][j] += weight[jj + 0][0] * in_left[i][j + jj + width_left];

							for (; jj <= RADIUS && jj < tile_size - j; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							for (; jj <= RADIUS; jj++)
								o[i][j] += weight[jj + 0][0] * in_right[i][j + jj - tile_size];

							for (ii = 1; ii < tile_size - i; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];

							/* i + ii >= tile_size */
							for (ii = tile_size - i ; ii <= RADIUS; ii++)
								o[i][j] += weight[0][ii + 0] * in_down[i + ii - tile_size][j];

							for (ii = -RADIUS; ii < 0; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];
						}

					for (int ii, i = RADIUS; i < tile_size - RADIUS; i++)
						for (int jj, j = 0; j < RADIUS; j++)
						{
							for (ii = -RADIUS; ii <= RADIUS; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];

							/* j + jj < 0 */
							for (jj = -RADIUS; jj < -j; jj++)
								o[i][j] += weight[jj + 0][0] * in_left[i][j + jj + width_left];

							for (jj = -j; jj < 0; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							for (jj = 1; jj <= RADIUS; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];
						}

					for (int ii, i = RADIUS; i < tile_size - RADIUS; i++)
						for (int jj, j = tile_size - RADIUS; j < tile_size; j++)
						{
							for (ii = -RADIUS; ii <= RADIUS; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];

							for (jj = -RADIUS; jj < 0; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							for (jj = 1; jj < tile_size - j; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							/* j + jj >= tile_size */
							for (jj = tile_size - j; jj <= RADIUS; jj++)
								o[i][j] += weight[jj + 0][0] * in_right[i][j + jj - tile_size];
						}

					/* center */
					for (int ii, i = RADIUS; i < tile_size - RADIUS; i++)
						for (int jj, j = RADIUS; j < tile_size - RADIUS; j++)
						{
							for (ii = -RADIUS; ii <= RADIUS; ii++)
								o[i][j] += weight[0][ii + 0] * in_b[i + ii][j];

							for (jj = -RADIUS; jj < 0; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];

							for (jj = 1; jj <= RADIUS; jj++)
								o[i][j] += weight[jj + 0][0] * in_b[i][j + jj];
						}
				}
			}

		/* add constant to solution to force refresh of neighbor data, if any       */
		for (int bi = 0; bi < n_tiles; bi++)
			for (int bj = 0; bj < n_tiles; bj++)
			{
				const int tile_height = (bi == 0 || bi == n_tiles - 1) ? RADIUS : tile_size;
				const int tile_width  = (bj == 0 || bj == n_tiles - 1) ? RADIUS : tile_size;

				DTYPE (*block)[tile_width] = (DTYPE (*)[tile_width])in[bi][bj];
				if (!block) continue;

				OMP_TASK(inout([tile_height]block) firstprivate(tile_size))
				for (int i = 0; i < tile_height; i++)
					for (int j = 0; j < tile_width; j++)
						block[i][j] += 1.0;
			}
	}

	#pragma omp taskwait
	stop_roi(iter);
	stencil_time = wtime() - stencil_time;


	DTYPE norm = 0.0;

	for (int bi = 1; bi < n_tiles - 1; bi++)
		for (int bj = 1; bj < n_tiles - 1; bj++)
		{
			DTYPE *block = out[bi][bj];
			#pragma omp task reduction(+:norm) out([tile_size * tile_size]block)
			for (int ii = 0; ii < tile_size; ii++)
				for (int jj = 0; jj < tile_size; jj++)
					norm += ABS(block[jj + ii * tile_size]);
		}

	#pragma omp taskwait

	DTYPE f_active_points = (n - 2 * tile_size) * (n - 2 * tile_size);
	norm /= f_active_points;

	/*******************************************************************************
	** Analyze and output results.
	********************************************************************************/

	for (int bi = 0; bi < n_tiles; bi++)
		for (int bj = 0; bj < n_tiles; bj++)
		{
			if (in[bi][bj])  free(in[bi][bj]);
			if (out[bi][bj]) free(out[bi][bj]);
		}

	free(out);
	free(in);

	/* verify correctness                                                            */
	DTYPE reference_norm = (DTYPE)(iterations + 1) * (COEFY + COEFX);
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
