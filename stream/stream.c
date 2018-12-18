/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Adapted to StarSs by Rosa M. Badia (Barcelona Supercomputing Center)  */
/* This version does not insert barriers after each set of operations,   */
/* to promote task chaining in StarSs                                    */
/* Revision: $Id: stream.c,v 5.8 2007/02/19 23:57:39 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#include <err.h>
#include <sys/time.h>

#include <catchroi.h>


/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */
double mysecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}


int checktick()
{
	#define    M   20

	/*  Collect a sequence of M unique time values from the system. */
	double timesfound[M], t1 = mysecond();
	for (int i = 0; i < M; i++)
	{
		double t2;
		do
		{
			t2 = mysecond();
		}
		while (t2 - t1 < 1e-6);
		timesfound[i] = t1 = t2;
	}

	/*
	 * Determine the minimum difference between these M values.
	 * This result will be our estimate (in microseconds) for the
	 * clock granularity.
	 */

	int minDelta = 1000000;
	for (int i = 1; i < M; i++)
	{
		int delta = (int)(1e6 * (timesfound[i] - timesfound[i - 1]));
		if (delta > 0 && minDelta > delta)
			minDelta = delta;
	}

	return minDelta;
}



#define SMP

/** INCLUDE KERNELS HERE **/
#ifdef SMP
	#include "kernels/smp.c"
#else
	#include "kernels/cuda.h"
#endif

/* INSTRUCTIONS:
 *
 *  1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */


size_t N = ((12ULL*1024*1024*1024) / (3*8)), BSIZE;


//#if defined (INPUT_SET_LARGE)
//#elif defined (INPUT_SET_MEDIUM) // 1.125 G
//	#define N  ((30ULL*1024*1024*1024) / (3*8))
//#else // DEFAULT = SMALL // 512 M
//	#define N ((12ULL*1024*1024*1024) / (3*8))
//#endif

//# define N    2000000
//# define N    2*512*1024*1024UL
//# define N    512*1024*1024UL

# define NTIMES 10
# define OFFSET 0

// Definitions of BSIZE for OmpSs
//#define BSIZE ((N)/(128*16))
//#define BSIZE ((N)/(128*2))
//#define BSIZE ((N)/(8))
//#define BSIZE (_bsize)
// orig
//#define BSIZE ((N)/128)


/*
 *  3) Compile the code with full optimization.  Many compilers
 *     generate unreasonably bad code before the optimizer tightens
 *     things up.  If the results are unreasonably good, on the
 *     other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *  4) Mail the results to mccalpin@cs.virginia.edu
 *     Be sure to include:
 *      a) computer hardware model number and software revision
 *      b) the compiler flags
 *      c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"


static double **a = NULL, **b = NULL, **c = NULL;
static int nodes = 0;



void allocations()
{
	a = malloc(nodes * sizeof(*a));
	b = malloc(nodes * sizeof(*b));
	c = malloc(nodes * sizeof(*c));

	for (int i = 0; i < nodes; i++)
	{
		a[i] = CATCHROI_INSTRUMENT(malloc)(sizeof(double[N + OFFSET]));
		b[i] = CATCHROI_INSTRUMENT(malloc)(sizeof(double[N + OFFSET]));
		c[i] = CATCHROI_INSTRUMENT(malloc)(sizeof(double[N + OFFSET]));
		fprintf(stderr, "node %d: a=%p b=%p c=%p, using regular malloc\n", i, a[i], b[i], c[i]);

		if (!a[i] || !b[i] || !c[i]) {
			err(1, "error allocating");
		}
	}
}

void tuned_initialization()
{
	for (int i = 0; i < nodes; i++)
		for (unsigned long j = 0; j < N; j += BSIZE)
		{
			unsigned long this_bs = (j + BSIZE > N) ? N - j : BSIZE;
			init_kernel(a[i], b[i], c[i], this_bs, j);
		}
}


void copy_task()
{
	for (int i = 0; i < nodes; i++)
		//#pragma omp task in([N](a[i])) out([N](c[i])) label(copy)
		for (unsigned j = 0; j < N; j += BSIZE)
		{
			unsigned long this_bs = (j + BSIZE > N) ? N - j : BSIZE;
			copy_kernel(a[i], c[i], this_bs, j);
		}
}


void scale_task(double scalar)
{
	for (int i = 0; i < nodes; i++)
		//#pragma omp task in([N](c[i])) out([N](b[i])) label(scale)
		for (unsigned long j = 0; j < N; j += BSIZE)
		{
			unsigned long this_bs = (j + BSIZE > N) ? N - j : BSIZE;
			scale_kernel(b[i], c[i], scalar, this_bs, j);
		}
}


void add_task()
{
	for (int i = 0; i < nodes; i++)
		//#pragma omp task in([N](a[i]), [N](b[i])) out([N](c[i])) label(add)
		for (unsigned long j = 0; j < N; j += BSIZE)
		{
			unsigned long this_bs = (j + BSIZE > N) ? N - j : BSIZE;
			add_kernel(a[i], b[i], c[i], this_bs, j);
		}
}


void triad_task(double scalar)
{
	for (int i = 0; i < nodes; i++)
		//#pragma omp task in([N](b[i]), [N](c[i])) out([N](a[i])) label(triad)
		for (unsigned long j = 0; j < N; j += BSIZE)
		{
			unsigned long this_bs = (j + BSIZE > N) ? N - j : BSIZE;
			triad_kernel(a[i], b[i], c[i], scalar, this_bs, j);
		}
}


void checkSTREAMresults()
{
	/* this works with one node only */
	/* reproduce initialization */
	double aj = 1.0, bj = 2.0, cj = 0.0;
	/* a[] is modified during timing check */
	aj *= 2.;

	/* now execute timing loop */
	double scalar = 3.0;
	for (int k = 0; k < NTIMES; k++)
	{
		cj = aj;
		bj = scalar * cj;
		cj = aj + bj;
		aj = bj + scalar * cj;
	}
	aj *= (double)N;
	bj *= (double)N;
	cj *= (double)N;

	double asum = 0.0, bsum = 0.0, csum = 0.0;
	for (size_t j = 0; j < N; j++)
	{
		asum += a[0][j];
		bsum += b[0][j];
		csum += c[0][j];
	}
	#ifdef VERBOSE
	printf("Results Comparison: \n");
	printf("        Expected  : %f %f %f \n", aj, bj, cj);
	printf("        Observed  : %f %f %f \n", asum, bsum, csum);
	#endif

	double epsilon = 1.e-8;

	if (!(fabs(aj / asum - 1) <= epsilon))
	{
		printf("Failed Validation on array a[]\n");
		printf("        Expected  : %f \n", aj);
		printf("        Observed  : %f \n", asum);
	}
	else if (!(fabs(bj - bsum) / bsum <= epsilon))
	{
		printf("Failed Validation on array b[]\n");
		printf("        Expected  : %f \n", bj);
		printf("        Observed  : %f \n", bsum);
	}
	else if (!(fabs(cj - csum) / csum <= epsilon))
	{
		printf("Failed Validation on array c[]\n");
		printf("        Expected  : %f \n", cj);
		printf("        Observed  : %f \n", csum);
	}
	else
		printf("Solution Validates\n");
}


int main(int argc, char *argv[])
{
	checktick();

	nodes = 1;
	if (argc == 2) {
		N = strtoull(argv[1], NULL, 0);
	}
	BSIZE = N / 128;


#ifdef _OMPSS
	printf("threads is %d, block size %lu\n", omp_get_num_threads(), BSIZE);
#else
	printf("threads is %d, block size %lu\n", 1, BSIZE);
#endif

	/* --- SETUP --- determine precision and check timing --- */

	size_t BytesPerWord = sizeof(double);
	printf(HLINE);
	printf("STREAM version $Revision: 5.8 $\n");
	printf(HLINE);
	printf("This system uses %lu bytes per DOUBLE PRECISION word.\n", BytesPerWord);

	printf(HLINE);
	printf("Array size = %lu, Offset = %d\n", N, OFFSET);
	printf("Total memory required = %.1f MB.\n", (3.0 * BytesPerWord) * ((double)(N * nodes) / 1048576.0));
	printf("Each test is run %d times, but only\n", NTIMES);
	printf("the *best* time for each is used.\n");

	printf(HLINE);
	printf("Printing one line per active node...\n");
	allocations();

	/* Get initial value for system clock. */
	double total_time = mysecond();
	start_roi();

	tuned_initialization();

	#pragma omp taskwait noflush

	/*   --- MAIN LOOP --- repeat test cases NTIMES times ---
	 *   NB: use inout() dependencies to serialize the type of tasks */

	double scalar = 3.0, times[5][NTIMES] = {0};
	times[0][0] = mysecond();

	for (int k = 0; k < NTIMES; k++)
	{
		// c = a
		copy_task();

		#pragma omp task inout({c[n][i * BSIZE;BSIZE], n=0:nodes-1, i=0:N/BSIZE-1}) out(times[1][k]) label(timing_copy)
		times[1][k] = mysecond();

		// b = scalar * c
		scale_task(scalar);

		#pragma omp task inout({b[n][i * BSIZE;BSIZE], n=0:nodes-1, i=0:N/BSIZE-1}) out(times[2][k]) label(timing_scale)
		times[2][k] = mysecond();

		// c = a + b
		add_task();

		#pragma omp task inout({c[n][i * BSIZE;BSIZE], n=0:nodes-1, i=0:N/BSIZE-1}) out(times[3][k]) label(timing_add)
		times[3][k] = mysecond();

		// a = b + scalar * c
		triad_task(scalar);

		#pragma omp task inout({a[n][i * BSIZE;BSIZE], n=0:nodes-1, i=0:N/BSIZE-1}) out(times[4][k], times[0][k+1]) label(timing_triad)
		{
			times[4][k] = mysecond();
			if ((k + 1) < NTIMES)
				times[0][k + 1] = mysecond();
		}
	}

	#pragma omp taskwait

	total_time = mysecond() - total_time;
	stop_roi(-1);

	/*   --- SUMMARY --- */

	double avgtime[4] = {0}, maxtime[4] = {0}, mintime[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
	for (int k = 1; k < NTIMES; k++) /* note -- skip first iteration, as a warmup */
	{
		for (int j = 0; j < 4; j++)
		{
			double t = times[j + 1][k] - times[j][k];
			avgtime[j] += t;

			if (mintime[j] > t)
				mintime[j] = t;

			if (maxtime[j] < t)
				maxtime[j] = t;
		}
	}

	// labels and sizes of each operation
	char *label[4] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};
	double   bytes[4] = {2 * sizeof(double) *N, 2 * sizeof(double) *N, 3 * sizeof(double) *N, 3 * sizeof(double) *N};

	printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
	for (int j = 0; j < 4; j++)
	{
		avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

		printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
		    (1e-6 * nodes * bytes[j]) / mintime[j],
		    avgtime[j], mintime[j], maxtime[j]);
	}

	printf(HLINE);

	printf("TOTAL time (including initialization) =  %11.4f seconds\n", total_time);

	printf(HLINE);

	printf("========== STREAM TRIAD RESULTS =========\n");
	printf("  Execution time (sec): %f\n", mintime[3]);
	printf("  Performance (GFLOPS): %f\n", (1e-9 * nodes * bytes[3]) / mintime[3]);
	printf("  Memory alocated (GB): %.2lf\n", (3.0 * BytesPerWord) * ((double)(N * nodes) / (1024.0 * 1024.0 * 1024.0)));
	printf("=========================================\n");

	/* --- Check Results --- */
	#if VALIDATE
	checkSTREAMresults();
	#endif

	printf(HLINE);

	return 0;
}
