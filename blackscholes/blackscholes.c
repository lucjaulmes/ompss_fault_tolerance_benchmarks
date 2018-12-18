// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice
// Hall, John C. Hull,
//
// OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis and Iulian Brumar - Barcelona Supercomputing Center

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <err.h>

#include <catchroi.h>


#define BSIZE_UNIT 1024
#define NUM_RUNS 100


//Precision to use for calculations
#define fptype float


typedef struct OptionData_
{
	fptype *spot;       // spot price
	fptype *strike;     // strike price
	fptype *rate;       // risk-free interest rate
	fptype *divq;       // dividend rate
	fptype *volatility; // volatility
	fptype *time;       // time to maturity or option expiration in years (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
	char   *type;       // Option type.  "P"=PUT, "C"=CALL
	fptype *divs;       // dividend vals (not used in this test)
	fptype *DGrefval;   // DerivaGem Reference Value
} OptionData;


// Same as OptionData, just one of each
struct
{
	fptype spot, strike, rate, divq, volatility, time;
	char   type;
	fptype divs, DGrefval;
} data_init[] =
{
#include "optionData.txt"
};

//Total number of options in optionData.txt
#define MAX_OPTIONS (sizeof(data_init)/sizeof(*data_init))


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

fptype CNDF(fptype x)
{
	// Check for negative value of x
	int sign = x < 0.0;
	if (sign)
		x = -x;

	// Compute NPrimeX term common to both four & six decimal accuracy calcs
	fptype NPrimeofX_K2 = exp(-x * x / 2);
	fptype K2 = 1.0 / (1.0 + 0.2316419 * x);

	NPrimeofX_K2 *= inv_sqrt_2xPI * K2;

	fptype y   = 1.330274429;
	y = y * K2 - 1.821255978;
	y = y * K2 + 1.781477937;
	y = y * K2 - 0.356563782;
	y = y * K2 + 0.319381530;

	y *= NPrimeofX_K2;

	if (sign)
		return y;
	else
		return 1.0 - y;
}


static inline
fptype BlkSchlsEqEuroNoDiv(fptype spot, fptype strike, fptype riskFreeRate, fptype volatility, fptype time, char otype)
{
	fptype sqrtTime = sqrt(time);
	fptype logTerm = log(spot / strike);
	fptype powerTerm = (volatility * volatility) / 2;
	fptype futureValue = strike * exp(- riskFreeRate * time);

	fptype den = volatility * sqrtTime;
	fptype d1 = ((riskFreeRate + powerTerm) * time + logTerm) / den;
	fptype d2 = d1 - den;

	fptype Nofd1 = CNDF(d1);
	fptype Nofd2 = CNDF(d2);

	if (otype == 'C')
		return (spot * Nofd1) - (futureValue * Nofd2);
	else // 'P'
		return (futureValue * (1.0 - Nofd2)) - (spot * (1.0 - Nofd1));
}


int bs_calc(int numOptions, int bsize, OptionData *opt, fptype *prices)
{
	int i, numError = 0;
	(void)bsize;

	// Adding nogroup causes segfaults somehow.
	// They wouldn't make a lot of sense anyway.

	#pragma omp taskloop grainsize(bsize) in(opt->spot[i;bsize], opt->strike[i;bsize], opt->rate[i;bsize], opt->volatility[i;bsize], opt->time[i;bsize], opt->type[i;bsize]) \
			out(prices[i;bsize]) label(BlkSchlsEqEuroNoDiv)
	for (i = 0; i < numOptions; i++)
	{
		// Calling main function to calculate option value based on Black & Sholes's equation.
		prices[i] = BlkSchlsEqEuroNoDiv(opt->spot[i], opt->strike[i], opt->rate[i], opt->volatility[i], opt->time[i], opt->type[i]);
	}

	#ifdef ERR_CHK
	#pragma omp taskloop grainsize(bsize) in(prices[i;bsize]) reduction(+:numError) label(check)
	for (i = 0; i < numOptions; i ++)
	{
		fptype priceDelta = opt->DGrefval[i] - prices[i];
		int ok = fabs(priceDelta) < 1e-4;
		// NB: check for result OK, not for minimal error size. Otherwise "nan" passes.
		if (!ok)
		{
			printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n", i, prices[i], opt->DGrefval[i], priceDelta);
			numError++;
		}
	}
	#endif

	return numError;
}


int main(int argc, char **argv)
{
	// Handle parameters
	if (argc < 2 || argc > 4)
		errx(1, "Usage:  %s numOptions [blocksize] [runs]", argv[0]);

	int numOptions = atoi(argv[1]);

	int bsize = BSIZE_UNIT, runs = NUM_RUNS;
	if (argc > 2)
		bsize = atoi(argv[2]);
	if (argc > 3)
		runs = atoi(argv[3]);

	if (bsize > numOptions || numOptions < 0)
		errx(1, "ERROR: Block size %d larger than number of options %d. Please reduce the block size, or use larger data size.", bsize, numOptions);

	// Allocate space for data
#define LINESIZE 64
	OptionData options =
	{
		.spot       = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype)),
		.strike     = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype)),
		.rate       = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype)),
		.divq       = NULL,
		.volatility = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype)),
		.time       = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype)),
		.type       = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(char)),
		.divs       = NULL,
		.DGrefval   = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype))
	};
	fptype *prices  = CATCHROI_INSTRUMENT(aligned_alloc)(LINESIZE, numOptions * sizeof(fptype));

	// Read input data from file without the files.
	for (int i = 0, j = 0; i < numOptions; i++, j++)
	{
		if (j == MAX_OPTIONS)
			j = 0;

		options.spot[i]       = data_init[j].spot;
		options.strike[i]     = data_init[j].strike;
		options.rate[i]       = data_init[j].rate;
		options.volatility[i] = data_init[j].volatility;
		options.time[i]       = data_init[j].time;
		options.type[i]       = data_init[j].type;
		options.DGrefval[i]   = data_init[j].DGrefval;
	}

	printf("Num of Options: %d\n", numOptions);
	printf("Num of Runs: %d\n", runs);
	printf("Size of data: %lu\n", numOptions * (7 * sizeof(fptype) + sizeof(char)));

	struct timeval start, stop;

	gettimeofday(&start, NULL);
	start_roi();

	// Do work
	int numError = 0;
	for (int i = 0; i < runs; i++)
		numError += bs_calc(numOptions, bsize, &options, prices);

	stop_roi(-1);
	gettimeofday(&stop, NULL);

	#ifdef ERR_CHK
	printf("Num Errors: %d\n", numError);
	#endif

	unsigned long elapsed = (stop.tv_sec - start.tv_sec) * 1000000  + stop.tv_usec - start.tv_usec;
	printf("par_sec_time_us:%lu\n", elapsed);

	free(options.spot);
    free(options.strike);
    free(options.rate);
    free(options.divq);
    free(options.volatility);
    free(options.time);
    free(options.type);
    free(options.DGrefval);
	free(prices);

	return 0;
}
