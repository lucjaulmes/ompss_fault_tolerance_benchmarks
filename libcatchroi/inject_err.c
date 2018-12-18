#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <err.h>
#include <errno.h>
#include <sys/time.h>
#include <execinfo.h>

#include "inject_err.h"


#ifndef MEASURE_CACHE_PRESENCE
#define MEASURE_CACHE_PRESENCE 0
#endif // MEASURE_CACHE_PRESENCE

typedef struct _seu
{
	volatile int64_t *pos;
	int64_t mask;
	double time;
	int n_bits;
	short flip, undo;
	_Atomic int inj;
	float min_protect;
#if MEASURE_CACHE_PRESENCE
	uint64_t miss_time, hit_time, test_time;
#endif // MEASURE_CACHE_PRESENCE
} seu_t;


static int inject_n_bits = 0, inject_region = -1, inject_undo = 0, inject_flip = 1;
static int64_t inject_put = 0, inject_page = -1;
static double inject_mtbf = 0.;

static seu_t *seu = NULL;
pthread_t injector_thread = {0};


char *token(char **next)
{
	char *ret = NULL;

	for (; **next; (*next)++)
		if (ret == NULL && !isspace(**next))
			ret = *next;
		else if (ret != NULL && isspace(**next))
		{
			**next = '\0';
			(*next)++;
			break;
		}

	return ret;
}


void inject_parse_env()
{
#define required_argument 1
#define no_argument 0
	static struct
	{
		const char *name;
		int has_arg, *flag, val;
	} long_options[] =
	{
		{"n_bits",  required_argument, NULL, 'n'},
		{"vector",  required_argument, NULL, 'v'},
		{"page",    required_argument, NULL, 'a'},
		{"mtbf",    required_argument, NULL, 'm'},
		{"seed",    required_argument, NULL, 's'},
		{"put",     required_argument, NULL, 'p'},
		{"undo",    no_argument,       NULL, 'u'},
	};

	char *argstr = getenv("INJECT");

	if (argstr == NULL)
		return;

	int seed = 0;

	// Home-made option parsing, like getopts except don't call on it, since it interferes with calling it from main.
	for (char *optarg = NULL, *optnext = argstr, *optstr = token(&optnext); optstr != NULL; optstr = token(&optnext), optarg = NULL)
	{
		int optlen = strlen(optstr);
		int tok_opts[optlen - 1], n_tok_opts = 0;

		// Parse a single long option
		if (optlen > 2 && optstr[0] == '-' && optstr[1] == '-')
		{
			for (size_t lo = 0; lo < sizeof(long_options) / sizeof(*long_options); lo++)
				if (strcmp(optstr + 2, long_options[lo].name) == 0)
				{
					*tok_opts = long_options[lo].val;
					n_tok_opts = 1;

					if (long_options[lo].has_arg == required_argument)
						optarg = token(&optnext);

					break;
				}
		}
		// Parse a short option(s)
		else if (optlen > 1 && optstr[0] == '-' && optstr[1] != '-')
			// loop on optstr chars, each is (potentially) a short option
			for (n_tok_opts = 0; n_tok_opts < optlen - 1; )
			{
				tok_opts[n_tok_opts] = 0;
				for (size_t lo = 0; lo < sizeof(long_options) / sizeof(*long_options); lo++)
					if ((int)optstr[1 + n_tok_opts] == long_options[lo].val)
					{
						tok_opts[n_tok_opts] = long_options[lo].val;
						if (long_options[lo].has_arg == required_argument)
						{
							// -o<val>
							if (optlen > n_tok_opts + 2)
								optarg = optstr + n_tok_opts + 2;
							// -o <val>
							else
								optarg = token(&optnext);
							// stop loop on short options
							optlen = n_tok_opts;
						}
						break;
					}

				// unrecognized option => break
				if (!tok_opts[n_tok_opts])
					break;
				else
					n_tok_opts++;
			}
		// unrecognized option or '--': break out of here
		else
			break;

		for (int opt_n = 0, opt = tok_opts[opt_n]; opt_n < n_tok_opts; opt = tok_opts[++opt_n])
		{
			// the usual switch() just as if we called opt = getopt(...), even sets optarg.
			// NB. options n and p should be mutually exclusive
			switch (opt)
			{
			case 'n':
				inject_n_bits = atoi(optarg);
				break;
			case 'v':
				inject_region = atoi(optarg);
				break;
			case 'a':
				inject_page   = strtoll(optarg, NULL, 0);
				break;
			case 'm':
				inject_mtbf   = atof(optarg);
				break;
			case 's':
				seed          = atoi(optarg);
				break;
			case 'u':
				inject_undo   = 1;
				break;
			case 'p':
				inject_flip   = 0;
				inject_put    = strtoll(optarg, NULL, 0);
			}
		}
	}

	if (seed == 0)
	{
		struct timeval t;
		gettimeofday(&t, NULL);
		seed = t.tv_sec + t.tv_usec;
	}
	srand(seed);
}


// from x in a uniform distribution between 0 and 1, get y according to weibull distribution
double weibull(const double lambda, const double k, const double x)
{
	double y, inv_k = 1 / k;
	y = - log1p(-x); // - log(1 - x)
	y = pow(y, inv_k);
	y *= lambda; // where lambda ~ mean time between faults

	return y;
}

// for x uniform between 0 and 1, return -lambda * log(1 - x)
double exponential(const double lambda, const double x)
{
	double y = - log1p(-x); // - log(1 - x)
	y *= lambda;

	return y;
}

inline int64_t pick_bits(const int n_bits)
{
	int64_t flip_bits = 0LL;

	/* Pick distinct bits in [0,63] to flip, simultaneously.
	 * Store them in sorted order, to remember which are already flipped. */
	int i, j, b, distinct_bits[n_bits];
	for (i = 0; i < n_bits; i++)
	{
		/* new bit to flip, as the b-th _unflipped_ bit */
		b = ((double)rand() / RAND_MAX) * (64 - i);

		for (j = i - 1; j >= 0; --j)
		{
			if (distinct_bits[j] >= b + j + 1)
				distinct_bits[j + 1] = distinct_bits[j];
			else
				break;
		}

		distinct_bits[j + 1] = b + j + 1;
		flip_bits ^= 1LL << (b + j + 1);
	}

	return flip_bits;
}


static inline
void sleep_ns(double ns)
{
	struct timespec next_sim_fault, remainder;

	next_sim_fault.tv_sec  = (long long)floor(ns / 1e9);
	next_sim_fault.tv_nsec = (long long)floor(ns - 1e9 * next_sim_fault.tv_sec);

	if (nanosleep(&next_sim_fault, &remainder) != 0)
		fprintf(stderr, "Nanosleep skipped %d.%09d of %d.%09d sleeping time\n",
		        (int)remainder.tv_sec, (int)remainder.tv_nsec, (int)next_sim_fault.tv_sec, (int)next_sim_fault.tv_nsec);
}


inline void
clflush(volatile void *p)
{
    asm volatile ("clflush (%0)" :: "r"(p));
}


inline uint64_t
rdtsc()
{
    unsigned long a, d;
    asm volatile ("cpuid; rdtsc" : "=a" (a), "=d" (d) :: "ebx", "ecx");
    return a | ((uint64_t)d << 32);
}


void* inject_err(void* ignore __attribute__((unused)))
{
	// default cancellability state + nanosleep is a cancellation point
	sleep_ns(seu->time);

#if MEASURE_CACHE_PRESENCE
	volatile int64_t get, *ptr = seu->pos;
	uint64_t time[5];

	// get access time
    time[0] = rdtsc();
	get = *ptr;
    time[1] = rdtsc();

	// flush out of the cache
    clflush((volatile void*)ptr);

	// successively get miss and hit time
    time[2] = rdtsc();
	get = *ptr;
    time[3] = rdtsc();
	get = *ptr;
    time[4] = rdtsc();

	if (seu->flip)
		*ptr ^= seu->mask;
	else
	{
		*ptr = seu->mask;
		seu->mask = (int64_t)get;
	}
	seu->inj++;

	seu->test_time = time[1] - time[0];
	seu->miss_time = time[3] - time[2];
	seu->hit_time  = time[4] - time[3];
#else  // MEASURE_CACHE_PRESENCE
	int64_t *ptr = (int64_t*)seu->pos;
	if (seu->flip)
		*ptr ^= seu->mask;
	else
	{
		int64_t get = *ptr;
		*ptr = seu->mask;
		seu->mask = get;
	}
	seu->inj++;
#endif // MEASURE_CACHE_PRESENCE

	return NULL;
}


void inject_start()
{
	if (seu == NULL)
		return;

	void *d = &seu->mask;
	printf("inject_flip:%d inject_mask:%#016lx inject_dbl:%g inject_addr:%p inject_back:%d inject_time:%.09fs\n",
			seu->flip, seu->mask, *(double*)d, (void*)seu->pos, seu->undo, seu->time / 1e9);
	fflush(stdout);

	if (!pthread_create(&injector_thread, NULL, &inject_err, (void*)seu) == 0)
		err(errno, "Failed to create the injector thread");
}


void inject_stop()
{
	if (seu == NULL)
		return;

	/* Print whether we flipped anything or whether the inject region stopped earlier */
#if MEASURE_CACHE_PRESENCE
	printf("inject_done:%d time_hit:%ld time_miss:%ld time_test:%ld\n", seu->inj, seu->hit_time, seu->miss_time, seu->test_time);
#else
	printf("inject_done:%d\n", seu->inj);
#endif // MEASURE_CACHE_PRESENCE

	pthread_join(injector_thread, NULL);

	if (seu->inj > 0 && seu->undo)
	{
		if (seu->flip)
			*seu->pos ^= seu->mask;
		else
		{
			int64_t tmp = *seu->pos;
			*seu->pos = seu->mask;
			seu->mask = tmp;
		}
	}

	free(seu);
	fflush(stdout);
}


void register_target_region(int id, void *target_ptr, size_t target_size)
{
	// WARNING: hardcoded 4K page size
	int64_t target_page_pos = inject_page * 4096;
	inject_page -= (target_size + 4095) / 4096;

	if ((target_page_pos < 0 && inject_region < 0)|| (target_page_pos >= 0 && inject_page >= 0) || (inject_region >= 0 && id != inject_region)
			|| (inject_n_bits <= 0 && inject_flip > 0) || inject_mtbf <= 0)
		return;

	seu = calloc(1, sizeof(*seu));

	double rand_pos = (double)rand() / RAND_MAX;
	size_t flip_word;

	if (inject_region >= 0)
		flip_word = rand_pos * target_size / sizeof(int64_t);
	else
	{
		size_t page_size = (int64_t)target_size >= target_page_pos + 4096 ? 4096 : target_size - target_page_pos;
		flip_word = (target_page_pos + page_size * rand_pos) / sizeof(int64_t);
	}

	seu->inj = 0;
	seu->pos = ((int64_t*)target_ptr) + flip_word;
	seu->time = inject_mtbf * (double)rand() / RAND_MAX;
	seu->mask = inject_flip ? pick_bits(inject_n_bits) : inject_put;
	seu->flip = inject_flip;
	seu->undo = inject_undo;
	seu->n_bits = inject_n_bits;
}
