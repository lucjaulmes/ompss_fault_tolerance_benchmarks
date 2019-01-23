#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

// things that should be defined globally : constants, functions, etc.
#ifndef RECOMPUTE_GRADIENT_FREQ
#define RECOMPUTE_GRADIENT_FREQ 50
#endif

#ifdef UNUSED
#elif defined(__GNUC__)
	#define UNUSED __attribute__((unused))
#elif defined(__LCLINT__)
	#define UNUSED /*@unused@*/
#elif defined(__cplusplus)
	#define UNUSED
#endif

#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

// a few global vars for parameters that everyone needs to know
extern int nb_blocks;
extern int max_it;
extern int *block_bounds;

static inline void set_block_end(const int b, const int pos)
{
	block_bounds[b+1] = pos;
}

static inline int get_block_start(const int b)
{
	return block_bounds[b];
}

static inline int get_block_end(const int b)
{
	return block_bounds[b+1];
}

// shorthands so we can use the same function with and without MPI
static inline int world_block(const int b) {return b;}
static inline double *localrank_ptr(double *p) {return p;}

#endif // GLOBAL_H_INCLUDED

