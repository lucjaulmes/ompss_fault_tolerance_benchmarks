#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/mman.h>

#include "inject_err.h"
#include "catchroi.h"

static struct timeval start_time, stop_time;


void start_roi()
{
	inject_start();
	gettimeofday(&start_time, NULL);
}

void stop_roi(int it)
{
	gettimeofday(&stop_time, NULL);
	inject_stop();
	long long time = 1000000 * (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec);
	if (it >= 0)
		printf("solve_time:%lld iterations:%d\n", time, it);
	else
		printf("solve_time:%lld\n", time);
}

// Have TaskSim automatically catch our beginning/end of ROI
void __parsec_roi_begin () __attribute__ ((noinline, alias ("start_roi")));
void __parsec_roi_end () __attribute__ ((noinline, alias ("stop_roi")));


#ifdef CATCHROI_OVERRIDE_NAMES

/* Undef the names of common memory functions, so we can access the normal names (only in this file) */
#undef malloc
#undef calloc
#undef realloc
#undef aligned_alloc
#undef memalign
#undef posix_memalign
#undef free

#undef mmap
#undef munmap

#endif // CATCHROI_OVERRIDE_NAMES

/* Define the implementations that will intercept calls with preloading,
 * simply calling prefixed functions with ALLOC_IN_LIB flag. */

static int alloc_counter = 0;

void* CATCHROI_INSTRUMENT(malloc)(size_t size)
{
	void *ptr = malloc(size);
	register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(calloc)(size_t elt_count, size_t elt_size)
{
	void *ptr = calloc(elt_count, elt_size);
	register_target_region(alloc_counter, ptr, elt_count * elt_size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, elt_count * elt_size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(realloc)(void *ptr, size_t size)
{
	ptr = realloc(ptr, size);
	register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(aligned_alloc)(size_t align, size_t size)
{
#if __GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ > 2)
	void *ptr = aligned_alloc(align, size);
#else
	void *ptr = NULL;
	posix_memalign(&ptr, align, size);
#endif
	register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(memalign)(size_t align, size_t size)
{
	void *ptr = memalign(align, size);
	register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

int   CATCHROI_INSTRUMENT(posix_memalign)(void **ptr, size_t align, size_t size)
{
	int ret = posix_memalign(ptr, align, size);
	register_target_region(alloc_counter, *ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, *ptr, size);
	return ret;
}

void  CATCHROI_INSTRUMENT(free)(void *ptr)
{
	free(ptr);
}


void* CATCHROI_INSTRUMENT(mmap)(void *start, size_t len, int prot, int mem_flags, int fd, off_t off)
{
	void *ptr = mmap(start, len, prot, mem_flags, fd, off);
	register_target_region(alloc_counter, ptr, len);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, len);
	return ptr;
}

int   CATCHROI_INSTRUMENT(munmap)(void *ptr, size_t free_size)
{
	return munmap(ptr, free_size);
}


