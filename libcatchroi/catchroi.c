#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>

#include "catchroi.h"

const int clockid = CLOCK_MONOTONIC_RAW;
uint64_t time_zero_ns = 0;

uint64_t getns()
{
    struct timespec get_time;
    clock_gettime(clockid, &get_time);

    return (uint64_t)(get_time.tv_sec * 1000000000 + get_time.tv_nsec) - time_zero_ns;
}

void __attribute__((constructor)) setup_time_zero()
{
	time_zero_ns = getns();
}


static struct timeval start_time, stop_time;

static enum {BEFORE_ROI = 0, DURING_ROI = 1, AFTER_ROI = 2} when = BEFORE_ROI;
static _Atomic int taskcount[3] = {0};


void start_roi()
{
	when = DURING_ROI;
	gettimeofday(&start_time, NULL);
}

void stop_roi(int it)
{
	gettimeofday(&stop_time, NULL);
	long long time = 1000000 * (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec);
	if (it >= 0)
		printf("solve_time:%lld iterations:%d\n", time, it);
	else
		printf("solve_time:%lld\n", time);

	when = AFTER_ROI;
}

// NB: must be thread-safe.
void task_started(int id)
{
	(void)id;
}

// NB: must be thread-safe (taskcounts are atomic)
void task_ended(int id)
{
	// ignore main
	if (id > 1)
		taskcount[when]++;
}

int roi_progress()
{
	return taskcount[DURING_ROI];
}

void __attribute__((destructor)) report()
{
	if (taskcount[BEFORE_ROI] + taskcount[DURING_ROI] + taskcount[AFTER_ROI]) {
		printf("tasks executed before/during/after ROI: %d, %d, %d\n", taskcount[BEFORE_ROI], taskcount[DURING_ROI], taskcount[AFTER_ROI]);
	} else {
		printf("no tasks reported from nanox instrumentation plugin.\n");
	}
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

mem_region_callback_t *register_target_region = NULL;
void register_mem_region_callback(mem_region_callback_t *callback)
{
	register_target_region = callback;
}

static int alloc_counter = 0;


void* CATCHROI_INSTRUMENT(malloc)(size_t size)
{
	void *ptr = malloc(size);
	if (register_target_region) register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(calloc)(size_t elt_count, size_t elt_size)
{
	void *ptr = calloc(elt_count, elt_size);
	if (register_target_region) register_target_region(alloc_counter, ptr, elt_count * elt_size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, elt_count * elt_size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(realloc)(void *ptr, size_t size)
{
	ptr = realloc(ptr, size);
	if (register_target_region) register_target_region(alloc_counter, ptr, size);
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
	if (register_target_region) register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

void* CATCHROI_INSTRUMENT(memalign)(size_t align, size_t size)
{
	void *ptr = memalign(align, size);
	if (register_target_region) register_target_region(alloc_counter, ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, size);
	return ptr;
}

int CATCHROI_INSTRUMENT(posix_memalign)(void **ptr, size_t align, size_t size)
{
	int ret = posix_memalign(ptr, align, size);
	if (register_target_region) register_target_region(alloc_counter, *ptr, size);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, *ptr, size);
	return ret;
}

void CATCHROI_INSTRUMENT(free)(void *ptr)
{
	free(ptr);
}


void* CATCHROI_INSTRUMENT(mmap)(void *start, size_t len, int prot, int mem_flags, int fd, off_t off)
{
	void *ptr = mmap(start, len, prot, mem_flags, fd, off);
	if (register_target_region) register_target_region(alloc_counter, ptr, len);
	printf("alloc_%d [%p;%ld]\n", alloc_counter++, ptr, len);
	return ptr;
}

int   CATCHROI_INSTRUMENT(munmap)(void *ptr, size_t free_size)
{
	return munmap(ptr, free_size);
}
