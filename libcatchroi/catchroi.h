#ifndef CATCHROI_H
#define CATCHROI_H

#ifdef __cplusplus
extern "C" {
#endif

void start_roi();
void stop_roi(int it);


#ifndef DISABLE_MEMORY_INSTRUMENTATION


#define PASTE_PREFIX(a, b) a ## b
#define CATCHROI_INSTRUMENT(f) PASTE_PREFIX(_libcatchroi_, f)

/* Define a set of functions that replace memory management functions, and take an extra argument:
 * a flag to identify whether the call comes from the library or the instrumented code. */
#include <stdlib.h>
void* CATCHROI_INSTRUMENT(malloc)(size_t);
void* CATCHROI_INSTRUMENT(calloc)(size_t, size_t);
void* CATCHROI_INSTRUMENT(realloc)(void*, size_t);
void* CATCHROI_INSTRUMENT(aligned_alloc)(size_t, size_t);
void* CATCHROI_INSTRUMENT(memalign)(size_t, size_t);
int   CATCHROI_INSTRUMENT(posix_memalign)(void**, size_t, size_t);
void  CATCHROI_INSTRUMENT(free)(void*);

#include <sys/mman.h>
void* CATCHROI_INSTRUMENT(mmap)(void*, size_t, int, int, int, off_t);
int   CATCHROI_INSTRUMENT(munmap)(void*, size_t);


#ifdef CATCHROI_OVERRIDE_NAMES
/* Redefine common names of memory functions, so that all functions after the include are intercepted automatically */

#define malloc(...)           CATCHROI_INSTRUMENT(malloc)(__VA_ARGS__)
#define calloc(...)           CATCHROI_INSTRUMENT(calloc)(__VA_ARGS__)
#define realloc(...)          CATCHROI_INSTRUMENT(realloc)(__VA_ARGS__)
#define aligned_alloc(...)    CATCHROI_INSTRUMENT(aligned_alloc)(__VA_ARGS__)
#define memalign(...)         CATCHROI_INSTRUMENT(memalign)(__VA_ARGS__)
#define posix_memalign(...)   CATCHROI_INSTRUMENT(posix_memalign)(__VA_ARGS__)
#define free(...)             CATCHROI_INSTRUMENT(free)(__VA_ARGS__)

#define mmap(...)             CATCHROI_INSTRUMENT(mmap)(__VA_ARGS__)
#define munmap(...)           CATCHROI_INSTRUMENT(munmap)(__VA_ARGS__)

#endif // CATCHROI_OVERRIDE_NAMES



#else  // DISABLE_MEMORY_INSTRUMENTATION
#define CATCHROI_INSTRUMENT(f) f
#endif



#ifdef __cplusplus
}
#endif


#endif // CATCHROI_H
