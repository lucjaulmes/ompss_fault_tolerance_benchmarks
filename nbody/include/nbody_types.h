#ifndef nbody_types_h
#define nbody_types_h

#ifndef __OPENCL_VERSION__
	#include <stddef.h> //Header not avaiable in OCL
#endif


typedef struct
{
	float x; /* x   */
	float y; /* y   */
	float z; /* z   */
} coord_t;


typedef coord_t *__restrict__ const coord_ptr_t;
typedef   float *__restrict__ const float_ptr_t;

#endif /* #ifndef nbody_types_h */
