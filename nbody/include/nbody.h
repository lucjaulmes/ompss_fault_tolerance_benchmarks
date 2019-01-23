#ifndef nbody_h
#define nbody_h

#ifndef BIGO
	#define BIGO N
#endif

#include <stddef.h>
#include "nbody_types.h"

#define PART 1024

#define STR(s) #s
#define XSTR(s) STR(s)


#ifdef _OMPSS
# ifndef TASKLOOP
#  define TASKLOOP 1
# endif
extern int GS;
#endif

// Define macros to make the code cleaner: OMP_TASK(blah) is #pragma omp task blah
// with the macros inside blah expanded to their values.
#define EVAL_PRAGMA(x) _Pragma (#x)
#define DO_PRAGMA(x) EVAL_PRAGMA(x)

// round up(size / block) - 1 => (size + block - 1) / block - 1
#define ALL_BLOCKS(arr, size, block) {arr[j*block;block], j=0:(size - 1)/block}


typedef struct
{
	size_t input_size;
	size_t check_size;
	size_t position_offset;
	size_t mass_offset;
	char name[1000];
} nbody_file_t;

typedef struct
{
	float domain_size_x;
	float domain_size_y;
	float domain_size_z;
	float mass_maximum;
	float time_interval;
	int   seed;
	const char *name;
	const int timesteps;
	const int num_particles;
} nbody_conf_t;

typedef struct
{
	coord_ptr_t const positions;
	coord_ptr_t const velocities;
	coord_ptr_t const forces;
	float_ptr_t const masses;
	coord_ptr_t const remote_positions;
	float_ptr_t const remote_masses;
	const int num_particles;
	const int timesteps;
	nbody_file_t file;
} nbody_t;


/* common.c */
#if USE_MPI
void setup_mpi(int argc, char *argv[]);
static inline int get_commsize() { extern int commsize; return commsize; }
#else
#define get_commsize() 1
#endif
extern int rank;

nbody_t nbody_setup(const nbody_conf_t *const conf);
void nbody_save_particles(const nbody_t *nbody);
void nbody_free(const nbody_t *nbody);
void nbody_check(const nbody_t *nbody);

double wall_time();

void print_stats(double n_blocks, int timesteps, double elapsed_time);

#if USE_MPI
	void exchange_particles(void *sendbuf, void *recvbuf, int size);
#endif

void solve_nbody(coord_ptr_t local_positions, coord_ptr_t velocities, coord_ptr_t forces, float_ptr_t local_masses, coord_ptr_t remote_positions, float_ptr_t remote_masses,
                 const int n_blocks, const int timesteps, const float time_interval);

void update_particles(coord_ptr_t positions, coord_ptr_t velocities, coord_ptr_t forces, float_ptr_t masses, const int n_blocks, const float time_interval);

void calculate_local_forces(coord_ptr_t forces, coord_ptr_t positions, float_ptr_t masses, int bs);

void calculate_forces(coord_ptr_t forces, coord_ptr_t positions1, float_ptr_t masses1, coord_ptr_t positions2, float_ptr_t masses2, int bs);


#endif /* #ifndef nbody_h */
