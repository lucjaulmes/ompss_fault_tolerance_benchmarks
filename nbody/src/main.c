/* nbody.c */

#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <string.h>
#include <unistd.h>

#if USE_MPI
	#include <mpi.h>
#endif

static const float default_domain_size_x = 1.0e+10; /* m  */
static const float default_domain_size_y = 1.0e+10; /* m  */
static const float default_domain_size_z = 1.0e+10; /* m  */
static const float default_mass_maximum  = 1.0e+28; /* kg */
static const float default_time_interval = 1.0e+0;  /* s  */
static const int   default_seed          = 12345;
static const char *default_name          = "data/nbody_soa";
static const int   default_num_particles = 16384;
static const int   default_timesteps     = 10;

int main(int argc, char **argv)
{
	int num_particles = default_num_particles, timesteps = default_timesteps, arg_read = 0;

	if (argc > arg_read + 1)
		num_particles = atoi(argv[++arg_read]);

	if (argc > arg_read + 1)
		timesteps = atoi(argv[++arg_read]);

	#if TASKLOOP
	int n_tasks = nanos_omp_get_num_threads();
	if (argc > arg_read + 1)
		n_tasks = atoi(argv[++arg_read]);
	#endif

	#if USE_MPI
	argv[arg_read] = argv[0];
	setup_mpi(argc - arg_read, argv + arg_read);
	#endif

	num_particles /= get_commsize();

	const int MIN_PARTICLES = 128;
	num_particles = ((num_particles - 1) | (MIN_PARTICLES - 1)) + 1;

	if (num_particles < MIN_PARTICLES || timesteps <= 0)
		errx(1, "Need %d >= %d particles per process, and %d > 0 timesteps\n", num_particles, MIN_PARTICLES, timesteps);
	#if TASKLOOP
	if (n_tasks <= 0)
		errx(1, "Need %d > 0 tasks per thread\n", n_tasks);

	GS = num_particles / n_tasks;
	if (num_particles != GS * n_tasks)
		errx(1, "The number of particles must be equally dividable among parallel blocks"
		     ": %d * %d != %d\n", n_tasks, GS, num_particles);

	if (rank == 0)
		printf("O(%s), processes: %d, particles: %d, particles in process: %d, tasks per loop: %d, timesteps: %d\n", XSTR(BIGO), get_commsize(), num_particles * get_commsize(), num_particles, n_tasks, timesteps);
	#else
	if (rank == 0)
		printf("O(%s), processes: %d, particles: %d, particles in process: %d, timesteps: %d\n", XSTR(BIGO), get_commsize(), num_particles * get_commsize(), num_particles, timesteps);
	#endif

	#if USE_MPI
	char node_name[MPI_MAX_PROCESSOR_NAME];
	int n_len;
	MPI_Get_processor_name(node_name, &n_len);
	printf("Node name: %s\n", node_name);
	#else
	printf("No-MPI run\n");
	#endif

	const nbody_conf_t conf = {default_domain_size_x, default_domain_size_y, default_domain_size_z,
	                           default_mass_maximum, default_time_interval, default_seed, default_name,
	                           timesteps, num_particles
	                          };

	const nbody_t nbody = nbody_setup(&conf);

	const double start = wall_time();
	solve_nbody(nbody.positions, nbody.velocities, nbody.forces, nbody.masses, nbody.remote_positions, nbody.remote_masses, num_particles, timesteps, conf.time_interval);;
	const double end = wall_time();

	if (rank == 0)
		printf("Total execution time: %g s\n", end - start);

	nbody_save_particles(&nbody);

	nbody_check(&nbody);

	nbody_free(&nbody);

	#if USE_MPI
	MPI_Finalize();
	#endif

	return 0;
}
