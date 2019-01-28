#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#if USE_MPI
	#include <mpi.h>
#endif

#include "nbody.h"

#if TASKLOOP
extern int GS; // grainsize
#endif

nbody_conf_t parse_args(int argc, char *argv[])
{
	nbody_conf_t conf = {1e10 /* m */, 1e10 /* m */, 1e10 /* m */, 1e28 /* kg */, 1e0 /* s */,
						  16384, 10, 12345, 0, "data/nbody_soa",
#if TASKLOOP
						  nanos_omp_get_num_threads(),
#endif
	};

#define no_argument 0
#define required_argument 1
#define optional_argument 2
	struct option long_options[] =
	{
		{"particles",	required_argument,	NULL, 'n'},
		{"size_x",		required_argument,	NULL, 'x'},
		{"size_y",		required_argument,	NULL, 'y'},
		{"size_z",		required_argument,	NULL, 'z'},
		{"mass",		required_argument,	NULL, 'm'},
		{"interval",	required_argument,	NULL, 'i'},
		{"timesteps",	required_argument,	NULL, 't'},
		{"seed",		required_argument,	NULL, 's'},
		{"filename",	required_argument,	NULL, 'f'},
#if TASKLOOP
		{"blocking",	required_argument,	NULL, 'b'},
#endif
		{"check",		no_argument,		NULL, 'c'},
		{"dump",		no_argument,		NULL, 'd'},
		{"help",		no_argument,		NULL, 'h'},
		{NULL,			0,					NULL, '\0'}
	};

	char optstring[2 * sizeof(long_options) / sizeof(*long_options)], *optchar = optstring;
	for (struct option *opt = long_options; opt->name != NULL; ++opt)
	{
		*optchar++ = opt->val;
		if (opt->has_arg)
			*optchar++ = ':';
	}
	*optchar = '\0';


	for (int opt = 0; opt != -1;)
	{
		opt = getopt_long(argc, argv, optstring, long_options, NULL);
		switch (opt) {
			case 'n':
				conf.num_particles = strtol(optarg, NULL, 0);
				break;
			case 'x':
				conf.domain_size_x = strtod(optarg, NULL);
				break;
			case 'y':
				conf.domain_size_y = strtod(optarg, NULL);
				break;
			case 'z':
				conf.domain_size_z = strtod(optarg, NULL);
				break;
			case 'm':
				conf.mass_maximum  = strtod(optarg, NULL);
				break;
			case 'i':
				conf.time_interval = strtod(optarg, NULL);
				break;
			case 't':
				conf.timesteps = strtol(optarg, NULL, 0);
				break;
			case 's':
				conf.seed = strtol(optarg, NULL, 0);
				break;
			case 'f':
				conf.name = strndup(optarg, MAX_FILENAME_LEN);
				break;
			case 'd':
				conf.check = DUMP_RESULT;
				break;
			case 'c':
				conf.check = CHECK_RESULT;
				break;
#if TASKLOOP
			case 'b':
				conf.num_tasks = strtol(optarg, NULL, 0);
				break;
#endif
			case 0:
			case -1:
				break;
			case 'h':
			default:
				printf("Usage: %s [options]\nPossible options are:\n", argv[0]);
				for (struct option *opt = long_options; opt->name != NULL; ++opt)
					printf("    -%c, --%s%*s\n", opt->val, opt->name, 15 - (int)strlen(opt->name), opt->has_arg ? " val":"");
				exit(opt == '?' || opt == ':');
		}
	}

	return conf;
}

int main(int argc, char *argv[])
{
	nbody_conf_t conf = parse_args(argc, argv);

	#if USE_MPI
	int arg_read = optind;
	optind = 0;
	argv[arg_read] = argv[0];
	setup_mpi(argc - arg_read, argv + arg_read);
	#endif

	const int MIN_PARTICLES = 128;
	int local_num_particles = conf.num_particles / commsize;
	local_num_particles = ((local_num_particles - 1) | (MIN_PARTICLES - 1)) + 1;
	conf.num_particles = local_num_particles * commsize;

	if (local_num_particles < MIN_PARTICLES)
		errx(1, "Need %d >= %d particles per process\n", local_num_particles, MIN_PARTICLES);

	if (conf.timesteps <= 0)
		errx(1, "Need and %d > 0 timesteps\n", conf.timesteps);

	if (rank == 0)
		printf("O(%s), processes: %d, particles: %d, particles in process: %d, ", XSTR(BIGO), commsize, conf.num_particles, local_num_particles);

	#if TASKLOOP
	if (conf.num_tasks <= 0)
		errx(1, "Need %d > 0 tasks per thread\n", conf.num_tasks);
	else
		printf("tasks per loop: %d, ", conf.num_tasks);

	GS = local_num_particles / conf.num_tasks;
	if (local_num_particles % conf.num_tasks)
		errx(1, "The number of particles must be equally dividable among parallel blocks: %d * %d != %d\n",
				conf.num_tasks, GS, local_num_particles);
	#endif

	if (rank == 0)
		printf("timesteps: %d\n", conf.timesteps);

	#if USE_MPI
	char node_name[MPI_MAX_PROCESSOR_NAME];
	int n_len;
	MPI_Get_processor_name(node_name, &n_len);
	printf("Node name: %s\n", node_name);
	#else
	printf("No-MPI run\n");
	#endif

	const nbody_t nbody = nbody_setup(&conf);

	const double start = wall_time();
	solve_nbody(nbody.positions, nbody.velocities, nbody.forces, nbody.masses,
				nbody.remote_positions, nbody.remote_masses, local_num_particles, conf.timesteps, conf.time_interval);
	const double end = wall_time();

	if (rank == 0)
		printf("Total execution time: %g s\n", end - start);

	if (conf.check == DUMP_RESULT)
		nbody_save_particles(&nbody);
	else if (conf.check == CHECK_RESULT)
		nbody_check(&nbody);

	nbody_free(&nbody);

	#if USE_MPI
	MPI_Finalize();
	#endif

	return 0;
}
