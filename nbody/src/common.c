#include "nbody.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <ieee754.h>
#include <string.h>
#include <err.h>

#include <catchroi.h>

#if USE_MPI
	#include <mpi.h>
#endif

int rank = 0;


#if USE_MPI
static const char *mpi_thread_support[] = {"SINGLE", "FUNNELED", "SERIALIZED", "MULTIPLE"};
int commsize = 0;

void setup_mpi(int argc, char *argv[])
{
	int requested = MPI_THREAD_MULTIPLE, provided;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);

	if (provided > (int)(sizeof(mpi_thread_support) / sizeof(*mpi_thread_support)))
		err(1, "Unknown level of MPI thread support %d", provided);
	else if (requested != provided)
		warnx("Expected multiple thread mpi support %s, got %s instead", mpi_thread_support[requested], mpi_thread_support[provided]);
}

void exchange_particles(void *sendbuf, void *recvbuf, int size)
{
	if (commsize <= 1)
		errx(1, "Exchanging particles with a single running node");

	int src = (rank + commsize - 1) % commsize;
	int dst = (rank + 1) % commsize;

	if (sendbuf != recvbuf)
		MPI_Sendrecv(sendbuf, size, MPI_BYTE, dst, 0, recvbuf, size, MPI_BYTE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	else
		MPI_Sendrecv_replace(sendbuf, size, MPI_BYTE, dst, 0, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
#endif


void particle_init(const nbody_conf_t *const conf, coord_t *position, float *mass)
{
	position->x = conf->domain_size_x * ((float) random() / ((float)RAND_MAX + 1.0));
	position->y = conf->domain_size_y * ((float) random() / ((float)RAND_MAX + 1.0));
	position->z = conf->domain_size_z * ((float) random() / ((float)RAND_MAX + 1.0));
	*mass       = conf->mass_maximum  * ((float) random() / ((float)RAND_MAX + 1.0));
}

void nbody_generate_particles(const nbody_conf_t *conf, nbody_file_t *file)
{
	char fname[1024];
	sprintf(fname, "%s.in", file->name);

	if (access(fname, F_OK) == 0)
		return;

	const int fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IRGRP | S_IROTH);
	void *const ptr = mmap(NULL, file->input_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);

	if (fd < 0 || ftruncate(fd, file->input_size) != 0 || ptr == MAP_FAILED)
		err(1, "Error creating and mapping file %s with size %zu", fname, file->input_size);

	const int total_num_particles = conf->num_particles * get_commsize();
	coord_ptr_t positions = ptr + 0;
	float_ptr_t masses    = ptr + file->check_size;

	/* Probably really inefficient, but we'll do it only once */
	for (int i = 0; i < total_num_particles; i++)
		particle_init(conf, (coord_t *)(positions + i), (float *)(masses + i));

	if (munmap(ptr, file->input_size) != 0 || close(fd) != 0)
		err(1, "Error closing file %s", fname);
}

void nbody_check(const nbody_t *nbody)
{
	char fname[1024];
	sprintf(fname, "%s.ref", nbody->file.name);
	if (access(fname, F_OK) != 0)
	{
		if (rank == 0)
			warnx("No file %s: skipping check", fname);
		return;
	}

	const int fd = open(fname, O_RDONLY, 0);
	coord_t *world_positions = mmap(NULL, nbody->file.check_size, PROT_READ, MAP_SHARED, fd, 0);

	if (fd == 0 || world_positions == MAP_FAILED)
		err(1, "Error opening and mapping file %s with size %zu", fname, nbody->file.check_size);

	coord_ptr_t positions = world_positions + nbody->file.position_offset / sizeof(coord_t);
	double error = 0.0;
	int count = 0;
	for (int i = 0; i < nbody->num_particles; i++)
	{
		if ((nbody->positions[i].x != positions[i].x) ||
		    (nbody->positions[i].y != positions[i].y) ||
		    (nbody->positions[i].z != positions[i].z))
		{
			error += fabs(((nbody->positions[i].x - positions[i].x) * 100.0) / positions[i].x) +
			         fabs(((nbody->positions[i].y - positions[i].y) * 100.0) / positions[i].y) +
			         fabs(((nbody->positions[i].z - positions[i].z) * 100.0) / positions[i].z);

			count++;
		}
	}

	munmap(world_positions, nbody->file.check_size);

	double relative_error = error / (3.0 * count);
	#if USE_MPI
	if ((count * 100.0) / nbody->num_particles > 0.6 || relative_error > 0.000008)
		printf("At rank %d relative error[%d]: %f\n", rank, count, relative_error);

	// double local_error = error; int local_count = count; error = 0.0; count = 0;
	MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &error, &error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &count, &count, 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);
	relative_error = error / (3.0 * count);
	#endif

	if (rank == 0)
	{
		if ((count * 100.0) / (get_commsize() * nbody->num_particles) > 0.6 || relative_error > 0.000008)
			printf("Relative error[%d]: %f\n", count, relative_error);
		else
			printf("Result validation: OK\n");
	}
}


void nbody_load_particles(coord_t *positions, float *masses, nbody_file_t *file, const int num_particles)
{
	char fname[1024];
	sprintf(fname, "%s.in", file->name);

	const int fd = open(fname, O_RDONLY, 0);
	void *ptr = mmap(NULL, file->input_size, PROT_READ, MAP_SHARED, fd, 0);

	if (fd < 0 || ptr == MAP_FAILED)
		err(1, "Error loading particiles from file %s", fname);

	memcpy(positions, (void *)(ptr + file->position_offset), num_particles * sizeof(*positions));
	memcpy(masses, (void *)(ptr + file->mass_offset),     num_particles * sizeof(*masses));

	if (munmap(ptr, file->input_size) != 0 || close(fd) != 0)
		err(1, "Could not close file %s properly", fname);
}


void *nbody_alloc(const size_t size)
{
	void *const space = CATCHROI_INSTRUMENT(mmap)(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	/* for the sake of touching all the pages before running */
	memset(space, 0, size);

	if (space == MAP_FAILED)
		err(1, "Error allocating memory");

	return space;
}

coord_t *nbody_alloc_coord(const nbody_conf_t *const conf)
{
	size_t size = conf->num_particles * sizeof(coord_t);
	void *const space = CATCHROI_INSTRUMENT(mmap)(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	/* for the sake of touching all the pages before running */
	memset(space, 0, size);

	if (space == MAP_FAILED)
		err(1, "Error allocating memory");

	return space;
}

void *nbody_alloc_position_mass(const nbody_conf_t *const conf)
{
	size_t pos_size  = conf->num_particles * sizeof(coord_t);
	size_t mass_size = conf->num_particles * sizeof(float);

	// This is a trick so that we can instrument the 2 contiguous memory regions separately while still guaranteeing they are contiguous.
	void *space = mmap(NULL, pos_size + mass_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	munmap(space, pos_size + mass_size);

	void *const pos  = CATCHROI_INSTRUMENT(mmap)(space, pos_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	void *const mass = CATCHROI_INSTRUMENT(mmap)(space + pos_size, mass_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	if (pos != space || space + pos_size != mass)
		err(1, "Pos & mass allocations not contiguous");

	/* for the sake of touching all the pages before running */
	memset(space, 0, pos_size + mass_size);

	if (space == MAP_FAILED)
		err(1, "Error allocating memory");

	return space;
}

nbody_file_t nbody_setup_file(const nbody_conf_t *const conf)
{
	nbody_file_t file;
	const int total_num_particles = conf->num_particles * get_commsize();
	file.check_size = total_num_particles * sizeof(coord_t);
	file.input_size = total_num_particles * sizeof(float) + file.check_size;

	file.position_offset = conf->num_particles * sizeof(coord_t) * rank;
	file.mass_offset     = conf->num_particles * sizeof(float)  * rank + file.check_size;

	sprintf(file.name, "%s-%s-%d-%d-%d", conf->name, XSTR(BIGO), total_num_particles, 1, conf->timesteps); // 1 is there for file name compatibility
	return file;
}

nbody_t nbody_setup(const nbody_conf_t *const conf)
{
	nbody_file_t file = nbody_setup_file(conf);

	if (rank == 0)
		nbody_generate_particles(conf, &file);

	/* NB. when allocating, force positions and masses to be contiguous */
	void *local_buffer = nbody_alloc_position_mass(conf);

	#if USE_MPI
	void *remote_buffer = nbody_alloc_position_mass(conf);

	if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
		err(1, "Failed waiting for all ranks after generating particles");
	#endif

	nbody_t nbody =
	{
		.positions          = (coord_ptr_t)local_buffer,
		.velocities         = nbody_alloc_coord(conf),
		.forces             = nbody_alloc_coord(conf),
		.masses             = (float_ptr_t)(local_buffer + conf->num_particles * sizeof(coord_t)),
		#if USE_MPI
		.remote_positions   = (coord_ptr_t)remote_buffer,
		.remote_masses      = (float_ptr_t)(remote_buffer + conf->num_particles * sizeof(coord_t)),
		#endif
		.num_particles      = conf->num_particles,
		.timesteps          = conf->timesteps,
		.file               = file
	};
	#if USE_MPI
	printf("remote_positions=%p-%p\n", (void*)nbody.remote_positions, (void*)(nbody.remote_positions + conf->num_particles));
	printf("remote_masses=%p-%p\n",    (void*)nbody.remote_masses,    (void*)(nbody.remote_masses + conf->num_particles));
	#endif

	nbody_load_particles(nbody.positions, nbody.masses, &file, conf->num_particles);

	#if USE_MPI
	if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
		err(1, "Failed waiting for all ranks setting up particles");
	#endif

	return nbody;
}


void nbody_save_particles(const nbody_t *const nbody)
{
	char fname[1024];
	sprintf(fname, "%s.out", nbody->file.name);

	#if USE_MPI
	MPI_File outfile;
	MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);

	MPI_File_set_view(outfile, nbody->file.position_offset, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
	MPI_File_write(outfile, nbody->positions, nbody->num_particles * sizeof(coord_t), MPI_BYTE, MPI_STATUS_IGNORE);

	MPI_File_close(&outfile);
	MPI_Barrier(MPI_COMM_WORLD);
	#else
	int fd = open(fname, O_WRONLY | O_CREAT);
	write(fd, nbody->positions, nbody->num_particles * sizeof(coord_t));
	close(fd);
	#endif
}

void nbody_free(const nbody_t *const nbody)
{
	{
		const size_t size = nbody->num_particles * sizeof(coord_t);
		munmap(nbody->velocities, size);
		munmap(nbody->forces, size);
	}

	{
		const size_t size = nbody->num_particles * (sizeof(coord_t) + sizeof(float));
		#if USE_MPI
		munmap(nbody->remote_positions, size);
		#endif
		munmap(nbody->positions, size);
	}
}

double wall_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)(ts.tv_sec)  + (double) ts.tv_nsec * 1.0e-9;
}

void print_stats(double n_particles, int timesteps, double elapsed_time)
{
	double total_particles      = get_commsize() * n_particles; // Each process has n_particles
	double total_ints_per_sec   = total_particles * total_particles * timesteps / elapsed_time;
	double process_ints_per_sec = total_ints_per_sec / get_commsize();
	printf("Non-offload execution time: %g s \n", elapsed_time);
	printf("Million interactions per second: %g (total: %g)\n", process_ints_per_sec / 1.0E6,      total_ints_per_sec / 1.0E6);
	printf("GFLOPS (20 FLOP per inter): %g (total: %g)\n",      process_ints_per_sec * 20 / 1.0E9, total_ints_per_sec * 20 / 1.0E9);
}
