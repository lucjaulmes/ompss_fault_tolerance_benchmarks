/* nbody.c */
/*
#pragma offload_attribute (push,target(mic))
#include "include/kernels.h"
#pragma offload_attribute (pop)
*/
#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


void solve_nbody(particles_block_t *__restrict__ local, particles_block_t *__restrict__ tmp, force_block_t *__restrict__ forces,
                 const int n_blocks, const int timesteps, const float time_interval)
{
	const double start = wall_time();
	for (int t = 0; t < timesteps; t++)
	{
		particles_block_t *remote = local;
		for (int i = 0; i < commsize; i++)
		{
			//#pragma omp task inout([n_blocks]local, [n_blocks]remote) in([n_blocks]forces)
			calculate_forces(forces, local, remote, n_blocks);

			//#pragma omp task in([n_blocks]remote) out([n_blocks]tmp)
			exchange_particles(remote, tmp, n_blocks, i);
			remote = tmp;
		}
		//#pragma omp task inout([n_blocks]local) inout([n_blocks]forces)
		update_particles(n_blocks, local, forces, time_interval);
	}

	//#pragma omp taskwait
	const double end = wall_time();
	if (rank == 0)
		print_stats(n_blocks, timesteps, end - start);
}
