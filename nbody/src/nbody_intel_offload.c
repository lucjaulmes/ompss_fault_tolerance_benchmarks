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


#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)

void solve_nbody(particles_block_t *local, particles_block_t *tmp, force_block_t *forces,
                 const int n_blocks, const int timesteps, const float time_interval)
{
	const double start = wall_time();
	for (int t = 0; t < timesteps; t++)
	{
		particles_block_t *remote = local;
		for (int i = 0; i < commsize; i++)
		{
			//#pragma offload target(mic) in(local[0:n]) in(remote[0:n_blocks]) inout(forces[0:n_blocks])
#pragma offload target(mic) in(local[0:n_blocks] : alloc_if(t==0) free_if(0)) \
                                     in(remote[0:n_blocks] : free_if(i==commsize-1) alloc_if(i==0 || i==1)) \
                                     out(forces[0:n_blocks] : alloc_if(t==0) free_if(0))
			calculate_forces(forces, local, remote, n_blocks);


			exchange_particles(remote, tmp, n_blocks, t * commsize + i);
			remote = tmp;
		}
		//#pragma offload target(mic) inout(local[0:n_blocks]) in(forces[0:n_blocks])
#pragma offload target(mic) inout(local[0:n_blocks] : alloc_if(0) free_if(t==timesteps-1)) \
                                  inout(forces[0:n_blocks] : alloc_if(0) free_if(t==timesteps-1))
		update_particles(n_blocks, local, forces, time_interval);
	}

	const double end = wall_time();
	if (rank == 0)
		print_stats(n_blocks, timesteps, end - start);
}
