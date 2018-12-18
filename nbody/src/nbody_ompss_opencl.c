/* nbody.c */

#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>





void solve_nbody(particles_block_t *const local, particles_block_t *const tmp, force_block_t *const forces,
                 const int n_blocks, const int timesteps, const float time_interval)
{
	#pragma omp target device(opencl) ndrange(1, n_blocks, 32) copy_deps
	#pragma omp task inout([bs]forces) in([bs]block1, [bs]block2)
	void calculate_forces(__global force_ptr_t forces, __global particle_ptr_t block1, __global particle_ptr_t block2, cint bs);

	const double start = wall_time();

	for (int t = 0; t < timesteps; t++)
	{
		particles_block_t *remote = local;
		for (int i = 0; i < commsize; i++)
		{
			calculate_forces(forces, local, remote, n_blocks);
			#pragma omp target device(smp) copy_deps
			#pragma omp task in([n_blocks] remote) out([n_blocks] tmp)
			exchange_particles(remote, tmp, n_blocks, i);
			remote = tmp;
		}

		#pragma omp target device(smp) copy_deps
		#pragma omp task inout([n_blocks]local) inout([n_blocks]forces)
		update_particles(n_blocks, local, forces, time_interval);
	}

	#pragma omp taskwait
	const double end = wall_time();
	if (rank == 0)
		print_stats(n_blocks, timesteps, end - start);
}
