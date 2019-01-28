/* nbody.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>
#include "catchroi.h"

#if USE_MPI
	#include <mpi.h>
#endif

#include "nbody.h"

int GS;

void solve_nbody(coord_ptr_t local_positions, coord_ptr_t velocities, coord_ptr_t forces,
				 float_ptr_t local_masses, coord_ptr_t remote_positions, float_ptr_t remote_masses,
                 const int n_part, const int timesteps, const float time_interval)
{
	#if USE_MPI
	if (local_positions + n_part != (coord_ptr_t)local_masses ||
	    remote_positions + n_part != (coord_ptr_t)remote_masses)
		errx(2, "Positions and masses need to be contiguous to be sent simultaneously in MPI_Sendrecv");
	#else
	(void)remote_positions;
	(void)remote_masses;
	#endif

	int t, send_size = n_part * (sizeof(*local_masses) + sizeof(*local_positions));
	const double start = wall_time();
	start_roi();

	for (t = 0; t < timesteps; t++)
	{
		// Calculate forces from local particles.
		calculate_local_forces(forces, local_positions, local_masses, n_part);

		#if USE_MPI
		coord_t *send_pos = local_positions;
		float *send_mass = local_masses;

		// Send and get particles from another rank commsize-1 times, and add contributions from their particles.
		// First send local then re-send the previous remote along in a circle.
		for (int i = 0; i < get_commsize() - 1; i++, send_pos = remote_positions, send_mass = remote_masses)
		{
			DO_PRAGMA(omp task in(ALL_BLOCKS(send_pos, n_part, GS), ALL_BLOCKS(send_mass, n_part, GS)) out(ALL_BLOCKS(remote_positions, n_part, GS), ALL_BLOCKS(remote_masses, n_part, GS)) label(exchange_particles))
			exchange_particles(send_pos, remote_positions, send_size);

			calculate_forces(forces, local_positions, local_masses, remote_positions, remote_masses, n_part);
		}
		#endif

		update_particles(local_positions, velocities, forces, local_masses, n_part, time_interval);
	}

	#pragma omp taskwait
	stop_roi(t);

	const double end = wall_time();
	if (rank == 0)
		print_stats(n_part, timesteps, end - start);
}
