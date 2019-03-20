// ************************************************************************
//
// miniAMR: stencil computations with boundary exchange and AMR.
//
// Copyright (2014) Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
// retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
// Questions? Contact Courtenay T. Vaughan (ctvaugh@sandia.gov)
//                    Richard F. Barrett (rfbarre@sandia.gov)
//
// ************************************************************************

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "block.h"
#include "timer.h"
#include "proto.h"
#include "proto_task.h"


//// Functions that create tasks and call actual kernels ////

// Generate check sum for a variable over all active blocks.
void check_sum(int var, int number, double *sum)
{
	double t = timer();

	// memset(sum + var, 0, number * sizeof(double));
	for (int in = 0; in < sorted_index[num_refine + 1]; in++)
	{
		int n = sorted_list[in].n;
		if (blocks[n].number >= 0)
		{
			typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];
			box_arr_t array = (box_arr_t)blocks[n].array;

			double *sum_var = sum + var; // reductions only take OpenMP-style deps, not (+: sum[var;number])
			#pragma omp task label(check_sum_task) inout(array[var;number]) reduction(+:[number]sum_var) \
				firstprivate(x_block_size, y_block_size, z_block_size, var, number) default(shared)
			for (int i = var; i < var + number; ++i)
				sum[i] += check_sum_task(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
		}
	}

	timer_cs_calc += timer() - t;
	total_red++;
}

// This routine does the stencil calculations.
void stencil_calc(int var, int number)
{
	double t = timer();
	typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];

	for (int in = 0; in < sorted_index[num_refine + 1]; in++)
	{
		int n = sorted_list[in].n;
		if (blocks[n].number >= 0)
		{
			box_arr_t array = (box_arr_t)blocks[n].array;
			if (stencil == 7)
			{
				#pragma omp task label(stencil7) inout(array[var;number]) \
					firstprivate(x_block_size, y_block_size, z_block_size, var, number) default(shared)
				for (int i = var; i < var + number; ++i)
					stencil_task7(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
			}
			else
			{
				#pragma omp task label(stencil27) inout(array[var;number]) \
					firstprivate(x_block_size, y_block_size, z_block_size, var, number) default(shared)
				for (int i = var; i < var + number; ++i)
					stencil_task27(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
			}
		}
	}

	timer_cs_all = timer_calc_all = timer() - t;
}

// This routine does both of the above, either merged or not based or merge_checksum
void stencil_calc_checksum(int var, int number, double *sum)
{
	double t = timer();
	if (!merge_checksum)
	{
		stencil_calc(var, number);
		check_sum(var, number, sum);
	}
	else
	{
		typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];

		for (int in = 0; in < sorted_index[num_refine + 1]; in++)
		{
			int n = sorted_list[in].n;
			if (blocks[n].number >= 0)
			{
				box_arr_t array = (box_arr_t)blocks[n].array;
				double *sum_var = sum + var; // reductions only take OpenMP-style deps, not (+: sum[var;number])
				if (stencil == 7)
				{
					#pragma omp task label(stencil7) inout(array[var;number]) reduction(+:[number]sum_var) \
						firstprivate(x_block_size, y_block_size, z_block_size, var, number) default(shared)
					for (int i = var; i < var + number; ++i)
					{
						stencil_task7(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
						sum[i] += check_sum_task(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
					}
				}
				else
				{
					#pragma omp task label(stencil27) inout(array[var;number]) reduction(+:[number]sum_var) \
						firstprivate(x_block_size, y_block_size, z_block_size, var, number) default(shared)
					for (int i = var; i < var + number; ++i)
					{
						stencil_task27(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
						sum[i] += check_sum_task(x_block_size + 2, y_block_size + 2, z_block_size + 2, array[i]);
					}
				}
			}
		}
	}
	timer_cs_all = timer_calc_all = timer() - t;
}


// Main driver for program.
void driver()
{
	double *sum = calloc(num_vars, sizeof(double));

	init();
	init_profile();
	counter_malloc_init = counter_malloc;
	size_malloc_init = size_malloc;

	double start_t = timer();

	if (num_refine || uniform_refine) refine(0);

	if (plot_freq)
		plot(0);

	nb_min = nb_max = global_active;
	int comm_stage = 0;
	for (int ts = 1; ts <= num_tsteps; ts++)
	{
		for (int stage = 0; stage < stages_per_ts; stage++, comm_stage++)
		{
			total_blocks += global_active;
			if (global_active < nb_min) nb_min = global_active;
			if (global_active > nb_max) nb_max = global_active;

			for (int start = 0; start < num_vars; start += comm_vars)
			{
				int number = comm_vars;
				if (number > num_vars - start)
					number = num_vars - start;

				comm(start, number, comm_stage);

				if (!(checksum_freq && (stage % checksum_freq) == 0))
					stencil_calc(start, number);
				else
				{
					stencil_calc_checksum(start, number, sum);

					#pragma omp task label(check_sum_val) inout(sum[start;number], grid_sum[start;number]) \
						firstprivate(start, number, report_diffusion, my_pe, ts, sum, grid_sum, tol)
					for (int var = start; var < (start + number); var++)
					{
						double check = fabs(sum[var] - grid_sum[var]) / grid_sum[var];

						if (report_diffusion && !my_pe)
							printf("%d var %d sum %lf old %lf diff %lf tol %lf\n", ts, var, sum[var], grid_sum[var], check, tol);

						if (!(check <= tol))
						{
							if (!my_pe)
								printf("Time step %d sum %lf (old %lf) variable %d difference too large\n", ts, sum[var], grid_sum[var], var);

							exit(1);
						}

						grid_sum[var] = sum[var];
						sum[var] = 0.0;
					}
				}
			}
		}

		if (num_refine && !uniform_refine)
		{
			double start_move_t = timer();
			move();

			// NB refine already adds to timer_refine_all
			timer_refine_all += timer() - start_move_t;

			if ((ts % refine_freq) == 0)
				refine(ts);
		}

		if (plot_freq && (ts % plot_freq) == 0)
			plot(ts);
	}
	#pragma omp taskwait

	timer_calc_all = timer_all = timer() - start_t;
    free(sum);
}
