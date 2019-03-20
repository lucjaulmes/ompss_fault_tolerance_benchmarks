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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "block.h"
#include "timer.h"
#include "proto.h"

int main(int argc, char **argv)
{
	int i, object_num = 0;

	init_globals();

	my_pe = 0;
	num_pes = 1;

	counter_malloc = 0;
	size_malloc = 0.0;

	/* set initial values */
	for (i = 1; i < argc; i++)
		if (!strcmp(argv[i], "--max_blocks"))
			max_num_blocks = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--target_active"))
			target_active = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--target_max"))
			target_max = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--target_min"))
			target_min = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--num_refine"))
			num_refine = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--block_change"))
			block_change = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--uniform_refine"))
			uniform_refine = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--nx"))
			x_block_size = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--ny"))
			y_block_size = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--nz"))
			z_block_size = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--num_vars"))
			num_vars = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--comm_vars"))
			comm_vars = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--init_x"))
			init_block_x = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--init_y"))
			init_block_y = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--init_z"))
			init_block_z = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--refine_freq"))
			refine_freq = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--report_diffusion"))
			report_diffusion = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--error_tol"))
			error_tol = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--num_tsteps"))
			num_tsteps = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--stages_per_ts"))
			stages_per_ts = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--checksum_freq"))
			checksum_freq = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--stencil"))
			stencil = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--permute"))
			permute = 1;
		else if (!strcmp(argv[i], "--report_perf"))
			report_perf = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--plot_freq"))
			plot_freq = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--code"))
			code = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--refine_ghost"))
			refine_ghost = 1;
		else if (!strcmp(argv[i], "--num_objects"))
		{
			num_objects = atoi(argv[++i]);
			objects = malloc(num_objects * sizeof(object));
			object_num = 0;
		}
		else if (!strcmp(argv[i], "--object"))
		{
			if (object_num >= num_objects)
			{
				printf("object number greater than num_objects\n");
				exit(-1);
			}
			objects[object_num].type = atoi(argv[++i]);
			objects[object_num].bounce = atoi(argv[++i]);
			objects[object_num].cen[0] = atof(argv[++i]);
			objects[object_num].cen[1] = atof(argv[++i]);
			objects[object_num].cen[2] = atof(argv[++i]);
			objects[object_num].move[0] = atof(argv[++i]);
			objects[object_num].move[1] = atof(argv[++i]);
			objects[object_num].move[2] = atof(argv[++i]);
			objects[object_num].size[0] = atof(argv[++i]);
			objects[object_num].size[1] = atof(argv[++i]);
			objects[object_num].size[2] = atof(argv[++i]);
			objects[object_num].inc[0] = atof(argv[++i]);
			objects[object_num].inc[1] = atof(argv[++i]);
			objects[object_num].inc[2] = atof(argv[++i]);
			object_num++;
		}
		else if (!strcmp(argv[i], "--original_comm"))
			original_comm = 1;
		else if (!strcmp(argv[i], "--merge_checksum"))
			merge_checksum = 1;
		else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
		{
			print_help_message();
			exit(0);
		}
		else
		{
			printf("** Error ** Unknown input parameter %s\n", argv[i]);
			print_help_message();
			exit(-1);
		}

	if (check_input())
		exit(-1);

	if (!block_change)
		block_change = num_refine;

	for (object_num = 0; object_num < num_objects; object_num++)
		for (i = 0; i < 3; i++)
		{
			objects[object_num].orig_cen[i] = objects[object_num].cen[i];
			objects[object_num].orig_move[i] = objects[object_num].move[i];
			objects[object_num].orig_size[i] = objects[object_num].size[i];
		}

	allocate();

	driver();

	profile();

	deallocate();

	exit(0);
}

// =================================== print_help_message ====================

void print_help_message()
{
	printf("(Optional) command line input is of the form: \n\n");
	const char opt_fmt[] = "%-20s   %s\n";

	printf(opt_fmt, "--nx",               "block size x (even && > 0)");
	printf(opt_fmt, "--ny",               "block size y (even && > 0)");
	printf(opt_fmt, "--nz",               "block size z (even && > 0)");
	printf(opt_fmt, "--init_x",           "initial blocks in x (> 0)");
	printf(opt_fmt, "--init_y",           "initial blocks in y (> 0)");
	printf(opt_fmt, "--init_z",           "initial blocks in z (> 0)");
	printf(opt_fmt, "--reorder",          "ordering of blocks if initial number > 1");
	printf(opt_fmt, "--max_blocks",       "maximun number of blocks per core");
	printf(opt_fmt, "--num_refine",       "number of levels of refinement (>= 0)");
	printf(opt_fmt, "--block_change",     "number of levels a block can change in a timestep (>= 0)");
	printf(opt_fmt, "--uniform_refine",   "if 1, then grid is uniformly refined");
	printf(opt_fmt, "--refine_freq",      "frequency (in timesteps) of checking for refinement");
	printf(opt_fmt, "--target_active",    "target number of blocks per core, none if 0 (>= 0)");
	printf(opt_fmt, "--target_max",       "max number of blocks per core, none if 0 (>= 0)");
	printf(opt_fmt, "--target_min",       "min number of blocks per core, none if 0 (>= 0)");
	printf(opt_fmt, "--num_vars",         "number of variables (> 0)");
	printf(opt_fmt, "--comm_vars",        "number of vars to communicate together");
	printf(opt_fmt, "--num_tsteps",       "number of timesteps (> 0)");
	printf(opt_fmt, "--stages_per_ts",    "number of comm/calc stages per timestep");
	printf(opt_fmt, "--checksum_freq",    "number of stages between checksums");
	printf(opt_fmt, "--stencil",          "7 or 27 point (27 will not work with refinement (except uniform))");
	printf(opt_fmt, "--error_tol",        "(e^{-error_tol} ; >= 0) ");
	printf(opt_fmt, "--report_diffusion", "none if 0 (>= 0)");
	printf(opt_fmt, "--report_perf",      "0, 1, 2");
	printf(opt_fmt, "--refine_freq",      "frequency (timesteps) of plotting (0 for none)");
	printf(opt_fmt, "--code",             "closely minic communication of different codes");
	printf(opt_fmt, "",                   "0 minimal sends, 1 send ghosts, 2 send ghosts and process on send\n");
	printf(opt_fmt, "--permute",          "altenates directions in communication");
	printf(opt_fmt, "--refine_ghost",     "use full extent of block (including ghosts) to determine if block is refined");
	printf(opt_fmt, "--num_objects",      "number of objects to cause refinement (>= 0)");
	printf(opt_fmt, "--object",           "type, position, movement, size, size rate of change");

	printf("All associated settings are integers except for objects\n");

	printf("\n\n");
	printf("Extra settings for the modified version\n");
	printf("=======================================\n\n");
	printf(opt_fmt, "--original_comm",    "(flag) uses the original halo communication routines");
	printf(opt_fmt, "",                   "(always original when `code != 0` or `stencil == 27`)");
	printf(opt_fmt, "--merge_checksum",   "(flag) does the checksum calculation in the same task");
	printf(opt_fmt, "",                   "as the stencil calculation");
}

// =================================== allocate ==============================

void allocate()
{
	box_size = sizeof(double[x_block_size + 2][y_block_size + 2][z_block_size + 2]);

	num_blocks = malloc((num_refine + 1) * sizeof(int));
	num_blocks[0] = num_pes * init_block_x * init_block_y * init_block_z;
	num_blocks[0] = init_block_x * init_block_y * init_block_z;

	blocks = malloc(max_num_blocks * sizeof(block));

	const size_t pagesize = sysconf(_SC_PAGESIZE);

	for (int n = 0; n < max_num_blocks; n++)
	{
		blocks[n].number = -1;
		blocks[n].array = aligned_alloc(pagesize, num_vars * box_size);
		/*
		for (int m = 0; m < num_vars; m++) {
			blocks[n].array[m] = aligned_alloc(pagesize, box_size);
		} */
	}

	sorted_list = malloc(max_num_blocks * sizeof(sorted_block));
	sorted_index = malloc((num_refine + 2) * sizeof(int));

	max_num_parents = max_num_blocks;  // Guess at number needed
	parents = malloc(max_num_parents * sizeof(parent));
	for (int n = 0; n < max_num_parents; n++)
		parents[n].number = -1;

	grid_sum = malloc(num_vars * sizeof(double));
	memset((void *) &grid_sum[0], 0, num_vars * sizeof(double));

	p8 = malloc((num_refine + 2) * sizeof(int));
	p2 = malloc((num_refine + 2) * sizeof(int));
	block_start = malloc((num_refine + 1) * sizeof(int));

	box_size = box_size / sizeof(double);
}

// =================================== deallocate ============================

void deallocate()
{
	for (int n = 0; n < max_num_blocks; n++)
		free(blocks[n].array);
	free(blocks);

	free(sorted_list);
	free(sorted_index);

	free(objects);

	free(grid_sum);

	free(p8);
	free(p2);

	free(num_blocks);
	free(parents);
	free(block_start);
}

int check_input()
{
	int error = 0;

	if (init_block_x < 1 || init_block_y < 1 || init_block_z < 1)
	{
		printf("initial blocks on processor must be positive\n");
		error++;
	}
	if (max_num_blocks < init_block_x * init_block_y * init_block_z)
	{
		printf("max_num_blocks not large enough\n");
		error++;
	}
	if (x_block_size < 1 || y_block_size < 1 || z_block_size < 1)
	{
		printf("block size must be positive\n");
		error++;
	}
	if (((x_block_size / 2) * 2) != x_block_size)
	{
		printf("block size in x direction must be even\n");
		error++;
	}
	if (((y_block_size / 2) * 2) != y_block_size)
	{
		printf("block size in y direction must be even\n");
		error++;
	}
	if (((z_block_size / 2) * 2) != z_block_size)
	{
		printf("block size in z direction must be even\n");
		error++;
	}
	if (target_active && target_max)
	{
		printf("Only one of target_active and target_max can be used\n");
		error++;
	}
	if (target_active && target_min)
	{
		printf("Only one of target_active and target_min can be used\n");
		error++;
	}
	if (target_active < 0 || target_active > max_num_blocks)
	{
		printf("illegal value for target_active\n");
		error++;
	}
	if (target_max < 0 || target_max > max_num_blocks ||
	    target_max < target_active)
	{
		printf("illegal value for target_max\n");
		error++;
	}
	if (target_min < 0 || target_min > max_num_blocks ||
	    target_min > target_active || target_min > target_max)
	{
		printf("illegal value for target_min\n");
		error++;
	}
	if (num_refine < 0)
	{
		printf("number of refinement levels must be non-negative\n");
		error++;
	}
	if (block_change < 0)
	{
		printf("number of refinement levels must be non-negative\n");
		error++;
	}
	if (num_vars < 1)
	{
		printf("number of variables must be positive\n");
		error++;
	}
	if (num_pes != npx * npy * npz)
	{
		printf("number of processors used %d does not match number allocated %d * %d * %d\n", num_pes, npx, npy, npz);
		error++;
	}
	if (stencil != 7 && stencil != 27)
	{
		printf("illegal value for stencil\n");
		error++;
	}
	if (stencil == 27 && num_refine && !uniform_refine)
		printf("WARNING: 27 point stencil with non-uniform refinement: answers may diverge\n");
	if (comm_vars == 0 || comm_vars > num_vars)
		comm_vars = num_vars;
	if (code < 0 || code > 2)
	{
		printf("code must be 0, 1, or 2\n");
		error++;
	}

	return (error);
}
