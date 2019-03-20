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
#include <stdlib.h>

#include "block.h"
#include "timer.h"
#include "proto.h"

// The routines in this file are used in the communication of ghost values
// between blocks, both on processor and off processor.

// Main communication routine that sends and recieves ghost values between
// blocks on different processors and directos blocks on the same processor
// to exchange their ghost values.
void comm(int start, int num_comm, int stage)
{
	int i, j, k, l, m, n, dir, o, in;
	int permutations[6][3] = { {0, 1, 2}, {1, 2, 0}, {2, 0, 1},
		{0, 2, 1}, {1, 0, 2}, {2, 1, 0}
	};
	double t0 = timer(), t1, t2;
	block *bp;

	if (!original_comm && !code && stencil == 7)
	{
		for (in = 0; in < sorted_index[num_refine + 1]; in++)
		{
			n = sorted_list[in].n;
			bp = &blocks[n];
			if (bp->number < 0) continue;

			double **east, **west, **north, **south, **up, **down;
			int east_f, west_f, north_f, south_f, up_f, down_f;

			//EAST
			east = malloc(4 * sizeof(double *));
			if (bp->nei_level[1] == -2)
			{
				east_f = 2;
				east[0] = bp->array;
				counter_bc[0]++;
			}
			else if (bp->nei_level[1] - bp->level == 0)
			{
				east_f = 0;
				east[0] = blocks[bp->nei[1][0][0]].array;
				counter_same[0]++;
			}
			else if (bp->nei_level[1] - bp->level == 1)
			{
				east_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						east[i * 2 + j] = blocks[bp->nei[1][i][j]].array;
				counter_diff[0] += 4;
			}
			else if (bp->nei_level[1] - bp->level == -1)
			{
				east_f = -1;
				m = bp->nei[1][0][0];
				east[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[0][i][j] == n)
							east_f -= i * 2 + j;
				counter_diff[0]++;
			}
			else
			{
				printf("ERROR: misconnected block, east: %d\n", bp->nei_level[1] - bp->level);
				exit(-1);
			}

			// WEST
			west = malloc(4 * sizeof(double *));
			if (bp->nei_level[0] == -2)
			{
				west_f = 2;
				west[0] = bp->array;
				counter_bc[0]++;
			}
			else if (bp->nei_level[0] - bp->level == 0)
			{
				west_f = 0;
				west[0] = blocks[bp->nei[0][0][0]].array;
				counter_same[0]++;
			}
			else if (bp->nei_level[0] - bp->level == 1)
			{
				west_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						west[i * 2 + j] = blocks[bp->nei[0][i][j]].array;
				counter_diff[0] += 4;
			}
			else if (bp->nei_level[0] - bp->level == -1)
			{
				west_f = -1;
				m = bp->nei[0][0][0];
				west[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[1][i][j] == n)
							west_f -= i * 2 + j;
				counter_diff[0]++;
			}
			else
			{
				printf("ERROR: misconnected block, west: %d\n", bp->nei_level[0] - bp->level);
				exit(-1);
			}

			// NORTH
			north = malloc(4 * sizeof(double *));
			if (bp->nei_level[3] == -2)
			{
				north_f = 2;
				north[0] = bp->array;
				counter_bc[1]++;
			}
			else if (bp->nei_level[3] - bp->level == 0)
			{
				north_f = 0;
				north[0] = blocks[bp->nei[3][0][0]].array;
				counter_same[1]++;
			}
			else if (bp->nei_level[3] - bp->level == 1)
			{
				north_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						north[i * 2 + j] = blocks[bp->nei[3][i][j]].array;
				counter_diff[1] += 4;
			}
			else if (bp->nei_level[3] - bp->level == -1)
			{
				north_f = -1;
				m = bp->nei[3][0][0];
				north[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[2][i][j] == n)
							north_f -= i * 2 + j;
				counter_diff[1]++;
			}
			else
			{
				printf("ERROR: misconnected block, north: %d\n", bp->nei_level[3] - bp->level);
				exit(-1);
			}

			// SOUTH
			south = malloc(4 * sizeof(double *));
			if (bp->nei_level[2] == -2)
			{
				south_f = 2;
				south[0] = bp->array;
				counter_bc[1]++;
			}
			else if (bp->nei_level[2] - bp->level == 0)
			{
				south_f = 0;
				south[0] = blocks[bp->nei[2][0][0]].array;
				counter_same[1]++;
			}
			else if (bp->nei_level[2] - bp->level == 1)
			{
				south_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						south[i * 2 + j] = blocks[bp->nei[2][i][j]].array;
				counter_diff[1] += 4;
			}
			else if (bp->nei_level[2] - bp->level == -1)
			{
				south_f = -1;
				m = bp->nei[2][0][0];
				south[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[3][i][j] == n)
							south_f -= i * 2 + j;
				counter_diff[1]++;
			}
			else
			{
				printf("ERROR: misconnected block, south: %d\n", bp->nei_level[2] - bp->level);
				exit(-1);
			}

			// UP
			up = malloc(4 * sizeof(double *));
			if (bp->nei_level[5] == -2)
			{
				up_f = 2;
				up[0] = bp->array;
				counter_bc[2]++;
			}
			else if (bp->nei_level[5] - bp->level == 0)
			{
				up_f = 0;
				up[0] = blocks[bp->nei[5][0][0]].array;
				counter_same[2]++;
			}
			else if (bp->nei_level[5] - bp->level == 1)
			{
				up_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						up[i * 2 + j] = blocks[bp->nei[5][i][j]].array;
				counter_diff[2] += 4;
			}
			else if (bp->nei_level[5] - bp->level == -1)
			{
				up_f = -1;
				m = bp->nei[5][0][0];
				up[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[4][i][j] == n)
							up_f -= i * 2 + j;
				counter_diff[2]++;
			}
			else
			{
				printf("ERROR: misconnected block, up: %d\n", bp->nei_level[5] - bp->level);
				exit(-1);
			}

			// DOWN
			down = malloc(4 * sizeof(double *));
			if (bp->nei_level[4] == -2)
			{
				down_f = 2;
				down[0] = bp->array;
				counter_bc[2]++;
			}
			else if (bp->nei_level[4] - bp->level == 0)
			{
				down_f = 0;
				down[0] = blocks[bp->nei[4][0][0]].array;
				counter_same[2]++;
			}
			else if (bp->nei_level[4] - bp->level == 1)
			{
				down_f = 1;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						down[i * 2 + j] = blocks[bp->nei[4][i][j]].array;
				counter_diff[2] += 4;
			}
			else if (bp->nei_level[4] - bp->level == -1)
			{
				down_f = -1;
				m = bp->nei[4][0][0];
				down[0] = blocks[m].array;
				for (i = 0; i < 2; ++i)
					for (j = 0; j < 2; ++j)
						if (blocks[m].nei[5][i][j] == n)
							down_f -= i * 2 + j;
				counter_diff[2]++;
			}
			else
			{
				printf("ERROR: misconnected block, down: %d\n", bp->nei_level[4] - bp->level);
				exit(-1);
			}

			comm_alt(east, east_f, west, west_f,
					north, north_f, south, south_f,
					up, up_f, down, down_f,
					bp->array, start, num_comm);
		}

		return;
	}

	for (o = 0; o < 3; o++)
	{
		if (permute)
			dir = permutations[stage % 6][o];
		else
			dir = o;
		t1 = timer();

		// While values are being sent over the mesh, go through and direct
		// blocks to exchange ghost values with other blocks that are on
		// processor.  Also apply boundary conditions for boundary of domain.
		for (in = 0; in < sorted_index[num_refine + 1]; in++)
		{
			n = sorted_list[in].n;
			bp = &blocks[n];
			if (bp->number >= 0)
				for (l = dir * 2; l < (dir * 2 + 2); l++)
				{
					if (bp->nei_level[l] == bp->level)
					{
						t2 = timer();
						if ((m = bp->nei[l][0][0]) > n)
						{
							on_proc_comm(n, m, l, start, num_comm);
							counter_same[dir] += 2;
						}
						timer_comm_same[dir] += timer() - t2;
					}
					else if (bp->nei_level[l] == (bp->level + 1))
					{
						t2 = timer();
						for (i = 0; i < 2; i++)
							for (j = 0; j < 2; j++)
								if ((m = bp->nei[l][i][j]) > n)
								{
									on_proc_comm_diff(n, m, l, i, j, start, num_comm);
									counter_diff[dir] += 2;
								}
						timer_comm_diff[dir] += timer() - t2;
					}
					else if (bp->nei_level[l] == (bp->level - 1))
					{
						t2 = timer();
						if ((m = bp->nei[l][0][0]) > n)
						{
							k = dir * 2 + 1 - l % 2;
							for (i = 0; i < 2; i++)
								for (j = 0; j < 2; j++)
									if (blocks[m].nei[k][i][j] == n)
									{
										on_proc_comm_diff(m, n, k, i, j, start, num_comm);
										counter_diff[dir] += 2;
									}
						}
						timer_comm_diff[dir] += timer() - t2;
					}
					else if (bp->nei_level[l] == -2)
					{
						t2 = timer();
						apply_bc(l, bp, start, num_comm);
						counter_bc[dir]++;
						timer_comm_bc[dir] += timer() - t2;
					}
					else
					{
						printf("ERROR: misconnected block\n");
						exit(-1);
					}
				}
		}
		timer_comm_dir[dir] += timer() - t1;
	}
	timer_comm_all += timer() - t0;
}

// Routine that does on processor communication between two blocks that
// are at the same level of refinement.
void on_proc_comm(int n, int n1, int l, int start, int num_comm)
{
	typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];

	int is, ie, js, je;
	block *bp, *bp1;

	/* Determine direction and then exchange data across the face
	*/
	if (!code)
	{
		if ((l / 2) == 0)         /* West, East */
		{
			if ((l % 2) == 0)      /* West */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* East */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_EW) inout(barray1[start;num_comm], barray[start;num_comm])  \
				firstprivate(start, num_comm, x_block_size, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int j = 1; j <= y_block_size; j++)
					for (int k = 1; k <= z_block_size; k++)
					{
						array1[x_block_size + 1][j][k] = array[1][j][k];
						array[0][j][k] = array1[x_block_size][j][k];
					}
			}
		}
		else if ((l / 2) == 1)    /* South, North */
		{
			if ((l % 2) == 0)      /* South */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* North */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}
			if (stencil == 7)
			{
				is = 1;
				ie = x_block_size;
			}
			else
			{
				is = 0;
				ie = x_block_size + 1;
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_NS) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, is, ie, x_block_size, y_block_size, z_block_size) default(none)
			{
				for (int m = start; m < start + num_comm; m++)
				{
					typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
					box_t array1 = barray1[m];
					box_t array = barray[m];

					for (int i = is; i <= ie; i++)
						for (int k = 1; k <= z_block_size; k++)
						{
							array1[i][y_block_size + 1][k] = array[i][1][k];
							array[i][0][k] = array1[i][y_block_size][k];
						}
				}
			}
		}
		else if ((l / 2) == 2)    /* Down, Up */
		{
			if ((l % 2) == 0)      /* Down */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* Up */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}
			if (stencil == 7)
			{
				is = 1;
				ie = x_block_size;
				js = 1;
				je = y_block_size;
			}
			else
			{
				is = 0;
				ie = x_block_size + 1;
				js = 0;
				je = y_block_size + 1;
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_UD) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, is, ie, js, je, x_block_size, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int i = is; i <= ie; i++)
					for (int j = js; j <= je; j++)
					{
						array1[i][j][z_block_size + 1] = array[i][j][1];
						array[i][j][0] = array1[i][j][z_block_size];
					}
			}
		}
	}
	else      /* set all ghosts */
	{
		if ((l / 2) == 0)         /* West, East */
		{
			if ((l % 2) == 0)      /* West */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* East */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_allGH_EW) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, x_block_size, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int j = 0; j <= y_block_size + 1; j++)
					for (int k = 0; k <= z_block_size + 1; k++)
					{
						array1[x_block_size + 1][j][k] = array[1][j][k];
						array[0][j][k] = array1[x_block_size][j][k];
					}
			}
		}
		else if ((l / 2) == 1)    /* South, North */
		{
			if ((l % 2) == 0)      /* South */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* North */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_allGH_NS) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, x_block_size, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int i = 0; i <= x_block_size + 1; i++)
					for (int k = 0; k <= z_block_size + 1; k++)
					{
						array1[i][y_block_size + 1][k] = array[i][1][k];
						array[i][0][k] = array1[i][y_block_size][k];
					}
			}
		}
		else if ((l / 2) == 2)    /* Down, Up */
		{
			if ((l % 2) == 0)      /* Down */
			{
				bp = &blocks[n];
				bp1 = &blocks[n1];
			}
			else                   /* Up */
			{
				bp1 = &blocks[n];
				bp = &blocks[n1];
			}

			box_arr_t barray1 = (box_arr_t) bp1->array;
			box_arr_t barray = (box_arr_t) bp->array;
			#pragma omp task label(on_proc_comm_allGH_UD) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, x_block_size, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int i = 0; i <= x_block_size + 1; i++)
					for (int j = 0; j <= y_block_size + 1; j++)
					{
						array1[i][j][z_block_size + 1] = array1[i][j][1];
						array[i][j][0] = array1[i][j][z_block_size];
					}
			}
		}
	}
}

// Routine that does on processor communication between two blocks that are
// at different levels of refinement.  The order of the blocks that are
// being input determine which block is at a higher level of refinement.
void on_proc_comm_diff(int n, int n1, int l, int iq, int jq, int start, int num_comm)
{
	int i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3;
	block *bp, *bp1;

	bp = &blocks[n];
	bp1 = &blocks[n1];

	typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];
	box_arr_t barray1 = (box_arr_t) bp1->array;
	box_arr_t barray = (box_arr_t) bp->array;

	/* (iq, jq) quarter face on block n to whole face on block n1
	*/
	if (!code)
	{
		/* only have to communicate ghost values - bp is level, bp1 is level+1 -
		 * in 2 to 1 case get 0..block_half from one proc and
		 *                block_half+1..block_size+1 from another
		 * in 1 to 2 case get 0..block_size+1 from 0..block_half+1 or
		 *                block_half..block_size+1 with averages
		 */
		if ((l / 2) == 0)
		{
			if (l == 0)               /* West */
			{
				i0 = 0;
				i1 = 1;
				i2 = x_block_size + 1;
				i3 = x_block_size;
			}
			else                      /* East */
			{
				i0 = x_block_size + 1;
				i1 = x_block_size;
				i2 = 0;
				i3 = 1;
			}
			j1 = jq * y_block_half;
			k1 = iq * z_block_half;
			#pragma omp task label(on_proc_comm_diff_EW) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, i0, i1, i2, i3, j1, k1, y_block_half, z_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int j = 1; j <= y_block_half; j++)
					for (int k = 1; k <= z_block_half; k++)
					{
						array1[i2][2 * j - 1][2 * k - 1] =
						    array1[i2][2 * j - 1][2 * k  ] =
						        array1[i2][2 * j  ][2 * k - 1] =
						            array1[i2][2 * j  ][2 * k  ] =
						                array[i1][j + j1][k + k1] / 4.0;
						array[i0][j + j1][k + k1] =
						    array1[i3][2 * j - 1][2 * k - 1] +
						    array1[i3][2 * j - 1][2 * k  ] +
						    array1[i3][2 * j  ][2 * k - 1] +
						    array1[i3][2 * j  ][2 * k  ];
					}
			}
		}
		else if ((l / 2) == 1)
		{
			if (l == 2)               /* South */
			{
				j0 = 0;
				j1 = 1;
				j2 = y_block_size + 1;
				j3 = y_block_size;
			}
			else                      /* North */
			{
				j0 = y_block_size + 1;
				j1 = y_block_size;
				j2 = 0;
				j3 = 1;
			}
			i1 = jq * x_block_half;
			k1 = iq * z_block_half;
			#pragma omp task label(on_proc_comm_diff_NS) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, j0, j1, j2, j3, i1, k1, x_block_half, z_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int i = 1; i <= x_block_half; i++)
					for (int k = 1; k <= z_block_half; k++)
					{
						array1[2 * i - 1][j2][2 * k - 1] =
						    array1[2 * i - 1][j2][2 * k  ] =
						        array1[2 * i  ][j2][2 * k - 1] =
						            array1[2 * i  ][j2][2 * k  ] =
						                array[i + i1][j1][k + k1] / 4.0;
						array[i + i1][j0][k + k1] =
						    array1[2 * i - 1][j3][2 * k - 1] +
						    array1[2 * i - 1][j3][2 * k  ] +
						    array1[2 * i  ][j3][2 * k - 1] +
						    array1[2 * i  ][j3][2 * k  ];
					}
			}
		}
		else if ((l / 2) == 2)
		{
			if (l == 4)               /* Down */
			{
				k0 = 0;
				k1 = 1;
				k2 = z_block_size + 1;
				k3 = z_block_size;
			}
			else                      /* Up */
			{
				k0 = z_block_size + 1;
				k1 = z_block_size;
				k2 = 0;
				k3 = 1;
			}
			i1 = jq * x_block_half;
			j1 = iq * y_block_half;
			#pragma omp task label(on_proc_comm_diff_UD) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, k0, k1, k2, k3, i1, j1, x_block_half, y_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				for (int i = 1; i <= x_block_half; i++)
					for (int j = 1; j <= y_block_half; j++)
					{
						array1[2 * i - 1][2 * j - 1][k2] =
						    array1[2 * i - 1][2 * j  ][k2] =
						        array1[2 * i  ][2 * j - 1][k2] =
						            array1[2 * i  ][2 * j  ][k2] =
						                array[i + i1][j + j1][k1] / 4.0;
						array[i + i1][j + j1][k0] =
						    array1[2 * i - 1][2 * j - 1][k3] +
						    array1[2 * i - 1][2 * j  ][k3] +
						    array1[2 * i  ][2 * j - 1][k3] +
						    array1[2 * i  ][2 * j  ][k3];
					}
			}
		}
	}
	else      /* transfer ghosts */
	{
		if ((l / 2) == 0)
		{
			if (l == 0)               /* West */
			{
				i0 = 0;
				i1 = 1;
				i2 = x_block_size + 1;
				i3 = x_block_size;
			}
			else                      /* East */
			{
				i0 = x_block_size + 1;
				i1 = x_block_size;
				i2 = 0;
				i3 = 1;
			}
			j1 = jq * y_block_half;
			k1 = iq * z_block_half;
			j2 = y_block_size + 1;
			j3 = y_block_half + 1;
			k2 = z_block_size + 1;
			k3 = z_block_half + 1;
			#pragma omp task label(on_proc_comm_diff_allGH_EW) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, iq, jq, i0, i1, i2, i3, j1, j2, j3, k1, k2, k3,\
							y_block_half, z_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				array1[i2][0][0] = array[i1][j1][k1] / 4.0;
				for (int k = 1; k <= z_block_half; k++)
					array1[i2][0][2 * k - 1] =
					    array1[i2][0][2 * k  ] = array[i1][j1][k + k1] / 4.0;
				array1[i2][0][k2] = array[i1][j1][k3 + k1] / 4.0;
				if (jq == 0)
				{
					if (iq == 0)
						array[i0][0][0 ] = array1[i3][0][0 ];
					else
						array[i0][0][k2] = array1[i3][0][k2];
					for (int k = 1; k <= z_block_half; k++)
						array[i0][0][k + k1] = (array1[i3][0][2 * k - 1] +
						        array1[i3][0][2 * k  ]);
				}
				for (int j = 1; j <= y_block_half; j++)
				{
					array1[i2][2 * j - 1][0] =
					    array1[i2][2 * j  ][0] = array[i1][j + j1][k1] / 4.0;
					if (iq == 0)
						array[i0][j + j1][0 ] = (array1[i3][2 * j - 1][0 ] +
						        array1[i3][2 * j  ][0 ]);
					else
						array[i0][j + j1][k2] = (array1[i3][2 * j - 1][k2] +
						        array1[i3][2 * j  ][k2]);
					for (int k = 1; k <= z_block_half; k++)
					{
						array1[i2][2 * j - 1][2 * k - 1] =
						    array1[i2][2 * j - 1][2 * k  ] =
						        array1[i2][2 * j  ][2 * k - 1] =
						            array1[i2][2 * j  ][2 * k  ] =
						                array[i1][j + j1][k + k1] / 4.0;
						array[i0][j + j1][k + k1] =
						    array1[i3][2 * j - 1][2 * k - 1] +
						    array1[i3][2 * j - 1][2 * k  ] +
						    array1[i3][2 * j  ][2 * k - 1] +
						    array1[i3][2 * j  ][2 * k  ];
					}
					array1[i2][2 * j - 1][k2] =
					    array1[i2][2 * j  ][k2] = array[i1][j + j1][k3 + k1] / 4.0;
				}
				array1[i2][j2][0] = array[i1][j3 + j1][k1] / 4.0;
				for (int k = 1; k <= z_block_half; k++)
					array1[i2][j2][2 * k - 1] =
					    array1[i2][j2][2 * k  ] = array[i1][j3 + j1][k + k1] / 4.0;
				array1[i2][j2][k2] = array[i1][j3 + j1][k3 + k1] / 4.0;
				if (jq == 1)
				{
					if (iq == 0)
						array[i0][j2][0 ] = array1[i3][j2][0 ];
					else
						array[i0][j2][k2] = array1[i3][j2][k2];
					for (int k = 1; k <= z_block_half; k++)
						array[i0][j2][k + k1] = (array1[i3][j2][2 * k - 1] +
						        array1[i3][j2][2 * k  ]);
				}
			}
		}
		else if ((l / 2) == 1)
		{
			if (l == 2)               /* South */
			{
				j0 = 0;
				j1 = 1;
				j2 = y_block_size + 1;
				j3 = y_block_size;
			}
			else                      /* North */
			{
				j0 = y_block_size + 1;
				j1 = y_block_size;
				j2 = 0;
				j3 = 1;
			}
			i1 = jq * x_block_half;
			k1 = iq * z_block_half;
			i2 = x_block_size + 1;
			i3 = x_block_half + 1;
			k2 = z_block_size + 1;
			k3 = z_block_half + 1;
			#pragma omp task label(on_proc_comm_diff_allGH_NS) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, iq, jq, j0, j1, j2, j3, i1, i2, i3, k1, k2, k3,\
							x_block_half, z_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				array1[0][j2][0 ] = array[i1][j1][k1] / 4.0;
				for (int k = 1; k <= z_block_half; k++)
					array1[0][j2][2 * k - 1] =
					    array1[0][j2][2 * k  ] = array[i1][j1][k + k1] / 4.0;
				array1[0][j2][k2] = array[i1][j1][k3 + k1] / 4.0;
				if (jq == 0)
				{
					if (iq == 0)
						array[0][j0][0 ] = array1[0][j3][0 ];
					else
						array[0][j0][k2] = array1[0][j3][k2];
					for (int k = 1; k <= z_block_half; k++)
						array[0][j0][k + k1] = (array1[0][j3][2 * k - 1] +
						        array1[0][j3][2 * k  ]);
				}
				for (int i = 1; i <= x_block_half; i++)
				{
					array1[2 * i - 1][j2][0] =
					    array1[2 * i  ][j2][0] = array[i + i1][j1][k1] / 4.0;
					if (iq == 0)
						array[i + i1][j0][0 ] = (array1[2 * i - 1][j3][0 ] +
						        array1[2 * i  ][j3][0 ]);
					else
						array[i + i1][j0][k2] = (array1[2 * i - 1][j3][k2] +
						        array1[2 * i  ][j3][k2]);
					for (int k = 1; k <= z_block_half; k++)
					{
						array1[2 * i - 1][j2][2 * k - 1] =
						    array1[2 * i - 1][j2][2 * k  ] =
						        array1[2 * i  ][j2][2 * k - 1] =
						            array1[2 * i  ][j2][2 * k  ] =
						                array[i + i1][j1][k + k1] / 4.0;
						array[i + i1][j0][k + k1] =
						    array1[2 * i - 1][j3][2 * k - 1] +
						    array1[2 * i - 1][j3][2 * k  ] +
						    array1[2 * i  ][j3][2 * k - 1] +
						    array1[2 * i  ][j3][2 * k  ];
					}
					array1[2 * i - 1][j2][k2] =
					    array1[2 * i  ][j2][k2] = array[i + i1][j1][k3 + k1] / 4.0;
				}
				array1[i2][j2][0 ] = array[i3 + i1][j1][k1] / 4.0;
				for (int k = 1; k <= z_block_half; k++)
					array1[i2][j2][2 * k - 1] =
					    array1[i2][j2][2 * k  ] = array[i3 + i1][j1][k + k1] / 4.0;
				array1[i2][j2][k2] = array[i3 + i1][j1][k3 + k1] / 4.0;
				if (jq == 1)
				{
					if (iq == 0)
						array[i2][j0][0 ] = array1[i2][j3][0 ];
					else
						array[i2][j0][k2] = array1[i2][j3][k2];
					for (int k = 1; k <= z_block_half; k++)
						array[i2][j0][k + k1] = (array1[i2][j3][2 * k - 1] +
						        array1[i2][j3][2 * k  ]);
				}
			}
		}
		else if ((l / 2) == 2)
		{
			if (l == 4)               /* Down */
			{
				k0 = 0;
				k1 = 1;
				k2 = z_block_size + 1;
				k3 = z_block_size;
			}
			else                      /* Up */
			{
				k0 = z_block_size + 1;
				k1 = z_block_size;
				k2 = 0;
				k3 = 1;
			}
			i1 = jq * x_block_half;
			j1 = iq * y_block_half;
			i2 = x_block_size + 1;
			i3 = x_block_half + 1;
			j2 = y_block_size + 1;
			j3 = y_block_half + 1;
			#pragma omp task label(on_proc_comm_diff_allGH_UD) inout(barray1[start;num_comm], barray[start;num_comm]) \
				firstprivate(start, num_comm, iq, jq, k0, k1, k2, k3, i1, i2, i3, j1, j2, j3, \
							x_block_half, y_block_half, y_block_size, z_block_size) default(none)
			for (int m = start; m < start + num_comm; m++)
			{
				typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
				box_t array1 = barray1[m];
				box_t array = barray[m];

				array1[0][0 ][k2] = array[i1][j1][k1] / 4.0;
				for (int j = 1; j <= y_block_half; j++)
					array1[0][2 * j - 1][k2] =
					    array1[0][2 * j  ][k2] = array[i1][j + j1][k1] / 4.0;
				array1[0][j2][k2] = array[i1][j3 + j1][k1] / 4.0;
				if (jq == 0)
				{
					if (iq == 0)
						array[0][0 ][k0] = array1[0][0 ][k3];
					else
						array[0][j2][k0] = array1[0][j2][k3];
					for (int j = 1; j <= y_block_half; j++)
						array[0][j + j1][k0] = (array1[0][2 * j - 1][k3] +
						        array1[0][2 * j  ][k3]);
				}
				for (int i = 1; i <= x_block_half; i++)
				{
					array1[2 * i - 1][0][k2] =
					    array1[2 * i  ][0][k2] = array[i + i1][j1][k1] / 4.0;
					if (iq == 0)
						array[i + i1][0][k0] = (array1[2 * i - 1][0][k3] +
						        array1[2 * i  ][0][k3]);
					else
						array[i + i1][j2][k0] = (array1[2 * i - 1][j2][k3] +
						        array1[2 * i  ][j2][k3]);
					for (int j = 1; j <= y_block_half; j++)
					{
						array1[2 * i - 1][2 * j - 1][k2] =
						    array1[2 * i - 1][2 * j  ][k2] =
						        array1[2 * i  ][2 * j - 1][k2] =
						            array1[2 * i  ][2 * j  ][k2] =
						                array[i + i1][j + j1][k1] / 4.0;
						array[i + i1][j + j1][k0] =
						    array1[2 * i - 1][2 * j - 1][k3] +
						    array1[2 * i - 1][2 * j  ][k3] +
						    array1[2 * i  ][2 * j - 1][k3] +
						    array1[2 * i  ][2 * j  ][k3];
					}
					array1[2 * i - 1][j2][k2] =
					    array1[2 * i  ][j2][k2] = array[i + i1][j3 + j1][k1] / 4.0;
				}
				array1[i2][0 ][k2] = array[i3 + i1][j1][k1] / 4.0;
				for (int j = 1; j <= y_block_half; j++)
					array1[i2][2 * j - 1][k2] =
					    array1[i2][2 * j  ][k2] = array[i3 + i1][j + j1][k1] / 4.0;
				array1[i2][j2][k2] = array[i3 + i1][j3 + j1][k1] / 4.0;
				if (jq == 1)
				{
					if (iq == 0)
						array[i2][0 ][k0] = array1[i2][0 ][k3];
					else
						array[i2][j2][k0] = array1[i2][j2][k3];
					for (int j = 1; j <= y_block_half; j++)
						array[i2][j + j1][k0] = (array1[i2][2 * j - 1][k3] +
						        array1[i2][2 * j  ][k3]);
				}
			}
		}
	}
}

// Apply reflective boundary conditions to a face of a block.
void apply_bc(int l, block *bp, int start, int num_comm)
{
	typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];

	box_arr_t barray = (box_arr_t)bp->array;
	#pragma omp task label(apply_bc) inout(barray[start;num_comm]) \
		firstprivate(start, num_comm, l, code, stencil, x_block_size, y_block_size, z_block_size) default(none)
	{
		typedef double (*box_t)[y_block_size + 2][z_block_size + 2];
		int t = 0;
		int f = 1;

		if (!code && stencil == 7)
			switch (l)
			{
				case 1:
					t = x_block_size + 1;
					f = x_block_size;
				case 0:
					// TODO try: for(box_t array = barray[start]; array != barray[start + num_comm]; array++)
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int j = 1; j <= y_block_size; j++)
							for (int k = 1; k <= z_block_size; k++)
								array[t][j][k] = array[f][j][k];
					}
					break;

				case 3:
					t = y_block_size + 1;
					f = y_block_size;
				case 2:
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int i = 1; i <= x_block_size; i++)
							for (int k = 1; k <= z_block_size; k++)
								array[i][t][k] = array[i][f][k];
					}
					break;

				case 5:
					t = z_block_size + 1;
					f = z_block_size;
				case 4:
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int i = 1; i <= x_block_size; i++)
							for (int j = 1; j <= y_block_size; j++)
								array[i][j][t] = array[i][j][f];
					}
					break;
			}
		else
			switch (l)
			{
				case 1:
					t = x_block_size + 1;
					f = x_block_size;
				case 0:
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int j = 0; j <= y_block_size + 1; j++)
							for (int k = 0; k <= z_block_size + 1; k++)
								array[t][j][k] = array[f][j][k];
					}
					break;

				case 3:
					t = y_block_size + 1;
					f = y_block_size;
				case 2:
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int i = 0; i <= x_block_size + 1; i++)
							for (int k = 0; k <= z_block_size + 1; k++)
								array[i][t][k] = array[i][f][k];
					}
					break;

				case 5:
					t = z_block_size + 1;
					f = z_block_size;
				case 4:
					for (int var = start; var < start + num_comm; var++)
					{
						box_t array = barray[var];
						for (int i = 0; i <= x_block_size + 1; i++)
							for (int j = 0; j <= y_block_size + 1; j++)
								array[i][j][t] = array[i][j][f];
					}
					break;
			}
	}
}
