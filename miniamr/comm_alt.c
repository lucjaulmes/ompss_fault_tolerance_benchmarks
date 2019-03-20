#include <stdlib.h>
#include <stdio.h>
#include "block.h"
#include "proto.h"

static inline void comm_eq(double *from,
    double *to,
    int itot, int iinc,
    int jtot, int jinc)
{
	for (int i = 0; i < itot * iinc; i += iinc)
		for (int j = 0; j < jtot * jinc; j += jinc)
			to[i + j] = from[i + j];
}

static inline void comm_finetocoarse(double *from,
    double *to,
    int itot, int iinc,
    int jtot, int jinc)
{
	for (int i = 0; i < itot * iinc; i += iinc)
		for (int j = 0; j < jtot * jinc; j += jinc)
			to[i + j] =
			    (from[i * 2 + j * 2] +
			        from[i * 2 + iinc + j * 2 + jinc]) +
			    (from[i * 2 + j * 2 + jinc] +
			        from[i * 2 + iinc + j * 2]);
}

static inline void comm_coarsetofine(double *from,
    double *to,
    int itot, int iinc,
    int jtot, int jinc)
{
	for (int i = 0; i < itot * iinc; i += iinc)
		for (int j = 0; j < jtot * jinc; j += jinc)
			to[i * 2 + j * 2] =
			    to[i * 2 + j * 2 + jinc] =
			        to[i * 2 + iinc + j * 2] =
			            to[i * 2 + iinc + j * 2 + jinc] =
			                from[i + j] / 4.0;
}

void comm_alt(double *east_flat[4], int east_f,
    double *west_flat[4], int west_f,
    double *north_flat[4], int north_f,
    double *south_flat[4], int south_f,
    double *up_flat[4], int up_f,
    double *down_flat[4], int down_f,
    double *b_array_flat,
    int start, int number)
{
	typedef double (*box_arr_t)[x_block_size+2][y_block_size+2][z_block_size+2];

	int east_c = east_f == 1 ? 4 : 1;
	int west_c = west_f == 1 ? 4 : 1;
	int north_c = north_f == 1 ? 4 : 1;
	int south_c = south_f == 1 ? 4 : 1;
	int up_c = up_f == 1 ? 4 : 1;
	int down_c = down_f == 1 ? 4 : 1;

    box_arr_t *east = (box_arr_t*)east_flat;
    box_arr_t *west = (box_arr_t*)west_flat;
    box_arr_t *north = (box_arr_t*)north_flat;
    box_arr_t *south = (box_arr_t*)south_flat;
    box_arr_t *up = (box_arr_t*)up_flat;
    box_arr_t *down = (box_arr_t*)down_flat;
    box_arr_t b_array = (box_arr_t)b_array_flat;

	#pragma omp task label(comm_alt) inout(b_array[start;number]) \
        in({(east[i])[start;number],  i=0;east_c},  {(west[i])[start;number],  i=0;west_c}, \
           {(north[i])[start;number], i=0;north_c}, {(south[i])[start;number], i=0;south_c}, \
           {(up[i])[start;number],    i=0;up_c},    {(down[i])[start;number],  i=0;down_c}) \
	    firstprivate(start, number, east_f, west_f, north_f, south_f, up_f, down_f, east_c, west_c, north_c, south_c, up_c, down_c)
	{
		// EAST -> CENTER
		if (east_f == 2)
		{
			for (int var = start; var < start + number; ++var)
				comm_eq(&((east[0][var])[x_block_size][1][1]),
				    &((b_array[var])[x_block_size + 1][1][1]),
				    y_block_size, z_block_size + 2,
				    z_block_size, 1);
		}
		else if (east_f == 0)
		{
			for (int var = start; var < start + number; ++var)
				comm_eq(&((east[0][var])[1][1][1]),
				    &((b_array[var])[x_block_size + 1][1][1]),
				    y_block_size, z_block_size + 2,
				    z_block_size, 1);
		}
		else if (east_f < 0)
		{
			east_f = -east_f - 1;
			int ystart = 1 + (east_f % 2) * y_block_half;
			int zstart = 1 + (east_f / 2) * z_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((east[0][var])[1][ystart][zstart]),
				    &((b_array[var])[x_block_size + 1][1][1]),
				    y_block_half, z_block_size + 2,
				    z_block_half, 1);
		}
		else
		{
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int ystart = 1 + (i % 2) * y_block_half;
					int zstart = 1 + (i / 2) * z_block_half;
					comm_finetocoarse(&((east[i][var])[1][1][1]),
					    &((b_array[var])[x_block_size + 1][ystart][zstart]),
					    y_block_half, z_block_size + 2,
					    z_block_half, 1);
				}
		}

		fflush(stdout);

		// WEST -> CENTER
		if (west_f == 2)
		{
			for (int var = start; var < start + number; ++var)
				comm_eq(&((west[0][var])[1][1][1]),
				    &((b_array[var])[0][1][1]),
				    y_block_size, z_block_size + 2,
				    z_block_size, 1);
		}
		else if (west_f == 0)
		{
			for (int var = start; var < start + number; ++var)
				comm_eq(&((west[0][var])[x_block_size][1][1]),
				    &((b_array[var])[0][1][1]),
				    y_block_size, z_block_size + 2,
				    z_block_size, 1);
		}
		else if (west_f < 0)
		{
			west_f = -west_f - 1;
			int ystart = 1 + (west_f % 2) * y_block_half;
			int zstart = 1 + (west_f / 2) * z_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((west[0][var])[x_block_size][ystart][zstart]),
				    &((b_array[var])[0][1][1]),
				    y_block_half, z_block_size + 2,
				    z_block_half, 1);
		}
		else
		{
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int ystart = 1 + (i % 2) * y_block_half;
					int zstart = 1 + (i / 2) * z_block_half;
					comm_finetocoarse(&((west[i][var])[x_block_size][1][1]),
					    &((b_array[var])[0][ystart][zstart]),
					    y_block_half, z_block_size + 2,
					    z_block_half, 1);
				}
		}

		fflush(stdout);

		// NORTH -> CENTER
		if (north_f == 2)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((north[0][var])[1][y_block_size][1]),
				    &((b_array[var])[1][y_block_size + 1][1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    z_block_size, 1);
		else if (north_f == 0)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((north[0][var])[1][1][1]),
				    &((b_array[var])[1][y_block_size + 1][1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    z_block_size, 1);
		else if (north_f < 0)
		{
			north_f = -north_f - 1;
			int xstart = 1 + (north_f % 2) * x_block_half;
			int zstart = 1 + (north_f / 2) * z_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((north[0][var])[xstart][1][zstart]),
				    &((b_array[var])[1][y_block_size + 1][1]),
				    x_block_half, (z_block_size + 2) * (y_block_size + 2),
				    z_block_half, 1);
		}
		else
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int xstart = 1 + (i % 2) * x_block_half;
					int zstart = 1 + (i / 2) * z_block_half;
					comm_finetocoarse(&((north[i][var])[1][1][1]),
					    &((b_array[var])[xstart][y_block_size + 1][zstart]),
					    x_block_half, (z_block_size + 2) * (y_block_size + 2),
					    z_block_half, 1);
				}

		// SOUTH -> CENTER
		if (south_f == 2)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((south[0][var])[1][1][1]),
				    &((b_array[var])[1][0][1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    z_block_size, 1);
		else if (south_f == 0)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((south[0][var])[1][y_block_size][1]),
				    &((b_array[var])[1][0][1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    z_block_size, 1);
		else if (south_f < 0)
		{
			south_f = -south_f - 1;
			int xstart = 1 + (south_f % 2) * x_block_half;
			int zstart = 1 + (south_f / 2) * z_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((south[0][var])[xstart][y_block_size][zstart]),
				    &((b_array[var])[1][0][1]),
				    x_block_half, (z_block_size + 2) * (y_block_size + 2),
				    z_block_half, 1);
		}
		else
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int xstart = 1 + (i % 2) * x_block_half;
					int zstart = 1 + (i / 2) * z_block_half;
					comm_finetocoarse(&((south[i][var])[1][y_block_size][1]),
					    &((b_array[var])[xstart][0][zstart]),
					    x_block_half, (z_block_size + 2) * (y_block_size + 2),
					    z_block_half, 1);
				}

		// UP -> CENTER
		if (up_f == 2)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((up[0][var])[1][1][z_block_size]),
				    &((b_array[var])[1][1][z_block_size + 1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    y_block_size, z_block_size + 2);
		else if (up_f == 0)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((up[0][var])[1][1][1]),
				    &((b_array[var])[1][1][z_block_size + 1]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    y_block_size, z_block_size + 2);
		else if (up_f < 0)
		{
			up_f = -up_f - 1;
			int xstart = 1 + (up_f % 2) * x_block_half;
			int ystart = 1 + (up_f / 2) * y_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((up[0][var])[xstart][ystart][1]),
				    &((b_array[var])[1][1][z_block_size + 1]),
				    x_block_half, (z_block_size + 2) * (y_block_size + 2),
				    y_block_half, z_block_size + 2);
		}
		else
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int xstart = 1 + (i % 2) * x_block_half;
					int ystart = 1 + (i / 2) * y_block_half;
					comm_finetocoarse(&((up[i][var])[1][1][1]),
					    &((b_array[var])[xstart][ystart][z_block_size + 1]),
					    x_block_half, (z_block_size + 2) * (y_block_size + 2),
					    y_block_half, z_block_size + 2);
				}

		// DOWN -> CENTER
		if (down_f == 2)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((down[0][var])[1][1][1]),
				    &((b_array[var])[1][1][0]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    y_block_size, z_block_size + 2);
		else if (down_f == 0)
			for (int var = start; var < start + number; ++var)
				comm_eq(&((down[0][var])[1][1][z_block_size]),
				    &((b_array[var])[1][1][0]),
				    x_block_size, (z_block_size + 2) * (y_block_size + 2),
				    y_block_size, z_block_size + 2);
		else if (down_f < 0)
		{
			down_f = -down_f - 1;
			int xstart = 1 + (down_f % 2) * x_block_half;
			int ystart = 1 + (down_f / 2) * y_block_half;
			for (int var = start; var < start + number; ++var)
				comm_coarsetofine(&((down[0][var])[xstart][ystart][z_block_size]),
				    &((b_array[var])[1][1][0]),
				    x_block_half, (z_block_size + 2) * (y_block_size + 2),
				    y_block_half, z_block_size + 2);
		}
		else
			for (int var = 0; var < start + number; ++var)
				for (int i = 0; i < 4; ++i)
				{
					int xstart = 1 + (i % 2) * x_block_half;
					int ystart = 1 + (i / 2) * y_block_half;
					comm_finetocoarse(&((down[i][var])[1][1][z_block_size]),
					    &((b_array[var])[xstart][ystart][0]),
					    x_block_half, (z_block_size + 2) * (y_block_size + 2),
					    y_block_half, z_block_size + 2);

					// printf("Fine to coarse %d %d\n", xstart, ystart);
				}


		free(north);
		free(south);
		free(east);
		free(west);
		free(up);
		free(down);
	}
}

