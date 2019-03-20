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
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#if _POSIX_C_SOURCE < 199309L
	#include <sys/time.h>
#endif

#include "block.h"
#include "proto.h"
#include "timer.h"

static inline double wtime()
{
	#if _POSIX_C_SOURCE < 199309L
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (tv.tv_sec + 1e-6 * tv.tv_usec);
	#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (ts.tv_sec + 1e-9 * ts.tv_nsec);
	#endif
}

double timer()
{
	// return((((double) clock())/((double) CLOCKS_PER_SEC)));
	return wtime();
}



// Write block information (level and center) to plot file.
void plot(int ts)
{
	double t = timer();

	char fname[20];
	snprintf(fname, sizeof(fname), "plot.%d", ts);
	FILE *fp = fopen(fname, "w");

	int total_num_blocks = 0;
	for (int i = 0; i <= num_refine; i++)
		total_num_blocks += num_blocks[i];

	fprintf(fp, "%d %d %d %d %d\n", total_num_blocks, num_refine,
	    npx * init_block_x, npy * init_block_y, npz * init_block_z);
	fprintf(fp, "%d\n", num_active);

	block *bp = blocks;
	for (int n = 0; n < max_active_block; n++, bp++)
		if (bp->number >= 0)
			fprintf(fp, "%d %d %d %d\n", bp->level, bp->cen[0], bp->cen[1], bp->cen[2]);

	fclose(fp);

	timer_plot += timer() - t;
}
