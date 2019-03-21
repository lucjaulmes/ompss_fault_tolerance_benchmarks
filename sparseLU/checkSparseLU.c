#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <err.h>
#include <string.h>
#include <assert.h>

#include "catchroi.h"

#define FLOATS_PER_LINE 15

#ifdef SPU_CODE
#include <spu_intrinsics.h>
#endif

typedef float (*p_block_t)[];

p_block_t allocate_clean_block_instrumented(int BS, int i, int j, int it)
{
	//printf("A_%d,%d (%d)\n", i, j, it);
	p_block_t p = CATCHROI_INSTRUMENT(calloc)(BS * BS, sizeof(float));
	if (p == NULL)
		err(1, "Can not allocate clean block");
	return p;
}


p_block_t allocate_clean_block(int BS)
{
	p_block_t p = calloc(BS * BS, sizeof(float));
	if (p == NULL)
		err(1, "Can not allocate clean block");
	return p;
}


void genmat(int NB, int BS, p_block_t A[NB][NB])
{
	int i, j, ii, jj;

	/* structure */
	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
		{
			if (ii == jj || ii == jj - 1 || ii - 1 == jj || (
				/* both even and sub-diagonal is a multiple of 3 */
				(ii % 2 == 0) && (jj % 2 == 0)
				&& (ii > jj || (ii % 3 == 0))
				&& (ii < jj || (jj % 3 == 0))
			))
				A[ii][jj] = allocate_clean_block_instrumented(BS, ii, jj, -1);
			else
				A[ii][jj] = NULL;
		}

	/* Initialization */
	long int init_val = 1325;
	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
		{
			float (*p)[BS] = A[ii][jj];

			if (p == NULL)
				continue;

			for (i = 0; i < BS; i++)
				for (j = 0; j < BS; j++)
				{
					init_val = (3125 * init_val) % 65536;
					p[i][j] = 0.0001;
					if (ii == jj)
					{
						if (i == j)
							p[i][j] = -20000 ;
						if (i - 1 == j || i == j - 1)
							p[i][j] = 10000 ;
					}
				}
		}
}


#pragma omp task in([BS]ref_block, [BS]to_comp) reduction(+: [BS]col_diffs)
void sum_col_diffs(int BS, float (*ref_block)[BS], float (*to_comp)[BS], float *col_diffs)
{
	for (int i = 0; i < BS; i++)
		for (int j = 0; j < BS; j++)
			col_diffs[j] += fabsf(ref_block[i][j] - to_comp[i][j]);
}

#pragma omp task in([BS]ref_block, [BS]to_comp) out(*max_diff)
void maxdiff(int BS, float (*ref_block)[BS], float (*to_comp)[BS], float *max_diff)
{
	for (int i = 0; i < BS; i++)
		for (int j = 0; j < BS; j++)
		{
			float diff = fabsf(ref_block[i][j] - to_comp[i][j]);
			if (*max_diff < diff)
				*max_diff = diff;
		}
}


#pragma omp task in([BS]block) reduction(+: [BS]col_sums)
void sum_cols(int BS, float (*block)[BS], float *col_sums)
{
	for (int i = 0; i < BS; i++)
		for (int j = 0; j < BS; j++)
			col_sums[j] += fabsf(block[i][j]);
}


int compare_mat(int NB, int BS, p_block_t X[NB][NB], p_block_t Y[NB][NB], struct timeval *stop)
{
	p_block_t zero_block = allocate_clean_block(BS);

	float *colsum_diff = malloc(BS * sizeof(*colsum_diff)), *colsum_A = malloc(BS * sizeof(*colsum_diff));
	float colmax_diff = 0.0, colmax_A = 0.0, max_diff = 0.0;

	for (int jj = 0; jj < NB; jj++)
	{
		// Compute the infinity norm of X and |X-Y|, batched per column blocks
		memset(colsum_diff, 0, BS * sizeof(*colsum_diff));
		memset(colsum_A, 0, BS * sizeof(*colsum_A));
		for (int ii = 0; ii < NB; ii++)
		{
			if (X[ii][jj] || Y[ii][jj])
			{
				sum_col_diffs(BS, X[ii][jj] ?: zero_block, Y[ii][jj] ?: zero_block, colsum_diff);
				maxdiff(BS, X[ii][jj] ?: zero_block, Y[ii][jj] ?: zero_block, &max_diff);
			}

			if (X[ii][jj])
				sum_cols(BS, X[ii][jj], colsum_A);
		}

		#pragma omp taskwait // on colsum.... ?
		for (int j = 0; j < BS; j++)
		{
			if (colmax_diff < colsum_diff[j])
				colmax_diff = colsum_diff[j];
			if (colmax_A < colsum_A[j])
				colmax_A = colsum_A[j];
		}
	}

	free(colsum_diff);
	free(colsum_A);

	float final = colmax_diff / (colmax_A * NB * BS);
	printf("-- ||LU-A||_inf/(||A||_inf.n.eps) = %e\n", final / FLT_EPSILON);

	return isnan(final) || isinf(final) || (final > 60.0 * FLT_EPSILON);
}


#pragma omp task inout([BS]diag)
void lu0(int BS, float (*diag)[BS])
{
	int i, j, k;

	for (k = 0; k < BS; k++)
		for (i = k + 1; i < BS; i++)
		{
			diag[i][k] = diag[i][k] / diag[k][k];
			for (j = k + 1; j < BS; j++)
				diag[i][j] -= diag[i][k] * diag[k][j];
		}

}

#pragma omp task in([BS]diag) inout([BS]row)
void bdiv(int BS, float (*diag)[BS], float (*row)[BS])
{
	int i, j, k;

	for (i = 0; i < BS; i++)
		for (k = 0; k < BS; k++)
		{
			row[i][k] = row[i][k] / diag[k][k];
			for (j = k + 1; j < BS; j++)
				row[i][j] -= row[i][k] * diag[k][j];
		}
}


#pragma omp task in([BS]row, [BS]col) inout([BS]inner)
void bmod(int BS, float (*row)[BS], float (*col)[BS], float (*inner)[BS])
{
	int i, j, k;

	for (i = 0; i < BS; i++)
		for (j = 0; j < BS; j++)
			for (k = 0; k < BS; k++)
				inner[i][j] -= row[i][k] * col[k][j];

}

#pragma omp task in([BS]a, [BS]b) commutative([BS]c)
void block_mpy_add(int BS, float (*a)[BS], float (*b)[BS], float (*c)[BS])
{
	int i, j, k;

	for (i = 0; i < BS; i++)
		for (j = 0; j < BS; j++)
			for (k = 0; k < BS; k++)
				c[i][j] += a[i][k] * b[k][j];
}


#pragma omp task in([BS]diag) inout([BS]col)
void fwd(int BS, float (*diag)[BS], float (*col)[BS])
{
	int i, j, k;

	for (j = 0; j < BS; j++)
		for (k = 0; k < BS; k++)
			for (i = k + 1; i < BS; i++)
				col[i][j] -= diag[i][k] * col[k][j];

}

#pragma omp task in([BS]A) out([BS]L, [BS]U)
void split_block(int BS, float (*A)[BS], float (*L)[BS], float (*U)[BS])
{
	int i, j;

	for (i = 0; i < BS; i++)
		for (j = 0; j < BS; j++)
		{
			if (i == j)     L[i][j] = 1.0,      U[i][j] = A[i][j];
			else if (i > j) L[i][j] = A[i][j],  U[i][j] = 0.0;
			else            L[i][j] = 0.0,      U[i][j] = A[i][j];
		}
}


#pragma omp task in([BS]src) out([BS]dst)
void copy_block(int BS, float (*src)[BS], float (*dst)[BS])
{
	memcpy(dst, src, BS * BS * sizeof(**dst));
}

#pragma omp task out([BS]dst)
void clean_block(int BS, float (*dst)[BS])
{
	memset(dst, 0, BS * BS * sizeof(**dst));
}

void LU(int NB, int BS, p_block_t A[NB][NB])
{
	int ii, jj, kk;

	for (kk = 0; kk < NB; kk++)
	{
		lu0(BS, A[kk][kk]);

		for (jj = kk + 1; jj < NB; jj++)
			if (A[kk][jj] != NULL)
				fwd(BS, A[kk][kk], A[kk][jj]);

		for (ii = kk + 1; ii < NB; ii++)
			if (A[ii][kk] != NULL)
				bdiv(BS, A[kk][kk], A[ii][kk]);

		for (ii = kk + 1; ii < NB; ii++)
			if (A[ii][kk] != NULL)
				for (jj = kk + 1; jj < NB; jj++)
					if (A[kk][jj] != NULL)
					{
						if (A[ii][jj] == NULL)
							A[ii][jj] = allocate_clean_block_instrumented(BS, ii, jj, kk);
						bmod(BS, A[ii][kk], A[kk][jj], A[ii][jj]);
					}
	}

	#pragma omp taskwait
}

void split_mat(int NB, int BS, p_block_t LU[NB][NB], p_block_t L[NB][NB], p_block_t U[NB][NB])
{
	int ii, jj;

	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
		{
			/* split diagonal block */
			if (ii == jj)
			{
				L[ii][ii] = allocate_clean_block(BS);
				U[ii][ii] = allocate_clean_block(BS);
				split_block(BS, LU[ii][ii], L[ii][ii], U[ii][ii]);
			}
			else
			{
				/* copy non diagonal block to L or U */
				if (LU[ii][jj] == NULL)
				{
					L[ii][jj] = NULL;
					U[ii][jj] = NULL;
				}
				else if (ii > jj)
				{
					L[ii][jj] = allocate_clean_block(BS);
					U[ii][jj] = NULL;
					copy_block(BS, LU[ii][jj], L[ii][jj]);
				}
				else
				{
					L[ii][jj] = NULL;
					U[ii][jj] = allocate_clean_block(BS);
					copy_block(BS, LU[ii][jj], U[ii][jj]);
				}
			}
		}
}

void copy_mat(int NB, int BS, p_block_t src[NB][NB], p_block_t dst[NB][NB])
{
	int ii, jj;

	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
			if (src[ii][jj] != NULL)
			{
				dst[ii][jj] = allocate_clean_block(BS);
				copy_block(BS, src[ii][jj], dst[ii][jj]);
			}
			else
				dst[ii][jj] = NULL;
}

void clean_mat(int NB, int BS, p_block_t src[NB][NB])
{
	int ii, jj;

	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
			if (src[ii][jj] != NULL)
				clean_block(BS, src[ii][jj]);
}

/* C = A * B */
void sparse_matmult(int NB, int BS, p_block_t A[NB][NB], p_block_t B[NB][NB], p_block_t C[NB][NB])
{
	int ii, jj, kk;

	for (ii = 0; ii < NB; ii++)
		for (jj = 0; jj < NB; jj++)
			for (kk = 0; kk < NB; kk++)
				if (A[ii][kk] != NULL && B[kk][jj] != NULL)
				{
					if (C[ii][jj] == NULL)
						C[ii][jj] = allocate_clean_block(BS);
					block_mpy_add(BS, A[ii][kk], B[kk][jj], C[ii][jj]);
				}
}

int main(int argc, char* argv[])
{
	if (argc != 3)
		errx(0, "Usage: %s num_blocks block_size", argv[0]);

	const int NB = atoi(argv[1]), BS = atoi(argv[2]);
	p_block_t A[NB][NB];
	p_block_t origA[NB][NB];


	genmat(NB, BS, A);
	printf("Matrix generated\n");
	copy_mat(NB, BS, A, origA);
	printf("Matrix copied: starting ROI\n");

	struct timeval start, stop;
	gettimeofday(&start,NULL);
	start_roi();

	LU(NB, BS, A);

	stop_roi(-1);
	gettimeofday(&stop,NULL);
	printf("LU finished\n");

	p_block_t L[NB][NB];
	p_block_t U[NB][NB];
	split_mat(NB, BS, A, L, U);

	clean_mat(NB, BS, A);
	sparse_matmult(NB, BS, L, U, A);

	if (!compare_mat(NB, BS, origA, A, &stop))
		printf("matrices are identical\n");

	unsigned long elapsed = 1000000 * (stop.tv_sec - start.tv_sec) + stop.tv_usec - start.tv_usec;
	//printf ("Time %lu microsecs\n", elapsed);
	printf ("par_sec_time_us:%lu\n", elapsed);

	return 0;
}

