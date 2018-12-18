#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <catchroi.h>

#include "global.h"
#include "debug.h"

#include "matrix.h"

// matrix-vector multiplication, row major (W = A x V)
void mult(const Matrix *A, const double *V, double *W)
{
	int i, j, n = A->n;

	for(i=0; i < n; i++)
	{
		W[i] = 0;

		for(j=A->r[i]; j < A->r[i+1]; j++)
			W[i] += A->v[j] * V[ A->c[j] ];
	}
}

void print_matrix(FILE* f, const Matrix *A)
{
	int i, j, n = A->n;

	for(i=0; i < n; i++)
	{
		fprintf(f, "%4d   |  ", i);

		for(j=A->r[i]; j < A->r[i+1]; j++)
			fprintf(f, " [%4d ] % 1.2e ", A->c[j], A->v[j]);

		fprintf(f, "\n");
	}
}

void read_matrix(const int n, const int m, const long nnz, const int symmetric, Matrix *A, FILE* input_file)
{
	int Y, prevY = -1, X, i, j, k, pos = 0, *nb_subdiagonals = NULL;
	double val;

	if(symmetric)
		nb_subdiagonals = (int*)calloc(n, sizeof(int));

	A->n = n;
	A->m = m;

	char line[256], *next;
	for(i=0; i<nnz; i++)
	{
		next = fgets(line, sizeof(line), input_file);
		/* read & adjust from 1-based to 0-based:
		 * lines must be "X Y val" separated by a single space */
		X = atoi(next) - 1;
		while (*next++ != ' ');
		Y = atoi(next) - 1;
		while (*next++ != ' ');
		val = atof(next);

		// for debug purposes
		if(Y >= n || X >= m)
			continue;

		if(Y > prevY)
		{
			A->r[Y] = pos;

			// leave space for the subdiagonals elements
			if(symmetric)
				pos += nb_subdiagonals[Y];

			prevY = Y;
		}

		A->v[pos] = val;
		A->c[pos] = X;
		pos ++;

		if(symmetric && X > Y)
			nb_subdiagonals[X]++;
	}

	A->nnz = pos;
	A->r[A->n] = pos;

	if(symmetric)
	{
		// now let's fill in the subdiagonal part
		int *fill_row = malloc(n * sizeof(int));

		for(j=0; j<A->n; j++)
			fill_row[j] = A->r[j];

		for(i=0; i<A->n; i++)
			for(k = A->r[i] + nb_subdiagonals[i] ; k < A->r[i+1] ; k++)
			{
				if(i == A->c[k])
					continue;

				j = A->c[k];
				// now put (i,j) in (j,i)

				pos = fill_row[j];
				A->c[pos] = i;
				A->v[pos] = A->v[k];

				fill_row[j]++;
			}

		free(nb_subdiagonals);
		free(fill_row);
	}
}

// finite-difference method for a 3D Poisson's equation with a 7, 19 or 27 point stencil
void generate_Poisson3D(Matrix *A, const int p, const int stencil_points, const int start_row, const int end_row)
{
	int p2 = p * p, p3 = p2 * p, i, j=0, pos=0;

	const int    *stenc_c;
	const double *stenc_v;

	const int    stenc_c7[]  = { -p2,  -p,  -1,   0,   1,   p,  p2};
	const double stenc_v7[]  = {-1.0,-1.0,-1.0, 6.0,-1.0,-1.0,-1.0};

	const double r = 1.0;
	const int    stenc_c19[] =
	{
		       -p2-p,          -p2-1,  -p2+0, -p2+1,          -p2+p,
		 -p-1,    -p,    -p+1,    -1,      0,     1,     p-1,     p,     p+1,
		        p2-p,           p2-1,   p2+0,  p2+1,           p2+p
	};
	const double stenc_v19[] =
	{
		     -1-r,      -1-r,    -8*r+4,   -1-r,      -1-r,
		-2, -6+2*r, -2, -6+2*r, 32+16*r, -6+2*r, -2, -6+2*r, -2,
		     -1-r,      -1-r,    -8*r+4,   -1-r,      -1-r
	};

	const int    stenc_c27[] =
	{
		-p2-p-1, -p2-p, -p2-p+1, -p2-1,  -p2+0, -p2+1, -p2+p-1, -p2+p, -p2+p+1,
		   -p-1,    -p,    -p+1,    -1,      0,     1,     p-1,     p,     p+1,
		 p2-p-1,  p2-p,  p2-p+1,  p2-1,   p2+0,  p2+1,  p2+p-1,  p2+p,  p2+p+1
	};
	const double stenc_v27[] =
	{
		   -2-r,  -8+10*r,    -2-r,  -8+10*r, -100*r+40,  -8+10*r,    -2-r,  -8+10*r,    -2-r,
		-20+2*r, -80+20*r, -20+2*r, -80+20*r, 400+200*r, -80+20*r, -20+2*r, -80+20*r, -20+2*r,
		   -2-r,  -8+10*r,    -2-r,  -8+10*r, -100*r+40,  -8+10*r,    -2-r,  -8+10*r,    -2-r
	};

	if(stencil_points == 7)
	{
		stenc_c = stenc_c7;
		stenc_v = stenc_v7;
	}
	else if(stencil_points == 19)
	{
		stenc_c = stenc_c19;
		stenc_v = stenc_v19;
	}
	else if(stencil_points == 27)
	{
		stenc_c = stenc_c27;
		stenc_v = stenc_v27;
	}
	else
		// this should be impossible, but silences compiler warnings
		return;

	// to compute the nnz, we just need to know that each stencil point at distance |d| from the diagonal
	// will be excluded from the matrix on d lines, otherwise each stencil point is on each line
	A->nnz = stencil_points * p3;
	for(i=0; i<stencil_points; i++)
		A->nnz -= abs(stenc_c[i]);

	// let's only do the part here.
	for(j=start_row; j<end_row; j++)
	{
		A->r[j-start_row] = pos;
		for(i=0; i<stencil_points; i++)
			if(j + stenc_c[i] >= 0 && j + stenc_c[i] < A->n)
			{
				A->c[pos] = j + stenc_c[i];
				A->v[pos] = stenc_v[i];
				pos++;
			}
	}

	// point to just beyond last element
	A->r[j-start_row] = pos;
}

void allocate_matrix(const int n, const int m, const long nnz, Matrix *A)
{
	A->n = n;
	A->m = m;
	A->nnz = nnz;


	A->r = (int*)CATCHROI_INSTRUMENT(calloc)((n+1), sizeof(int));

	A->c = (int*)CATCHROI_INSTRUMENT(calloc)(nnz, sizeof(int));
	A->v = (double*)CATCHROI_INSTRUMENT(calloc)(nnz, sizeof(double));

	if(! A->v || ! A->c || ! A->r)
	{
		fprintf(stderr, "Allocating sparse matrix of size %d rows and %ld non-zeros failed !\n", n, nnz);
		exit(2);
	}
}

void deallocate_matrix(Matrix *A)
{
	free(A->r);
	free(A->c);

	if(A->v)
		free(A->v);
}

