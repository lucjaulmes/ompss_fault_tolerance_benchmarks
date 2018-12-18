#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

typedef struct Matrix
{
	int n, m, *c, *r;
	long nnz;
	double *v;
} Matrix;

typedef enum
{
	UNKNOWN = 0,
	FROM_FILE = 0,
	POISSON3D
} matrix_type;


// general matrix-vector multiplication, row major (W = A x V)
void mult(const Matrix *A, const double *V, double *W);

// read the matrix data from a Matrix Market file (header already parsed)
void read_matrix(const int n, const int m, const long nnz, const int symmetric, Matrix *A, FILE* input_file);

// visual representation of matrix
void print_matrix(FILE* f, const Matrix *A);

// generate given lines from a synthetic matrix with given stencil and size p^3
void generate_Poisson3D(Matrix *A, const int p, const int stencil_points, const int mpi_rank, const int mpi_size);

// memory utility functions
void allocate_matrix(const int n, const int m, const long nnz, Matrix *A);

void deallocate_matrix(Matrix *A);

// 2 useful functions that don't fit somewhere else
static inline double norm(const int n, const double *v)
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * v[i];

	return r;
}

static inline double scalar_product(const int n, const double *v, const double *w)
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * w[i];

	return r;
}

#endif // MATRIX_H_INCLUDED

