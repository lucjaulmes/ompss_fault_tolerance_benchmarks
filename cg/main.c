#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <float.h>
#include <assert.h>

#include <catchroi.h>

#include "global.h"

#include "matrix.h"
#include "cg.h"
#include "mmio.h"
#include "debug.h"

// globals
int nb_blocks;
int *block_bounds;
int max_it = 0;

void set_blocks_sparse(Matrix *A, int *nb_blocks)
{
	block_bounds = (int*)calloc((*nb_blocks) + 1, sizeof(int));

	// compute block repartition now we have the matrix, arrange for block limits to be on fail block limits
	int i, pos = 0, next_stop = 0, ideal_bs = (A->nnz + *nb_blocks / 2) / *nb_blocks;
	for (i = 0; i < *nb_blocks - 1; i++)
	{
		next_stop += ideal_bs;

		while ( pos + 1 <= A->n && A->r[pos + 1] < next_stop )
			pos++;

		if ( pos + 1 <= A->n && A->r[pos + 1] - next_stop < next_stop - A->r[pos] )
			pos++;

		if (pos >= A->n)
		{
			fprintf(stderr, "Error while making blocks : end of block %d/%d is %d, beyond size of matrix %d ;"
			        " nb_blocks reduced to %d. You could try reducing -ps\n", i + 1, *nb_blocks, pos, A->n, i + 1);
			*nb_blocks = i + 1;
			break;
		}

		set_block_end(i, pos);

		// force to increment by at least 1
		pos++;
	}

	set_block_end( *nb_blocks - 1, A->n );
}

void usage(const char *arg0, const char *arg_err)
{
	if (arg_err != NULL)
		printf("Error near (or after) argument \"%s\"\n\n", arg_err);

	printf("Usage: %s [options] [<matrix-market-filename>|-synth name param] [, ...]\n"
	       " === Matrix === \n"
	       "  Either provide a path to a symmetric positive definite matrix in Matrix Market format\n"
	       "  or provide the -synth option for a synthetic matrix. Arguments are name param pairs :\n"
	       "    Poisson3D  p n  3D Poisson's equation using finite differences, matrix size n^3\n"
	       "                    with a p-points stencil : p one of 7, 19, 27 (TODO : 9 BCC, 13 FCC)\n"
	       " === run configuration options === \n"
	       "  -th    threads    Manually define number of threads.\n"
	       "  -nb    blocks     Defines the number of blocks in which to divide operations ;\n"
	       "                    their size will depend on the matrix' size.\n"
	       "  -r     runs       number of times to run a matrix solving.\n"
	       "  -cv    thres      Run until the error verifies ||b-Ax|| < thres * ||b|| (default 1e-10).\n"
	       "  -maxit N          Run no more than N iterations (default no limit).\n"
	       "  -seed  s          Initialize seed of each run with s. If 0 use different (random) seeds.\n"
	       "All options apply to every following input file. You may re-specify them for each file.\n\n", arg0);
	exit(1);
}

// we return how many parameters we consumed, -1 for error
int read_param(int argsleft, char* argv[], int *runs, int *blocks, unsigned int *seed, double *cv_thres, matrix_type *matsource, int *stencil_points, int *matrix_size)
{
	// we want at least the integer and a matrix market file after
	if ( argsleft <= 2 )
		return argv[0][0] == '-' ? -1 : 0;

	if (strcmp(argv[0], "-r") == 0)
	{
		*runs = (int) strtol(argv[1], NULL, 10);

		if ( *runs < 0 )
			return -1;
	}
	else if (strcmp(argv[0], "-maxit") == 0)
	{
		max_it = (int) strtol(argv[1], NULL, 10);

		if ( max_it <= 0 )
			return -1;
	}
	else if (strcmp(argv[0], "-seed") == 0)
	{
		*seed = (unsigned int) strtol(argv[1], NULL, 10);
	}
	else if (strcmp(argv[0], "-cv") == 0)
	{
		*cv_thres = strtod(argv[1], NULL);

		if ( *cv_thres <= 1e-15 )
			return -1;
	}
	else if (strcmp(argv[0], "-th") == 0)
	{
		int th = (int) strtol(argv[1], NULL, 10);

		if ( th <= 0 )
			return -1;

		set_num_threads(th);
	}
	else if (strcmp(argv[0], "-nb") == 0)
	{
		*blocks = (int) strtol(argv[1], NULL, 10);

		if ( *blocks <= 0 )
			return -1;
	}
	else if (strcmp(argv[0], "-synth") == 0)
	{
		// we want at least the name and size and points parameter after
		if (argsleft <= 3)
			return -1;

		if (strcmp(argv[1], "Poisson3D") == 0)
		{
			*matsource = POISSON3D;

			*stencil_points = (int)strtol(argv[2], NULL, 10);

			if (*stencil_points != 7 && *stencil_points != 19 && *stencil_points != 27)
				return -1;
		}
		else
			return -1; // unrecognized

		*matrix_size = (int)strtol(argv[3], NULL, 10);

		if (*matrix_size <= 0)
			return -1;

		return 4;
	}
	else
		return 0; // no option regognized, consumed 0 parameters

	// by default consumed 2 options
	return 2;
}

FILE* get_infos_matrix(char *filename, int *n, int *m, int *nnz, int *symmetric)
{
	FILE* input_file = fopen(filename, "r");
	MM_typecode matcode;
	*nnz = 0;

	if (input_file == NULL)
	{
		printf("Error : file \"%s\" not valid (check path/read permissions)\n", filename);
		return NULL;
	}

	else if (mm_read_banner(input_file, &matcode) != 0)
		printf("Could not process Matrix Market banner of file \"%s\".\n", filename);

	else if (mm_is_complex(matcode))
		printf("Sorry, this application does not support Matrix Market type of file \"%s\" : [%s]\n",
		       filename, mm_typecode_to_str(matcode));

	else if ( !mm_is_array(matcode) && (mm_read_mtx_crd_size(input_file, m, n, nnz) != 0 || *m != *n) )
		printf("Sorry, this application does not support the not-array matrix in file \"%s\"\n", filename);

	else if ( mm_is_array(matcode) && (mm_read_mtx_array_size(input_file, m, n) != 0 || *m != *n) )
		printf("Sorry, this application does not support the array matrix in file \"%s\"\n", filename);

	else if ( !mm_is_symmetric(matcode) )
		printf("Sorry, this application does not support the non-symmetric matrix in file \"%s\"\n", filename);

	else // hurray, no reasons to fail
	{
		if ( *nnz == 0 )
			*nnz = (*m) * (*n);
		*symmetric = mm_is_symmetric(matcode);
		if ( max_it == 0 )
			max_it = 10 * (*n);
		return input_file;
	}

	// if we're here we failed at some point but opened the file
	fclose(input_file);
	return NULL;
}

// main function, where we parse arguments, read files, setup stuff and start the recoveries
int main(int argc, char* argv[])
{
	// no buffer on stdout so messages interleaved with stderr will be in right order
	setbuf(stdout, NULL);

	if (argc < 2)
		usage(argv[0], NULL);

	int i, j, f, nb_read, runs = 1;
	unsigned int seed = 0;
	int stencil_points, size_param;
	matrix_type matsource = UNKNOWN;
	double cv_thres = 1e-10;

	nb_blocks = get_num_threads();

	// Iterate over parameters (usually open files)
	for (f = 1; f < argc; f += nb_read)
	{
		nb_read = read_param(argc - f, &argv[f], &runs, &nb_blocks, &seed, &cv_thres, &matsource, &stencil_points, &size_param);

		// error happened
		if ( nb_read < 0 )
			usage(argv[0], argv[f]);

		// if no parameters read, next param must be a matrix file. Read it (and consume parameter)
		else if (nb_read == 0 || matsource != UNKNOWN)
		{
			int n;
			Matrix matrix;
			char mat_name[200];

			if (matsource == FROM_FILE)
			{
				nb_read = 1;
				int m, lines_in_file, symmetric;
				FILE* input_file = get_infos_matrix(argv[f], &n, &m, &lines_in_file, &symmetric);

				if (input_file == NULL)
				{
					// Oops! was not a valid matrix market file
					usage(argv[0], argv[f]);
				}

				allocate_matrix(n, m, lines_in_file * (1 + symmetric), &matrix);
				read_matrix(n, m, lines_in_file, symmetric, &matrix, input_file);

				set_blocks_sparse(&matrix, &nb_blocks);

				fclose(input_file);

				strcpy(mat_name, argv[f]);
			}
			else if (matsource == POISSON3D)
			{
				// here all block etc. repartition is static, so we can load balance in advance
				n = size_param * size_param * size_param;

				if ( max_it == 0 )
					max_it = 10 * n;

				allocate_matrix(n, n, n * stencil_points, &matrix);
				generate_Poisson3D(&matrix, size_param, stencil_points, 0, n);

				set_blocks_sparse(&matrix, &nb_blocks);

				sprintf(mat_name, "Poisson3D-%d-%d", stencil_points, size_param);
			}
			else
			{
				// Oops! invalid matrix type, somehow?
				usage(argv[0], argv[f]);
			}

			char header[500];

			sprintf(header, "executable:%s matrix:%s problem_size:%d nb_threads:%d nb_blocks:%d srand_seed:%u max_it:%d convergence_at:%e\n",
			        argv[0], mat_name, n, get_num_threads(), nb_blocks, seed, max_it, cv_thres);


			// set some parameters that we don't want to pass through solve_cg
			printf(header);

#if VERBOSE >= SHOW_TOOMUCH
			print_matrix(stderr, &matrix);
#endif

			// a few vectors for rhs of equation, solution and verification
			double *b, *x, *s;
			x = (double*)CATCHROI_INSTRUMENT(calloc)(n, sizeof(double));
			b = (double*)CATCHROI_INSTRUMENT(calloc)(n, sizeof(double));
			s = (double*)calloc(n, sizeof(double));

			// interesting stuff is here
			for (j = 0; j < runs; j++)
			{
				// seed = 0 -> random : time for randomness, +j to get different seeds even if solving < 1s
				unsigned int real_seed = seed == 0 ? time(NULL) + j : seed;
				printf("run:%d seed:%u\n", j, real_seed);

				srand(real_seed);

				// generate random rhs to problem
				double range = (double) 1;

				for (i = 0; i < n; i++)
				{
					b[i] = ((double)rand() / (double)RAND_MAX) * range - range / 2;
					x[i] = 0.0;
				}

				solve_cg(&matrix, b, x, cv_thres);

				// compute verification
				mult(&matrix, x, s);

				// do displays (verification, error)
				double err = 0, norm_b = norm(n, b);
				for (i = 0; i < n; i++)
				{
					double e_i = b[i] - s[i];
					err += e_i * e_i;
				}

				printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err / norm_b));
			}

			// deallocate everything we have allocated for several solvings
			deallocate_matrix(&matrix);
			free(b);
			free(x);
			free(s);
			free(block_bounds);

			// reset so we can pass a second matrix as parameters
			matsource = UNKNOWN;
		}
	}

	return 0;
}
