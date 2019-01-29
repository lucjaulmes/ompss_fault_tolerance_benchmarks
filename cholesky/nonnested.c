#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <errno.h>
#include <err.h>
//#include <linux/getcpu.h>

#if USE_PAPI
#include <papi.h>
#endif

#include <sched.h>
#include <omp.h>

#include <catchroi.h>

// Configuration options
#define CONVERT_TASK 1
#define CONVERT_ON_REQUEST 0
#define USE_AFFINITY 0
#define DOUBLE_PREC 1

#define POTRF_SMP 1
#define POTRF_NESTED 0
#define TRSM_SMP 1
#define TRSM_NESTED 0
#define SYRK_SMP 1
#define SYRK_NESTED 0
#define GEMM_SMP 1
#define GEMM_NESTED 0

#define USE_PAPI 0
#define MAGMA_BLAS 0
#define MKL 0
#define ATLAS 1

#define STOP_SCHED 0
//int NUM_NODES;

#if !(POTRF_SMP==1 && TRSM_SMP==1 && SYRK_SMP==1 && GEMM_SMP==1)
	#include <cuda_runtime.h>
#endif
// Checking constraints
// If converting matrix under request, mark convert functions as tasks
#if CONVERT_ON_REQUEST
	#if !CONVERT_TASK
		#undef CONVERT_TASK
		#define CONVERT_TASK 1
	#endif
	#define CONVERT_ONREQ_ONLY
#else
	#define CONVERT_ONREQ_ONLY __attribute__((unused))
#endif

#if CONVERT_TASK
	#define CONVERT_TASK_ONLY
#else
	#define CONVERT_TASK_ONLY __attribute__((unused))
#endif

// Include GPU kernel's library: MAGMA or CUBLAS
#if MAGMA_BLAS
	#include <magma.h>
#else
	#if MKL
		#include <mkl.h>
	#else
		#if ATLAS
			#include <cblas.h>
		#else
			#include <cublas.h>
		#endif
	#endif
#endif

#define PASTE(a, b) a ## b
#define PASTE3(a, b, c) a ## b ## c
#define SMP(f) PASTE(smp_, f)
#define OMP(f) PASTE(omp_, f)
#define TILE(f) PASTE(f, _tile)


// Define macros to make the code cleaner: OMP_TASK(blah) is #pragma omp task blah
// with the macros inside blah expanded to their values.
#define DO_PRAGMA(x) _Pragma (#x)
#define OMP_TASK(dependencies, extras) DO_PRAGMA(omp task dependencies extras)


#if POTRF_NESTED
	#define POTRF(x1, x2) smp_cholesky(x1, x2);
#else
	#define POTRF(x1, x2) TILE(potrf)(x1, x2);
#endif

#if TRSM_NESTED
	#define TRSM(x1, x2, x3) OMP(trsm)(x1, x2, x3);
#else
	#define TRSM(x1, x2, x3) TILE(trsm)(x1, x2, x3);
#endif

#if SYRK_NESTED
	#define SYRK(x1, x2, x3) OMP(syrk)(x1, x2, x3);
#else
	#define SYRK(x1, x2, x3) TILE(syrk)(x1, x2, x3);
#endif

#if GEMM_NESTED
	#define GEMM(x1, x2, x3, x4) OMP(gemm)(x1, x2, x3, x4);
#else
	#define GEMM(x1, x2, x3, x4) TILE(gemm)(x1, x2, x3, x4);
#endif


#ifdef USE_AFFINITY
	#define AFFINITY target device(smp) copy_deps
#else
	#define AFFINITY
#endif

#define CUDA_TARGET target device(cuda)


#if POTRF_SMP
	#define POTRF_TARGET AFFINITY
#else
	#define POTRF_TARGET CUDA_TARGET untied
#endif

#if TRSM_SMP
	#define TRSM_TARGET AFFINITY
#else
	#define TRSM_TARGET CUDA_TARGET
#endif

#if SYRK_SMP
	#define SYRK_TARGET AFFINITY
#else
	#define SYRK_TARGET CUDA_TARGET
#endif

#if GEMM_SMP
	#define GEMM_TARGET AFFINITY untied
#else
	#define GEMM_TARGET CUDA_TARGET
#endif


#if CONVERT_ON_REQUEST
	#define TRY_GATHER_BLOCK(x1, x2, x3, x4)      \
    do {                                          \
		if (x4 == NULL)                           \
		{                                         \
			x4 = malloc_block(x2);                \
			gather_block(x1, x2, x3, x4);         \
		}                                         \
	} while (0)
#else
	#define TRY_GATHER_BLOCK(x1, x2, x3, x4) do {} while (0)
#endif


// define SD: the prefix s or d for single or double precision
#ifdef DOUBLE_PREC
	#define REAL				double
	#define LAPACK(funcname)	PASTE3(d, funcname, _)
	#define	CBLAS(funcname)		PASTE(cblas_d, funcname)
	#define GPU(funcname)		PASTE(gpu_d_, funcname)
	#define GPUBLAS(funcname)	PASTE(gpu_blas_d_, funcname)
	#define accepted_error		1.0e-15
	#define	SCAN_REAL			"%lg"
#else
	#define REAL				float
	#define LAPACK(funcname)	PASTE3(s, funcname, _)
	#define	CBLAS(funcname)		PASTE(cblas_s, funcname)
	#define GPU(funcname)		PASTE(gpu_s_, funcname)
	#define GPUBLAS(funcname)	PASTE(gpu_blas_s_, funcname)
	#define accepted_error		1.0e-1
	#define	SCAN_REAL			"%g"
#endif


#if MAGMA_BLAS
	#define gpu_d_potrf     magma_dpotrf_gpu
	#define gpu_blas_d_gemm magmablas_dgemm
	#define gpu_blas_d_trsm magmablas_dtrsm
	#define gpu_blas_d_syrk magmablas_dsyrk
	#define gpu_s_potrf     magma_spotrf_gpu
	#define gpu_blas_s_gemm magmablas_sgemm
	#define gpu_blas_s_trsm magmablas_strsm
	#define gpu_blas_s_syrk cublasSsyrk		  // = magmablas_ssyrk
#else
	#define gpu_blas_d_gemm cublasDgemm
	#define gpu_blas_d_trsm cublasDtrsm
	#define gpu_blas_d_syrk cublasDsyrk
	#define gpu_blas_s_gemm cublasSgemm
	#define gpu_blas_s_trsm cublasStrsm
	#define gpu_blas_s_syrk cublasSsyrk
#endif


#if MKL
	#define LONG int
#else
	#define LONG long
#endif


void LAPACK(laset)(const char *UPLO, const int *M, const int *N, const REAL *ALPHA, const REAL *BETA, REAL *A, const int *LDA);
void LAPACK(gemm)(const char *transa, const char *transb, int *l, int *n, int *m, REAL *alpha, const void *a, int *lda, void *b, int *ldb, REAL *beta, void *c, int *ldc);
REAL LAPACK(lange)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *work);
void LAPACK(lacpy)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *b, const int *ldb);
void LAPACK(larnv)(const int *idist, int *iseed, const int *n, REAL *x);
void LAPACK(potrf)(const char *uplo, const int *n, REAL *a, const int *lda, LONG *info);
void LAPACK(trsm)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, REAL *alpha, REAL *a, int *lda, REAL *b, int *ldb);
void LAPACK(trmm)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, REAL *alpha, REAL *a, int *lda, REAL *b, int *ldb);
void LAPACK(syrk)(char *uplo, char *trans, int *n, int *k, REAL *alpha, REAL *a, int *lda, REAL *beta, REAL *c, int *ldc);

float get_time();
static int check_factorization(int, REAL *, REAL *, char, REAL);

//void gpu_spotf2_var1_(char *, int *, unsigned int *, int *, int *);
void gpu_spotrf_var1_(char *, int *, unsigned int *, int *, int *, int *);

void cholesky(REAL *Alin, REAL **Ah, int ts, int nt);


enum blas_order_type
{
	blas_rowmajor = 101,
	blas_colmajor = 102
};

enum blas_cmach_type
{
	blas_base      = 151,
	blas_t         = 152,
	blas_rnd       = 153,
	blas_ieee      = 154,
	blas_emin      = 155,
	blas_emax      = 156,
	blas_eps       = 157,
	blas_prec      = 158,
	blas_underflow = 159,
	blas_overflow  = 160,
	blas_sfmin     = 161
};

enum blas_norm_type
{
	blas_one_norm       = 171,
	blas_real_one_norm  = 172,
	blas_two_norm       = 173,
	blas_frobenius_norm = 174,
	blas_inf_norm       = 175,
	blas_real_inf_norm  = 176,
	blas_max_norm       = 177,
	blas_real_max_norm  = 178
};


static void __attribute__((unused))
BLAS_ge_norm(enum blas_order_type order, enum blas_norm_type norm, int m, int n, const REAL *a, int lda, REAL *res)
{
	int i, j;
	char rname[] = "BLAS_ge_norm";

	if (order != blas_colmajor)
		errx(1, "%s %d %d %d", rname, -1, order, 0);

	if (norm == blas_frobenius_norm)
	{
		register float anorm = 0.0f;
		for (j = n; j; --j)
		{
			for (i = m; i; --i)
			{
				anorm += a[0] * a[0];
				a++;
			}
			a += lda - m;
		}

		if (res)
			*res = sqrt(anorm);
	}
	else if (norm == blas_inf_norm)
	{
		float anorm = 0.0f;
		for (i = 0; i < m; ++i)
		{
			register float v = 0.0f;
			for (j = 0; j < n; ++j)
				v += abs(a[i + j * lda]);

			if (anorm < v)
				anorm = v;
		}

		if (res)
			*res = anorm;
	}
	else
		errx(1, "%s %d %d %d", rname, -2, norm, 0);
}


static REAL
BLAS_dpow_di(REAL x, int n)
{
	REAL rv = 1.0;

	if (n < 0)
	{
		n = -n;
		x = 1.0 / x;
	}

	for (; n; n >>= 1, x *= x)
		if (n & 1)
			rv *= x;

	return rv;
}


static REAL
BLAS_dfpinfo(enum blas_cmach_type cmach)
{
	REAL eps = 1.0, r = 1.0, o = 1.0, b = 2.0;
	int t = 53, l = 1024, m = -1021;
	char rname[] = "BLAS_dfpinfo";

	if (sizeof(eps) == sizeof(float))
	{
		t = 24;
		l = 128;
		m = -125;
	}
	else
	{
		t = 53;
		l = 1024;
		m = -1021;
	}

	/* for (i = 0; i < t; ++i) eps *= half; */
	eps = BLAS_dpow_di(b, -t);
	/* for (i = 0; i >= m; --i) r *= half; */
	r = BLAS_dpow_di(b, m - 1);

	o -= eps;
	/* for (i = 0; i < l; ++i) o *= b; */
	o *= BLAS_dpow_di(b, l - 1) * b;

	switch (cmach)
	{
		case blas_eps: return eps;
		case blas_sfmin: return r;
		default: errx(1, "%s %d %d %d", rname, -1, cmach, 0);
	}

	return 0.0;
}


void add_to_diag_hierarchical(REAL **matrix, int ts, int nt, float alpha)
{
	int i;

	for (i = 0; i < nt * ts; i++)
		matrix[(i / ts) * nt + (i / ts)][(i % ts) * ts + (i % ts)] += alpha;
}


void add_to_diag(REAL *matrix, int n, REAL alpha)
{
	int i;
	for (i = 0; i < n; i++)
		matrix[i + i * n] += alpha;
}


double gtod_ref_time_sec = 0.0;

float get_time()
{
	double t, norm_sec;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	// If this is the first invocation of through dclock(), then initialize the
	// "reference time" global variable to the seconds field of the tv struct.
	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double) tv.tv_sec;

	// Normalize the seconds field of the tv struct so that it is relative to the
	// "reference time" that was recorded during the first invocation of dclock().
	norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

	// Compute the number of seconds since the reference time.
	t = norm_sec + tv.tv_usec * 1.0e-6;

	return (float) t;
}


/*------------------------------------------------------------------------
 *  Robust Check the factorization of the matrix A2
 */
static int check_factorization(int N, REAL *A1, REAL *A2, char uplo, REAL eps)
{
	int i, j;
	char NORM = 'I', LE = 'L', TR = 'T', NU = 'N', RI = 'R';

	REAL *L2       = calloc(N * N, sizeof(REAL));
	REAL *work     = calloc(N, sizeof(REAL));
	REAL alpha     = 1.0;

	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, N, N, A1, N, &Anorm);
	REAL Anorm = LAPACK(lange)(&NORM, &N, &N, A1, &N, work);

	/* Dealing with L'L or U'U  */
	LAPACK(lacpy)(&uplo, &N, &N, A2, &N, L2, &N);

	if (uplo == 'U')
		/* L2 = 1 * A2^T * L2 */
		LAPACK(trmm)(&LE, &uplo, &TR, &NU, &N, &N, &alpha, A2, &N, L2, &N);
	else
		/* L2 = 1 * L2 * A2^T */
		LAPACK(trmm)(&RI, &uplo, &TR, &NU, &N, &N, &alpha, A2, &N, L2, &N);

	/* Compute the residual || A - L'L || */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A1[j * N + i] -= L2[j * N + i];

	REAL Rnorm = LAPACK(lange)(&NORM, &N, &N, A1, &N, work);
	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, N, N, A1, N, &Rnorm);

	free(L2);
	free(work);

	printf("============\n");
	printf("Checking the Cholesky Factorization \n");
	printf("-- ||L'L-A||_inf/(||A||_inf.N.eps) = %e\n", Rnorm / (Anorm * N * eps));

	int info_factorization = isnan(Rnorm / (Anorm * N * eps)) || isinf(Rnorm / (Anorm * N * eps)) || (Rnorm / (Anorm * N * eps) > 60.0);

	if (info_factorization)
		printf("-- Factorization is suspicious ! \n");
	else
		printf("-- Factorization is CORRECT ! \n");

	return info_factorization;
}


//--------------------- check results ----------------------------------------------

REAL sckres(int n, REAL *A, int lda, REAL *L, int ldl)
{
	REAL zero = 0.0, minus_one = -1.0, one = 1.0, dummy = 9;
	char NORM = '1', T = 'T', N = 'N', U = 'U';
	int nminus_one = n - 1;

	REAL nrm = LAPACK(lange)(&NORM, &n, &n, A, &lda, &dummy);

	LAPACK(laset)(&U, &nminus_one, &nminus_one, &zero, &zero, &L[ldl], &ldl);
	LAPACK(gemm)(&N, &T, &n, &n, &n, &minus_one, L, &ldl, L, &ldl, &one, A, &lda);

	REAL nrm2 = LAPACK(lange)(&NORM, &n, &n, A, &lda, &dummy);

	return nrm2 / nrm;
}


//----------------------------------------------------------------------------------
//			 Changes in data storage
//----------------------------------------------------------------------------------

void print_linear_matrix(int n, REAL *matrix)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			printf("%g ", matrix[i * n + j]);
		printf("\n");
	}
}


void read_matrix(char *filename, int n, REAL *matrix, REAL *checksum)
{
	int i = 0;
	if (filename != NULL)
	{
		FILE *matrix_file = fopen(filename, "r");
		if (!matrix_file)
			err(1, "Could not open matrix file %s", filename);

		while (i < n * n && fscanf(matrix_file, SCAN_REAL, &matrix[i]) != EOF)
			i++;

		// Finished reading matrix; read checksum
		if (fscanf(matrix_file, SCAN_REAL, checksum) == EOF)
			errx(1, "Invalid matrix file: could not read checksum");
	}
	else
	{
		fprintf(stderr, "No matrix file, initializing matrix with random values\n");
		int ISEED[] = {0, 0, 0, 1}, intONE = 1;

		for (i = 0; i < n * n; i += n)
			LAPACK(larnv)(&intONE, &ISEED[0], &n, &matrix[i]);

		int j;
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
			{
				matrix[j * n + i] = matrix[j * n + i] + matrix[i * n + j];
				matrix[i * n + j] = matrix[j * n + i];
			}

		add_to_diag(matrix, n, (REAL) n);

		*checksum = 0.0;
	}
}


OMP_TASK(in([N*N]Alin) out([ts*ts]A) label(gather_block), AFFINITY untied)
static void CONVERT_TASK_ONLY gather_block(int N, int ts, REAL *Alin, REAL *A)
{
	int i, j;
	for (i = 0; i < ts; i++)
		for (j = 0; j < ts; j++)
			A[i * ts + j] = Alin[i * N + j];
}


OMP_TASK(in([ts*ts]A) inout([N*N]Alin) label(scatter_block), AFFINITY untied)
static void CONVERT_TASK_ONLY scatter_block(int N, int ts, REAL *A, REAL *Alin)
{
	int i, j;
	for (i = 0; i < ts; i++)
		for (j = 0; j < ts; j++)
			Alin[i * N + j] = A[i * ts + j];
}


static void convert_to_blocks(int ts, int DIM, int N, REAL Alin[N][N], REAL *A[DIM][DIM])
{
	int i, j;
#if CONVERT_TASK
	for (i = 0; i < DIM; i++)
		for (j = 0; j < DIM; j++)
			gather_block(N, ts, &Alin[i * ts][j * ts], A[i][j]);
#else
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[j / ts][i / ts][(j % ts) * ts + i % ts] = Alin[j][i];
#endif
}


static void convert_to_linear(int ts, int DIM, int N, REAL *A[DIM][DIM], REAL Alin[N][N])
{
	int i, j;
#if CONVERT_TASK
	for (i = 0; i < DIM; i++)
		for (j = 0; j < DIM; j++)
			if (A[i][j] != NULL)
				scatter_block(N, ts, A[i][j], (REAL *) &Alin[i * ts][j * ts]);
#else
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			Alin[j][i] = A[j / ts][i / ts][(j % ts) * ts + i % ts];
#endif
}


#if CONVERT_ON_REQUEST
static REAL *malloc_block(int ts)
{
	REAL *block = malloc(ts * ts * sizeof(REAL));

	if (block == NULL)
		err(1, "ALLOCATION ERROR (Ah block of %d elements)", ts);

	return block;
}
#endif

//----------------------------------------------------------------------------------
//			 TASKS FOR CHOLESKY
//----------------------------------------------------------------------------------


OMP_TASK(inout([NB*NB]A) label(TILE(potrf)), POTRF_TARGET)
void TILE(potrf)(REAL *A, int NB)
{
	char L = 'L';
#if POTRF_SMP
#  if ATLAS
	LONG INFO;
	LAPACK(potrf)(&L, &NB, A, &NB, &INFO);
#  else
	mpla
	LONG INFO;
	LAPACK(potrf)(&L, &NB, A, &NB, &INFO);
#  endif
#else
	// Performing Cholesky on GPU
	LONG INFO;
#if MAGMA_BLAS
	GPU(potrf)(L, NB, A, NB, &INFO);
#else
	int block = 32;
	REAL *address = A;
	gpu_spotrf_var1_(&L, &NB, (unsigned int *) &address, &NB, &block, &INFO);
#endif
#endif
}


OMP_TASK(in([NB*NB]A, [NB*NB]B) inout([NB*NB]C) label(TILE(gemm)), GEMM_TARGET)
void TILE(gemm)(REAL *A, REAL *B, REAL *C, int NB)
{
	unsigned char TR = 'T', NT = 'N';
	REAL DONE = 1.0, DMONE = -1.0;
#if GEMM_SMP
#if ATLAS
	CBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasTrans, NB, NB, NB, -1.0, A, NB, B, NB, 1.0, C, NB);
#else
	LAPACK(gemm)(&NT, &TR, &NB, &NB, &NB, &DMONE, A, &NB, B, &NB, &DONE, C, &NB);
#endif //ATLAS
#else
	GPUBLAS(gemm)(NT, TR, NB, NB, NB, DMONE, A, NB, B, NB, DONE, C, NB);
#endif
}


OMP_TASK(in([NB*NB]T) inout([NB*NB]B) label(TILE(trsm)), TRSM_TARGET)
void TILE(trsm)(REAL *T, REAL *B, int NB)
{
	char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
	REAL DONE = 1.0;
#if TRSM_SMP
#if ATLAS
	CBLAS(trsm)(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, NB, NB, 1.0, T, NB, B, NB);
#else //MKL
	LAPACK(trsm)(&RI, &LO, &TR, &NU, &NB, &NB, &DONE, T, &NB, B, &NB );
#endif //ATLAS
	// Performing STRSM on GPU
#else
	GPUBLAS(trsm)(RI, LO, TR, NU, NB, NB, DONE, T, NB, B, NB );
#endif
}


OMP_TASK(in([NB*NB]A) inout([NB*NB]C) label(TILE(syrk)), SYRK_TARGET)
void TILE(syrk)(REAL *A, REAL *C, int NB)
{
	//int x = 0;
	//for(int i = 0; i<5999999999; i++)
	//	x = i * x + 2;

	//int y = x + x;
	//printf("in syrk tile\n");

	unsigned char LO = 'L', NT = 'N';
	REAL DONE = 1.0, DMONE = -1.0;
#if POTRF_SMP
#if ATLAS
	CBLAS(syrk)(CblasColMajor, CblasLower,CblasNoTrans, NB, NB, -1.0, A, NB, 1.0, C, NB);
#else //MKL
	LAPACK(syrk)(&LO, &NT, &NB, &NB, &DMONE, A, &NB, &DONE, C, &NB );
#endif //ATLAS
#else
	// Performing SSYRK on GPU
	GPUBLAS(syrk)(LO, NT, NB, NB, DMONE, A, NB, DONE, C, NB );
#endif
}


//----------------------------------------------------------------------------------
//			 END TASKS FOR CHOLESKY
//----------------------------------------------------------------------------------

void cholesky(REAL CONVERT_ONREQ_ONLY *Alin, REAL** Ah, int ts, int nt)
{
	int i, j, k;

#if CONVERT_ON_REQUEST
	int N = nt * ts;
#endif

#if STOP_SCHED
	NANOS_SAFE(nanos_stop_scheduler());
	NANOS_SAFE(nanos_wait_until_threads_paused());
#endif

	// Shuffle across sockets
	for (k = 0; k < nt; k++) {
		TRY_GATHER_BLOCK(N, ts, &Alin[k * ts * N + k * ts], Ah[k * nt + k]);
		for (i = 0; i < k; i++)
		{
			TRY_GATHER_BLOCK(N, ts, &Alin[i * ts * N + i * ts], Ah[i * nt + i]);
			SYRK(Ah[i * nt + k], Ah[k * nt + k], ts);
		}

		// Diagonal Block factorization and panel permutations
		POTRF(Ah[k * nt + k], ts);

		// update trailing matrix
		for (i = k + 1; i < nt; i++)
		{
			for (j = 0; j < k; j++)
			{
				TRY_GATHER_BLOCK(N, ts, &Alin[k * ts * N + j * ts], Ah[k * nt + j]);
				TRY_GATHER_BLOCK(N, ts, &Alin[i * ts * N + j * ts], Ah[j * nt + i]);
				GEMM(Ah[j * nt + i], Ah[j * nt + k], Ah[k * nt + i], ts);
			}
			TRY_GATHER_BLOCK(N, ts, &Alin[k * ts * N + i * ts], Ah[k * nt + i]);
			TRSM(Ah[k * nt + k], Ah[k * nt + i], ts);
		}
	}
#if STOP_SCHED
	NANOS_SAFE(nanos_start_scheduler());
	NANOS_SAFE(nanos_wait_until_threads_unpaused());
#endif

	#pragma omp taskwait
}

//--------------------------- MAIN --------------------
int main(int argc, char *argv[])
{
	if (argc != 3 && argc != 4)
		errx(1, "Usage: %s size block_size [matrix_file]", argv[0]);

	int n = atoi(argv[1]);
	int ts = atoi(argv[2]);
	int nt = n / ts;

	char *filename = NULL;
	if (argc == 4)
		filename = argv[3];

	//nanos_get_num_sockets(&NUM_NODES);

	// Allocations

	REAL *matrix = CATCHROI_INSTRUMENT(aligned_alloc)(4096, n * n * sizeof(REAL));
	if (matrix == NULL)
		err(1, "ALLOCATION ERROR");

	REAL *original_matrix = CATCHROI_INSTRUMENT(aligned_alloc)(4096, n * n * sizeof(REAL));
	if (original_matrix == NULL)
		err(1, "ALLOCATION ERROR\n");

	int i;
	REAL **Ah = malloc(nt * nt * sizeof(REAL *));
	if (Ah == NULL)
		err(1, "ALLOCATION ERROR (Ah)");

	for (i = 0; i < nt * nt; i++)
	{
		Ah[i] = CATCHROI_INSTRUMENT(aligned_alloc)(4096, ts * ts * sizeof(REAL));
		if (Ah[i] == NULL)
			err(1, "ALLOCATION ERROR (Ah[%d])", i);
	}


	// Initialize matrix and copy
	REAL checksum;
	read_matrix(filename, n, matrix, &checksum);
	memcpy(original_matrix, matrix, n * n * sizeof(*matrix));

#if USE_PAPI
	if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
		err(2, "ERROR in library init!\n");

	unsigned int native = 0x0;
	int EventSet = PAPI_NULL;
	if(PAPI_create_eventset(&EventSet) != PAPI_OK)
		errx(2, "ERROR creating event set!")
	if(PAPI_event_name_to_code("rapl:::PP0_ENERGY:PACKAGE0", &native) != PAPI_OK)
		errx(2, "ERROR creating native event")
	if(PAPI_add_event(EventSet, native) != PAPI_OK)
		errx(2, "ERROR adding event for Energy!")
#endif

	//long_long values[10] = {0};
	//printf("converting to blocks....")

#if !CONVERT_ON_REQUEST
	convert_to_blocks(ts, nt, n, (REAL(*)[n])matrix, (REAL*(*)[nt])Ah);
#endif
	//    #pragma omp taskwait

	float t1 = get_time();
	start_roi();

#if USE_PAPI
	if(PAPI_start(EventSet) != PAPI_OK)
		errx(2, "ERROR in papi start!")
#endif

	// Actual work
	cholesky(matrix, Ah, ts, nt);

#if USE_PAPI
	#pragma omp taskwait
	if (PAPI_stop(EventSet, values) != PAPI_OK)
		errx(2, "ERROR in papi stop!")
#endif

	float time = get_time() - t1;
	stop_roi(-1);

#if USE_PAPI
	if(PAPI_read(EventSet, values) != PAPI_OK)
		errx(2, "ERROR read failed for Energy!")
#endif

	//printf("Succesfull energy measurement! value = %lld\n", values[0]);
	convert_to_linear(ts, nt, n, (REAL*(*)[nt]) Ah, (REAL(*)[n])matrix);

	//#pragma omp taskwait

	REAL eps = BLAS_dfpinfo(blas_eps);
	int info_factorization = check_factorization(n, original_matrix, matrix, 'L', eps);

	float gflops = (1.0 / 3.0) * n * n * n / (time * 1e9);

	// Print configuration
	fprintf(stderr, "\tCONVERT_TASK   " " %d\n", CONVERT_TASK);
	fprintf(stderr, "\tCONVERT_ON_REQUEST %d\n", CONVERT_ON_REQUEST);
	fprintf(stderr, "\tPOTRF_SMP"       " %d\n", POTRF_SMP);
	fprintf(stderr, "\tPOTRF_NESTED"    " %d\n", POTRF_NESTED);
	fprintf(stderr, "\tMAGMA_BLAS"      " %d\n", MAGMA_BLAS);
	fprintf(stderr, "\tTRSM_SMP"        " %d\n", TRSM_SMP);
	fprintf(stderr, "\tSYRK_NESTED"     " %d\n", SYRK_NESTED);
	fprintf(stderr, "\tGEMM_SMP"        " %d\n", GEMM_SMP);
	fprintf(stderr, "\tGEMM_NESTED"     " %d\n", GEMM_NESTED);

	// Print results
	printf("============ CHOLESKY RESULTS ============\n");
	printf("  matrix size:          %dx%d\n", n, n);
	printf("  block size:           %dx%d\n", ts, ts);
	printf("  block count:          %dx%d\n", nt, nt);
	printf("  time (s):             %f\n", time);
	printf("  performance (gflops): %f\n", gflops);
	printf("==========================================\n");
	//printf( "%f ", time);


	// Free blocked matrix
	for (i = 0; i < nt * nt; i++)
		free(Ah[i]);
	free(Ah);
	free(matrix);
	free(original_matrix);

	return 0;
}

