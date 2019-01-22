#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <err.h>
//#include <cuda_runtime.h>

#include <catchroi.h>

// Configuration options
#define CONVERT_TASK 0
#define CONVERT_REQUEST 0
#define USE_AFFINITY 0

#define POTRF_SMP 1
#define POTRF_NESTED 0
#define TRSM_SMP 1
#define TRSM_NESTED 0
#define SYRK_SMP 1
#define SYRK_NESTED 0
#define GEMM_SMP 1
#define GEMM_NESTED 0
#define TRTRI_SMP 1
#define TRTRI_NESTED 0
#define TRMM_SMP 1
#define TRMM_NESTED 0
#define LAUUM_SMP 1
#define LAUUM_NESTED 0
#define CHOLESKY_INVERSE_SMP 1

#define MAGMA_BLAS 0
#define MKL 0

#if !(POTRF_SMP==1 && TRSM_SMP==1 && SYRK_SMP==1 && GEMM_SMP==1)
	#include <cuda_runtime.h>
#endif
// Checking constraints
// If converting matrix under request, mark convert functions as tasks
#if CONVERT_REQUEST
	#if !CONVERT_TASK
		#undef CONVERT_TASK
		#define CONVERT_TASK 1
	#endif
#endif

// Include GPU kernel's library: MAGMA or CUBLAS
#if MAGMA_BLAS
	#include <magma.h>
#else
	#if MKL
		#include <mkl.h>
	#else
		#include <cblas.h>
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
	#define POTRF(...)	SMP(cholesky)(__VA_ARGS__)
#else
	#define POTRF(...)	TILE(potrf)(__VA_ARGS__)
#endif

#if TRSM_NESTED
	#define TRSM(...)	OMP(trsm)(__VA_ARGS__)
#else
	#define TRSM(...)	TILE(trsm)(__VA_ARGS__)
#endif

#if SYRK_NESTED
	#define SYRK(...)	OMP(syrk)(__VA_ARGS__)
#else
	#define SYRK(...)	TILE(syrk)(__VA_ARGS__)
#endif

#if GEMM_NESTED
	#define GEMM(...)	OMP(gemm)(__VA_ARGS__)
#else
	#define GEMM(...)	TILE(gemm)(__VA_ARGS__)
#endif

#if LAUUM_NESTED
	#define LAUUM(...)	OMP(lauum)(__VA_ARGS__)
#else
	#define LAUUM(...)	TILE(lauum)(__VA_ARGS__)
#endif

#if TRTRI_NESTED
	#define TRTRI(...)	OMP(dtrtri)(__VA_ARGS__)
#else
	#define TRTRI(...)	TILE(trtri)(__VA_ARGS__)
#endif

#if TRMM_NESTED
	#define TRMM(...)	OMP(trmm)(__VA_ARGS__)
#else
	#define TRMM(...)	TILE(trmm)(__VA_ARGS__)
#endif


#if USE_AFFINITY
	#define AFFINITY target device(smp) copy_deps
#else
	#define AFFINITY
#endif

#define CUDA_TARGET target device(cuda)


#if POTRF_SMP
	#define POTRF_TARGET AFFINITY priority(100000) untied
#else
	#define POTRF_TARGET CUDA_TARGET
#endif

#if TRSM_SMP
	#define TRSM_TARGET AFFINITY untied
#else
	#define TRSM_TARGET CUDA_TARGET
#endif

#if SYRK_SMP
	#define SYRK_TARGET AFFINITY untied
#else
	#define SYRK_TARGET CUDA_TARGET
#endif

#if GEMM_SMP
	#define GEMM_TARGET AFFINITY untied
#else
	#define GEMM_TARGET CUDA_TARGET
#endif

#if LAUUM_SMP
	#define LAUUM_TARGET AFFINITY untied priority(100000)
#else
	#define LAUUM_TARGET TARGET_CUDA
#endif

#if TRTRI_SMP
	#define TRTRI_TARGET AFFINITY priority(100000) untied
#else
	#define TRTRI_TARGET CUDA_TARGET
#endif

#if TRMM_SMP
	#define TRMM_TARGET AFFINITY priority(priority) untied
#else
	#define TRMM_TARGET CUDA_TARGET
#endif


#if CONVERT_REQUEST
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


// define calling conventions: the prefix s or d for single or double precision
// trailing _ to call (Fortran) functions avaiable via lapack
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
REAL LAPACK(gemm)(const char *transa, const char *transb, int *l, int *n, int *m, REAL *alpha, const REAL *a, int *lda, REAL *b, int *ldb, REAL *beta, REAL *c, int *ldc);
REAL LAPACK(lange)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *work);
REAL LAPACK(lansy)(const char *norm, const char *UPLO, const int *n, const REAL *a, const int *lda, REAL *work);
void LAPACK(lacpy)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *b, const int *ldb);
void LAPACK(larnv)(const int *idist, int *iseed, const int *n, REAL *x);
void LAPACK(potrf)(const char *uplo, const int *n, REAL *a, const int *lda, LONG *info);
void LAPACK(trsm)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, REAL *alpha, REAL *a, int *lda, REAL *b, int *ldb);
void LAPACK(trmm)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, REAL *alpha, REAL *a, int *lda, REAL *b, int *ldb);
void LAPACK(syrk)(char *uplo, char *trans, int *n, int *k, REAL *alpha, REAL *a, int *lda, REAL *beta, REAL *c, int *ldc);
void LAPACK(trtri)(const char *uplo, const char *diag, const int *n, REAL *a, const int *lda, LONG *info);
void LAPACK(symm)(const char *side, const char *uplo, const int *m, const int *n, REAL *alpha, REAL *a, int *lda, REAL *b, int *ldb, REAL *beta, REAL *c, int *ldc);
void LAPACK(lauum)(const char *uplo, const int *n, REAL *a, const int *lda, int *info);

float get_time();
static int check_factorization(int, REAL *, REAL *, char, REAL);
static int check_inverse(int, int, int, REAL **, REAL *, char, REAL);

//void gpu_spotf2_var1_(char *, int *, unsigned int *, int *, int *);
void gpu_spotrf_var1_(char *, int *, unsigned int *, int *, int *, int *);

void cholesky_inverse(int ts, int nt, REAL *Ah[nt][nt]);


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


static void
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


static void inf_norm(int N, REAL *A, REAL *norm)
{
	char NORM = 'I';
	REAL *work = calloc(N, sizeof(REAL));

	*norm = LAPACK(lange)(&NORM, &N, &N, A, &N, work);

	free(work);
}


static void inf_norm_sym(int N, REAL *A, REAL *norm, char uplo)
{
	char NORM = 'I';
	REAL *work = calloc(N, sizeof(REAL));

	*norm = LAPACK(lansy)(&NORM, &uplo, &N, A, &N, work);

	free(work);
}


/*------------------------------------------------------------------------
 *  Robust Check the factorization of the matrix A2
 */
static int check_factorization(int N, REAL *A1, REAL *A2, char uplo, REAL eps)
{
	int i, j;
	char LE = 'L', TR = 'T', NU = 'N', RI = 'R';

	REAL *L2       = calloc(N * N, sizeof(REAL));
	REAL *work     = calloc(N, sizeof(REAL));
	REAL alpha     = 1.0;

	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, N, N, A1, N, &Anorm);
	REAL Anorm = 0., Rnorm = 0.;
	inf_norm(N, A1, &Anorm);

	#pragma omp task inout([N * N]A2)
	{
		/* Dealing with L'L or U'U  */
		LAPACK(lacpy)(&uplo, &N, &N, A2, &N, L2, &N);

		if (uplo == 'U')
			/* L2 = 1 * A2^T * L2 */
			LAPACK(trmm)(&LE, &uplo, &TR, &NU, &N, &N, &alpha, A2, &N, L2, &N);
		else
			/* L2 = 1 * L2 * A2^T */
			LAPACK(trmm)(&RI, &uplo, &TR, &NU, &N, &N, &alpha, A2, &N, L2, &N);
	}

	#pragma omp task in([N * N]A2) inout([N * N]A1)
	{
		/* Compute the residual || A - L'L || */
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
				A1[j * N + i] -= L2[j * N + i];
	}

	inf_norm(N, A1, &Rnorm);
	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, N, N, A1, N, &Rnorm);

	#pragma omp taskwait noflush

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


static void convert_to_blocks(int ts, int DIM, int N, REAL   Alin[N][N], REAL *A[DIM][DIM]);
static void convert_to_linear(int ts, int DIM, int N, REAL *A[DIM][DIM], REAL   Alin[N][N]);

//-----------------------check inverse---------------------------

static int check_inverse(int nt, int ts, int n, REAL **Ah, REAL *matrix, char uplo, REAL eps)
{
	REAL *product = calloc(n * n, sizeof(REAL));
	REAL *inverse = calloc(n * n, sizeof(REAL));
	convert_to_linear(ts, nt, n, (REAL *(*)[nt])Ah, (REAL(*)[n])inverse);

	REAL alpha = -1.0, beta  = 0.0;
	char side = uplo == 'U' ? 'L' : 'R';

	LAPACK(symm)(&side, &uplo, &n, &n, &alpha, inverse, &n, matrix, &n, &beta, product, &n);

	/* Add the identity matrix to product */
	add_to_diag(product, n, 1.0);

	REAL Rnorm = 0., Anorm = 0., Ainvnorm = 0.;
	inf_norm_sym(n, inverse, &Ainvnorm, uplo);
	inf_norm(n, matrix, &Anorm);
	inf_norm(n, product, &Rnorm);

	if (getenv("PLASMA_TESTING_VERBOSE"))
		printf("||A||_1=%f\n||Ainv||_1=%f\n||Id - A*Ainv||_1=%e\n\n", Anorm, Ainvnorm, Rnorm);

	REAL result = Rnorm / (Anorm * Ainvnorm * n * eps);
	printf("============\n");
	printf("Checking the Residual of the inverse \n");
	printf("-- ||Id - A * Ainv||_1 / (||A||_1 ||Ainv||_1 N eps) = %e\n", result);

	int info_inverse = isnan(Ainvnorm) || isinf(Ainvnorm) || isnan(result) || isinf(result) || (result > 60.0);

	if (info_inverse)
		printf("-- The inverse is suspicious ! \n");
	else
		printf("-- The inverse is CORRECT ! \n");

	free(product);
	free(inverse);

	return info_inverse;
}


static int check_inverse_blocked(int nt, int ts, int n, REAL **B, REAL *matrix, char uplo, REAL eps)
{
	/* A = matrix, B = Ah = inverse, C = product */
	REAL **A = calloc(nt * nt, sizeof(REAL *)), **C = calloc(nt * nt, sizeof(REAL *));
	REAL alpha = -1.0, beta = 1.0;
	char side = uplo == 'U' ? 'L' : 'R', normal = 'N', transposed = 'T';
	int i, j, k;

	REAL *product = calloc(n * n, sizeof(REAL));
	REAL *inverse = calloc(n * n, sizeof(REAL));

	if (A == NULL || B == NULL || C == NULL)
		err(1, "ALLOCATION ERROR");

	for (i = 0; i < nt * nt; i++)
	{
		A[i] = calloc(ts * ts, sizeof(REAL));
		C[i] = calloc(ts * ts, sizeof(REAL));
		if (A[i] == NULL || C[i] == NULL)
			err(1, "ALLOCATION ERROR (B[%d] or C[%d])", i, i);
	}

	convert_to_blocks(ts, nt, n, (REAL(*)[n])matrix, (REAL *(*)[nt])A);

	for (i = 0; i < nt; i++)
		#pragma omp task commutative([ts * ts](C[i * nt + i])) firstprivate(ts)
		/* C[ii] = Id */
		add_to_diag(C[i * nt + i], ts, 1.0);

	for (i = 0; i < nt; i++)
		for (j = 0; j < nt; j++)
			for (k = 0; k < nt; k++)
			{
				REAL *Akj = A[k * nt + j], *Cij = C[i * nt + j], *Bxy;
				char transB;

				/* use B[ik] or B[ki] based on which side of the diagonal is valid */
				if ((i > k && uplo == 'U') || (i < k && uplo == 'L'))
				{
					Bxy = B[i * nt + k];
					transB = normal;
				}
				else
				{
					Bxy = B[k * nt + i];
					transB = transposed;
				}

				#pragma omp task in([ts * ts]Akj, [ts * ts]Bxy) commutative([ts * ts]Cij) firstprivate(side, uplo, normal, transB, ts, i, k)
				{
					if (i == k)
						/* C[ij] -= B[ii] * A[ij] or A[ij] * B[ii] */
						LAPACK(symm)(&side, &uplo, &ts, &ts, &alpha, Bxy, &ts, Akj, &ts, &beta, Cij, &ts);
					else if (uplo == 'U')
						/* C[ij] -= B[ik] * A[kj] or t(B[ki]) * A[kj] */
						LAPACK(gemm)(&transB, &normal, &ts, &ts, &ts, &alpha, Bxy, &ts, Akj, &ts, &beta, Cij, &ts);
					else
						/* C[ij] -= A[kj] * B[ik] or A[kj] * t(B[ki]) */
						LAPACK(gemm)(&normal, &transB, &ts, &ts, &ts, &alpha, Akj, &ts, Bxy, &ts, &beta, Cij, &ts);
				}
			}

	#pragma omp taskwait

	REAL Rnorm = 0., Anorm = 0., Ainvnorm = 0.;

	#pragma omp task out([n * n]inverse)
	convert_to_linear(ts, nt, n, (REAL *(*)[nt])B, (REAL(*)[n])inverse);
	#pragma omp task out([n * n]product)
	convert_to_linear(ts, nt, n, (REAL *(*)[nt])C, (REAL(*)[n])product);

	#pragma omp task in([n * n]inverse) out(Ainvnorm)
	inf_norm_sym(n, inverse, &Ainvnorm, uplo);
	#pragma omp task in([n * n]matrix) out(Anorm)
	inf_norm(n, matrix, &Anorm);
	#pragma omp task in([n * n]product) out(Rnorm)
	inf_norm(n, product, &Rnorm);

	#pragma omp taskwait

	if (getenv("PLASMA_TESTING_VERBOSE"))
		printf("||A||_1=%f\n||Ainv||_1=%f\n||Id - A*Ainv||_1=%e\n\n", Anorm, Ainvnorm, Rnorm);

	REAL result = Rnorm / (Anorm * Ainvnorm * n * eps);
	printf("============\n");
	printf("Checking the Residual of the inverse \n");
	printf("-- ||Id - A * Ainv||_1 / (||A||_1 ||Ainv||_1 N eps) = %e\n", result);

	int info_inverse = isnan(Ainvnorm) || isinf(Ainvnorm) || isnan(result) || isinf(result) || (result > 60.0);

	if (info_inverse)
		printf("-- The inverse is suspicious ! \n");
	else
		printf("-- The inverse is CORRECT ! \n");

	for (i = 0; i < nt * nt; i++)
	{
		free(A[i]);
		free(C[i]);
	}
	free(A);
	free(C);

	free(inverse);
	free(product);

	return info_inverse;
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


void read_matrix(char *filename, int n, REAL matrix[n][n], REAL *checksum)
{
	int i = 0;
	if (filename != NULL)
	{
		FILE *matrix_file = fopen(filename, "r");
		if (!matrix_file)
			err(1, "Could not open matrix file %s", filename);

		while (i < n * n && fscanf(matrix_file, SCAN_REAL, &matrix[i / n][i % n]) != EOF)
			i++;

		// Finished reading matrix; read checksum
		if (fscanf(matrix_file, SCAN_REAL, checksum) == EOF)
			errx(1, "Invalid matrix file: could not read checksum");
	}
	else
	{
		fprintf(stderr, "No matrix file, initializing matrix with random values\n");
		int ISEED[] = {0, 0, 0, 1}, intONE = 1;

		for (i = 0; i < n; i ++)
			LAPACK(larnv)(&intONE, &ISEED[0], &n, matrix[i]);

		int j;
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
			{
				REAL sum = matrix[j][i] + matrix[i][j];
				matrix[j][i] = sum;
				matrix[i][j] = sum;
			}

		add_to_diag((REAL*)matrix, n, (REAL) n);

		*checksum = 0.0;
	}
}


#if CONVERT_TASK
OMP_TASK(in([N*(ts-1)+ts]Alin) out([ts*ts]A) label(gather_block), AFFINITY)
static void gather_block(int N, int ts, REAL *Alin, REAL *A)
{
	int i, j;
	for (i = 0; i < ts; i++)
		for (j = 0; j < ts; j++)
			A[i * ts + j] = Alin[i * N + j];
}
#endif


#if CONVERT_TASK
OMP_TASK(in([ts*ts]A) out([N*(ts-1)+ts]Alin) label(scatter_block), AFFINITY)
static void scatter_block(int N, int ts, REAL *A, REAL *Alin)
{
	int i, j;
	for (i = 0; i < ts; i++)
		for (j = 0; j < ts; j++)
			Alin[i * N + j] = A[i * ts + j];
}
#endif


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


#if CONVERT_REQUEST
static REAL *malloc_block(int ts)
{
	REAL *block = malloc(ts * ts * sizeof(REAL));

	if (block == NULL)
		err(1, "ALLOCATION ERROR (Ah block of %lld elements)", ts);

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
	LONG INFO;
	LAPACK(potrf)(&L, &NB, A, &NB, &INFO);
#else
	// Performing Cholesky on GPU
	int INFO;
#if MAGMA_BLAS
	GPU(potrf)(L, NB, A, NB, &INFO);
#else
	int block = 32;
	REAL *address = A;
	gpu_spotrf_var1_(&L, &NB, (unsigned int *) &address, &NB, &block, &INFO);
#endif
#endif
}


OMP_TASK(inout([NB*NB]A) label(TILE(lauum)), LAUUM_TARGET)
void TILE(lauum)(REAL *A, int NB)
{
	char L = 'L';
	int INFO;
	LAPACK(lauum)(&L, &NB, A, &NB, &INFO);
}


OMP_TASK(in([NB*NB]T) inout([NB*NB]B), TRMM_TARGET)
void TILE(trmm)(REAL *T, REAL *B, int NB, char side, char trans, char diag, REAL alpha, unsigned priority)
{
	char uplo = 'L';
	LAPACK(trmm)(&side, &uplo, &trans, &diag, &NB, &NB, &alpha, T, &NB, B, &NB);
}


OMP_TASK(inout([NB*NB]A), TRTRI_TARGET)
void TILE(trtri)(REAL *A, int NB, char diag)
{
	char L = 'L';
	LONG INFO;
	LAPACK(trtri)(&L, &diag, &NB, A, &NB, &INFO);
}


// commutative(C) ?
OMP_TASK(in([NB*NB]A, [NB*NB]B) inout([NB*NB]C) label(TILE(gemm)), GEMM_TARGET)
void TILE(gemm)(REAL *A, REAL *B, REAL *C, int NB, char transa, char transb, REAL alpha, REAL beta)
{
#if GEMM_SMP
	LAPACK(gemm)(&transa, &transb, &NB, &NB, &NB, &alpha, A, &NB, B, &NB, &beta, C, &NB);
#else
	REAL DONE = 1.0, DMONE = -1.0;
	GPUBLAS(gemm)(transa, transb, NB, NB, NB, DMONE, A, NB, B, NB, DONE, C, NB);
#endif
}


OMP_TASK(in([NB*NB]T) inout([NB*NB]B) label(TILE(trsm)), priority(priority) TRSM_TARGET)
void TILE(trsm)(REAL *T, REAL *B, int NB, char side, char trans, REAL alpha, unsigned priority)
{
	char LO = 'L';
#if TRSM_SMP
	char di = 'N';
	LAPACK(trsm)(&side, &LO, &trans, &di, &NB, &NB, &alpha, T, &NB, B, &NB);
#else
	REAL DONE = 1.0;
	GPUBLAS(trsm)(RI, LO, trans, NU, NB, NB, DONE, T, NB, B, NB);
#endif
}


// commutative(C) ?
OMP_TASK(in([NB*NB]A) inout([NB*NB]C), priority(priority) SYRK_TARGET)
void TILE(syrk)(REAL *A, REAL *C, int NB, char trans, REAL alpha, unsigned priority)
{
	char LO = 'L';
	REAL DONE = 1.0;
#if POTRF_SMP
	LAPACK(syrk)(&LO, &trans, &NB, &NB, &alpha, A, &NB, &DONE, C, &NB);
#else
	REAL DMONE = -1.0;
	GPUBLAS(syrk)(LO, NT, NB, NB, DMONE, A, NB, DONE, C, NB);
#endif
}



//----------------------------------------------------------------------------------
//			 END TASKS FOR CHOLESKY
//----------------------------------------------------------------------------------

//*****************
//Stage 1: Cholesky factorization
void cholesky_inverse(int ts, int nt, REAL *Ah[nt][nt])
{
	int i, j, k;

	for (i = 0; i < nt; i++)
	{
		/* --------------------- Stage 1 : Factorize A = L*t(L) ---------------------- */
		POTRF(Ah[i][i], ts);

		// Triangular systems
		for (j = i + 1; j < nt; j++)
			TRSM(Ah[i][i], Ah[i][j], ts, 'R', 'T', 1.0, (nt - j) + 10);

		// update trailing matrix
		for (j = i + 1; j < nt; j++)
		{
			for (k = i + 1; k < j; k++)
				GEMM(Ah[i][j], Ah[i][k], Ah[k][j], ts, 'N', 'T', -1.0, 1.0);
			SYRK(Ah[i][j], Ah[j][j], ts, 'N', -1.0, (nt - j) + 10);

			//SYRK (Ah[i][j], Ah[j][j], ts, 'N', -1.0, (nt - j) + 10);
			//for (k = i + 1; k < j; k++)
			//	GEMM(Ah[i][j], Ah[i][k], Ah[k][j], ts, 'N', 'T', -1.0, 1.0);
		}


		/* ----------------------- Stage 2 : Invert A = L^(-1) ------------------------ */
		for (j = i + 1; j < nt; j++)
		{
			TRSM(Ah[i][i], Ah[i][j], ts, 'R', 'N', -1.0, (nt - j) + 10);
			for (k = 0; k < i; k++)
				GEMM(Ah[i][j], Ah[k][i], Ah[k][j], ts, 'N', 'N', 1.0, 1.0);
		}

		for (j = 0; j < i; j++)
			TRSM(Ah[i][i], Ah[j][i], ts, 'L', 'N', 1.0, (nt - j) + 10);

		TRTRI(Ah[i][i], ts, 'N');


		/* --------------- Stage 3 : Compute A^(-1) = t(L^(-1)) * L^(-1) --------------- */
		for (k = 0; k < i; k++)
		{
			SYRK(Ah[k][i], Ah[k][k], ts, 'T', 1.0, (nt - k) + 10);
			for (j = k + 1; j < i; j++)
				GEMM(Ah[j][i], Ah[k][i], Ah[k][j], ts, 'T', 'N', 1.0, 1.0);
		}

		for (k = 0; k < i; k++)
			TRMM(Ah[i][i], Ah[k][i], ts, 'L', 'T', 'N', 1.0, (nt - k) + 10);

		LAUUM(Ah[i][i], ts);
	}

	#pragma omp taskwait noflush
}


// MKL and others need to allocate buffers before computing.
void warmup()
{
	int block = 512;
	REAL *A = (REAL *) malloc(block * block * sizeof(REAL));
	REAL *B = (REAL *) malloc(block * block * sizeof(REAL));
	REAL *C = (REAL *) malloc(block * block * sizeof(REAL));

	char transa = 'N';
	char transb = 'N';

	REAL DONE = 1.0, DMONE = -1.0;

	LAPACK(gemm)(&transa, &transb, &block, &block, &block, &DONE, A, &block, B, &block, &DMONE, C, &block);

	free(A);
	free(B);
	free(C);
}

//--------------------------- MAIN --------------------
int main(int argc, char *argv[])
{
	if (argc != 3 && argc != 4)
		errx(1, "Usage: inverse size block_size [matrix_file]");

	const long long n = atoll(argv[1]);
	const long long ts = atoll(argv[2]);
	const long long nt = n / ts;

	if (n % ts)
		err(1, "The block size %lld must divide the matrix size %lld", ts, n);

#if USE_NUMA
	nanos_get_num_sockets(&num_nodes);
	fprintf(stderr, "Running with %d nodes\n", num_nodes);
#endif

	// Allocate matrix
	REAL *matrix          = malloc(n * n * sizeof(REAL));
	REAL **Ah             = malloc(nt * nt * sizeof(REAL *));

	if (matrix == NULL || Ah == NULL)
		err(1, "ALLOCATION ERROR");

#if NANOS_API_COPIES_API >= 1004
	#pragma omp register([n*n]matrix)
#endif

	int i;
	for (i = 0; i < nt * nt; i++)
	{
		Ah[i] = CATCHROI_INSTRUMENT(aligned_alloc)(getpagesize(), ts * ts * sizeof(REAL));
		if (Ah[i] == NULL)
			err(1, "ALLOCATION ERROR (Ah[%d])", i);

#if NANOS_API_COPIES_API >= 1004
		REAL *block = Ah[i];
		#pragma omp register([ts*ts]block)
#endif
	}

	REAL checksum;
	read_matrix(argc == 4 ? argv[3] : NULL, n, (REAL(*)[n])matrix, &checksum);


#if !CONVERT_REQUEST
	convert_to_blocks(ts, nt, n, (REAL(*)[n]) matrix, (REAL * (*)[nt]) Ah);
	#pragma omp taskwait noflush
#endif

	float time = get_time();
	start_roi();

	cholesky_inverse(ts, nt, (REAL*(*)[nt])Ah);

	stop_roi(-1);
	time = get_time() - time;

	#pragma omp taskwait noflush

	char uplo = 'L';
	REAL eps = BLAS_dfpinfo(blas_eps);
	// check_inverse(nt, ts, n, Ah, matrix, uplo, eps);
	check_inverse_blocked(nt, ts, n, Ah, matrix, uplo, eps);

	float gflops = ((n * n * n) + (n * n) + n) / (time * 1e9);

	// Print configuration
	printf("\tCONVERT_TASK " "%d\n", CONVERT_TASK);
	printf("\tCONVERT_REQUEST %d\n", CONVERT_REQUEST);
	printf("\tPOTRF_SMP "    "%d\n", POTRF_SMP);
	printf("\tPOTRF_NESTED " "%d\n", POTRF_NESTED);
	printf("\tMAGMA_BLAS "   "%d\n", MAGMA_BLAS);
	printf("\tTRSM_SMP "     "%d\n", TRSM_SMP);
	printf("\tTRSM_NESTED "  "%d\n", TRSM_NESTED);
	printf("\tSYRK_SMP "     "%d\n", SYRK_SMP);
	printf("\tSYRK_NESTED "  "%d\n", SYRK_NESTED);
	printf("\tGEMM_SMP "     "%d\n", GEMM_SMP);
	printf("\tGEMM_NESTED "  "%d\n", GEMM_NESTED);
	//printf("\tCHECK_THRESH " "%g\n", 60 * eps);

	// Print results
	printf("============ CHOLESKY RESULTS ============\n");
	printf("  matrix size:          %lldx%lld\n", n, n);
	printf("  block size:           %lldx%lld\n", ts, ts);
	printf("  time (s):             %f\n", time);
	printf("  performance (gflops): %f\n", gflops);
	printf("==========================================\n");

	// Free blocked matrix
	for (i = 0; i < nt * nt; i++)
		free(Ah[i]);

	free(Ah);
	free(matrix);

	return 0;
}
