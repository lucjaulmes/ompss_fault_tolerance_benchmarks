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


void LAPACK(laset)(const char *UPLO, const int *M, const int *n, const REAL *ALPHA, const REAL *BETA, REAL *A, const int *LDA);
void LAPACK(gemm)(const char *transa, const char *transb, const int *l, const int *n, const int *m, REAL *alpha, const void *a, const int *lda, void *b, const int *ldb, REAL *beta, void *c, const int *ldc);
REAL LAPACK(lange)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *work);
void LAPACK(lacpy)(const char *norm, const int *m, const int *n, const REAL *a, const int *lda, REAL *b, const int *ldb);
void LAPACK(larnv)(const int *idist, int *iseed, const int *n, REAL *x);
void LAPACK(potrf)(const char *uplo, const int *n, REAL *a, const int *lda, LONG *info);
void LAPACK(trsm)(char *side, char *uplo, char *transa, char *diag, const int *m, const int *n, REAL *alpha, REAL *a, const int *lda, REAL *b, const int *ldb);
void LAPACK(trmm)(char *side, char *uplo, char *transa, char *diag, const int *m, const int *n, REAL *alpha, REAL *a, const int *lda, REAL *b, const int *ldb);
void LAPACK(syrk)(char *uplo, char *trans, const int *n, const int *k, REAL *alpha, REAL *a, const int *lda, REAL *beta, REAL *c, const int *ldc);

float get_time();
static int check_factorization(int, REAL *, REAL *, char, REAL);

//void gpu_spotf2_var1_(char *, int *, unsigned int *, int *, int *);
void gpu_spotrf_var1_(char *, int *, unsigned int *, int *, int *, int *);

void cholesky(const int n, const int nt, const int ts, REAL (*Alin)[n], REAL (*(*Ah)[nt])[ts]);


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


void add_to_diag_hierarchical(REAL **matrix, const int ts, const int nt, float alpha)
{
	int i;

	for (i = 0; i < nt * ts; i++)
		matrix[(i / ts) * nt + (i / ts)][(i % ts) * ts + (i % ts)] += alpha;
}


void add_to_diag(const int n, REAL (*matrix)[n], REAL alpha)
{
	for (int i = 0; i < n; i++)
		matrix[i][i] += alpha;
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
static int check_factorization(const int n, REAL *A1, REAL *A2, char uplo, REAL eps)
{
	int i, j;
	char NORM = 'I', LE = 'L', TR = 'T', NU = 'N', RI = 'R';

	REAL *L2       = calloc(n * n, sizeof(REAL));
	REAL *work     = calloc(n, sizeof(REAL));
	REAL alpha     = 1.0;

	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, n, n, A1, n, &Anorm);
	REAL Anorm = LAPACK(lange)(&NORM, &n, &n, A1, &n, work);

	/* Dealing with L'L or U'U  */
	LAPACK(lacpy)(&uplo, &n, &n, A2, &n, L2, &n);

	if (uplo == 'U')
		/* L2 = 1 * A2^T * L2 */
		LAPACK(trmm)(&LE, &uplo, &TR, &NU, &n, &n, &alpha, A2, &n, L2, &n);
	else
		/* L2 = 1 * L2 * A2^T */
		LAPACK(trmm)(&RI, &uplo, &TR, &NU, &n, &n, &alpha, A2, &n, L2, &n);

	/* Compute the residual || A - L'L || */
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			A1[j * n + i] -= L2[j * n + i];

	REAL Rnorm = LAPACK(lange)(&NORM, &n, &n, A1, &n, work);
	//BLAS_ge_norm(blas_colmajor, blas_inf_norm, n, n, A1, n, &Rnorm);

	free(L2);
	free(work);

	printf("============\n");
	printf("Checking the Cholesky Factorization \n");
	printf("-- ||L'L-A||_inf/(||A||_inf.n.eps) = %e\n", Rnorm / (Anorm * n * eps));

	int info_factorization = isnan(Rnorm / (Anorm * n * eps)) || isinf(Rnorm / (Anorm * n * eps)) || (Rnorm / (Anorm * n * eps) > 60.0);

	if (info_factorization)
		printf("-- Factorization is suspicious ! \n");
	else
		printf("-- Factorization is CORRECT ! \n");

	return info_factorization;
}


//--------------------- check results ----------------------------------------------

REAL sckres(const int n, REAL *A, int lda, REAL *L, int ldl)
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

void print_linear_matrix(const int n, REAL (*matrix)[n])
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			printf("%g ", matrix[i][j]);
		printf("\n");
	}
}


void read_matrix(char *filename, const int n, REAL (*matrix)[n], REAL *checksum)
{
	if (filename != NULL)
	{
		FILE *matrix_file = fopen(filename, "r");
		if (!matrix_file)
			err(1, "Could not open matrix file %s", filename);

		for (int i = 0; i < n * n; i++)
			if (fscanf(matrix_file, SCAN_REAL, &matrix[0][i]) == EOF)
				break;

		// Finished reading matrix; read checksum
		if (fscanf(matrix_file, SCAN_REAL, checksum) == EOF)
			errx(1, "Invalid matrix file: could not read checksum");
	}
	else
	{
		fprintf(stderr, "No matrix file, initializing matrix with random values\n");
		int ISEED[] = {0, 0, 0, 1}, intONE = 1;

		for (int i = 0; i < n; i++)
			LAPACK(larnv)(&intONE, &ISEED[0], &n, matrix[i]);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
			{
				matrix[j][i] = matrix[j][i] + matrix[i][j];
				matrix[i][j] = matrix[j][i];
			}

		add_to_diag(n, matrix, (REAL) n);

		*checksum = 0.0;
	}
}


#if CONVERT_TASK
// NB: with full-deps and support for changing dimensions: in(Alin_ij[0;ts][0;ts]) out([ts]Aij)
// now we need to flatten A to conform to future uses and could reference concurrent([n * n]full_matrix)
OMP_TASK(out([ts * ts](A[0])) label(gather_block), AFFINITY untied)
#endif
static void
gather_block(const int n, const int ts, REAL (*Alin)[n], REAL (*A)[ts])
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++)
			A[i][j] = Alin[i][j];
}


#if CONVERT_TASK
// NB: with full-deps and support for changing dimensions: in([ts]A) out(Alin_ij[0;ts][0;ts])
OMP_TASK(in([ts * ts](A[0])) label(scatter_block), AFFINITY untied)
#endif
static void
scatter_block(const int n, const int ts, REAL (*A)[ts], REAL (*Alin)[n])
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++)
			Alin[i][j] = A[i][j];
}


static void
#if CONVERT_ON_REQUEST
__attribute__((unused))
#endif
convert_to_blocks(const int n, const int nt, const int ts, REAL (*Alin)[n], REAL (*(*A)[nt])[ts])
{
	for (int i = 0; i < nt; i++)
		for (int j = 0; j < nt; j++)
		{
			REAL (*Aij)[ts] = A[i][j];
			REAL (*Alin_ij)[n] = (REAL (*)[n]) &Alin[i * ts][j * ts];
			gather_block(n, ts, Alin_ij, Aij);
		}
}


static void convert_to_linear(const int n, const int nt, const int ts, REAL (*(*A)[nt])[ts], REAL (*Alin)[n])
{
	for (int i = 0; i < nt; i++)
		for (int j = 0; j < nt; j++)
		{
			REAL (*Aij)[ts] = A[i][j];
			REAL (*Alin_ij)[n] = (REAL (*)[n]) &Alin[i * ts][j * ts];
			scatter_block(n, ts, Aij, Alin_ij);
		}
}


static inline REAL *malloc_block(const int ts)
{
	REAL *block = CATCHROI_INSTRUMENT(aligned_alloc)(4096, ts * ts * sizeof(REAL));

	if (block == NULL)
		err(1, "ALLOCATION ERROR (Ah block of %d elements)", ts);

	return block;
}


static inline void check_block(int CONVERT_ONREQ_ONLY n, int CONVERT_ONREQ_ONLY ts,
								REAL CONVERT_ONREQ_ONLY (*linaddr)[n],
								REAL CONVERT_ONREQ_ONLY (**blockaddr)[ts])
{
#if CONVERT_ON_REQUEST
	if (*blockaddr == NULL)
	{
		*blockaddr = (REAL(*)[ts])malloc_block(ts);
		gather_block(n, ts, linaddr, *blockaddr);
	}
#endif
}

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
	char TR = 'T', NT = 'N';
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

	char LO = 'L', NT = 'N';
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

void cholesky(const int n, const int nt, const int ts, REAL (*Alin)[n], REAL (*(*Ah)[nt])[ts])
{
	int i, j, k;

#if STOP_SCHED
	NANOS_SAFE(nanos_stop_scheduler());
	NANOS_SAFE(nanos_wait_until_threads_paused());
#endif

	// Shuffle across sockets
	for (k = 0; k < nt; k++) {
		check_block(n, ts, (REAL (*)[n]) &Alin[k * ts][k * ts], &Ah[k][k]);
		for (i = 0; i < k; i++)
		{
			check_block(n, ts, (REAL (*)[n]) &Alin[i * ts][i * ts], &Ah[i][i]);
			SYRK(Ah[i][k][0], Ah[k][k][0], ts);
		}

		// Diagonal Block factorization and panel permutations
		POTRF((REAL*)Ah[k][k], ts);

		// update trailing matrix
		for (i = k + 1; i < nt; i++)
		{
			for (j = 0; j < k; j++)
			{
				check_block(n, ts, (REAL (*)[n]) &Alin[k * ts][j * ts], &Ah[k][j]);
				check_block(n, ts, (REAL (*)[n]) &Alin[i * ts][j * ts], &Ah[j][i]);
				GEMM((REAL*)Ah[j][i], (REAL*)Ah[j][k], (REAL*)Ah[k][i], ts);
			}
			check_block(n, ts, (REAL (*)[n]) &Alin[k * ts][i * ts], &Ah[k][i]);
			TRSM((REAL*)Ah[k][k], (REAL*)Ah[k][i], ts);
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

	const int n = atoi(argv[1]);
	const int ts = atoi(argv[2]);
	const int nt = n / ts;


	char *filename = NULL;
	if (argc == 4)
		filename = argv[3];

	//nanos_get_num_sockets(&NUM_NODES);

	// Allocations

	REAL (*matrix)[n] = (REAL(*)[n]) aligned_alloc(4096, n * n * sizeof(REAL));
	if (matrix == NULL)
		err(1, "ALLOCATION ERROR");

	REAL (*original_matrix)[n] = aligned_alloc(4096, n * n * sizeof(REAL));
	if (original_matrix == NULL)
		err(1, "ALLOCATION ERROR\n");

	REAL  (*(*Ah)[nt])[ts] = (REAL (*(*)[nt])[ts])malloc(nt * nt * sizeof(REAL *));
	if (Ah == NULL)
		err(1, "ALLOCATION ERROR (Ah)");

	for (int i = 0; i < nt; i++)
		for (int j = 0; j < nt; j++)
		{
			Ah[i][j] = (REAL (*)[ts])malloc_block(ts);
			if (Ah[i][j] == NULL)
				err(1, "ALLOCATION ERROR (Ah[%d][%d])", i, j);
		}


	// Initialize matrix and copy
	REAL checksum;
	read_matrix(filename, n, matrix, &checksum);
	memcpy(original_matrix[0], matrix[0], n * n * sizeof(matrix[0][0]));

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

#if !CONVERT_ON_REQUEST
	convert_to_blocks(n, nt, ts, matrix, Ah);
	#pragma omp taskwait
#endif

	float t1 = get_time();
	start_roi();

#if USE_PAPI
	if(PAPI_start(EventSet) != PAPI_OK)
		errx(2, "ERROR in papi start!")
#endif

	// Actual work
	cholesky(n, nt, ts, matrix, Ah);

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

	//printf("Succesful energy measurement! value = %lld\n", values[0]);
	convert_to_linear(n, nt, ts, Ah, matrix);

#if CONVERT_TASK
	#pragma omp taskwait
#endif

	REAL eps = BLAS_dfpinfo(blas_eps);
	int info_factorization = check_factorization(n, (REAL*)original_matrix, (REAL*)matrix, 'L', eps);

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
	for (int i = 0; i < nt; i++)
		for (int j = 0; j < nt; j++)
			free(Ah[i][j]);

	free(Ah);
	free(matrix);
	free(original_matrix);

	return 0;
}

