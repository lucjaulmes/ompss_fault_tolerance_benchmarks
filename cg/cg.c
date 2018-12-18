#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/types.h>
#include <unistd.h>

#include <catchroi.h>

#include "global.h"
#include "debug.h"

#include "cg.h"

#if VERBOSE >= SHOW_TASKINFO
void *first_p = NULL;
static inline int which_p_copy(const void *ptr)
{
	return (ptr == first_p ? 0 : 1);
};
#endif

void scalar_product_task(const double *p, const double *Ap, double* r)
{
	int i;
	for (i = 0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <p, Ap>
		#pragma omp task concurrent(*r) in(p[s:e-1], Ap[s:e-1]) firstprivate(s, e) label(dotp) priority(10) no_copy_deps
		{
			double local_r = 0;
			for (int k = s; k < e; k++)
				local_r += p[k] * Ap[k];

			#pragma omp atomic
			*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <p[%d], Ap> block %d finished = %e\n", which_p_copy(p), world_block(i), local_r);
		}
	}
}

void norm_task(const double *v, double* r)
{
	int i;
	for (i = 0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- || v ||
		#pragma omp task concurrent(*r) in(v[s:e-1]) firstprivate(s, e) label(norm) priority(10) no_copy_deps
		{
			double local_r = 0;
			for (int k = s; k < e; k++)
				local_r += v[k] * v[k];

			#pragma omp atomic
			*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", world_block(i), local_r);
		}
	}
}

void update_gradient(double *gradient, double *Ap, double *alpha)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		#pragma omp task in(*alpha, Ap[s:e-1]) inout(gradient[s:e-1]) firstprivate(s, e) label(update_gradient) priority(10) no_copy_deps
		{
			for (int k = s; k < e; k++)
				gradient[k] -= (*alpha) * Ap[k];

			log_err(SHOW_TASKINFO, "Updating gradient part %d finished = %e with alpha = %e\n", world_block(i), norm(e - s, &(gradient[s])), *alpha);
		}
	}
}

void recompute_gradient(const Matrix *A, double *gradient, double *iterate, const double *b)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Aiterate <- A * iterate
		#pragma omp task in(A->r[s:e-1], A->c[A->r[s]:A->r[e]-1], A->v[A->r[s]:A->r[e]-1], {iterate[get_block_start(b):get_block_end(b)-1], b=0:nb_blocks-1}, b[s:e-1]) out(gradient[s:e-1]) firstprivate(s, e) label(recompute_gradient) priority(10) no_copy_deps
		{
			for (int l = s; l < e; l++)
			{
				gradient[l] = b[l];

				for (int k = A->r[l]; k < A->r[l + 1]; k++)
					gradient[l] -= A->v[k] * iterate[ A->c[k] ];
			}

			log_err(SHOW_TASKINFO, "A * x part %d finished = %e\n", world_block(i), norm(e - s, &(Aiterate[s])));
		}
	}
}

void update_p(double *p, double *old_p, double *gradient, double *beta)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// p <- beta * old_p + gradient
		#pragma omp task in(*beta, gradient[s:e-1], old_p[s:e-1]) out(p[s:e-1]) firstprivate(s, e) label(update_p) priority(10) no_copy_deps
		{
			for (int k = s; k < e; k++)
				p[k] = (*beta) * old_p[k] + gradient[k];

			log_err(SHOW_TASKINFO, "Updating p[%d from %d] part %d finished = %e with beta = %e\n", which_p_copy(p), which_p_copy(old_p), world_block(i), norm(e - s, &(p[s])), *beta);
		}
	}
}

void compute_Ap(const Matrix *A, double *p, double *Ap)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Ap <- A * p
		#pragma omp task in(A->r[s:e-1], A->c[A->r[s]:A->r[e]-1], A->v[A->r[s]:A->r[e]-1], {p[get_block_start(b):get_block_end(b)-1], b=0:nb_blocks-1}) out(Ap[s:e-1]) firstprivate(s, e) label(Axp) priority(20) no_copy_deps
		{
			for (int l = s; l < e; l++)
			{
				Ap[l] = 0;

				for (int k = A->r[l]; k < A->r[l + 1]; k++)
					Ap[l] += A->v[k] * p[ A->c[k] ];
			}

			log_err(SHOW_TASKINFO, "A * p[%d] part %d finished = %e\n", which_p_copy(localrank_ptr(p)), world_block(i), norm(e - s, &(Ap[s])));
		}
	}
}

void update_iterate(double *iterate, double *p, double *alpha)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// iterate <- iterate - alpha * p
		#pragma omp task in(*alpha, p[s:e-1]) inout(iterate[s:e-1]) firstprivate(s, e) label(update_iterate) priority(5) no_copy_deps
		{
			for (int k = s; k < e; k++)
				iterate[k] += (*alpha) * p[k];

			log_err(SHOW_TASKINFO, "Updating it (from p[%d]) part %d finished = %e with alpha = %e\n", which_p_copy(p), world_block(i), norm(e - s, &(iterate[s])), *alpha);
		}
	}
}

#pragma omp task in(*err_sq, *old_err_sq) out(*beta) label(compute_beta) priority(100) no_copy_deps
void compute_beta(const double *err_sq, const double *old_err_sq, double *beta)
{

	// on first iterations of a (re)start, old_err_sq should be INFINITY so that beta = 0
	*beta = *err_sq / *old_err_sq;

	log_err(SHOW_TASKINFO, "Computing beta finished : err_sq = %e ; old_err_sq = %e ; beta = %e\n", *err_sq, *old_err_sq, *beta);
}

#pragma omp task inout(*normA_p_sq, *err_sq) out(*alpha, *old_err_sq) label(compute_alpha) priority(100) no_copy_deps
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *alpha)
{

	*alpha = *err_sq / *normA_p_sq ;
	*old_err_sq = *err_sq;

	log_err(SHOW_TASKINFO, "Computing alpha finished : normA_p_sq = %+e ; err_sq = %e ; alpha = %e\n", *normA_p_sq, *err_sq, *alpha);

	// last consumer of these values : let's 0 them so the scalar product doesn't need to
	*err_sq = 0.0;
	*normA_p_sq = 0.0;
}

static inline void swap(double **v, double **w)
{
	double *swap = *v;
	*v = *w;
	*w = swap;
}

void solve_cg(const Matrix *A, const double *b, double *iterate, double convergence_thres)
{
	// do some memory allocations
	double norm_b, thres_sq;
	int r = -1, do_update_gradient = 0;
	double normA_p_sq = 0.0, err_sq = 0.0, old_err_sq = INFINITY, alpha = 0.0, beta = 0.0;

	double *p, *old_p, *Ap, *gradient;
	p        = (double*)CATCHROI_INSTRUMENT(calloc)(A->n, sizeof(double));
	old_p    = (double*)CATCHROI_INSTRUMENT(calloc)(A->n, sizeof(double));
	Ap       = (double*)CATCHROI_INSTRUMENT(calloc)(A->n, sizeof(double));
	gradient = (double*)CATCHROI_INSTRUMENT(calloc)(A->n, sizeof(double));


	// some parameters pre-computed, and show some informations
	norm_b = norm(A->n, b);
	thres_sq = convergence_thres * convergence_thres * norm_b;
#if VERBOSE >= SHOW_DBGINFO && (!defined PERFORMANCE || defined EXTRAE_EVENTS)
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);
#endif

#if VERBOSE >= SHOW_TASKINFO
	first_p = p;
#endif

	start_roi();

	// real work starts here

	for (r = 0; r < max_it; r++)
	{
		if ( --do_update_gradient > 0 )
		{
			update_gradient(gradient, Ap, &alpha);

			norm_task(gradient, &err_sq);

			compute_beta(&err_sq, &old_err_sq, &beta);

			update_iterate(iterate, old_p, &alpha);
		}
		else
		{
			// our initial guess is always 0, don't bother updating it
			if ( r > 0 )
				update_iterate(iterate, old_p, &alpha);

			recompute_gradient(A, gradient, iterate, b);

			norm_task(gradient, &err_sq);

			compute_beta(&err_sq, &old_err_sq, &beta);
		}

		update_p(p, old_p, gradient, &beta);

		compute_Ap(A, p, Ap);

		scalar_product_task(p, Ap, &normA_p_sq);

		// when reaching this point, all tasks of loop should be created.
		// then waiting starts: should be released halfway through the loop.
		// We want this to be after alpha on normal iterations, after AxIt in recompute iterations
		if ( !do_update_gradient )
		{
			#pragma omp taskwait on(old_err_sq) //, {iterate[get_block_start(i):get_block_end(i)-1], i=0:nb_blocks})
		}
		else
		{
			#pragma omp taskwait on(old_err_sq)
		}
		// swapping p's so we reduce pressure on the execution of the update iterate tasks
		// now output-dependencies are not conflicting with the next iteration but the one after
		{
			swap(&p, &old_p);

			//if( r > 0 )
			//  log_convergence(r-1, old_err_sq);

			if ( old_err_sq <= thres_sq )
				break;

			if ( do_update_gradient <= 0 )
				do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
		}

		// if we will recompute the gradient, prepare to listen for incoming iterate exchanges in compute_alpha
		compute_alpha(&err_sq, &normA_p_sq, &old_err_sq, &alpha);
	}

	#pragma omp taskwait
	// end of the math, showing infos
	stop_roi(r);
	//log_convergence(r, old_err_sq);


	log_out("CG method finished iterations:%d with error:%e\n", r, sqrt((err_sq == 0.0 ? old_err_sq : err_sq) / norm_b));

	free(p);
	free(old_p);
	free(Ap);
	free(gradient);
}
