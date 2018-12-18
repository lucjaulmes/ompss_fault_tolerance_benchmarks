#include <algorithm>

#include "matrix.hh"
#include "heat.hh"
#include "algorithms_inner.hh"

/* wrap the inner algorithms inside tasks */
static inline void heat_step_inplace(const unsigned long BS, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right,
		double *sum, const bool check)
{
	if (check)
	{
		#pragma omp task label(heat_step) inout([BS*BS]mu, [BS]top, [BS]bottom, [BS]left, [BS]right) reduction(+:[1]sum) no_copy_deps
		*sum += inner_heat_step_inplace(BS, mu, top, wr_top, bottom, wr_bottom, left, wr_left, right, wr_right);
	}
	else
	{
		#pragma omp task label(heat_step) inout([BS*BS]mu, [BS]top, [BS]bottom, [BS]left, [BS]right) no_copy_deps
		inner_heat_step_inplace_nocheck(BS, mu, top, wr_top, bottom, wr_bottom, left, wr_left, right, wr_right);
	}
}


static inline void heat_step(const unsigned long BS, double *oldmu, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right,
		double *sum, const bool check)
{
	if (check)
	{
		#pragma omp task label(heat_step) in([BS*BS]oldmu) out([BS*BS]mu) inout([BS]top, [BS]bottom, [BS]left, [BS]right) reduction(+:[1]sum) no_copy_deps
		*sum += inner_heat_step(BS, oldmu, mu, top, wr_top, bottom, wr_bottom, left, wr_left, right, wr_right);
	}
	else
	{
		#pragma omp task label(heat_step) in([BS*BS]oldmu) out([BS*BS]mu) inout([BS]top, [BS]bottom, [BS]left, [BS]right) no_copy_deps
		inner_heat_step_nocheck(BS, oldmu, mu, top, wr_top, bottom, wr_bottom, left, wr_left, right, wr_right);
	}
}


void relax_gauss(Matrix &m, bool check, double *residual)
{
	if (check)
		*residual = 0.0;

	long BS = m.BS;
	long NB = m.NB;
	long max_diag = 2 * NB - 1;

	/* Iterate over all blocks based on their distance to the diagonal (in increasing order),
	 * and update each block in place. */
	for (long diag = 0; diag < max_diag; ++diag)
	{
		long start_i = std::max(0L, diag - NB + 1);
		long max_i = std::min(diag + 1, NB);
		for (long i = start_i; i < max_i; ++i)
		{
			long j = diag - i;
			heat_step_inplace(BS, m.block(i, j),
					m.top(i, j),	i != 0,
					m.bottom(i, j),	i != NB - 1,
					m.left(i, j),	j != 0,
					m.right(i, j),	j != NB - 1,
					residual, check);
		}
	}

	if (check)
	{
		#pragma omp taskwait
	}
}

void relax_jacobi(Matrix &m, bool check, double *residual)
{
	if (check)
		*residual = 0.0;

	++m; // change of matrix!
	long BS = m.BS;
	long NB = m.NB;

	/* Iterate over all blocks in arbitrary order and update from old into new. */
	for (long i = 0; i < NB; ++i)
	{
		for (long j = 0; j < NB; ++j)
		{
			heat_step(BS, m.oldblock(i, j), m.block(i, j),
					m.top(i, j),	i != 0,
					m.bottom(i, j),	i != NB - 1,
					m.left(i, j),	j != 0,
					m.right(i, j),	j != NB - 1,
					residual, check);
		}
	}

	if (check)
	{
		#pragma omp taskwait
	}
}


void relax_redblack(Matrix &m, bool check, double *residual)
{
	if (check)
		*residual = 0.0;

	long BS = m.BS;
	long NB = m.NB;

	/* Iterate over all blocks with (i + j % 2 == 0) in arbitrary order and update in place,
	 * then do the same for all blocks with (i + j % 2 == 1). */
	for (long rb = 0; rb < 2; ++rb)
	{
		for (long j = 0; j < NB; ++j)
		{
			long i_start = (j + rb) % 2;
			for (long i = i_start; i < NB; i += 2)
			{
				heat_step_inplace(BS, m.block(i, j),
						m.top(i, j),	i != 0,
						m.bottom(i, j),	i != NB - 1,
						m.left(i, j),	j != 0,
						m.right(i, j),	j != NB - 1,
						residual, check);
			}
		}
	}

	if (check)
	{
		#pragma omp taskwait
	}
}


