#ifndef CG_H_INCLUDED
#define CG_H_INCLUDED

#include "global.h"
#include "matrix.h"

void solve_cg(const Matrix *A, const double *b, double *iterate, double convergence_thres);

// all the algorithmical steps of CG that will be subdivided into tasks :
void update_gradient(double *gradient, double *Ap, double *alpha);
void recompute_gradient_mvm(const Matrix *A, double *iterate, double *Aiterate);
void recompute_gradient_update(double *gradient, double *Aiterate, const double *b);
void update_p(double *p, double *old_p, double *gradient, double *beta);
void update_iterate(double *iterate, double *p, double *alpha);
void compute_Ap(const Matrix *A, double *p, double *Ap);

void scalar_product_task(const double *p, const double *Ap, double* r);
void norm_task(const double *v, double* r);

void compute_beta(const double *err_sq, const double *old_err_sq, double *beta);
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *alpha);

#endif // CG_H_INCLUDED
