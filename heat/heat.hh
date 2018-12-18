#ifndef _HEAT_HH
#define _HEAT_HH

#include "matrix.hh"

void relax_jacobi(Matrix &m, bool check, double *residual);
void relax_gauss(Matrix &m, bool check, double *residual);
void relax_redblack(Matrix &m, bool check, double *residual);

#endif
