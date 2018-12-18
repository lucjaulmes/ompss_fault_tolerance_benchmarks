#ifndef _HEAT_INNER_HH
#define _HEAT_INNER_HH

double inner_heat_step_inplace(const unsigned long BS, double *mu,
		       double *top, const bool wr_top,
		       double *bottom, const bool wr_bottom,
		       double *left, const bool wr_left,
		       double *right, const bool wr_right);

double inner_heat_step(const unsigned long BS, double *oldmu, double *mu,
		       double *top, const bool wr_top,
		       double *bottom, const bool wr_bottom,
		       double *left, const bool wr_left,
		       double *right, const bool wr_right);

void inner_heat_step_inplace_nocheck(const unsigned long BS, double *mu,
		       double *top, const bool wr_top,
		       double *bottom, const bool wr_bottom,
		       double *left, const bool wr_left,
		       double *right, const bool wr_right);

void inner_heat_step_nocheck(const unsigned long BS, double *oldmu, double *mu,
		       double *top, const bool wr_top,
		       double *bottom, const bool wr_bottom,
		       double *left, const bool wr_left,
		       double *right, const bool wr_right);

#endif
