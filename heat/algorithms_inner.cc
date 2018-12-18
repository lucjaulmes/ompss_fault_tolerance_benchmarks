#include <algorithm>

#include "algorithms_inner.hh"

/* Average adjacent items while iterating over dest/cur rows:
 * for i=1..BS, do dest[i] = average(cur[i - 1], prev[i], cur[i + 1], next[i]),
 * where left/right are dest[-1] and dest[BS].
 * NB: the order is correct to be a Gauss-Seidel implementation if dest == cur. */
static inline
void update_row_nodiff(const unsigned long BS, double *dest_row, double left, double *prev_row, double *cur_row, double *next_row, double right)
{
	dest_row[0] = (left + cur_row[1] + prev_row[0] + next_row[0]) / 4;

	for (unsigned long j = 1; j < BS-1; ++j)
		dest_row[j] = (cur_row[j-1] + cur_row[j+1] + prev_row[j] + next_row[j]) / 4;

	dest_row[BS-1] = (cur_row[BS-2] + right + prev_row[BS-1] + next_row[BS-1]) / 4;
}


/* Same as update_row_nodiff, but return the sum of (dest[i] - cur[i])^2 */
static inline
double update_row(const unsigned long BS, double *dest_row, double left, double *prev_row, double *cur_row, double *next_row, double right)
{
	double diff = 0.0, new_val;

	new_val = (left + cur_row[1] + prev_row[0] + next_row[0]) / 4;
	diff += (cur_row[0] - new_val) * (cur_row[0] - new_val);
	dest_row[0] = new_val;

	for (unsigned long j = 1; j < BS-1; ++j)
	{
		new_val = (cur_row[j-1] + cur_row[j+1] + prev_row[j] + next_row[j]) / 4;
		diff += (cur_row[j] - new_val) * (cur_row[j] - new_val);
		dest_row[j] = new_val;
	}

	new_val = (cur_row[BS-2] + right + prev_row[BS-1] + next_row[BS-1]) / 4;
	diff += (cur_row[BS-1] - new_val) * (cur_row[BS-1] - new_val);
	dest_row[BS-1] = new_val;

	return diff;
}


/* put in mu[(i, j)] the average of mu[(i +/- 1, j +/-1)], using top, bottom, left, right
 * whenever the indexes would be outside of boundaries */
double inner_heat_step_inplace(const unsigned long BS, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right)
{
	typedef double (*rm)[BS];
	rm u = reinterpret_cast<rm>(mu);
	unsigned long i;
	double diff = 0.0;

	diff += update_row(BS, u[0], left[0], top, u[0], u[0 + 1], right[0]);

	for (i = 1; i < BS - 1; ++i)
		diff += update_row(BS, u[i], left[i], u[i - 1], u[i], u[i + 1], right[i]);

	diff += update_row(BS, u[BS - 1], left[BS - 1], u[BS - 2], u[BS - 1], bottom, right[BS - 1]);

	// Write to the halos if needed
	if (wr_top)
		std::copy(u[0], u[1], top);

	if (wr_bottom)
		std::copy(u[BS - 1], u[BS], bottom);

	if (wr_left)
		for (i = 0; i < BS; ++i)
			left[i] = u[i][0];

	if (wr_right)
		for (i = 0; i < BS; ++i)
			right[i] = u[i][BS - 1];

	return diff;
}

/* put in mu[(i, j)] the average of mu[(i +/- 1, j +/-1)], using top, bottom, left, right
 * whenever the indexes would be outside of boundaries */
void inner_heat_step_inplace_nocheck(const unsigned long BS, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right)
{
	typedef double (*rm)[BS];
	rm u = reinterpret_cast<rm>(mu);
	unsigned long i;

	update_row_nodiff(BS, u[0], left[0], top, u[0], u[1], right[0]);

	for (i = 1; i < BS - 1; ++i)
		update_row_nodiff(BS, u[i], left[i], u[i - 1], u[i], u[i + 1], right[i]);

	update_row_nodiff(BS, u[BS-1], left[BS-1], u[BS-2], u[BS-1], bottom, right[BS-1]);


	// Write to the halos if needed
	if (wr_top)
		std::copy(u[0], u[1], top);

	if (wr_bottom)
		std::copy(u[BS - 1], u[BS], bottom);

	if (wr_left)
		for (i = 0; i < BS; ++i)
			left[i] = u[i][0];

	if (wr_right)
		for (i = 0; i < BS; ++i)
			right[i] = u[i][BS - 1];
}


/* put in mu[(i, j)] the average of old_mu[(i +/- 1, j +/-1)], using top, bottom, left, right
 * whenever the indexes would be outside of boundaries */
void inner_heat_step_nocheck(const unsigned long BS, double *oldmu, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right)
{
	typedef double (*rm)[BS];
	rm u = reinterpret_cast<rm>(mu);
	rm old_u = reinterpret_cast<rm>(oldmu);
	unsigned long i;

	update_row_nodiff(BS, u[0], left[0], top, old_u[0], old_u[1], right[0]);

	for (i = 1; i < BS - 1; ++i)
		update_row_nodiff(BS, u[i], left[i], old_u[i - 1], old_u[i], old_u[i + 1], right[i]);

	update_row_nodiff(BS, u[BS-1], left[BS-1], old_u[BS-2], old_u[BS-1], bottom, right[BS-1]);


	// Write to the halos if needed
	if (wr_top)
		std::copy(u[0], u[1], top);

	if (wr_bottom)
		std::copy(u[BS - 1], u[BS], bottom);

	if (wr_left)
		for (i = 0; i < BS; ++i)
			left[i] = u[i][0];

	if (wr_right)
		for (i = 0; i < BS; ++i)
			right[i] = u[i][BS - 1];
}


/* put in mu[(i, j)] the average of old_mu[(i +/- 1, j +/-1)], using top, bottom, left, right
 * whenever the indexes would be outside of boundaries */
double inner_heat_step(const unsigned long BS, double *oldmu, double *mu,
		double *top, const bool wr_top,
		double *bottom, const bool wr_bottom,
		double *left, const bool wr_left,
		double *right, const bool wr_right)
{
	typedef double (*rm)[BS];
	rm u = reinterpret_cast<rm>(mu);
	rm old_u = reinterpret_cast<rm>(oldmu);
	unsigned long i;
	double diff = 0.0;

	diff += update_row(BS, u[0], left[0], top, old_u[0], old_u[0 + 1], right[0]);

	for (i = 1; i < BS - 1; ++i)
		diff += update_row(BS, u[i], left[i], old_u[i - 1], old_u[i], old_u[i + 1], right[i]);

	diff += update_row(BS, u[BS - 1], left[BS - 1], old_u[BS - 2], old_u[BS - 1], bottom, right[BS - 1]);


	// Write to the halos if needed
	if (wr_top)
		std::copy(u[0], u[1], top);

	if (wr_bottom)
		std::copy(u[BS - 1], u[BS], bottom);

	if (wr_left)
		for (i = 0; i < BS; ++i)
			left[i] = u[i][0];

	if (wr_right)
		for (i = 0; i < BS; ++i)
			right[i] = u[i][BS - 1];


	return diff;
}
