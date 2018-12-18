#ifndef _MATRIX_HH
#define _MATRIX_HH

struct heatsrc_t
{
	double posx;
	double posy;
	double range;
	double temp;
};

struct Matrix
{
	unsigned long NB;
	unsigned long BS;

private:

	/* u contains up to 2 matrices, as linearized arrays of blocks indexed by (row * NB + col),
	 * each block is a linearized array of values indexed by (row * BS + col). */
	double **u[2]; // u matrix
	double **vertical[2];   // left and right halos (containers)
	double **horizontal[2]; // top and bottom halos (containers)
	const bool dbuffer;
	int current;

public:

	Matrix(unsigned long NB, unsigned long BS, bool dbuffer=false);
	~Matrix();

	void setBorder(const heatsrc_t &src);

	/* convenience function to get block (i, j) at current matrix (or only matrix if unique) */
	inline double * block(unsigned long i, unsigned long j)
	{
		return u[dbuffer ? current : 0][i*NB + j];
	}

	/* convenience function to get block (i, j) at previous matrix (or only matrix if unique) */
	inline double * oldblock(unsigned long i, unsigned long j)
	{
		return u[dbuffer ? 1-current : 0][i*NB + j];
	}

	inline double * top(unsigned long i, unsigned long j)
	{
		return horizontal[dbuffer ? 1-current : 0][i*NB + j];
	}

	inline double * bottom(unsigned long i, unsigned long j)
	{
		return horizontal[dbuffer ? current : 0][(i+1)*NB + j];
	}

	inline double * left(unsigned long i, unsigned long j)
	{
		return vertical[dbuffer ? 1-current : 0][i*(NB+1) + j];
	}

	inline double * right(unsigned long i, unsigned long j)
	{
		return vertical[dbuffer ? current : 0][i*(NB+1) + j+1];
	}

	/* increments 'current' modulo 2 to select which matrix is current (when using 2 matrices) */
	inline Matrix & operator++()
	{
		current = (current + 1)%2;

		return *this;
	}
};
#endif
