#include <cmath>
#include <algorithm>
#include <iostream>
#include <unistd.h>

#include <catchroi.h>

#include "matrix.hh"

static void border_step(const long BS, const double limit, const double range, const double temp, long x, const double src_x, const double y_sq, double *b);

static inline void fill_zero(const long size, double *b)
{
	std::fill(&b[0], &b[size], 0.0);
}

Matrix::Matrix(unsigned long NB, unsigned long BS, bool dbuffer)
	: NB(NB), BS(BS), dbuffer(dbuffer), current(0)
{
	const unsigned long page_size = sysconf(_SC_PAGESIZE);

	/* Allocate the matrix blocks array, then every array of blocks, aligned to memory page boundary */
	u[0] = new double*[NB*NB];
	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned long j = 0; j < NB; ++j)
		{
			if ((u[0][i*NB+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*BS*sizeof(double))) == NULL)
				std::cerr << "There was an error allocating matrix `u'!" << std::endl;

			fill_zero(BS*BS, u[0][i*NB+j]);
		}

	/* Allocate NB rows and NB + 1 columns, each of size BS */
	vertical[0] = new double*[NB*(NB + 1)];
	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned long j = 0; j < NB + 1; ++j)
		{

			if ((vertical[0][i*(NB+1)+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*sizeof(double))) == NULL)
				std::cerr << "There was an error allocating vertical halo!" << std::endl;

			fill_zero(BS, vertical[0][i*(NB+1)+j]);
		}

	/* Allocate NB + 1 rows and NB columns, each of size BS */
	horizontal[0] = new double*[(NB + 1)*NB];
	for (unsigned long i = 0; i < NB + 1; ++i)
		for (unsigned long j = 0; j < NB; ++j)
		{
			if ((horizontal[0][i*NB+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*sizeof(double))) == NULL)
				std::cerr << "There was an error allocating horizontal halo!" << std::endl;

			fill_zero(BS, horizontal[0][i*NB+j]);
		}

	if (dbuffer)
	{
		/* same allocation, second matrix */
		u[1] = new double*[NB*NB];
		for (unsigned long i = 0; i < NB; ++i)
			for (unsigned long j = 0; j < NB; ++j)
			{
				if ((u[1][i*NB+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*BS*sizeof(double))) == NULL)
					std::cerr << "There was an error allocating matrix `u'!" << std::endl;

				// Init not needed because data will be overwritten before being used.
				//fill_zero(BS*BS, u[1][i*NB+j]);
			}

		/* vertical halos, NB rows x NB + 1 columns: reuse first and last column, allocate the rest anew */
		vertical[1] = new double*[NB*(NB + 1)];
		for (unsigned long i = 0; i < NB; ++i)
		{
			vertical[1][i*(NB+1)+0] = vertical[0][i*(NB+1)+0];
			vertical[1][i*(NB+1)+NB] = vertical[0][i*(NB+1)+NB];
			for (unsigned long j = 1; j < NB; ++j)
			{
				if ((vertical[1][i*(NB+1)+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*sizeof(double))) == NULL)
					std::cerr << "There was an error allocating vertical halo!" << std::endl;

				fill_zero(BS, vertical[1][i*(NB+1)+j]);
			}
		}

		/* horizontal halos, NB + 1 rows x NB columns: reuse first and last column, allocate the rest anew */
		horizontal[1] = new double*[(NB + 1)*NB];
		for (unsigned long j = 0; j < NB; ++j)
		{
			horizontal[1][0*NB+j] = horizontal[0][0*NB+j];
			horizontal[1][NB*NB+j] = horizontal[0][NB*NB+j];
			for (unsigned long i = 1; i < NB; ++i)
			{
				if ((horizontal[1][i*NB+j] = (double*)CATCHROI_INSTRUMENT(aligned_alloc)(page_size, BS*sizeof(double))) == NULL)
					std::cerr << "There was an error allocating horizontal halo!" << std::endl;

				fill_zero(BS, horizontal[1][i*NB+j]);
			}
		}
	}
}

Matrix::~Matrix()
{
	for (unsigned long i = 0; i < NB*NB; ++i)
		free(u[0][i]);

	delete[] u[0];


	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned long j = 0; j < NB + 1; ++j)
			free(vertical[0][i*(NB+1)+j]);

	delete[] vertical[0];


	for (unsigned long i = 0; i < NB + 1; ++i)
		for (unsigned long j = 0; j < NB; ++j)
			free(horizontal[0][i*NB+j]);

	delete[] horizontal[0];


	if (dbuffer)
	{
		for (unsigned long i = 0; i < NB*NB; ++i)
			free(u[1][i]);

		delete[] u[1];


		for (unsigned long i = 0; i < NB; ++i)
			for (unsigned long j = 1; j < NB; ++j)
				free(vertical[1][i*(NB+1)+j]);

		delete[] vertical[1];


		for (unsigned long i = 1; i < NB; ++i)
			for (unsigned long j = 0; j < NB; ++j)
				free(horizontal[1][i*NB+j]);

		delete[] horizontal[1];
	}
}

void Matrix::setBorder(const heatsrc_t &src)
{
	double limit = NB*BS-1;

	// TOP row
	double y_sq = std::pow(src.posy, 2.0);
	for (unsigned long bl = 0; bl < NB; ++bl)
		border_step(BS, limit, src.range, src.temp, bl*BS, src.posx, y_sq, top(0, bl));

	// RIGHT column
	y_sq = std::pow(1.0 - src.posx, 2.0);
	for (unsigned long bl = 0; bl < NB; ++bl)
		border_step(BS, limit, src.range, src.temp, bl*BS, src.posy, y_sq, right(bl, NB-1));

	// BOTTOM row
	y_sq = std::pow(1.0 - src.posy, 2.0);
	for (unsigned long bl = 0; bl < NB; ++bl)
		border_step(BS, limit, src.range, src.temp, bl*BS, src.posx, y_sq, bottom(NB-1, bl));

	// LEFT column
	y_sq = std::pow(src.posx, 2.0);
	for (unsigned long bl = 0; bl < NB; ++bl)
		border_step(BS, limit, src.range, src.temp, bl*BS, src.posy, y_sq, left(bl, 0));
}


static void border_step(const long BS, const double limit, const double range, const double temp, long x, const double src_x, const double y_sq, double *b)
{
	/* for all within block */
	for (long i = 0; i < BS; ++i)
	{
		/* dist(src, here) = sqrt( y² + ((x+i)/limit - src)² ) */
		double dist = std::sqrt(std::pow(double(x+i)/limit - src_x, 2.0) + y_sq);

		if (dist <= range)
			/* (range - dist) / range = 1..0 on dist=0..range,
			 * => fill b with temp, decreasing linearly with distance until range */
			b[i] += (range-dist) / range * temp;
	}
}
