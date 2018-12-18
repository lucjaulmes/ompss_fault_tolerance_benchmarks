#ifndef _MISC_HH
#define _MISC_HH

#include <iostream>

#include "matrix.hh"

struct algoparam_t
{
	unsigned blocks;		// number of blocks
	unsigned maxiter;       // maximum number of iterations
	unsigned checkiter;     // number of iterations to check convergence
	unsigned resolution;    // spatial resolution
	int algorithm;          // 0=>Jacobi, 1=>Gauss
	Matrix *m;
	unsigned visres;        // visualization resolution
	double convergence;
	double *uvis;
	unsigned numsrcs;     // number of heat sources
	heatsrc_t *heatsrcs;

	algoparam_t();
	~algoparam_t();
	int read_input(std::istream &infile);
	int init();
	void print_params() const;
	void write_image(std::ostream &file);
	void dump_values(std::ostream &outfile);
	double check_values(std::istream &infile);
	int coarsen();
};

double wtime();

#endif
