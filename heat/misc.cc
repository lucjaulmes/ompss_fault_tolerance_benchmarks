#include <limits>
#include <sys/time.h>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#include "misc.hh"

double wtime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);

	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

algoparam_t::algoparam_t()
	: m(NULL),
	uvis(NULL),
	heatsrcs(NULL)
{
}

algoparam_t::~algoparam_t()
{
	if (m != NULL)
		delete m;

	if (heatsrcs != NULL)
		delete[] heatsrcs;

	if (uvis != NULL)
		delete[] uvis;
}

int algoparam_t::read_input(std::istream& infile)
{
	algoparam_t *param = this;

	unsigned i;
	std::string buf;

	std::getline(infile, buf);
	std::istringstream ss(buf);
	ss >> blocks;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> maxiter;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> checkiter;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> resolution;

	param->visres = param->resolution;
	//param->visres = 256;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> algorithm;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> convergence;

	buf.clear();
	std::getline(infile, buf);
	ss.str(buf);
	ss >> numsrcs;

	//(param->heatsrcs) =
	//  (heatsrc_t*) malloc(sizeof(heatsrc_t) * (param->numsrcs));
	heatsrcs = new heatsrc_t[numsrcs];

	for (i = 0; i < param->numsrcs; i++)
	{
		buf.clear();
		std::getline(infile, buf);
		ss.str(buf);
		ss >> heatsrcs[i].posx >> heatsrcs[i].posy >> heatsrcs[i].range >> heatsrcs[i].temp;
	}
	return 1;
}

int algoparam_t::init()
{
	if (resolution % blocks != 0)
	{
		std::cerr << "The number of blocks must divide the resolution!" << std::endl;
		exit(1);
	}
	m = new Matrix(blocks, resolution / blocks, algorithm == 0);

	for (unsigned long i = 0; i < numsrcs; ++i)
		m->setBorder(heatsrcs[i]);

	return 1;
}

void algoparam_t::print_params() const
{
	std::cout << "Blocks            : " << blocks << '\n';
	std::cout << "Iterations        : " << maxiter << '\n';
	std::cout << "Check iterations  : " << checkiter << '\n';
	std::cout << "Resolution        : " << resolution << '\n';
	std::cout << "Convergence       : " << convergence << '\n';
	std::cout << "Algorithm         : " << algorithm << " (";

	if (algorithm == 0)      std::cout << "Jacobi";
	else if (algorithm == 1) std::cout << "Gauss-Seidel";
	else if (algorithm == 2) std::cout << "Red-Black";

	std::cout << ")\nNum. Heat sources : " << numsrcs << '\n';

	std::ios_base::fmtflags save_flags(std::cout.flags());
	std::cout  << std::fixed << std::setprecision(2);
	for (unsigned i = 0; i < numsrcs; i++)
	{
		std::cout << "  " << std::setw(2) << (i + 1) << ": ("
				<< std::setw(2) << heatsrcs[i].posx << ", "
				<< std::setw(2) << heatsrcs[i].posy << ") "
				<< std::setw(2) << heatsrcs[i].range << ' '
				<< std::setw(2) << heatsrcs[i].temp << '\n';
	}
	std::cout.flags(save_flags);

	std::cout << std::flush;
}

void algoparam_t::write_image(std::ostream &f)
{
	if (uvis == NULL)
	{
		uvis = new double[visres * visres];
		std::fill(&uvis[0], &uvis[visres * visres - 1], 0.0);
	}

	coarsen();

	unsigned sizex = visres;
	unsigned sizey = visres;

	unsigned char r[1024], g[1024], b[1024];
	unsigned i, j, k;

	double min, max;

	j = 1023;

	// prepare RGB table
	for (i = 0; i < 256; i++)
	{
		r[j]=255; g[j]=i; b[j]=0;
		j--;
	}
	for (i = 0; i < 256; i++)
	{
		r[j]=255-i; g[j]=255; b[j]=0;
		j--;
	}
	for (i = 0; i < 256; i++)
	{
		r[j]=0; g[j]=255; b[j]=i;
		j--;
	}
	for (i = 0; i < 256; i++)
	{
		r[j]=0; g[j]=255-i; b[j]=255;
		j--;
	}

	min =  std::numeric_limits<double>::max();
	max = -std::numeric_limits<double>::max();

	// find minimum and maximum
	for (i = 0; i < sizey; i++)
		for (j = 0; j < sizex; j++)
		{
			double v = uvis[i * sizex + j];
			if (max < v)
				max = v;
			if (min > v)
				min = v;
		}


	f << "P3\n" << sizex << ' ' << sizey << '\n' << 255 << '\n';

	for (i = 0; i < sizey; i++)
	{
		for (j = 0; j < sizex; j++)
		{
			//k = (int)(1024.0*(u[i*sizex+j]-min)/(max-min));
			// gmiranda: 1024 is out of bounds
			k = static_cast<int>(1023.0 * (uvis[i * sizex + j] - min) / (max - min));
			f << r[k] << ' ' << g[k] << ' ' << b[k] << "  ";
		}
		f << '\n';
	}
}

void algoparam_t::dump_values(std::ostream& outfile)
{
	unsigned long NB = m->NB;
	unsigned long BS = m->BS;

	float row[resolution + 1];

	row[0] = resolution;
	for (unsigned i = 0; i < resolution; ++i)
		row[i + 1] = i / row[0];
	outfile.write((char*)row, sizeof(row));


	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned ii = 0; ii < BS; ++ii)
		{
			row[0] = (i * BS + ii) / float(resolution);

			float *pos = row + 1;
			for (unsigned long j = 0; j < NB; ++j)
			{
				double *matrix_row = m->block(i, j) + ii * BS;
				for (unsigned jj = 0; jj < BS; ++jj)
					*pos++ = matrix_row[jj];
			}

			outfile.write((char*)row, sizeof(row));
		}
}

double algoparam_t::check_values(std::istream& infile)
{
	unsigned long NB = m->NB;
	unsigned long BS = m->BS;

	float row[resolution + 1];
	infile.read((char*)row, sizeof(row));

	if (row[0] != float(resolution))
	{
		std::cerr << "Error: wrong float resolution in the check values file\n";
		return std::numeric_limits<double>::infinity();
	}

	double err = 0.0;
	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned ii = 0; ii < BS; ++ii)
		{
			infile.read((char*)row, sizeof(row));
			float *pos = row + 1, diff;

			for (unsigned long j = 0; j < NB; ++j)
			{
				double *matrix_row = m->block(i, j) + ii * BS;

				for (unsigned jj = 0; jj < BS; ++jj)
				{
					diff = *pos++ - float(matrix_row[jj]);
					err += diff * diff;
				}
			}
		}

	return err;
}

int algoparam_t::coarsen()
{
	unsigned long NB = m->NB;
	unsigned long BS = m->BS;

	for (unsigned long i = 0; i < NB; ++i)
		for (unsigned long j = 0; j < NB; ++j)
		{
			unsigned long extra = 0;
			for (unsigned ii = 0; ii < BS; ++ii)
			{
				for (unsigned jj = 0; jj < BS; ++jj)
					uvis[i * NB * BS * BS + j * BS + extra + jj] = m->block(i, j)[ii * m->BS + jj];

				extra += resolution;
			}
		}

	return 1;
}
