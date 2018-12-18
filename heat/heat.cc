#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>

#include <catchroi.h>

#include "heat.hh"
#include "misc.hh"

void heat(algoparam_t &param, double start_time)
{
	double runtime = wtime();
	start_roi();

	double residual = 0.0, avg_dist = std::numeric_limits<double>::infinity();
	unsigned long iter;

	for (iter = 1; avg_dist >= param.convergence && iter <= param.maxiter; iter++)
	{
		bool check_convergence = (iter % param.checkiter) == 0 or (iter == param.maxiter);

		switch (param.algorithm)
		{
			case 0:
				relax_jacobi(*param.m, check_convergence, &residual);
				break;
			case 1:
				relax_gauss(*param.m, check_convergence, &residual);
				break;
			case 2:
				relax_redblack(*param.m, check_convergence, &residual);
				break;
			default:
				std::cerr << "Unknown algorithm selected (" << param.algorithm << ')' << std::endl;
				exit(1);
		}

		// check_convergence implies taskwait at the end of relax_xxx()
		if (check_convergence)
			// Get the euclidian norm divided by number of elements.
			avg_dist = sqrt(residual) / (param.resolution * param.resolution);
	}

	stop_roi(iter);

	double flop = iter * 11.0 * param.resolution * param.resolution;
	double end_time = wtime();

	runtime = end_time - runtime;

	std::cout << std::fixed << std::setprecision(3)
		<< "Time: " << runtime << " /-> Time total: " << end_time - start_time
		<< " (" << flop / 1e9 << " GFlop => " << flop / runtime / 1e6 << " MFlop/s)\n";
	std::cout << std::scientific << "Convergence to residual=" << residual << ", avg_dist=" << avg_dist << ": " << iter << " iterations\n";
}


void usage(char *argv0)
{
	std::cerr << "Usage: " << argv0 << " <input file> [--image output_file] [--dump result_file|--check dump_file]\n\n";
}


int main(int argc, char **argv)
{
	double start_time = wtime();

	if (argc < 2)
	{
		usage(argv[0]);
		return 1;
	}

	// read and config_check input file
	std::ifstream infile(argv[1]);
	algoparam_t param;
	bool config_check = false;

	if (infile.fail())
		std::cerr << "\nError: Cannot open \"" << argv[1] << "\" for reading.\n\n";

	else if (not param.read_input(infile))
		fprintf(stderr, "\nError: Error parsing input file.\n\n");

	else if (param.algorithm < 0 or param.algorithm >= 3)
		std::cerr << "Unknown algorithm selected (" << param.algorithm << ')' << std::endl;

	else
	{
		param.print_params();
		config_check = true;
	}

	infile.close();

	if (not config_check or not param.init())
	{
		if (config_check)
			fprintf(stderr, "Error in Solver initialization.\n\n");

		usage(argv[0]);
		return 1;
	}

	// parse optional command line options
	char *imgfile = NULL, *dumpfile = NULL, *checkfile = NULL;
	for (int i = 2; i < argc; i += 2)
	{
		std::string option(argv[i]);

		if (option.compare("--image") == 0)
		{
			imgfile = argv[i + 1];
			std::cout << "Writing image to " << imgfile << '\n';
		}
		else if (option.compare("--dump") == 0)
		{
			dumpfile = argv[i + 1];
			std::cout << "Dumping values to " << dumpfile << '\n';
		}
		else if (option.compare("--check") == 0)
		{
			checkfile = argv[i + 1];
			std::cout << "Checking values against " << checkfile << '\n';
		}
	}

	// do the work
	heat(param, start_time);

	// for plots
	if (imgfile != NULL)
	{
		std::ofstream resstream(imgfile, std::ios::out); //  | std::ios::binary ?
		if (!resstream.good())
		{
			fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", imgfile);
			usage(argv[0]);
			return 1;
		}

		param.write_image(resstream);
	}

	// error checking
	if (dumpfile != NULL)
	{
		std::ofstream dumpstream(dumpfile, std::ios::out | std::ios::binary);
		param.dump_values(dumpstream);
	}
	if (checkfile != NULL)
	{
		std::ifstream checkstream(checkfile, std::ios::in | std::ios::binary);
		double residual = param.check_values(checkstream);
		std::cout << "Difference with golden run: " << sqrt(residual) / (param.resolution * param.resolution) << '\n';
	}

	std::cout << "Time after extras: " << std::fixed << std::setprecision(3) << wtime() - start_time << '\n';

	return 0;
}
