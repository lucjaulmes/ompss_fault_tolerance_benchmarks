#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <err.h>

#include "kmeans.h"

int main(int argc, char *argv[])
{
	if (argc < 3 || argc > 5)
		errx(0, "Usage: %s num_points dimension output_file [seed]", argv[0]);

	Header h;

	std::istringstream(argv[1]) >> h.num;
	std::istringstream(argv[2]) >> h.dim;

	std::ofstream of(argv[3], std::ios::binary);
	if (!of.good())
		err(1, "Can not open %s", argv[3]);

	long seed = time(NULL);
	if (argc > 4)
		std::istringstream(argv[4]) >> seed;

	std::mt19937 gen(seed);
	std::uniform_real_distribution<> dis(0, 100.0);

	of.write(reinterpret_cast<char *>(&h), sizeof(h));
	for (long i = 0; i < h.num * h.dim; ++i)
    {
        double coord = dis(gen);
        of.write(reinterpret_cast<char *>(&coord), sizeof(coord));
    }

	of.close();
	return 0;
}
