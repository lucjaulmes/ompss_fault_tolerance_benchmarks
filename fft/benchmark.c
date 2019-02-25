#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <err.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <stdint.h>
#include <math.h>
#include <complex.h>

#include <catchroi.h>

#include "fft.h"
#include "ref.h"


int open_numpype()
{
	int fd[2];
	if (pipe(fd))
		err(1, "pipe failed");
	int p = fork();

	if (p < 0)
		err(1, "fork failed");

	else if (p == 0)
	{
		// child
		close(fd[1]);
		dup2(fd[0], STDIN_FILENO);
		close(fd[0]);
		char *argv[] = {"python3", NULL};
		execvp(argv[0], argv);
	}

	// parent
	close(fd[0]);
	return fd[1];
}


double lehmer_random(double *seed)
{
	double d2 = .2147483647e10;
	*seed = fmod(16807. * (*seed), d2);
	return (*seed - 1.0) / (d2 - 1.0);
}


complex_struct * copy_space_to_ref(uint64_t n, double complex *a)
{
	complex_struct *b = malloc(n * sizeof(*b));
	for (size_t i = 0; i < n; i++)
	{
		b[i].Re = creal(a[i]);
		b[i].Im = cimag(a[i]);
	}

	return b;
}

void diff_space_to_ref(uint64_t n, double complex *a, complex_struct *b)
{
	double max_diff = 0, avg_diff = 0;
	for (size_t i = 0; i < n; i++)
	{
		double diff = cabs(b[i].Re + I * b[i].Im - a[i]);
		if (!(diff <= max_diff))
			max_diff = diff;
		avg_diff += diff / n;
	}

	printf("Over %lu elements, max diff %g avg diff %g\n", n, max_diff, avg_diff);
}


void diff_space(uint64_t n, double complex *a, double complex *b)
{
	double max_diff = 0, avg_diff = 0;
	for (size_t i = 0; i < n; i++)
	{
		double diff = cabs(b[i] - a[i]);
		if (!(diff <= max_diff))
			max_diff = diff;
		avg_diff += diff;
	}

	avg_diff /= n;

	printf("Over %lu elements, max diff %g avg diff %g\n", n, max_diff, avg_diff);
	if (avg_diff <= 1e-8) {
		printf("Solution validates\n");
	} else {
		printf("Solution differs from reference\n");
	}
}


double complex * get_random_matrix(uint64_t n, double seed)
{
	double complex *M = CATCHROI_INSTRUMENT(malloc)(n * sizeof(*M));

	for (uint64_t i = 0; i < n; i++)
		M[i] = lehmer_random(&seed) + I * lehmer_random(&seed);

	return M;
}


void write_numpy_matrix(char *filename, ssize_t mat_elems, double complex *M)
{
	ssize_t mat_size = mat_elems * sizeof(*M);
	int tmp = mkstemp(filename);

	if (tmp < 0 || write(tmp, M, mat_size) != mat_size)
		err(1, "Error writing array for numpy");

	close(tmp);
}


void usage(int exitval, char *argv0)
{
	printf("Usage: %s [-p] [-r] [[-d|-c] file] nx [ny [nz]]\n"
	       "Where the size of the d-dimension matrix to transform is nx * ny * nz. Need to be powers of 2.\n"
		   "  -p   : check results using python (numpy.fft.ffn())\n"
		   "  -r   : check results using ref\n"
		   "  -d   : dump results to file\n"
		   "  -c   : check results from file\n",
	       argv0);
	exit(exitval);
}

int main(int argc, char *argv[])
{
	int numpy_check = 0, ref_check = 0, ndims = 0, opt;
	double seed = 564321;
	char *fpath = NULL, check = 0;
	char numpy_mat_before[] = "fft_numpy.XXXXXX", numpy_mat_after[] = "fft_numpy.XXXXXX";

	while ((opt = getopt(argc, argv, "rps:d:c:")) != -1)
	{
		if (opt == 'p')
			numpy_check = 1;
		else if (opt == 'r')
			ref_check = 1;
		else if (opt == 's')
			seed = strtod(optarg, NULL);
		else if (opt == 'c' || opt == 'd')
		{
			fpath = strdup(optarg);
			check = opt == 'c' ? 1 : 2;
		}
	}

	ndims = argc - optind;
	if (ndims < 1 || ndims > 3)
		usage(0, argv[0]);


	uint64_t dims[ndims], total = 1;
	for (int i = 0; i < ndims; i++)
	{
		dims[i] = strtoull(argv[i + optind], NULL, 0);
		if (!dims[i] || dims[i] != (1ULL << (ffs(dims[i]) - 1)))
			usage(1, argv[0]);
		total *= dims[i];
	}


	complex double *space = get_random_matrix(total, seed);
	complex_struct *ref_space = NULL;

	int fd;
	if (numpy_check)
	{
		fd = open_numpype();

		dprintf(fd, "import numpy as np\ndims = (%lu", dims[0]);
		for (int d = 1; d < ndims; d++)
			dprintf(fd, ", %lu", dims[d]);
		dprintf(fd, ")\n");

		write_numpy_matrix(numpy_mat_before, total, space);
		dprintf(fd, "a = np.memmap('%s', dtype=np.complex128, mode='r+', shape=dims, order = 'C')\n", numpy_mat_before);
	}

	if (ref_check)
		ref_space = copy_space_to_ref(total, space);


	if (ndims == 3)
		fft3D(dims[0], dims[1], dims[2], (complex double (*)[dims[1]][dims[2]])space, 1);
	if (ndims == 2)
		fft2D(dims[0], dims[1], (complex double (*)[dims[1]])space, 1);
	if (ndims == 1)
		fft(dims[0], space, 1);


	if (ref_check)
	{
		if (ndims == 3)
			ref_fft3D(ref_space, dims[0], dims[1], dims[2], 1);
		if (ndims == 2)
			ref_fft2D(ref_space, dims[0], dims[1], 1);
		if (ndims == 1)
			ref_fft(ref_space, dims[0], 1);
		diff_space_to_ref(total, space, ref_space);
	}

	if (numpy_check)
	{
		write_numpy_matrix(numpy_mat_after, total, space);
		dprintf(fd, "b = np.memmap('%s', dtype=np.complex128, mode='r+', shape=dims, order = 'C')\n", numpy_mat_after);
		dprintf(fd, "print(\"max error is\", np.absolute(np.fft.fftn(a) - b).max())\n");

		close(fd);
		wait(NULL);

		unlink(numpy_mat_before);
		unlink(numpy_mat_after);
	}

	if (check == 2)
	{
		printf("Writing the output to %s\n", fpath);
		int fd = open(fpath, O_CREAT | O_TRUNC | O_WRONLY, 0644);
		uint64_t long_dims = ndims;
		write(fd, &long_dims, sizeof(long_dims));
		write(fd, dims, ndims * sizeof(*dims));
		write(fd, space, total * sizeof(*space));
		close(fd);
	}
	else if (check == 1)
	{
		printf("Checking the output against %s\n", fpath);
		struct stat st;
		stat(fpath, &st);
		int fd = open(fpath, O_RDONLY);
		if (!st.st_size || fd < 0)
			err(1, "Error opening verification file");

		struct { uint64_t ndims, dims[]; } *s = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

		if ((void*)s == MAP_FAILED || (uint64_t)ndims != s->ndims || memcmp(s->dims, dims, ndims * sizeof(*dims)))
			errx(1, "Wrong parameters in verification file");

		complex double *data = (complex double*)(s->dims + s->ndims);
		diff_space(total, space, data);
	}

	free(space);
	free(ref_space);

	return 0;
}
