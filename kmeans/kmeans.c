#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <err.h>
#include <catchroi.h>

#include "kmeans.h"

#ifndef PAGE_SIZE
	#define PAGE_SIZE sysconf(_SC_PAGESIZE)
#endif


static inline
double wtime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);

	return tv.tv_sec + 1e-6 * tv.tv_usec;
}


static inline
double distance_sq(const long dim, const double a[dim], const double b[dim])
{
	double dist = 0.0;
	for (long i = 0; i < dim; i++)
	{
		double diff = a[i] - b[i];
		dist += diff * diff;
	}

	return dist;
}


static inline
void assign_point(const int dim, const double point[dim], int point_assignment, long *num_assigned_points, double (*sum_assigned_points)[dim])
{
	for (int j = 0; j < dim; j++)
	{
		register double *sum_j = sum_assigned_points[point_assignment] + j, pj = point[j];
		#pragma omp atomic
		*sum_j += pj;
	}
	#pragma omp atomic
	num_assigned_points[point_assignment]++;
}


int assign_point_to_nearest(const int dim, const double point[dim], const int num_centres, const double (*centres)[dim], int *assignment, long *num_assigned_points, double (*sum_assigned_points)[dim])
{
	// find closest centre to point
	int best = 0;
	double best_dist = distance_sq(dim, point, centres[best]);
	for (int j = 1; j < num_centres; j++)
	{
		double current_dist = distance_sq(dim, point, centres[j]);
		if (current_dist < best_dist)
		{
			best = j;
			best_dist = current_dist;
		}
	}

	int changed = (*assignment != best);

	// transfer point to better centre
	*assignment = best;
	assign_point(dim, point, best, num_assigned_points, sum_assigned_points);

	return changed;
}


void recalculate_centres(int num_centres, int const dim, long *num_points, double (*sum_points)[dim], double (*centres)[dim])
{
	for (int i = 0; i < num_centres; i++)
	{
		if (num_points[i])
			for (int k = 0; k < dim; k++)
				centres[i][k] = sum_points[i][k] / num_points[i];

		// reset for the next reductions
		memset(sum_points[i], 0, sizeof(sum_points[i]));
		num_points[i] = 0;
	}
}


static inline
double *load_from_file(const char *infile, int64_t *p, int64_t *d)
{
	int fd = open(infile, O_RDONLY);
	if (fd < 0)
		err(1, "Could not open file %s", infile);

	read(fd, p, sizeof(*p));
	read(fd, d, sizeof(*d));

	double *points = CATCHROI_INSTRUMENT(malloc)((*p) * (*d) * sizeof(*points));
	read(fd, points, (*p) * (*d) * sizeof(*points));

	close(fd);
	fprintf(stderr, "File %s has been read\n", infile);

	return points;
}


static inline
void save_to_file(const char *outfile, int32_t ncentres, int32_t dim, int64_t n_points, double (*centres)[dim], int *assignment)
{
	int fd = open(outfile, O_WRONLY | O_CREAT | O_TRUNC);

	write(fd, &ncentres, sizeof(ncentres));
	write(fd, &dim, sizeof(dim));
	write(fd, &n_points, sizeof(n_points));
	write(fd, centres, ncentres * sizeof(*centres));
	write(fd, assignment, n_points * sizeof(int));

	close(fd);
}


static inline
void compare_with_file(const char *checkfile, int32_t ncentres, int32_t dim, int n_points, const double (*points)[dim], double (*centres)[dim], int *assignment)
{
	int fd = open(checkfile, O_RDONLY);
	struct stat sb;
	fstat(fd, &sb);
	struct RHeader *cmp = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

	if (cmp == MAP_FAILED)
		err(1, "ERROR: Can not open file %s for comparison", checkfile);

	else if (cmp->ncent != ncentres || cmp->dim != dim || cmp->num != n_points)
		errx(1, "ERROR: Can not compare with results of different configuration\n");

	// centres after header, assignments after that
	double (*cmp_centres)[dim] = (double(*)[dim])(cmp + 1);
	int *cmp_assignment = (int*)(cmp_centres + ncentres);

	// Average euclidian distance between centres
	double d = 0;
	for (int i = 0; i < ncentres; i++)
		d += sqrt(distance_sq(dim, centres[i], cmp_centres[i]));
	d /= ncentres;

	// Number of mis-attributed points
	long m = 0;
	// Also compute within-cluster sum of squares (WCSS)
	double wcss = 0, cmp_wcss = 0;
	#pragma omp taskloop reduction(+:wcss, cmp_wcss) num_tasks(10) // group => implicit taskwait
	for (int i = 0; i < n_points; i++)
	{
		m += (assignment[i] != cmp_assignment[i]);
		wcss += sqrt(distance_sq(dim, points[i], centres[assignment[i]]));
		cmp_wcss += sqrt(distance_sq(dim, points[i], cmp_centres[cmp_assignment[i]]));
	}

	munmap(cmp, sb.st_size);
	close(fd);

	printf("Average euclidian distance of centre positions to reference %g\n", d);
	printf("Number of points distributed to different centres %ld / %d\n", m, n_points);
	printf("Quality of solution: within-cluster sum of squares (WCSS) %.17e (vs for reference %.17e)\n", wcss, cmp_wcss);
	printf("Comparison: %.17e\n", wcss / cmp_wcss - 1.);
}

static inline
double wcss(int32_t dim, int n_points, const double (*points)[dim], double (*centres)[dim], int *assignment)
{
	// Compute within-cluster sum of squares (WCSS)
	double wcss = 0;
	#pragma omp taskloop reduction(+:wcss) num_tasks(10) // group => implicit taskwait
	for (int i = 0; i < n_points; i++)
		wcss += sqrt(distance_sq(dim, points[i], centres[assignment[i]]));

	printf("Quality of solution: within-cluster sum of squares (WCSS) %.17e\n", wcss);
	return wcss;
}

static inline
void recompute_centres_from_assignments(int32_t ncentres, int32_t dim, int n_points, const double (*points)[dim], double (*centres)[dim], double (*sum_assigned_points)[dim], long *num_assigned_points, int *assignment, int chunk_size)
{
	memset(centres, 0, ncentres * sizeof(centres[0]));
	memset(num_assigned_points, 0, ncentres * sizeof(num_assigned_points[0]));
	memset(sum_assigned_points, 0, ncentres * sizeof(num_assigned_points[0]));
	(void)chunk_size;

	// NB: group => implicit taskwait
	#pragma omp taskloop reduction(+: [ncentres]num_assigned_points, [ncentres]sum_assigned_points) grainsize(chunk_size)
	for (int i = 0; i < n_points; i++)
		assign_point(dim, points[i], assignment[i], num_assigned_points, sum_assigned_points);

	recalculate_centres(ncentres, dim, num_assigned_points, sum_assigned_points, centres);
}


int main(int argc, char **argv)
{
	// Parameters
	int nblocks = 0, ncentres = 0, max_its = 500;
	double threshold = 0.01, ref_wcss = -1.;
	char *infile = NULL, *outfile = NULL, *checkfile = NULL;
	long seed = 0;

	int opt, help = 0;
	while ((opt = getopt(argc, argv, "b:k:i:o:c:m:t:w:s:h")) != -1)
		switch (opt)
		{
			case 'i': infile    = optarg; break;
			case 'o': outfile   = optarg; break;
			case 'c': checkfile = optarg; break;
			case 'b': nblocks   = strtol(optarg, NULL, 10); break;
			case 'k': ncentres  = strtol(optarg, NULL, 10); break;
			case 'm': max_its   = strtol(optarg, NULL, 10); break;
			case 's': seed      = strtol(optarg, NULL, 10); break;
			case 't': threshold = strtod(optarg, NULL); break;
			case 'w': ref_wcss  = strtod(optarg, NULL); break;
			case 'h':
			default:
				if (opt != 'h') warnx("Unrecognized option %c %s", (char)opt, optarg);
				help++;
		}

	if (argc == 1 || help)
		errx(0, "Usage: %s [-i infile] [-o outfile] [-b nblocks] [-k ncenters] [-t threshold] [-m max_iterations] [-s seed]", argv[0]);

	else if (infile == NULL || nblocks == 0 || ncentres == 0)
		errx(2, "Wrong parameters: input file %s, %d blocks and %d centres.", infile, nblocks, ncentres);

	else if (seed)
		srand(seed);

	// Alloc and load from file
	int64_t np, d;
	const double *flat_points = load_from_file(infile, &np, &d);

	const int n_points = np, dim = d, chunk_size = (n_points + nblocks - 1) / nblocks;

	// Cast points now we now dim, allocate, and initialize to all-0s, except assignment to all-1s because 0 is a valid assignment
	const double (*points)[dim] = (double(*)[dim])flat_points;
	double (*centres)[dim] = CATCHROI_INSTRUMENT(calloc)(ncentres, sizeof(*centres));
	double (*sum_assigned_points)[dim] = CATCHROI_INSTRUMENT(calloc)(ncentres, sizeof(*centres));
	long *num_assigned_points = CATCHROI_INSTRUMENT(calloc)(ncentres, sizeof(*num_assigned_points));
	int *assignment = memset(CATCHROI_INSTRUMENT(malloc)(n_points * sizeof(*assignment)), ~0, n_points * sizeof(*assignment));

	// Initialize centres using random points
	for (int i = 0; i < ncentres; i++)
		memcpy(centres[i], points[rand() % n_points], sizeof(*points));

	// Actual work

	printf("Points: %d (%d dimensions)\nCenters: %d\n", n_points, dim, ncentres);
	int its = 0;
	double runtime = wtime(), delta = -1;

	start_roi();

	for (its = 0; its < max_its; its++)
	{
		long changed = 0;

		// Distribute points to closest centre
		#pragma omp taskloop grainsize(chunk_size) nogroup in([ncentres]centres, points[p;chunk_size]) inout(assignment[p;chunk_size]) \
				reduction(+:changed, [ncentres]sum_assigned_points, [ncentres]num_assigned_points)
		for (long p = 0; p < n_points; p++)
			changed += assign_point_to_nearest(dim, points[p], ncentres, centres, assignment + p, num_assigned_points, sum_assigned_points);

		// move centres that have any points to the average position of their assigned points
		#pragma omp task inout([ncentres]num_assigned_points, [ncentres]sum_assigned_points) out([ncentres]centres)
		recalculate_centres(ncentres, dim, num_assigned_points, sum_assigned_points, centres);

		// Create tasks in parallel with centres being averaged - if possible
		#pragma omp taskwait on(changed)
		delta = changed / (double)n_points;
		if (delta <= threshold)
			break;
	}

	#pragma omp taskwait

	// Done
	stop_roi(its);

	runtime = wtime() - runtime;
	printf("Iterations: %d\nDelta: %10g\nRuntime: %g\n", its, delta, runtime);

	// Save and quit

	if (outfile != NULL)
		save_to_file(outfile, ncentres, dim, n_points, centres, assignment);

	if (checkfile != NULL)
		compare_with_file(checkfile, ncentres, dim, n_points, points, centres, assignment);
	else
	{
		recompute_centres_from_assignments(ncentres, dim, n_points, points, centres, sum_assigned_points, num_assigned_points, assignment, chunk_size);
		double run_wcss = wcss(dim, n_points, points, centres, assignment);
		if (ref_wcss >= 0.)
			printf("Verification: relative increase in wcss=%g\n", run_wcss / ref_wcss - 1);
	}

	free((void*)points);
	free(centres);
	free(assignment);
	free(sum_assigned_points);
	free(num_assigned_points);
}
