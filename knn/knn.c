#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <err.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include "catchroi.h"

/*
 * Size threshold for array chunks.  Training and test arrays will not fit in
 * the localsotre, training information but it generally won't fit in the local
 * store; therefore, they will have to be splitted in smaller pieces.  This
 * macro controls the size threshold of the array blocks.
 *
 * Use only power of two values.
 */
#define BLOCK_SIZE  32768
//#define BLOCK_SIZE  8192 /* SMP version */

#define debug_log(...) do{ if(0) fprintf(stderr, __VA_ARGS__); } while(0)



static inline
float dist_sq(const int dim, float a[dim], float b[dim])
{
	float d, s = 0;
	for (int i = 0; i < dim; i++)
	{
		d = a[i] - b[i];
		s += d * d;
	}
	return s;
}


void insert_smaller(const int K, int label, float distance, float *local_dist, int nearest[K])
{
	// we known local_dist[0] > distance: shift all dist/records down one slot until we find where to insert
	int rr;
	for (rr = 1; rr < K && !(local_dist[rr] < distance); rr++)
	{
		local_dist[rr - 1] = local_dist[rr];
		nearest[rr - 1] = nearest[rr];
	}
	local_dist[rr - 1] = distance;
	nearest[rr - 1] = label;
}


void insert_new(const int K, int label, float distance, float local_dist[K], int nearest[K])
{
	// shift smaller items up until we find where to insert
	// NB. This evicts the smallest item which is "empty", as we known there are empty spots
	int rr;
	for (rr = K - 2; rr >= 0 && !(local_dist[rr] > distance); rr--)
	{
		local_dist[rr + 1] = local_dist[rr];
		nearest[rr + 1] = nearest[rr];
	}
	local_dist[rr + 1] = distance;
	nearest[rr + 1] = label;
}


/******************************************************************************
 *                                                                            *
 * Find the classes of the K nearest neighbours.							  *
 *                                                                            *
 ******************************************************************************/
static inline
void compare(const int K, const int DIMENSIONS, float train[DIMENSIONS], int label, float record[DIMENSIONS], int nearest[K], float neighbor_distances[K], int *neighbor_count)
{
	float distance = dist_sq(DIMENSIONS, record, train);

	// If we do not already have K items, add it
	if (*neighbor_count < K)
		insert_new(++*neighbor_count, label, distance, neighbor_distances, nearest);

	// The new item is smaller than the biggest in the local_distance stack
	else if (!(neighbor_distances[0] <= distance))
		insert_smaller(K, label, distance, neighbor_distances, nearest);
}


/******************************************************************************
 *                                                                            *
 * Assign the class of each test point as the most frequent among the K       *
 * nearest neighbours.                                                        *
 *                                                                            *
 ******************************************************************************/
static inline
int get_best_classlabels(const int K, const int CLASSES, int nearest[K])
{
	int classcounts[CLASSES];
	memset(classcounts, 0, sizeof(classcounts));

	/* Count nearest neighbour classes */
	for (int j = 0; j < K; j++)
		classcounts[nearest[j]]++;

	/* Select the most popular one */
	int max_class_id   = -1;
	int max_classcount =  0;
	for (int j = 0; j < CLASSES; j++)
		if (classcounts[j] > max_classcount)
		{
			max_class_id   = j;
			max_classcount = classcounts[j];
		}

	return max_class_id;
}



/******************************************************************************
 *                                                                            *
 * Make a timer snapshot and, optionally, print the elapsed time since the    *
 * last time the function was called.                                         *
 *                                                                            *
 ******************************************************************************/
long time_int(int print)
{
	static struct timeval last_t = {0};
	struct timeval t;

	if (gettimeofday(&t, NULL) == -1)
		err(9, "failed calling gettimeofday");

	long us = (t.tv_sec  - last_t.tv_sec) * 1000000 + (t.tv_usec - last_t.tv_usec);

	if (print)
		printf("Time spent [%.2fs] \n", us / 1e6);

	last_t = t;
	return us;
}



/******************************************************************************
 *                                                                            *
 * Utilities to load and save configurations and results from a file.         *
 *                                                                            *
 ******************************************************************************/
typedef struct _config
{
	int train_size, points_size, dimension, neighbors, classes, seed;
} config_t;


void save_to_file(config_t *conf, int *classification, char *fname)
{
	int fd = open(fname, O_WRONLY | O_CREAT | O_TRUNC);
	if (fd < 0)
		err(1, "Impossible to open file to save config and results %s", fname);

	write(fd, conf, sizeof(*conf));
	write(fd, classification, conf->points_size * sizeof(*classification));

	close(fd);
}


// stores conf struct and results in a single file
int *load_from_file(config_t *conf, char *fname)
{
	int fd = open(fname, O_RDONLY);
	if (fd < 0)
		err(1, "Impossible to load config from file %s", fname);

	read(fd, conf, sizeof(*conf));

	int *check_classes = malloc(conf->points_size * sizeof(*check_classes));
	read(fd, check_classes, conf->points_size * sizeof(*check_classes));

	close(fd);

	return check_classes;
}


void load_from_args(config_t *conf, char *args[])
{
	conf->train_size  = atoi(args[0]);
	conf->points_size = atoi(args[1]);
	conf->dimension   = atoi(args[2]);
	conf->neighbors   = atoi(args[3]);
	conf->classes     = atoi(args[4]);
	conf->seed        = time(NULL);
}


/******************************************************************************
 *                                                                            *
 * Main function.                                                             *
 *                                                                            *
 ******************************************************************************/
int main(int argc, char *argv[])
{
	config_t conf = {0};
	int *check = NULL;
	char *outfile = argc == 7 ? argv[6] : NULL;
	if (argc == 6 || argc == 7)
		load_from_args(&conf, argv + 1);
	else if (argc == 2)
		check = load_from_file(&conf, argv[1]);
	else
		errx(2, "Usage: %s <labeled_pts> <points_to_label> <dim> <k_neighbors> <N_number_of_classes> [output_file]\nor %s input_file\n"
		     "eg: %s 200000 5000 48 30 20\n", argv[0], argv[0], argv[0]);


	const int TRAINING_RECORDS = conf.train_size;
	const int TEST_RECORDS     = conf.points_size;
	const int DIMENSIONS       = conf.dimension;
	const int K                = conf.neighbors;
	const int CLASSES          = conf.classes;

	/* Compute training and test block sizes to friendly values: smallest power of 2 count of blocks to fill BLOCK_SIZE space */
	size_t test_record_size    = (DIMENSIONS + K) * sizeof(float) + (K + 1) * sizeof(int);
	size_t train_record_size   = DIMENSIONS * sizeof(float) + sizeof(int);
	size_t test_block_records  = (TEST_RECORDS + 7) / 8;
	size_t train_block_records = (TRAINING_RECORDS + 7) / 8;
	size_t test_block_size     = test_record_size * test_block_records;
	size_t train_block_size    = train_record_size * train_block_records;


	printf("***********************************************\n");
	printf("***********************************************\n");
	printf("******                                   ******\n");
	printf("******         CellSs KNN DRIVER         ******\n");
	printf("******  BARCELONA SUPERCOMPUTING CENTER  ******\n");
	printf("******                                   ******\n");
	printf("******     ORIGINAL CBE VERSION FROM     ******\n");
	printf("******       OHIO STATE UNIVERSITY       ******\n");
	printf("******                                   ******\n");
	printf("***********************************************\n");
	printf("***********************************************\n\n");
	printf("BLOCK_SIZE is %d\n",                   BLOCK_SIZE);
	printf("Number of labeled samples is %d\n",    TRAINING_RECORDS);
	printf("Number of points to label is %d\n",    TEST_RECORDS);
	printf("Number of dimensions is %d\n",         DIMENSIONS);
	printf("Number of neighbors (k) is %d\n",      K);
	printf("Number of classes is %d\n\n",          CLASSES);
	printf("Training record size is %zd\n",        train_record_size);
	printf("Training block size is %zd\n",         train_block_size);
	printf("Training block contains %zd records\n",train_block_records);
	printf("Test record size is %zd\n",            test_record_size);
	printf("Test block size is %zd\n",             test_block_size);
	printf("Test block contains %zd records\n\n",  test_block_records);

	size_t localstore_usage = TRAINING_RECORDS * (DIMENSIONS * sizeof(float) + sizeof(int))	/* training data */
							+ TEST_RECORDS * DIMENSIONS * sizeof(float)						/* points */
							+ TEST_RECORDS * K * (sizeof(float) + sizeof(int))				/* neighbors */
							+ TEST_RECORDS * sizeof(float) + TEST_RECORDS * sizeof(int);	/* points' classes and neighbor counts */

	printf("Predicted localstore usage is %zd bytes\n\n", localstore_usage);

	/* Allocate space for data arrays */
	float (*training_points)[DIMENSIONS] = CATCHROI_INSTRUMENT(aligned_alloc)(128, TRAINING_RECORDS * sizeof(*training_points));
	int    *training_classes             = CATCHROI_INSTRUMENT(aligned_alloc)(128, TRAINING_RECORDS * sizeof(*training_classes));
	float (*test_points)[DIMENSIONS]     = CATCHROI_INSTRUMENT(aligned_alloc)(128, TEST_RECORDS * sizeof(*test_points));
	int   (*neighbor_classes)[K]         = CATCHROI_INSTRUMENT(aligned_alloc)(128, TEST_RECORDS * sizeof(*neighbor_classes));
	float (*neighbor_distances)[K]       = CATCHROI_INSTRUMENT(aligned_alloc)(128, TEST_RECORDS * sizeof(*neighbor_distances));
	int    *classification               = CATCHROI_INSTRUMENT(aligned_alloc)(128, TEST_RECORDS * sizeof(*classification));
	int    *neighbor_count               = CATCHROI_INSTRUMENT(aligned_alloc)(128, TEST_RECORDS * sizeof(*neighbor_count));

	memset(neighbor_distances, 0, TEST_RECORDS * sizeof(*neighbor_distances));
	memset(neighbor_count,     0, TEST_RECORDS * sizeof(*neighbor_count));

	srand(conf.seed);
	time_int(0);

	/* Make training data: random labeled samples and coordinates */
	for (int i = 0; i < TRAINING_RECORDS; i++)
	{
		for (int j = 0; j < DIMENSIONS; j++)
			training_points[i][j] = rand() / 10000000.0;

		training_classes[i] = rand() % CLASSES; /* Set class */
	}

	/* Make unknown data: random points */
	for (int i = 0; i < TEST_RECORDS; i++)
	{
		for (int j = 0; j < DIMENSIONS; j++)
			test_points[i][j] = rand() / 10000000.0;
		for (int j = 0; j < K; j++)
			neighbor_classes[i][j] = -1;
	}

	time_int(1);

	/* Done setting up, work starts here */

	time_int(0);
	start_roi();

	/* Launch the distance calculation tasks */
	for (int j = 0; j < TRAINING_RECORDS; j += train_block_records)
	{
		int train_until = j + train_block_records - 1;
		if (train_until >= TRAINING_RECORDS)
			train_until =  TRAINING_RECORDS - 1;

		#pragma omp taskloop nogroup grainsize(test_block_records) \
				in(training_points[j:train_until], training_classes[j:train_until]) \
				inout(test_points[i;test_block_records], neighbor_classes[i;test_block_records], neighbor_distances[i;test_block_records], neighbor_count[i;test_block_records])
		for (int i = 0; i < TEST_RECORDS; i++)
			for (int jj = j; jj <= train_until; jj++)
				compare(K, DIMENSIONS, training_points[jj], training_classes[jj], test_points[i], neighbor_classes[i], neighbor_distances[i], neighbor_count + i);
	}
	/* And now the class selection tasks */
	#pragma omp taskloop grainsize(test_block_records) in(neighbor_classes[i;test_block_records]) out(classification[i;test_block_records])
	for (int i = 0; i < TEST_RECORDS; i++)
		classification[i] = get_best_classlabels(K, CLASSES, neighbor_classes[i]);
	#pragma omp taskwait
	stop_roi(-1);
	long elapsed_usec = time_int(1);


	if (check)
	{
		int wrong = 0;
		for (int i = 0; i < TEST_RECORDS; i++)
			wrong += (check[i] != classification[i]);
		printf("Verification: %d / %d points classified differently\n", wrong, TEST_RECORDS);
		free(check);
	}
	else
	{
		printf("sample of the computed class labels ... \n");
		for (int i = 1; i < TEST_RECORDS; i++)
		{
			if (i % 50 == 0)
				printf("rec %6d = cls %3d    ", i, classification[i]);
			if (i % 200 == 0)
				printf("\n");
		}
		printf("\n\n");
	}

	if (outfile)
		save_to_file(&conf, classification, outfile);

	free(training_points);
	free(training_classes);
	free(test_points);
	free(neighbor_classes);
	free(classification);
	free(neighbor_count);
	free(neighbor_distances);

	printf("par_sec_time_us:%ld\n", elapsed_usec);

	return 0;
}
