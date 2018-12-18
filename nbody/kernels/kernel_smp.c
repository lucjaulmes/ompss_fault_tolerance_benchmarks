#include "nbody.h"
#include <math.h>
#include <stdbool.h>
#include <strings.h>
#include <string.h>

static const float gravitational_constant =  6.6726e-11; /* N(m/kg)2 */

#if TASKLOOP
	/* OmpSs task-loops */
	#define OMP_LOOP(deps, args) DO_PRAGMA(omp taskloop grainsize(GS) nogroup deps args)
	#define SIMD_LOOP(args)
#elif defined(_OMPSS)
	/* OmpSs for-loops, no SIMD and no "parallel" in loop */
	#define OMP_LOOP(deps, args) DO_PRAGMA(omp for args)
	#define SIMD_LOOP(args)
#else
	/* OpenMP task-loops, explicit parallel sections for for loops and simd pragma */
	#define OMP_LOOP(deps, args) DO_PRAGMA(omp parallel for args)
	#define SIMD_LOOP(args) DO_PRAGMA(simd args)
#endif


/* define (non-zero) values to get the #ifs to work */
#define N2 1
#define NlogN 2
#define N 3
#define VALUE(X) X

void calculate_local_forces(coord_ptr_t forces, coord_ptr_t positions, float_ptr_t masses, int n_particles)
{
	//pragma omp task in([n_blocks]local_masses, [n_blocks]local_positions) out([n_blocks]forces) label(local_forces)
	OMP_LOOP(in(ALL_BLOCKS(masses, n_particles, GS), ALL_BLOCKS(positions, n_particles, GS)) out(forces[i; GS]), label(local_forces_loop))
	for (int i = 0; i < n_particles; i++)
	{
		float fx = 0, fy = 0, fz = 0;
		const float px = positions[i].x, py = positions[i].y, pz = positions[i].z, mult = (masses[i] * gravitational_constant);

		#if VALUE(BIGO) == N2
		for (int j = 0; j < n_particles; j++)
		#elif VALUE(BIGO) == NlogN
		for (int j = 0; j < ffs(n_particles) - 1; j++)
		#elif VALUE(BIGO) == N
		int j = i + 1;
		if (j < n_particles)
		#else
#error unknown value of BIGO
		#endif
		{
			const float diff_x = positions[j].x - px;
			const float diff_y = positions[j].y - py;
			const float diff_z = positions[j].z - pz;

			const float distance_squared = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			const float distance = sqrtf(distance_squared); // (Optimization) Intel compiler detects and emits rsqrt for this sqrt

			float force = masses[j] / (distance_squared * distance) * mult;

			//This "if" MUST be after calculating force
			if (distance_squared == 0.0f) force = 0.0f;
			fx += force * diff_x;
			fy += force * diff_y;
			fz += force * diff_z;
		}
		// NB overwrite previous iterations' values
		forces[i] = (coord_t) {fx, fy, fz};
	}
}

void calculate_forces(coord_ptr_t forces, coord_ptr_t positions1, float_ptr_t masses1, coord_ptr_t positions2, float_ptr_t masses2, int n_particles)
{
	//pragma omp task in([n_blocks]local_masses, [n_blocks]local_positions, [n_blocks]local_masses, [n_blocks]remote_positions) inout([n_blocks]forces) label(remote_forces)
	OMP_LOOP(in(masses1[i; GS], positions1[i; GS], ALL_BLOCKS(masses2, n_particles, GS), ALL_BLOCKS(positions2, n_particles, GS)) out(forces[i; GS]), label(forces_loop))
	for (int i = 0; i < n_particles; i++)
	{
		float fx = 0, fy = 0, fz = 0;
		const float px = positions1[i].x, py = positions1[i].y, pz = positions1[i].z, mult = (masses1[i] * gravitational_constant);

		#if VALUE(BIGO) == N2
		for (int j = 0; j < n_particles; j++)
		#elif VALUE(BIGO) == NlogN
		for (int j = 0; j < ffs(n_particles) - 1; j++)
		#elif VALUE(BIGO) == N
		int j = i + 1;
		if (j < n_particles)
		#else
#error unknown value of BIGO
		#endif
		{
			const float diff_x = positions2[j].x - px;
			const float diff_y = positions2[j].y - py;
			const float diff_z = positions2[j].z - pz;

			// (Optimization) Intel compiler detects and emits rsqrt for this sqrt
			const float distance_squared = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			const float distance = sqrtf(distance_squared);

			const float force = masses2[j] / (distance_squared * distance) * mult;

			fx += force * diff_x;
			fy += force * diff_y;
			fz += force * diff_z;
		}
		// NB add to current iterations' values
		forces[i].x += fx;
		forces[i].y += fy;
		forces[i].z += fz;
	}
}


static inline
void update_part(coord_t *position, coord_t *velocity, const float_t mass, const coord_t *force, const float time_interval)
{
	const float time_by_mass       = time_interval / mass;
	const float half_time_interval = time_interval / 2;

	const float velocity_change_x = force->x * time_by_mass;
	const float velocity_change_y = force->y * time_by_mass;
	const float velocity_change_z = force->z * time_by_mass;
	const float position_change_x = velocity->x + velocity_change_x * half_time_interval;
	const float position_change_y = velocity->y + velocity_change_y * half_time_interval;
	const float position_change_z = velocity->z + velocity_change_z * half_time_interval;

	velocity->x += velocity_change_x;
	velocity->y += velocity_change_y;
	velocity->z += velocity_change_z;
	position->x += position_change_x;
	position->y += position_change_y;
	position->z += position_change_z;
}


void update_particles(coord_ptr_t positions, coord_ptr_t velocities, coord_ptr_t forces, float_ptr_t masses, const int n_particles, const float time_interval)
{
	//pragma omp task inout([n_blocks]local_positions, [n_blocks]velocities) in([n_blocks]local_masses, [n_blocks]forces) label(update_particles)
	OMP_LOOP(inout(positions[i; GS], velocities[i; GS]) in(masses[i; GS], forces[i; GS]), label(update_loop))
	for (int i = 0; i < n_particles; i++)
		update_part(positions + i, velocities + i, masses[i], forces + i, time_interval);
}
