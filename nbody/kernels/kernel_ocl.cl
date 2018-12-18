#include "nbody_types.h"

__constant single_force ZERO = { 0.0f, 0.0f, 0.0f};
__constant float gravitational_constant =  6.6726e-11; /* N(m/kg)2 */

inline single_force calculate_forces_part(__global particles_block_t * restrict part1,
                                     __global particles_block_t * restrict part2)
{

   const float diff_x = part2->position_x[0] - part1->position_x[0];
   const float diff_y = part2->position_y[0] - part1->position_y[0];
   const float diff_z = part2->position_z[0] - part1->position_z[0];

   const float distance_squared = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
   const float distance = rsqrt(distance_squared); //native_rsqrt seems to perform worse
   const float force = (part1->mass[0]*(distance*distance*distance))*(part2->mass[0])*gravitational_constant;

   const single_force ret = {force * diff_x,
                        force * diff_y,
                        force * diff_z};
   if (distance_squared == 0.0f) return  ZERO; else return ret;
   //return ret;
}

//This version doesn't calculate any usefull result, it's designed to benchmark computation vs comunication overheads
__kernel void calculate_forces_N2(__global force_block_t * restrict forces, __global particles_block_t * restrict block1, __global particles_block_t * restrict block2, cint bs)
{
     size_t i = get_global_id(0);
	if (i > bs ) return;

      force_block_t total = forces[i];

      for (int j = 0; j < bs; j++) {
         const single_force force = calculate_forces_part(block1+i, block2+j);
         total.x[0] += force.x;
         total.y[0] += force.y;
         total.z[0] += force.z;
      }
      forces[i] = total;
}

//This version doesn't calculate usefull results, it's designed to benchmark computation vs comunication impact
__kernel void calculate_forces_NlogN(__global force_block_t * restrict forces, __global particles_block_t * restrict block1, __global particles_block_t * restrict block2, cint bs)
{
     size_t i = get_global_id(0);
	if (i > bs ) return;

      force_block_t total = forces[i];
	  int logbs = log2i(bs);
      for (int j = 0; j < logbs; j++) {
         const single_force force = calculate_forces_part(block1+i, block2+i);
         total.x[0] += force.x;
         total.y[0] += force.y;
         total.z[0] += force.z;
      }
      forces[i] = total;
}

//This version doesn't calculate usefull results, it's designed to benchmark computation vs comunication impact
__kernel void calculate_forces_N(__global force_block_t * restrict forces, __global particles_block_t * restrict block1, __global particles_block_t * restrict block2, cint bs)
{
     size_t i = get_global_id(0);
	if (i > bs ) return;

      force_block_t total = forces[i];

      //for (int j = 0; j < bs; j++) {
         const single_force force = calculate_forces_part(block1+i, block2+i);
         total.x[0] += force.x;
         total.y[0] += force.y;
         total.z[0] += force.z;
      //}
      forces[i] = total;
}


/*

inline void calculate_forces_part(float pos_x1, float pos_y1, float pos_z1, float mass1,
                                  float pos_x2, float pos_y2, float pos_z2, float mass2,
				   __global float * restrict const fx,
				   __global float * restrict const fy,
				   __global float * restrict const fz){
   const float diff_x = pos_x2 - pos_x1;
   const float diff_y = pos_y2 - pos_y1;
   const float diff_z = pos_z2 - pos_z1;

   const float distance_squared = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

   const float distance = sqrt(distance_squared);

   float force = ((mass1/(distance*distance_squared))*(mass2*gravitational_constant));

   if (distance_squared == 0.0f) force = 0.0;
   *fx += force*diff_x;
   *fy += force*diff_y;
   *fz += force*diff_z;
}

inline void calculate_forces_BLOCK(__global force_block_t * restrict forces,
				   __global particles_block_t * restrict block1,
				   __global particles_block_t * restrict block2)
{
   __global float * const x      = forces->x;
   __global float * const y      = forces->y;
   __global float * const z      = forces->z;

   __global float * const pos_x1 = block1->position_x;
   __global float * const pos_y1 = block1->position_y;
   __global float * const pos_z1 = block1->position_z;
   __global float * const mass1  = block1->mass ;

   __global float * const pos_x2 = block2->position_x;
   __global float * const pos_y2 = block2->position_y;
   __global float * const pos_z2 = block2->position_z;
   __global float * const mass2  = block2->mass;


   for (int i = 0; i < BLOCK_SIZE; i++) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
        calculate_forces_part(pos_x1[i], pos_y1[i], pos_z1[i], mass1[i],
			      pos_x2[j], pos_y2[j], pos_z2[j], mass2[j],
			      x+i, y+i, z+i);
      }
   }
}


__kernel void calculate_forces_N2(__global force_block_t * restrict forces, __global particles_block_t * restrict block1, __global particles_block_t * restrict block2, cint bs)
{
//     size_t lid = get_local_id(0);
//     size_t lsize = get_local_size(0);
     size_t gid0 = get_global_id(0);
     size_t gid1 = get_global_id(1);
//     size_t gsize = get_global_size(0);
//     printf("%llu %llu %llu %llu\n", gid, gsize, lid, lsize);
      calculate_forces_BLOCK(forces+gid0, block1+gid0, block2+gid1);
}
*/

