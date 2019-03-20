// ************************************************************************
//
// miniAMR: stencil computations with boundary exchange and AMR.
//
// Copyright (2014) Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
// retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
// Questions? Contact Courtenay T. Vaughan (ctvaugh@sandia.gov)
//                    Richard F. Barrett (rfbarre@sandia.gov)
//
// ************************************************************************

#include <stddef.h>

// main.c
void print_help_message();
void allocate();
void deallocate();
int check_input();

// block.c
void split_blocks();
void consolidate_blocks();
void add_sorted_list(int, int, int);
void del_sorted_list(int, int);
int find_sorted_list(int, int);

// check_sum.c
void check_sum(int, int, double *);

// comm.c
void comm(int, int, int);
void on_proc_comm(int, int, int, int, int);
void on_proc_comm_diff(int, int, int, int, int, int, int);
void apply_bc(int, block *, int, int);

// driver.c
void driver();

// init.c
void init_globals();
void init();

// move.c
void move();
void check_objects();
int check_block(double cor[3][2]);

// new_comm.c
void comm_alt(double * east[4], int east_f,
              double * west[4], int west_f,
              double * north[4], int north_f,
              double * south[4], int south_f,
              double * up[4], int up_f,
              double * down[4], int down_f,
              double * b_array,
              int start, int number);

// plot.c
void plot(int);

// profile.c
void profile();
void calculate_results();
void init_profile();

// refine.c
void refine(int);
int refine_level();
void reset_all();

// stencil.c
void stencil_calc(int, int);
void stencil_calc_checksum(int, int, double *);

// target.c
int reduce_blocks();
void add_blocks();
void zero_refine();

// util.c
double timer();

// debug.c
void print_par();
void print_comm(int);
void print_block(int, int);
void print_blocks(int);
void print_parents(int);
