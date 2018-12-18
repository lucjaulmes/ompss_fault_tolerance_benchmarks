# Application Name Here
Author: Luc Jaulmes
Original FFT code: Wang Jian-Sheng (1998, 2003)

## Issues
Current task granularity might not be optimal.

## Compilation instructions
Use any of the targets (perf, instr, debug, seq) to build the corresponding executable, e.g. `make perf` will build the default ompss executable.

## Running instructions
Pick between 1 and 3 dimensions and run: `./fft dim1 dim2 dim3`. One dimension will run a 1D fft, 2 a 2D fft, and 3 a 3D fft.
Use the `-r` flag  to check the results against the reference (sequential) implementation.
Use the `-p` flag  to check the results against `numpy.fft.fftn()`, which requires python3 with numpy installed (and probably won't work for too big input sizes).


## Inputs
Inputs are generated randomly using a Lehmer RNG. Use the `-s` flag to change the seed.

## Additional information
Example running a 2D case for a 2K x 64K grid (solve time is reported in micro seconds):
```
> NX_ARGS=--smp-workers=4\ --summary ./fft -r 2048 65536
MSG: [?] Nanos++ Initial Environment Summary
==========================================================
===================== Global Summary =====================
=== Nanos++ version:     0.15a
=== PID:                 77662
=== Num. worker threads: 4
=== System CPUs:         active[ 4-7 ] - inactive[ 0-3, 8-47 ]
=== Binding:             true
=== Prog. Model:         OmpSs
=== Priorities:          Not needed / Disabled
=== Plugin:              SMP PE Plugin
===  | PEs:              48
===  | Worker Threads:   4
==========================================================

solve_time:4563196
solve_time:23943273
Over 134217728 elements, max diff 0 avg diff 0
MSG: [?] Nanos++ Final Execution Summary
==========================================================
=== Application ended in 38 seconds
=== 110592 tasks have been executed
==========================================================
```

- The task granularity is rather small currently, as that is what I required.
  Currently in 2D the number of tasks is proportional to dim1 * log2(dim2), the task size is roughly dim2 / 2.
  This means results will scale with the 2nd dimension and overheads will increase more when changing the first dimension.
  I found 64K in dim2 to be reasonable for 4 threads.
- Task dependencies can be changed, in particular for multiple dimension cases tasks can be grouped for several rows together or nested from one dimension to the next.
- For scaling results, the baseline should probably be sequential code from fft.c as precomputing the omega values makes it much faster than the legacy code in ref.c. Maybe the code before the commit "Add pragmas and parallelize with OmpSs" is even a better sequential baseline, as it is just the legacy code modernized but not parallelized yet.
