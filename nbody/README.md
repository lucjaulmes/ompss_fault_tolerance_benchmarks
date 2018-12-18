# N-body

## Introduction
An N-body simulation numerically approximates the evolution of a system of
bodies in which each body continuously interacts with every other body.
A familiar example is an astrophysical simulation in which each body
represents a galaxy or an individual star, and the bodies attract each other
through the gravitational force, as in Figure 31-1.

N-body simulation arises in many other computational science problems as well.
For example, protein folding is studied using N-body simulation to calculate
electrostatic and van der Waals forces. Turbulent fluid flow simulation and
global illumination computation in computer graphics are other examples of
problems that use N-body simulation.

## Quick start
To compile this benchmark:

  1. `cd' to the root directory containing a Makefile and a few directories.

  2. Type `source scripts/cluster/setenv.sh' to set up the necessary
environment variables. Change *cluster* by either *deep* or *marenostrum*,
depending on which system you are using.

  3. Type `make' to compile the package.

  4. By default, the compiled binaries will be placed into `bin/' directory.

## Contents

Package files are structured as follows:
```
.
├── bin/
│   ├── nbody_ompss.N2.debug.intel64
│   ├── nbody_ompss.N2.instrumentation.intel64
│   └── nbody_ompss.N2.performance.intel64
├── extrae.xml
├── include/
│   ├── nbody.h
│   └── nbody_types.h
├── kernels/
│   ├── kernel_ocl.cl
│   └── kernel_smp.c
├── Makefile
├── README.md
├── scripts/
│   ├── deep/
│   ├── local/
│   └── marenostrum/
└── src/
    ├── common.c
    ├── iomp_syms.c
    ├── main.c
    ├── nbody_deep_offload_global.c
    ├── nbody_deep_offload_partial.c
    ├── nbody_intel_offload.c
    ├── nbody_iomp.c
    ├── nbody_ompss.c
    └── nbody_ompss_opencl.c
```

* bin/ directory contains the binaries generated with Makefile.
  * *nbody_ompss.complexity.version.intel64* is the sequential version of the
application. Several binaries are generated for the same program.
Each version has different purposes:
    * debug used to locate and fix problems in the application and the runtime
library source code.
    * instrumentation used to trace application execution for analysis.
    * performance most lightweight version that is used either for production
or for execution time measurements.
In addition, the algorithmic complexity can be modified to change pressure the
application applies to the memory hierarchy. The results are not compatible
between different complexities. Available values are:
    * N
    * NlogN
    * N2

* obj/ contains the intermediate object files for each version.

* include/ contains the header files of this benchmark.

* scripts/ contains an environment setup script and a job submission script for
both *deep* and *marenostrum3* clusters.

* src/ contains the source code of this benchmark.

* extrae.xml is a configuration file used by Extrae, a tracer tool that gathers
MPI and OmpSs events and generates a trace file. This file can be later
visualized by the user for many different purposes.

## Source file structure
There are two main different groups of source files:
* Common source files: these files are used regardless which version of the
benchmark the user is going to use.
* Dedicated source files: these files are either used or not depending on which
version of the benchmark is used.
  * Kernel files: implement the algorithmic part of the application (and the
heaviest computational-wise). *OpenCL* and *C* versions are provided.
  * Solver files: implement the solver (main loop) part of the benchmark.
Which file is compiled depends on the programming model of interest.

## Programming models implemented
This package contains different implementations using different programming
models.

These include:
* MPI + OmpSs: nbody_ompss.c 

* MPI + OmpSs with OpenCL: nbody_ompss_opencl.c

* MPI + OpenMP (by default uses Intel implementation): nbody_iomp.c

* MPI + Intel OpenMP with Xeon Phi offload directives: nbody_intel_offload.c

* MPI + OmpSs offload
  * Partial offload: the solver (control) part of the application is
executed in the host side. Kernels are executed in the accelerator side:
nbody_deep_offload_partial.c
  * Global offload: both solver and kernel parts are executed on the
accelerator side: nbody_deep_offload_global.c

