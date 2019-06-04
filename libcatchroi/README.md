# Library to let benchmarks' region of intereset be easily instrumented

This library offers three main functionalities:

1. An external (thus interposable) call for the start and end of the Region of Interest (ROI) of benchmarks.
	- The execution time of the benchmark (and optionally its number of iterations) are printed to stdout by these functions.
	- The function names are aliased to the standard parsec start and end of ROI function names.
	- The [`nx_catch_tdg`](../nx_catch_tdg/README.md) library can also report each tasks start and end.
		- The breakdown of task counts during and outside the ROI is printed at the end of any linked program's execution.

2. A mechanism to instrument C memory allocation calls:
	- either selected manually, by wrapping the function in a macro, e.g. `void *ptr = CATCHROI_INSTRUMENT(malloc)(size);`
	- or instrumenting all calls in a file simply by including `catchroi.h` with `CATCHROI_OVERRIDE_NAMES` defined.

3. Some error injection mechanisms, optionally compiled, and controlled through the `INJECT` environment variable.
