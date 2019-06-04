# Library to allow benchmarks' region of interest be easily instrumented

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
	Options for injection errors are:
	- Type of error (mutually exclusive):
		- `-n`, `--n_bits=BITS`	Flip BITS number of bits at the targeted address.
		- `-p`, `--put=VALUE`	Insert VALUE at the targeted address (e.g. 7ff8000000000000 for a double NaN).
		- `-d`, `--due`			Simulate a DUE by recording the following access(es) to the targeted address.

	- Error target (mutually exclusive):
		- `-v`, `--vector=VECT`	Target the VECT instrumented allocation.
		- `-a`, `--page=PAGE`	Target the PAGE of instrumented allocations.

		For example, with two 2-page allocations that are instrumented by `libcatchroi`,
		`-v1`, `-a2` and `-a3` all target the second allocation.

	- Remaining options:
		- `-m`, `--mtbf=TIME`	Inject with a TIME mean time between errors.
		- `-s`, `--seed=SEED`	Initialize rng with SEED.
		- `-u`, `--undo`		Undo the error injection (flip back bits or restore value before inserting VALUE).

	The option parsing is done by optparse (from https://github.com/skeeto/optparse)

