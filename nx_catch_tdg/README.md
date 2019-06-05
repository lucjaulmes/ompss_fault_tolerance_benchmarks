# Nanox instrumenation plugin to catch scheduling events

In particular, report task creations, their dependencies, and when they start and end executing.
This info can be reported in a csv output file and/or passed to the libcatchroi library.
The csv format has a header for every column until `dependencies` which reports the number of dependencies. The following columns are the start address (in hex), size, and dependency direction for every direction.

To use:
1. Compile with make. Options that can be set:
	- `DESTDIR` should be set to the nanox lib/ (or lib64/) directory
	- `NANOX_SRC` should be set to the root directory of the nanox source code
	- `NOCATCHROI` can be set to anything non-empty to remove interactions with libcatchroi
	- `CATCHROI_HOME` can be set to the install prefix of libcatchroi (containing include/ and lib/)
	- `INSTRUMENTATION_DEBUG` can be set to 1 to build (and install) the instrumentation-debug version of the library

2. Install the instrumentation plugin:
	- either in the nanox lib/instrumentation/ directory (with `make install` and `DESTDIR` set)
	- or somewhere in your `LD_LIBRARY_PATH`

3. Run with `NX_INSTRUMENTATION=catch_tasks_deps NX_TDG_OUT=/dev/stdout <instrumentation_binary> <args>`.
	This will output the TDG in csv format to standard output, any file name can be used instead.
	Omitting NX_TDG_OUT hides the tdg output and using `-` redirects the output to standard output.
