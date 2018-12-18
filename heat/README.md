# Heat (C++)

Author: Isaac Sánchez Barrera <isaac.sanchez@bsc.es>


## Issues

Using the default Nanos++ scheduler (BF) with version 0.11 (or newer)
of the runtime can kill the app due to SIGABRT or SIGSEGV. The error
is unrelated to the program itself (happens with simple dummy programs
too) and devs at PM are aware of the issue.


## Compilation instructions

Just `make`.

There are multiple `heat-*` binaries, using `make all` you can compile
all of them (by default, just `heat` and `heat-cd`). The meanings of
the suffixes are

- `-cd`: copy_deps
- `-noinit`: data is not initialised in tasks
- `-i`: instrumentation
- `-d`: debug
- `-seq`: sequential


## Running instructions

`./heat <num-blocks> <input-file>`


## Inputs If needed, how to generate/obtain and use.

See `inputs/test.dat`, `inpus/test-jacobi.dat` and
`inputs/test-rb.dat`. Self-explanatory.

At the end of every line of the input file there can be a comment (but
comments cannot be on lines of their own).

Added `inputs/gem5-test.dat` for gem5 simulations with a reasonably large
memory footprint (~1GB) and managable simulation time.


## Additional information

- Tested with the following (Isaac):
  * Mercurium 2.0.0 (abc233dd, 14th Oct 2016) and Nanox++ 0.10
    (3550efdd, 16th Jun 2016). Backend GCC 5.1.0 (MareNostrum).
- The code is NUMA-aware for two sockets when using Nanos++ with
  `--schedule=socket --no-socket-auto-detect`.

### Output using `--summary`

For `test.dat` on a Dell laptop using `heat`:

```
MSG: [?] ========== Nanos++ Initial Environment Summary ==========
MSG: [?] === PID:                 19080
MSG: [?] === Num. worker threads: 4
MSG: [?] === System CPUs:         active[ 0-3 ] - inactive[  ]
MSG: [?] === Binding:             true
MSG: [?] === Prog. Model:         OmpSs
MSG: [?] === Priorities:          Not needed / disabled
MSG: [?] === Plugin:              SMP PE Plugin
MSG: [?] ===  | PEs:              4
MSG: [?] ===  | Worker Threads:   4
MSG: [?] =========================================================

[application output]

MSG: [?] ============ Nanos++ Final Execution Summary ============
MSG: [?] === Application ended in 2 seconds
MSG: [?] === 80088 tasks have been executed
MSG: [?] =========================================================
```


For `test.dat` on a Dell laptop using `heat-cd`:

```
MSG: [?] ========== Nanos++ Initial Environment Summary ==========
MSG: [?] === PID:                 19174
MSG: [?] === Num. worker threads: 4
MSG: [?] === System CPUs:         active[ 0-3 ] - inactive[  ]
MSG: [?] === Binding:             true
MSG: [?] === Prog. Model:         OmpSs
MSG: [?] === Priorities:          Not needed / disabled
MSG: [?] === Plugin:              SMP PE Plugin
MSG: [?] ===  | PEs:              4
MSG: [?] ===  | Worker Threads:   4
MSG: [?] =========================================================

[application output]

MSG: [?] ============ Nanos++ Final Execution Summary ============
MSG: [?] === Application ended in 4 seconds
MSG: [?] === 80088 tasks have been executed
MSG: [?] =========================================================
```
