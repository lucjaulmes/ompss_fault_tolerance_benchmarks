# MiniAMR
MiniAMR is a proxy application for
- Original code: from Mantevo project (MiniAMR serial 1.0 version,
  https://mantevo.org/download/)
- OmpSs version: Isaac Sánchez Barrera <isaac.sanchez@bsc.es>


## Issues

- The code might not work properly with `copy_deps` due to the
  `concurrent` clauses. Nanos++ spits lots of warnings because of this
  when `copy_deps` is active, not sure of the correctness.



## Compilation instructions

Just `make`. The possible targets are
- `ompss`: it generates a parallel binary (`miniAMR.x`)
- `seq`: it generates a sequential binary (`miniAMR-seq.x`)
- `instr`: it generates an instrumented binary (`miniAMR-i.x`)


## Running instructions and inputs

See README file. It is an adaptation from the MPI reference README.

When **not** using `--original_comm`, there are less tasks and the
runtime overhead is smaller. However, without the flag, communication
tasks have multidependencies that do various calls to `malloc` and
`free`.  This might have some problems with `copy_deps`.


## Additional information

- Tested with the following (Isaac):
  * Mercurium 2.0.0 (c25519f, 30th Aug 2017) and Nanox++ 0.14a
    (fa7ea58, 2nd Aug 2017). Backend GCC 6.2.0 (Nord III).
  * Mercurium 2.0.0 (c25519f, 30th Aug 2017) and Nanox++ 0.14a
    (fa7ea58, 2nd Aug 2017). Backend GCC 7.2.0 (MareNostrum IV).

### Output using `--summary`


In Nord III, 16 SMP workers. Input:

```
./bin/miniAMR \
--num_refine 4 --max_blocks 64000 --init_x 4 --init_y 2 --init_z 2 \
--nx 8 --ny 8 --nz 8 --num_objects 2 \
--object 2 0 -1.10 -1.10 -1.10 0.03 0.03 0.03 1.5 1.5 1.5 0.0 0.0 0.0 \
--object 2 0 0.5 0.5 1.76 0.0 0.0 -0.025 0.75 0.75 0.75 0.0 0.0 0.0 \
--num_tsteps 40 --checksum_freq 4 --stages_per_ts 16 --num_vars 40 \
--merge_checksum
```

Summary:

```
MSG: [?] Nanos++ Initial Environment Summary
==========================================================
===================== Global Summary =====================
=== Nanos++ version:     0.14a
=== PID:                 20547
=== Num. worker threads: 16
=== System CPUs:         active[ 0-15 ] - inactive[  ]
=== Binding:             true
=== Prog. Model:         OmpSs
=== Priorities:          Not needed / Disabled
=== Plugin:              SMP PE Plugin
===  | PEs:              16
===  | Worker Threads:   16
==========================================================


[application output]

MSG: [?] Nanos++ Final Execution Summary
==========================================================
=== Application ended in 246 seconds
=== 13062160 tasks have been executed
==========================================================
```
