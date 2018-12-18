### Symmetric Matrix Inversion ####
Code from Guillermo Miranda (guillermo.miranda@bsc.es)

#### Issues ####
None.

#### Compilation Instructions ####
The prerequisites for this benchmark are GCC/ICC, OmpSs and MKL.
To satisfy these requirements on MareNostrum for example, run:

"module load gcc"
"module load ompss"
"module load MKL"


Run "make" to generate a standard OmpSs binary for the benchmark
Run "make instr" to generate an instrumented OmpSs binary for the benchmark

#### Execution Instructions ####

Usage: ./inverse size block_size check_result matrix_file

size: Matrix dimension (N)
block_size: Block dimension for parallel decomposition
check_result: 0 or 1 - Whether or not to verify the result
matrix_file: An existing valid matrix input, or a nonexisting filename for a self constructed randomly generated matrix

#### Example ####
./inverse 2048 256 0 ./matrix.txt

#### Example Output ####
Running with 2 nodes
Matrix file not found, initializing matrix with random values
============ CHOLESKY RESULTS ============
  matrix size:          2048x2048
  block size:           256x256
  time (s):             0.043290
  performance (gflops): 198.524597
==========================================

#### Additional Information ####
Code is OmpSs socket-aware scheduler enabled. To use this instead of the default scheduler, set NX_SCHEDULE=socket before invoking the benchmark.
