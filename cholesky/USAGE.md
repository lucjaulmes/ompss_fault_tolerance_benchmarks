# Cholesky Factorization

Author: Unknown (modified BAR version)

added by Dimitrios Chasapis <dimitrios.chasapis@bsc.es>

## Dependencies
Requires some version of blas and lapack, e.g. mkl, atlas etc

## Compilation instructions
make

## Running instructions and inputs

./cholesky size block_size check_result matrix_file num_iterations

e.g. ./cholesky 8192 512 0 file 10

## Additional information

- Tested with the following (Dimitrios):
  	* Mercurium 2.0 (backend gcc 4.9.3) and Nanox++ 0.7a

