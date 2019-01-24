## OmpSs check benchmarks

This is a collection of benchmarks using the [OmpSs programming model](https://pm.bsc.es/) that include a verification of their results, either algorithmical or comparing with a reference output.
These codes are provided without guarantee of any kind.

#### Code organisation

Each directory provides build targets `perf`, `instr`, `debug`, `seq`, `install`, `uninstall`.
The first three are parallel builds, respectively built with and linked against the `performance`, `instrumentation`, and `debug` libraries of the Nanox++ runtime. `seq` builds a sequential binary.
All builds require a built `libcatchroi` at `$CATCHROI_HOME` (defaults to `<repo_root>/libcatchroi`). `install` and `uninstall` targets respect the `$DESTDIR` variable.


#### Benchmarks used for evaluation

There is a library, `libcatchroi`, that provides timing information, which is useful to interpose the Region Of Interest (ROI) start and end calls.
It also provides some error injection capabilities. The rest of the subdirectories are benchmarks, listed below.
The "Verif." column indicates how the benchmark's output verification is performed.

Name		 | Benchmark description					| Category						| Verif    | Origin |
-------------|------------------------------------------|-------------------------------|----------|--------|
Blackscholes | Option pricing							| Partial Differential Equation	| built-in | [PARSEC benchmarks](http://parsec.cs.princeton.edu/parsec3-doc.htm)<sup>4,5</sup>
Cholesky	 | Cholesky factorization					| Dense linear algebra			| built-in | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
CG			 | Conjugate Gradient						| Sparse linear algebra			| built-in | matrices from [SuiteSparse](https://sparse.tamu.edu/)<sup>2</sup>
DGEMM		 | Matrix multiplication					| Dense linear algebra			| built-in | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
FFT			 | Fast Fourier Transform					| Spectral method				| ref. run | [Wang Jian-Sheng](https://www.physics.nus.edu.sg/~phywjs/CZ5101/cz5101.html)
Gauss-Seidel | Heat diffusion, Gauss-Seidel solver		| Structured grid				| ref. run | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
Jacobi		 | Heat diffusion, Jacobi solver			| Structured grid				| ref. run | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
KNN			 | K-nearest neighbours						| Machine learning				| ref. run | [Heterogeneous Computer Architecture (HCA) group at BSC](https://wiki.hca.bsc.es/dokuwiki/start)
K Means		 | K-means clustering						| Machine learning				| ref. run | [Heterogeneous Computer Architecture (HCA) group at BSC](https://wiki.hca.bsc.es/dokuwiki/start)
N-body		 | Astrophysical simulation					| N-body method					| built-in | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
PRK2 stencil | Parallel Research Kernels stencil		| Stencil operation				| built-in | [Parallel Research Kernels](https://github.com/ParRes/Kernels)<sup>3</sup>
Red-black	 | Heat diffusion, red-black solver			| Structured grid				| ref. run | [BSC Application Repository](https://pm.bsc.es/gitlab/benchmarks)
SMI			 | Symmetric matrix inverse					| Dense linear algebra			| built-in | Guillermo Miranda
Stream		 | Stream Triad								| Memory bandwidth benchmark	| built-in | John D. McCalpin


<small>

The BSC benchmarks repository was previously hosted at [https://pm.bsc.es/projects/bar](https://pm.bsc.es/projects/bar).
SuiteSparse was previously the *UF Sparse Matrix Collection* hosted at [https://www.cise.ufl.edu/research/sparse/matrices/list_by_id.html](https://www.cise.ufl.edu/research/sparse/matrices/list_by_id.html).

1. J. D. McCalpin, “Memory Bandwidth and Machine Balance in Current High Performance Computers,” IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, pp. 19–25, 1995.
2. T. A. Davis and Y. Hu, “The University of Florida Sparse Matrix Collection,” ACM Transactions on Mathematical Software, vol. 38, no. 1, pp. 1:1–1:25, Dec. 2011.
3. R. F. V. der Wijngaart and T. G. Mattson, “The Parallel Research Kernels,” in IEEE High Performance Extreme Computing Conference, 2014, pp. 1–6.
4. C. Bienia, “Benchmarking Modern Multiprocessors,” PhD thesis, Princeton University, 2011.
5. D. Chasapis et al., “PARSECSs: Evaluating the Impact of Task Parallelism in the PARSEC Benchmark Suite,” ACM Transactions on Architecture and Code Optimization, vol. 12, no. 4, pp. 41:1–41:22, Dec. 2015.

</small>
