# Cholesky Factorization Optimization

Cholesky factorization of symmetric positive-definite matrices, parallelized with OpenMP.

## Building

Requires a C compiler with OpenMP support and `-lm`.

```
make IMPL=omp3          # build library, example, tests, benchmark
make IMPL=omp3 test     # build and run tests (requires LAPACK)
make IMPL=omp3 bench    # build and run benchmark for one implementation
make bench-all          # benchmark all implementations, output to test/bench_results.csv
make plot               # run bench-all then generate plots (requires Python3 + matplotlib)
make clean              # remove build artifacts
```

Available implementations: `baseline`, `opt1`, `opt2`, `opt3`, `omp1`, `omp2`, `omp3`.

The library is built as a static archive at `build/libcholesky_<IMPL>.a`.

## Usage

Include the header and link against the library:

```c
#include "mphil_dis_cholesky.h"

double elapsed = cholesky(c, n);
```

- `c`: pointer to `n*n` doubles in row-major order, representing a symmetric positive-definite matrix. On return, the lower triangle contains L and the upper triangle contains L^T, where C = L L^T.
- `n`: matrix dimension (must satisfy 1 <= n <= 100,000).
- Returns wall-clock time in seconds, or a negative value if `n` is out of range.

To factorize your own symmetric positive-definite matrix, pass it as a row-major `double*` array. See `example/demo.c` for a minimal example:

```c
#include "mphil_dis_cholesky.h"

double c[25] = {
    12,  3,  3,  2, -1,
     3, 12,  4,  4,  2,
     3,  4, 16,  5,  2,
     2,  4,  5, 12,  4,
    -1,  2,  2,  4, 17
};
double elapsed = cholesky(c, 5);
// c now contains L (lower triangle) and L^T (upper triangle)
```

To try it quickly, paste your SPD matrix into `example/demo.c` and run:

```
make IMPL=omp3
gcc -O3 -fopenmp -Iinclude example/demo.c -Lbuild -lcholesky_omp3 -lm -o build/demo
./build/demo
```

Or compile your own code against the library:

```
make IMPL=omp3
gcc -O3 -fopenmp -Iinclude your_code.c -Lbuild -lcholesky_omp3 -lm -o your_program
```

## Example program

`example/main.c` demonstrates how to use the library. It generates a symmetric positive-definite matrix, runs the factorization, and prints the elapsed time and log-determinant.

```
make IMPL=omp3                                    # build everything
OMP_NUM_THREADS=64 ./build/cholesky_example 5000   # factorize a 5000x5000 matrix
```

Sample output:
```
Cholesky factorization: n = 5000
Time:    0.456432 s
log|C|:  -22955.6140135157
```

The matrix size `n` is passed as a command-line argument (default: 1000).

## Performance guidance

- Use `IMPL=omp3` for best performance. It combines single-threaded loop optimizations with OpenMP parallelization of the trailing submatrix update.
- Set `OMP_NUM_THREADS` to the number of physical cores available (not hyperthreads).
- For small matrices (n < ~500), overhead from thread creation may outweigh gains. Single-threaded implementations (`opt1`-`opt3`) may be faster in that regime.
- Compile with `-O3 -march=native` (the default) to enable vectorization for your specific CPU.

## Directory structure

```
include/   header file (mphil_dis_cholesky.h)
src/       library source files
example/   example program
test/      correctness tests and benchmark
```

#### Use of generative AI
Claude (Anthropic) was used to assist with code development and report writing. All outputs were reviewed and tested by the author, who takes full responsibility for the final submission.