#include <math.h>
#include "../include/mphil_dis_cholesky.h"

#ifdef _OPENMP
  #include <omp.h>            /* OpenMP header for parallel directives and omp_get_wtime() */
#else
  #include <time.h>           /* Fallback timer when compiled without OpenMP */
#endif

/*
 * First OpenMP-parallel Cholesky factorization (omp1):
 *   Built on top of opt1 (loop swap, hoisted invariant, reciprocal multiply).
 *   Parallelization strategy:
 *     - The outer p-loop is inherently sequential (each step depends on the previous).
 *     - The submatrix update (i loop) is the dominant O(n^2) work per step and
 *       each row i is independent, so we parallelize over i with OpenMP.
 *     - The row and column scaling loops are also parallelized; they are O(n)
 *       per step so the benefit is smaller, but it avoids an implicit barrier
 *       between serial and parallel regions.
 */
double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }

    /* Use omp_get_wtime() for portable wall-clock timing across threads.
     * Falls back to clock_gettime() when compiled without OpenMP. */
#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    struct timespec _ts_start, _ts_end;
    clock_gettime(CLOCK_MONOTONIC, &_ts_start);
#endif

    for (int p = 0; p < n; p++) {

        /* diagonal element — sequential, single scalar operation */
        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;
        double inv_diag = 1.0 / diag;  /* multiply is cheaper than divide */

        /* update row to right of diagonal — parallelize over j;
         * each c[p*n+j] is independent, no data races.
         * schedule(static): even work per iteration, static gives lowest overhead. */
        #pragma omp parallel for schedule(static)
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] *= inv_diag;
        }

        /* update column below diagonal — parallelize over i;
         * each c[i*n+p] is independent, no data races.
         * schedule(static): same rationale as above. */
        #pragma omp parallel for schedule(static)
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] *= inv_diag;
        }

        /* update submatrix — this is the dominant O(n^2) work per step.
         * Parallelize over i (outer loop): each row i is fully independent
         * since rows don't share any write targets.
         * schedule(static): rows have equal work (n-p-1 multiplies each),
         * so static partitioning balances the load evenly. */
        #pragma omp parallel for schedule(static)
        for (int i = p + 1; i < n; i++) {
            double c_ip = c[i*n + p];          /* hoisted loop invariant */
            for (int j = p + 1; j < n; j++) {
                c[i*n + j] -= c_ip * c[p*n + j];
            }
        }
    }

#ifdef _OPENMP
    double t_end = omp_get_wtime();
    return t_end - t_start;
#else
    clock_gettime(CLOCK_MONOTONIC, &_ts_end);
    return (_ts_end.tv_sec - _ts_start.tv_sec) +
           (_ts_end.tv_nsec - _ts_start.tv_nsec) * 1e-9;
#endif
}
