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
 *     - A single persistent parallel region wraps the entire p-loop to avoid
 *       repeated thread team creation/destruction (fork-join overhead).
 *     - The outer p-loop is inherently sequential (each step depends on the previous).
 *     - The diagonal update is done by one thread via #pragma omp single;
 *       its implicit barrier ensures all threads see the updated inv_diag.
 *     - The submatrix update (i loop) is the dominant O(n^2) work per step and
 *       each row i is independent, so we parallelize over i with #pragma omp for.
 *     - The row and column scaling loops are also parallelized; they are O(n)
 *       per step so the benefit is smaller, but it keeps threads busy.
 */
double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }
    const long N = n;  /* use long stride to avoid int overflow for n > 46340 */

    /* Use omp_get_wtime() for portable wall-clock timing across threads.
     * Falls back to clock_gettime() when compiled without OpenMP. */
#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    struct timespec _ts_start, _ts_end;
    clock_gettime(CLOCK_MONOTONIC, &_ts_start);
#endif

    double inv_diag = 0.0;  /* shared across threads; written in single, read in for */

    #pragma omp parallel    /* create thread team ONCE */
    {
        for (int p = 0; p < n; p++) {

            /* diagonal — only one thread computes this.
             * The implicit barrier after the single block ensures all threads
             * see the updated c[p*n+p] and inv_diag before proceeding. */
            #pragma omp single
            {
                double diag = sqrt(c[p*N + p]);
                c[p*N + p] = diag;
                inv_diag = 1.0 / diag;
            }

            /* update row to right of diagonal — parallelize over j;
             * each c[p*n+j] is independent, no data races.
             * schedule(static): even work per iteration, lowest overhead. */
            #pragma omp for schedule(static)
            for (int j = p + 1; j < n; j++)
                c[p*N + j] *= inv_diag;

            /* update column below diagonal — parallelize over i;
             * each c[i*n+p] is independent, no data races. */
            #pragma omp for schedule(static)
            for (int i = p + 1; i < n; i++)
                c[i*N + p] *= inv_diag;

            /* update submatrix — this is the dominant O(n^2) work per step.
             * Parallelize over i (outer loop): each row i is fully independent
             * since rows don't share any write targets.
             * schedule(static): rows have equal work (n-p-1 multiplies each),
             * so static partitioning balances the load evenly. */
            #pragma omp for schedule(static)
            for (int i = p + 1; i < n; i++) {
                double c_ip = c[i*N + p];      /* hoisted loop invariant */
                for (int j = p + 1; j < n; j++)
                    c[i*N + j] -= c_ip * c[p*N + j];
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