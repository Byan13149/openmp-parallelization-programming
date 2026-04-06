#include <math.h>
#include "../include/mphil_dis_cholesky.h"

#ifdef _OPENMP
  #include <omp.h>
#else
  #include <time.h>
#endif

/*
 * OpenMP-parallel Cholesky v3 (omp3): Symmetric blocked Cholesky + OpenMP
 *
 * Builds on opt3 (lower-triangle-only Phase C, transpose-copy Phase B).
 *
 * Parallelization:
 *   - Phase A: column scaling parallelized over i (same as omp2).
 *   - Phase B: transpose copy parallelized over k (each row independent).
 *   - Phase C: parallelized over i with schedule(guided). Guided scheduling
 *     is critical here: the lower-triangle loop means row i does work
 *     proportional to (i - pp - pb), so static scheduling would leave the
 *     last thread with ~44% of the work vs the ideal 25%. Guided assigns
 *     larger chunks first (covering heavy rows) and smaller chunks later,
 *     naturally balancing the triangular workload.
 */

#define NB 64

static inline int min_int(int a, int b) { return a < b ? a : b; }

double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }
    const long N = n;  /* use long stride to avoid int overflow for n > 46340 */

#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    struct timespec _ts_start, _ts_end;
    clock_gettime(CLOCK_MONOTONIC, &_ts_start);
#endif

    #pragma omp parallel
    {
        for (int pp = 0; pp < n; pp += NB) {
            int pb = min_int(NB, n - pp);

            /* === Mirror: sync diagonal block's upper triangle === */
            #pragma omp single
            {
                if (pp > 0) {
                    for (int i = pp + 1; i < pp + pb; i++)
                        for (int j = pp; j < i; j++)
                            c[j*N + i] = c[i*N + j];
                }
            }

            /* === Phase A: Factor panel === */
            for (int k = 0; k < pb; k++) {
                int pk = pp + k;

                #pragma omp single
                {
                    double diag = sqrt(c[pk*N + pk]);
                    c[pk*N + pk] = diag;
                }

                double inv = 1.0 / c[pk*N + pk];

                #pragma omp single
                {
                    for (int j = pk + 1; j < pp + pb; j++)
                        c[pk*N + j] *= inv;
                }

                #pragma omp for schedule(static)
                for (int i = pk + 1; i < n; i++)
                    c[i*N + pk] *= inv;

                #pragma omp single
                {
                    for (int i = pk + 1; i < pp + pb; i++) {
                        double c_ip = c[i*N + pk];
                        for (int j = pk + 1; j < pp + pb; j++)
                            c[i*N + j] -= c_ip * c[pk*N + j];
                    }
                }

                #pragma omp for schedule(static)
                for (int i = pp + pb; i < n; i++) {
                    double c_ip = c[i*N + pk];
                    for (int j = pk + 1; j < pp + pb; j++)
                        c[i*N + j] -= c_ip * c[pk*N + j];
                }
            }

            /* === Phase B: Transpose copy (parallelized over rows k) === */
            #pragma omp for schedule(static)
            for (int k = pp; k < pp + pb; k++)
                for (int j = pp + pb; j < n; j++)
                    c[k*N + j] = c[j*N + k];

            /* === Phase C: Lower-triangle-only trailing update ===
             * schedule(guided): balances triangular workload across threads.
             * Row i does (i - pp - pb + 1) iterations in the j-loop,
             * so later rows have more work. Guided assigns decreasing
             * chunk sizes, giving heavy rows to threads that finish first. */
            #pragma omp for schedule(guided)
            for (int i = pp + pb; i < n; i++) {
                for (int k = pp; k < pp + pb; k++) {
                    double c_ik = c[i*N + k];
                    for (int j = pp + pb; j <= i; j++)
                        c[i*N + j] -= c_ik * c[k*N + j];
                }
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
