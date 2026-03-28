#include <math.h>
#include "../include/mphil_dis_cholesky.h"

#ifdef _OPENMP
  #include <omp.h>
#else
  #include <time.h>
#endif

/*
 * OpenMP-parallel Cholesky v2 (omp2): Blocked (panel) Cholesky + OpenMP
 *
 * Builds on opt2 (blocked algorithm). Parallelization targets:
 *   - Phase A (panel factor): column scaling parallelized over i.
 *   - Phase C (trailing update): the dominant work — parallelized over
 *     rows i. Each row is independent (no write conflicts).
 *   - Phase B (upper solve): O(pb^2 * m), small relative to Phase C,
 *     parallelized over j within each row.
 *
 * Single parallel region to avoid repeated fork/join overhead.
 */

#define NB 64

static inline int min_int(int a, int b) { return a < b ? a : b; }

double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }

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

            /* === Phase A: Factor panel columns pp..pp+pb-1 === */
            for (int k = 0; k < pb; k++) {
                int pk = pp + k;

                #pragma omp single
                {
                    double diag = sqrt(c[pk*n + pk]);
                    c[pk*n + pk] = diag;
                }

                double inv = 1.0 / c[pk*n + pk];

                /* Scale row within panel — small, single-threaded is fine */
                #pragma omp single
                {
                    for (int j = pk + 1; j < pp + pb; j++)
                        c[pk*n + j] *= inv;
                }

                /* Scale column below pk — parallelize over i */
                #pragma omp for schedule(static)
                for (int i = pk + 1; i < n; i++)
                    c[i*n + pk] *= inv;

                /* Rank-1 update within diagonal block — small, single-threaded */
                #pragma omp single
                {
                    for (int i = pk + 1; i < pp + pb; i++) {
                        double c_ip = c[i*n + pk];
                        for (int j = pk + 1; j < pp + pb; j++)
                            c[i*n + j] -= c_ip * c[pk*n + j];
                    }
                }

                /* Rank-1 update on lower panel — parallelize over i */
                #pragma omp for schedule(static)
                for (int i = pp + pb; i < n; i++) {
                    double c_ip = c[i*n + pk];
                    for (int j = pk + 1; j < pp + pb; j++)
                        c[i*n + j] -= c_ip * c[pk*n + j];
                }
            }

            /* === Phase B: Solve upper rows to the right === */
            /* Sequential — O(pb^2 * m), small vs Phase C's O(m^2 * pb) */
            #pragma omp single
            {
                for (int k = 0; k < pb; k++) {
                    int pk = pp + k;
                    double inv = 1.0 / c[pk*n + pk];

                    for (int k2 = 0; k2 < k; k2++) {
                        double factor = c[pk*n + (pp + k2)];
                        for (int j = pp + pb; j < n; j++)
                            c[pk*n + j] -= factor * c[(pp + k2)*n + j];
                    }

                    for (int j = pp + pb; j < n; j++)
                        c[pk*n + j] *= inv;
                }
            }

            /* === Phase C: Trailing submatrix update (rank-NB) ===
             * Parallelize over rows i — each row is independent. */
            #pragma omp for schedule(static)
            for (int i = pp + pb; i < n; i++) {
                for (int k = pp; k < pp + pb; k++) {
                    double c_ik = c[i*n + k];
                    for (int j = pp + pb; j < n; j++)
                        c[i*n + j] -= c_ik * c[k*n + j];
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
