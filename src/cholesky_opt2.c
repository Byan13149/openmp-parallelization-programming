#include <math.h>
#include <time.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Single-threaded Optimization 2 (opt2): Blocked (panel) Cholesky
 *
 * Bottleneck in opt1:
 *   The submatrix update c[i][j] -= c[i][p]*c[p][j] is a rank-1 update
 *   applied n times (once per column p). The trailing submatrix of size
 *   ~n^2 doesn't fit in cache, so EVERY rank-1 update reads/writes it
 *   from main memory. Total memory traffic: O(n^3) doubles moved.
 *
 * Fix: blocked Cholesky processes NB columns at a time. Instead of n
 *   individual rank-1 updates, we accumulate NB of them and apply one
 *   rank-NB update (a matrix multiply: C -= A*B). The trailing matrix
 *   is accessed n/NB times instead of n times, reducing memory traffic
 *   by a factor of NB.
 *
 * Three phases per panel [pp, pp+NB):
 *   A. Factor the panel: unblocked Cholesky on the NB columns, updating
 *      the diagonal block and the column panel below it.
 *   B. Solve the upper rows: triangular solve for rows pp..pp+NB-1,
 *      columns pp+NB..n-1.
 *   C. Trailing update: rank-NB matrix multiply on the remaining submatrix.
 *      This is the dominant O(n^2 * NB) work and the cache-friendly part.
 */

#define NB 64

static inline int min_int(int a, int b) { return a < b ? a : b; }

double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }
    const long N = n;  /* use long stride to avoid int overflow for n > 46340 */

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int pp = 0; pp < n; pp += NB) {
        int pb = min_int(NB, n - pp);

        /* === Phase A: Factor panel columns pp..pp+pb-1 ===
         * Unblocked algorithm applied to the panel. Updates the diagonal
         * block AND the column panel below it (rows pp+pb..n-1), so that
         * L21 is ready for the trailing update in Phase C. */
        for (int k = 0; k < pb; k++) {
            int pk = pp + k;

            double diag = sqrt(c[pk*N + pk]);
            c[pk*N + pk] = diag;
            double inv = 1.0 / diag;

            /* Scale row within panel (upper triangle of diagonal block) */
            for (int j = pk + 1; j < pp + pb; j++)
                c[pk*N + j] *= inv;

            /* Scale column for ALL rows below pk (both in-block and below) */
            for (int i = pk + 1; i < n; i++)
                c[i*N + pk] *= inv;

            /* Rank-1 update within the diagonal block */
            for (int i = pk + 1; i < pp + pb; i++) {
                double c_ip = c[i*N + pk];
                for (int j = pk + 1; j < pp + pb; j++)
                    c[i*N + j] -= c_ip * c[pk*N + j];
            }

            /* Rank-1 update on the lower panel (rows below the block) */
            for (int i = pp + pb; i < n; i++) {
                double c_ip = c[i*N + pk];
                for (int j = pk + 1; j < pp + pb; j++)
                    c[i*N + j] -= c_ip * c[pk*N + j];
            }
        }

        /* === Phase B: Solve upper rows to the right ===
         * Triangular solve: for each row pk in the panel, compute
         * c[pk][j] for j >= pp+pb by applying updates from earlier
         * panel rows and scaling by the diagonal. */
        for (int k = 0; k < pb; k++) {
            int pk = pp + k;
            double inv = 1.0 / c[pk*N + pk];

            for (int k2 = 0; k2 < k; k2++) {
                double factor = c[pk*N + (pp + k2)];
                for (int j = pp + pb; j < n; j++)
                    c[pk*N + j] -= factor * c[(pp + k2)*N + j];
            }

            for (int j = pp + pb; j < n; j++)
                c[pk*N + j] *= inv;
        }

        /* === Phase C: Trailing submatrix update (rank-NB) ===
         * c[i][j] -= sum_{k=pp}^{pp+pb-1} c[i][k] * c[k][j]
         * for i, j >= pp+pb.
         *
         * This replaces pb individual rank-1 updates with one pass.
         * Loop order: i outer, k middle, j inner — gives stride-1
         * access on c[i*n+j] and c[k*n+j] in the inner loop. */
        for (int i = pp + pb; i < n; i++) {
            for (int k = pp; k < pp + pb; k++) {
                double c_ik = c[i*N + k];
                for (int j = pp + pb; j < n; j++)
                    c[i*N + j] -= c_ik * c[k*N + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) * 1e-9;

    return elapsed;
}
