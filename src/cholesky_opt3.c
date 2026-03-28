#include <math.h>
#include <time.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Single-threaded Optimization 3 (opt3): Symmetric blocked Cholesky
 *
 * Bottlenecks in opt2:
 *   1. Phase C updates BOTH triangles of the trailing submatrix, but only
 *      the lower triangle is needed — the upper is derived from it in Phase B.
 *      This wastes ~50% of the dominant Phase C FLOPs.
 *   2. Phase B is an O(NB^2 * m) triangular solve, but it just reproduces
 *      values that already exist in the lower panel: c[k][j] = c[j][k].
 *      The entire solve can be replaced by a transpose copy.
 *
 * Fix:
 *   - Phase B: replace triangular solve with copy c[k][j] = c[j][k].
 *     Eliminates O(NB^2 * m) FLOPs per panel, replaced by O(NB * m) copies.
 *   - Phase C: only update the lower triangle (j <= i).
 *     Halves the dominant O(m^2 * NB) work per panel.
 *   - Mirror: before each Phase A, copy the lower triangle of the diagonal
 *     block to its upper triangle (needed since the previous Phase C only
 *     wrote the lower triangle). This is O(NB^2) per panel — negligible.
 */

#define NB 64

static inline int min_int(int a, int b) { return a < b ? a : b; }

double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int pp = 0; pp < n; pp += NB) {
        int pb = min_int(NB, n - pp);

        /* === Mirror: sync diagonal block's upper triangle ===
         * The previous Phase C only updated the lower triangle, so upper
         * entries of this diagonal block are stale. Copy lower → upper
         * so Phase A's row scaling reads correct values.
         * O(NB^2) — negligible compared to Phase C. */
        if (pp > 0) {
            for (int i = pp + 1; i < pp + pb; i++)
                for (int j = pp; j < i; j++)
                    c[j*n + i] = c[i*n + j];
        }

        /* === Phase A: Factor panel columns pp..pp+pb-1 ===
         * Identical to opt2: unblocked Cholesky on the panel, updating
         * the diagonal block and the column panel below. */
        for (int k = 0; k < pb; k++) {
            int pk = pp + k;

            double diag = sqrt(c[pk*n + pk]);
            c[pk*n + pk] = diag;
            double inv = 1.0 / diag;

            /* Scale row within panel (upper of diagonal block) */
            for (int j = pk + 1; j < pp + pb; j++)
                c[pk*n + j] *= inv;

            /* Scale column for ALL rows below pk */
            for (int i = pk + 1; i < n; i++)
                c[i*n + pk] *= inv;

            /* Rank-1 update within diagonal block */
            for (int i = pk + 1; i < pp + pb; i++) {
                double c_ip = c[i*n + pk];
                for (int j = pk + 1; j < pp + pb; j++)
                    c[i*n + j] -= c_ip * c[pk*n + j];
            }

            /* Rank-1 update on lower panel (rows below block) */
            for (int i = pp + pb; i < n; i++) {
                double c_ip = c[i*n + pk];
                for (int j = pk + 1; j < pp + pb; j++)
                    c[i*n + j] -= c_ip * c[pk*n + j];
            }
        }

        /* === Phase B: Copy lower panel to upper (replaces triangular solve) ===
         * After Phase A, c[j][k] for j >= pp+pb contains L[j][k].
         * We need c[k][j] = L^T[k][j] = L[j][k] = c[j][k].
         * Simple transpose copy: O(NB * m) instead of O(NB^2 * m) solve. */
        for (int k = pp; k < pp + pb; k++)
            for (int j = pp + pb; j < n; j++)
                c[k*n + j] = c[j*n + k];

        /* === Phase C: Lower-triangle-only trailing update ===
         * c[i][j] -= sum_{k} c[i][k] * c[k][j]  for j <= i  (lower only).
         * Halves the FLOP count compared to opt2's full-matrix update.
         * The upper triangle is never needed directly — Phase B of the
         * next panel will derive it from the lower via copy. */
        for (int i = pp + pb; i < n; i++) {
            for (int k = pp; k < pp + pb; k++) {
                double c_ik = c[i*n + k];
                for (int j = pp + pb; j <= i; j++)
                    c[i*n + j] -= c_ik * c[k*n + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) * 1e-9;

    return elapsed;
}
