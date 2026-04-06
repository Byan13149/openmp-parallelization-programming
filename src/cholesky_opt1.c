#include <math.h>
#include <time.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Single-threaded Optimization (opt1):
 *   1. Swap loop order in submatrix update: i outer, j inner
 *      so inner loop accesses c[i*n+j] with stride 1
 *   2. Hoist c[i*n+p] out of the inner j loop
 *   3. Replace division by diag with multiplication by 1/diag
 */
double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }
    const long N = n;  /* use long stride to avoid int overflow for n > 46340 */

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int p = 0; p < n; p++) {

        /* diagonal element */
        double diag = sqrt(c[p*N + p]);
        c[p*N + p] = diag;
        double inv_diag = 1.0 / diag;  /* multiply is cheaper than divide */

        /* update row to right of diagonal */
        for (int j = p + 1; j < n; j++) {
            c[p*N + j] *= inv_diag;
        }

        /* update column below diagonal */
        for (int i = p + 1; i < n; i++) {
            c[i*N + p] *= inv_diag;
        }

        /* update submatrix — i outer, j inner for row-major cache access */
        for (int i = p + 1; i < n; i++) {
            double c_ip = c[i*N + p];          /* hoisted loop invariant */
            for (int j = p + 1; j < n; j++) {
                c[i*N + j] -= c_ip * c[p*N + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) * 1e-9;

    return elapsed;
}
