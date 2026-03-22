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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int p = 0; p < n; p++) {

        /* diagonal element */
        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;
        double inv_diag = 1.0 / diag;  /* multiply is cheaper than divide */

        /* update row to right of diagonal */
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] *= inv_diag;
        }

        /* update column below diagonal */
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] *= inv_diag;
        }

        /* update submatrix — i outer, j inner for row-major cache access */
        for (int i = p + 1; i < n; i++) {
            double c_ip = c[i*n + p];          /* hoisted loop invariant */
            for (int j = p + 1; j < n; j++) {
                c[i*n + j] -= c_ip * c[p*n + j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) * 1e-9;

    return elapsed;
}
