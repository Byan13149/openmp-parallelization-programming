#include <math.h>
#include <time.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Baseline single-threaded Cholesky factorization
 * In-place: overwrites c with L (lower) and L^T (upper)
 * Returns wall-clock time in seconds
 */
double cholesky(double *c, int n)
{
    if (n <= 0 || n > 100000) {
        return -1.0;
    }
    const long N = n;  /* use long stride to avoid int overflow for n > 46340 */

    // start timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int p = 0; p < n; p++) {

        // diagonal element
        double diag = sqrt(c[p*N + p]);
        c[p*N + p] = diag;

        // update row to the right of diagonal
        for (int j = p + 1; j < n; j++) {
            c[p*N + j] /= diag;
        }

        // update column below diagonal
        for (int i = p + 1; i < n; i++) {
            c[i*N + p] /= diag;
        }

        // update submatrix
        for (int j = p + 1; j < n; j++) {
            for (int i = p + 1; i < n; i++) {
                c[i*N + j] -= c[i*N + p] * c[p*N + j];
            }
        }
    }

    // end timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) * 1e-9;

    return elapsed;
}