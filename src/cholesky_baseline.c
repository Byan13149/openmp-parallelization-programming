#include <math.h>
#include <time.h>
#include "mphil_dis_cholesky.h"

/*
 * Baseline single-threaded Cholesky factorization
 * In-place: overwrites c with L (lower) and L^T (upper)
 * Returns wall-clock time in seconds
 */
double cholesky(double *c, int n)
{
    // start timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int p = 0; p < n; p++) {

        // diagonal element
        double diag = sqrt(c[p*n + p]);
        c[p*n + p] = diag;

        // update row to the right of diagonal
        for (int j = p + 1; j < n; j++) {
            c[p*n + j] /= diag;
        }

        // update column below diagonal
        for (int i = p + 1; i < n; i++) {
            c[i*n + p] /= diag;
        }

        // update submatrix
        for (int j = p + 1; j < n; j++) {
            for (int i = p + 1; i < n; i++) {
                c[i*n + j] -= c[i*n + p] * c[p*n + j];
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