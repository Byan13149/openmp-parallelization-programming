#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mphil_dis_cholesky.h"

/* Generate a symmetric positive-definite matrix using the corr function from the spec */
static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

static void fill_spd_matrix(double *c, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[n * i + j] = corr(i, j, n);
        }
        c[n * i + i] = 1.0;
    }
}

/* Compute log|C| = 2 * sum(log(L_pp)) from the factorized matrix */
static double log_determinant(const double *c, int n)
{
    double logdet = 0.0;
    for (int p = 0; p < n; p++) {
        logdet += log(c[n * p + p]);
    }
    return 2.0 * logdet;
}

int main(int argc, char *argv[])
{
    int n = 1000;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n <= 0 || n > 100000) {
        fprintf(stderr, "Error: n must be in [1, 100000], got %d\n", n);
        return 1;
    }

    printf("Cholesky factorization: n = %d\n", n);

    double *c = (double *)malloc((size_t)n * n * sizeof(double));
    if (!c) {
        fprintf(stderr, "Error: failed to allocate %zu bytes\n",
                (size_t)n * n * sizeof(double));
        return 1;
    }

    fill_spd_matrix(c, n);

    double elapsed = cholesky(c, n);

    double logdet = log_determinant(c, n);

    printf("Time:    %.6f s\n", elapsed);
    printf("log|C|:  %.10f\n", logdet);

    /* Quick sanity check with the 2x2 example from the spec */
    if (n == 2) {
        printf("L[0,0]=%.4f  L[0,1]=%.4f\n", c[0], c[1]);
        printf("L[1,0]=%.4f  L[1,1]=%.4f\n", c[2], c[3]);
    }

    free(c);
    return 0;
}
