/*
 * Benchmark program for a single implementation.
 * Prints CSV lines: impl,n,time,logdet,logdet_lapack
 *
 * Usage: ./bench_all <impl_name>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mphil_dis_cholesky.h"

extern void dpotrf_(const char *uplo, const int *n, double *a,
                    const int *lda, int *info);

static double corr(double x, double y, double s)
{
    return 0.99 * exp(-0.5 * 16.0 * (x - y) * (x - y) / s / s);
}

static void fill_spd_matrix(double *c, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            c[(long)n * i + j] = corr(i, j, n);
        c[(long)n * i + i] = 1.0;
    }
}

static double log_determinant(const double *c, int n)
{
    double s = 0.0;
    for (int p = 0; p < n; p++)
        s += log(c[(long)n * p + p]);
    return 2.0 * s;
}

/* Compute LAPACK reference log|C| using dpotrf */
static double lapack_logdet(int n)
{
    double *a = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!a) return 0.0;

    /* Fill in column-major order for LAPACK */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[(long)n * j + i] = corr(i, j, n);
    for (int i = 0; i < n; i++)
        a[(long)n * i + i] = 1.0;

    int info;
    char uplo = 'L';
    dpotrf_(&uplo, &n, a, &n, &info);

    double s = 0.0;
    if (info == 0)
        for (int p = 0; p < n; p++)
            s += log(a[(long)n * p + p]);

    free(a);
    return 2.0 * s;
}

int main(int argc, char *argv[])
{
    const char *impl = (argc > 1) ? argv[1] : "unknown";

    int sizes[] = {100, 200, 500, 1000, 2000, 3000, 5000, 10000};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        double *c = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        if (!c) { fprintf(stderr, "alloc failed n=%d\n", n); continue; }

        fill_spd_matrix(c, n);
        double t = cholesky(c, n);
        double logdet = log_determinant(c, n);
        double logdet_ref = lapack_logdet(n);

        fprintf(stderr, "  n=%5d  time=%.4fs  logdet=%.6f\n", n, t, logdet);
        printf("%s,%d,%.6f,%.20e,%.20e\n", impl, n, t, logdet, logdet_ref);
        fflush(stdout);

        free(c);
    }

    return 0;
}
