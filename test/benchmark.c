#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Benchmark program: runs cholesky() on the spec's corr() matrix
 * at several sizes and prints timing + log|C| for verification.
 *
 * Compile against EITHER cholesky_baseline.c or cholesky_opt1.c
 * to compare:
 *
 *   gcc -O2 -o test/bench_baseline test/benchmark.c src/cholesky_baseline.c -lm
 *   gcc -O2 -o test/bench_opt1     test/benchmark.c src/cholesky_opt1.c     -lm
 *
 *   ./test/bench_baseline
 *   ./test/bench_opt1
 */

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

int main(void)
{
    int sizes[] = {100, 500, 1000, 2000, 3000};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%8s %12s %16s\n", "n", "time (s)", "log|C|");
    printf("----------------------------------------\n");

    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        double *c = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        if (!c) { fprintf(stderr, "alloc failed n=%d\n", n); continue; }

        fill_spd_matrix(c, n);
        double t = cholesky(c, n);
        double logdet = log_determinant(c, n);

        printf("%8d %12.6f %16.6f\n", n, t, logdet);

        free(c);
    }

    return 0;
}
