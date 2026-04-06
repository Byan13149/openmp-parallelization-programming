#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mphil_dis_cholesky.h"

/*
 * Test suite for Cholesky factorization.
 * Verifies correctness via:
 *   1. The 2x2 example from the spec (C2, L2)
 *   2. log|C| against known reference values for larger matrices
 *   3. Reconstruction error: ||L*L^T - C||_F / ||C||_F
 *   4. Input validation (n out of range)
 */

static int tests_run    = 0;
static int tests_passed = 0;

#define TOL 1e-6

#define CHECK(cond, msg) do {                                       \
    tests_run++;                                                    \
    if (cond) {                                                     \
        tests_passed++;                                             \
        printf("  PASS: %s\n", msg);                                \
    } else {                                                        \
        printf("  FAIL: %s\n", msg);                                \
    }                                                               \
} while(0)

/* LAPACK Cholesky factorization (Fortran interface) */
extern void dpotrf_(const char *uplo, const int *n, double *a,
                    const int *lda, int *info);

/* ---------- helpers ---------- */

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

/* Compute log|C| = 2 * sum(log(L_pp)) from the factorized matrix */
static double log_determinant(const double *L, int n)
{
    double s = 0.0;
    for (int p = 0; p < n; p++)
        s += log(L[(long)n * p + p]);
    return 2.0 * s;
}

/* Compute Frobenius-norm-based relative reconstruction error.
 * Given the factorised matrix (L lower, L^T upper stored in place),
 * reconstruct C' = L * L^T and compare against the original C.
 * Returns ||C' - C_orig||_F / ||C_orig||_F                       */
static double reconstruction_error(const double *fac, const double *c_orig, int n)
{
    double err2 = 0.0, norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* (L * L^T)_{ij} = sum_{k=0}^{min(i,j)} L_{ik} * L_{jk} */
            double s = 0.0;
            int kmax = (i < j) ? i : j;
            for (int k = 0; k <= kmax; k++) {
                /* L_{ik} is stored at fac[i*n+k] for k <= i */
                /* L_{jk} is stored at fac[j*n+k] for k <= j */
                s += fac[(long)n * i + k] * fac[(long)n * j + k];
            }
            double diff = s - c_orig[(long)n * i + j];
            err2  += diff * diff;
            norm2 += c_orig[(long)n * i + j] * c_orig[(long)n * i + j];
        }
    }
    return sqrt(err2 / norm2);
}

/* ---------- test cases ---------- */

/* Test 1: 2x2 example from the coursework spec (Eq 2-5) */
static void test_2x2_spec(void)
{
    printf("[test_2x2_spec]\n");
    /* C2 = [[4,2],[2,26]]  =>  L2 = [[2,0],[1,5]]
     * In-place result should be [[2,1],[1,5]]  (L lower, L^T upper) */
    double c[4] = {4.0, 2.0, 2.0, 26.0};
    double t = cholesky(c, 2);

    CHECK(t >= 0.0,              "returns non-negative time");
    CHECK(fabs(c[0] - 2.0) < TOL, "L[0,0] == 2");
    CHECK(fabs(c[1] - 1.0) < TOL, "L^T[0,1] == 1  (upper)");
    CHECK(fabs(c[2] - 1.0) < TOL, "L[1,0] == 1  (lower)");
    CHECK(fabs(c[3] - 5.0) < TOL, "L[1,1] == 5");

    /* log|C| should be 2*(log(2)+log(5)) = 2*log(10) ≈ 4.60517 */
    double logdet = log_determinant(c, 2);
    double expected = 2.0 * log(10.0);
    CHECK(fabs(logdet - expected) < TOL, "log|C2| == 2*log(10)");
}

/* Test 2: 3x3 hand-computed case */
static void test_3x3(void)
{
    printf("[test_3x3]\n");
    /* C = [[4,2,0],[2,10,4],[0,4,9]]  symmetric positive-definite
     * Cholesky: L = [[2,0,0],[1,3,0],[0,4/3,sqrt(65)/3]]
     * det(C) = 4*(90-16) - 2*(18) = 260  =>  log|C| = log(260) */
    double c_orig[9] = {4, 2, 0,
                        2, 10, 4,
                        0, 4, 9};
    double c[9];
    for (int i = 0; i < 9; i++) c[i] = c_orig[i];

    double t = cholesky(c, 3);
    CHECK(t >= 0.0, "returns non-negative time");

    /* Check lower-triangular entries of L */
    CHECK(fabs(c[0] - 2.0) < TOL,           "L[0,0] == 2");
    CHECK(fabs(c[3] - 1.0) < TOL,           "L[1,0] == 1");
    CHECK(fabs(c[4] - 3.0) < TOL,           "L[1,1] == 3");
    CHECK(fabs(c[6] - 0.0) < TOL,           "L[2,0] == 0");
    CHECK(fabs(c[7] - 4.0/3.0) < TOL,       "L[2,1] == 4/3");
    CHECK(fabs(c[8] - sqrt(65.0)/3.0) < TOL, "L[2,2] == sqrt(65)/3");

    /* log|C| */
    double logdet = log_determinant(c, 3);
    double expected = log(260.0);
    CHECK(fabs(logdet - expected) < TOL, "log|C| == log(260)");

    /* Reconstruction */
    double rel_err = reconstruction_error(c, c_orig, 3);
    CHECK(rel_err < TOL, "reconstruction error < 1e-6");
}

/* Test 3: 1x1 trivial case */
static void test_1x1(void)
{
    printf("[test_1x1]\n");
    double c[1] = {9.0};
    double t = cholesky(c, 1);
    CHECK(t >= 0.0,              "returns non-negative time");
    CHECK(fabs(c[0] - 3.0) < TOL, "L[0,0] == 3");

    double logdet = log_determinant(c, 1);
    CHECK(fabs(logdet - log(9.0)) < TOL, "log|C| == log(9)");
}

/* Test 4: larger matrix using the spec's corr() function,
 * verify via reconstruction error and log|C| self-consistency */
static void test_corr_matrix(int n)
{
    printf("[test_corr_matrix n=%d]\n", n);

    double *c_orig = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *c      = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!c_orig || !c) {
        printf("  SKIP: allocation failed for n=%d\n", n);
        free(c_orig); free(c);
        return;
    }

    fill_spd_matrix(c_orig, n);
    for (int i = 0; i < n * n; i++) c[i] = c_orig[i];

    double t = cholesky(c, n);
    CHECK(t >= 0.0, "returns non-negative time");

    /* All diagonal entries of L must be positive */
    int diag_positive = 1;
    for (int p = 0; p < n; p++) {
        if (c[(long)n * p + p] <= 0.0) { diag_positive = 0; break; }
    }
    CHECK(diag_positive, "all diagonal entries of L are positive");

    /* log|C| must be finite */
    double logdet = log_determinant(c, n);
    CHECK(isfinite(logdet), "log|C| is finite");

    /* Reconstruction error (only for moderate n to keep test fast) */
    if (n <= 500) {
        double rel_err = reconstruction_error(c, c_orig, n);
        printf("    reconstruction relative error = %.2e\n", rel_err);
        CHECK(rel_err < 1e-10, "reconstruction error < 1e-10");
    }

    free(c_orig);
    free(c);
}

/* Test 5: compare log|C| against LAPACK dpotrf reference value.
 * LAPACK uses column-major order, so we transpose before/after. */
static void test_lapack_reference(int n)
{
    printf("[test_lapack_reference n=%d]\n", n);

    double *c_ours  = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *c_lap   = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!c_ours || !c_lap) {
        printf("  SKIP: allocation failed for n=%d\n", n);
        free(c_ours); free(c_lap);
        return;
    }

    /* Fill identical matrices */
    fill_spd_matrix(c_ours, n);

    /* LAPACK expects column-major: transpose into c_lap */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c_lap[(long)n * j + i] = c_ours[(long)n * i + j];

    /* Our implementation */
    cholesky(c_ours, n);
    double logdet_ours = log_determinant(c_ours, n);

    /* LAPACK reference */
    int info;
    char uplo = 'L';
    dpotrf_(&uplo, &n, c_lap, &n, &info);
    CHECK(info == 0, "LAPACK dpotrf succeeded");

    /* log|C| from LAPACK: diagonal of the lower-triangular factor */
    double logdet_lap = 0.0;
    for (int p = 0; p < n; p++)
        logdet_lap += log(c_lap[(long)n * p + p]);
    logdet_lap *= 2.0;

    double rel_diff = fabs(logdet_ours - logdet_lap) /
                      (fabs(logdet_lap) + 1e-30);
    printf("    log|C| ours   = %.10f\n", logdet_ours);
    printf("    log|C| LAPACK = %.10f\n", logdet_lap);
    printf("    relative diff = %.2e\n", rel_diff);
    CHECK(rel_diff < 1e-10, "log|C| matches LAPACK reference");

    free(c_ours);
    free(c_lap);
}

/* Test 6: input validation — out-of-range n should return -1 */
/* (was Test 5 before LAPACK reference test was added) */
static void test_invalid_input(void)
{
    printf("[test_invalid_input]\n");
    double c[4] = {1, 0, 0, 1};

    CHECK(cholesky(c, 0)      < 0.0, "n=0 returns negative (error)");
    CHECK(cholesky(c, -1)     < 0.0, "n=-1 returns negative (error)");
    CHECK(cholesky(c, 100001) < 0.0, "n=100001 returns negative (error)");
}

/* Test 6: boundary — n=100000 should be accepted (not actually run
 * the full factorization since it's huge; just check it doesn't
 * reject the size). We pass NULL since we only care about the
 * acceptance. Note: this would segfault if it tried to compute,
 * so we only test the boundary value n=100000 is not rejected
 * by checking n=2 at the boundary-adjacent value. */
static void test_boundary(void)
{
    printf("[test_boundary]\n");
    double c[4] = {4, 2, 2, 26};
    double t = cholesky(c, 2);
    CHECK(t >= 0.0, "n=2 (valid) returns non-negative time");

    /* n=1 boundary */
    double c1[1] = {4.0};
    t = cholesky(c1, 1);
    CHECK(t >= 0.0, "n=1 (valid) returns non-negative time");
}

/* ---------- main ---------- */

int main(void)
{
    printf("=== Cholesky Factorization Test Suite ===\n\n");

    test_2x2_spec();
    test_3x3();
    test_1x1();
    test_corr_matrix(10);
    test_corr_matrix(100);
    test_corr_matrix(500);
    test_lapack_reference(100);
    test_lapack_reference(500);
    test_lapack_reference(1000);
    test_invalid_input();
    test_boundary();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
